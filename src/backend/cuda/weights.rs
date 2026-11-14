use std::sync::Arc;

use super::driver::CudaDevice;
use crate::model::{config::ModelConfig, weights::ModelWeights};

pub struct CudaMatrix {
    pub d_ptr: usize,
    pub rows: usize,
    pub cols: usize,
    pub size_bytes: usize,
    pub is_q4: bool,
    pub is_ternary: bool,
    /// Ternary flavor: PTQ1_0 (28-byte/128-weight blocks) vs Q2_0 (10-byte/32-weight blocks).
    /// The `gemv_ternary` PTX kernel is PTQ1_0-only; Q2_0 matrices need `gemv_q2_0`.
    pub is_ptq1: bool,
}

pub struct CudaLayerWeights {
    pub q: CudaMatrix,
    pub k: CudaMatrix,
    pub v: CudaMatrix,
    pub o: CudaMatrix,
    pub gate: CudaMatrix,
    pub up: CudaMatrix,
    pub down: CudaMatrix,
}

pub struct CudaModelWeights {
    pub device: Arc<CudaDevice>,
    pub output: Option<CudaMatrix>,
    pub layers: Vec<CudaLayerWeights>,
    pub offloaded_layers: usize,
    /// Device pointers for per-layer RMSNorm weights (attn_norm, ffn_norm).
    /// Populated only when the whole model is offloaded (resident mode).
    pub attn_norms: Vec<usize>,
    pub ffn_norms: Vec<usize>,
    pub final_norm: usize,
    /// Per-layer contiguous [q_bias | k_bias | v_bias] device buffers.
    pub qkv_biases: Vec<usize>,
}

impl CudaModelWeights {
    pub fn upload(
        device: Arc<CudaDevice>,
        weights: &ModelWeights<'_>,
        _config: &ModelConfig,
        requested_gpu_layers: Option<usize>,
    ) -> Result<Self, String> {
        let upload_matrix = |matrix: &crate::model::weights::QuantizedMatrix<'_>| -> Result<CudaMatrix, String> {
            let is_q4 = matrix.is_q4();
            let is_ternary = matrix.is_ternary();
            let is_ptq1 = matches!(matrix, crate::model::weights::QuantizedMatrix::PTQ1(_));
            let raw_bytes = matrix.raw_bytes();
            // Repack each four Q8_0 blocks into one compact 128-weight group:
            // [4 x u16 scale][128 x i8 weight] = 136 bytes. This keeps every
            // u32 weight load aligned without the old 2-byte pad per block.
            let bytes: Vec<u8>;
            let size_bytes: usize;
            if !is_q4 && !is_ternary && matrix.cols().is_multiple_of(128) {
                let rows = matrix.rows();
                let cols = matrix.cols();
                let num_blocks = cols / 32;
                let old_stride = num_blocks * 34;
                let num_groups = cols / 128;
                let new_stride = num_groups * 136;
                let mut out = vec![0u8; rows * new_stride];
                for r in 0..rows {
                    let src = &raw_bytes[r * old_stride..(r + 1) * old_stride];
                    let dst = &mut out[r * new_stride..(r + 1) * new_stride];
                    for g in 0..num_groups {
                        let d = g * 136;
                        for b in 0..4 {
                            let s = (g * 4 + b) * 34;
                            dst[d + b * 2..d + b * 2 + 2]
                                .copy_from_slice(&src[s..s + 2]);
                            dst[d + 8 + b * 32..d + 8 + (b + 1) * 32]
                                .copy_from_slice(&src[s + 2..s + 34]);
                        }
                    }
                }
                size_bytes = out.len();
                bytes = out;
            } else {
                size_bytes = raw_bytes.len();
                bytes = raw_bytes.to_vec();
            }
            let d_ptr = device.alloc(size_bytes)?;
            device.copy_to_device(d_ptr, &bytes)?;

            Ok(CudaMatrix {
                d_ptr,
                rows: matrix.rows(),
                cols: matrix.cols(),
                size_bytes,
                is_q4,
                is_ternary,
                is_ptq1,
            })
        };

        let total_layers = weights.layers.len();
        // 80% of total GPU memory budget (safe limit ~3.2 GB on a 4 GB card)
        let safe_vram_budget = (device.total_mem_bytes as f64 * 0.80) as usize;

        // Estimate layer size
        let first_layer_bytes = if let Some(first) = weights.layers.first() {
            first.q.raw_bytes().len()
                + first.k.raw_bytes().len()
                + first.v.raw_bytes().len()
                + first.o.raw_bytes().len()
                + first.gate.raw_bytes().len()
                + first.up.raw_bytes().len()
                + first.down.raw_bytes().len()
        } else {
            1
        };

        let max_fitting_layers = safe_vram_budget / first_layer_bytes.max(1);
        let layers_to_upload = requested_gpu_layers
            .unwrap_or(max_fitting_layers)
            .min(total_layers);

        println!(
            "\x1b[36m[spark] Hybrid Layer Offloading: {}/{} layers on GPU VRAM (~{:.1} MB)\x1b[0m",
            layers_to_upload,
            total_layers,
            (layers_to_upload * first_layer_bytes) as f64 / (1024.0 * 1024.0)
        );

        let mut layers = Vec::with_capacity(layers_to_upload);
        for layer in weights.layers.iter().take(layers_to_upload) {
            layers.push(CudaLayerWeights {
                q: upload_matrix(&layer.q)?,
                k: upload_matrix(&layer.k)?,
                v: upload_matrix(&layer.v)?,
                o: upload_matrix(&layer.o)?,
                gate: upload_matrix(&layer.gate)?,
                up: upload_matrix(&layer.up)?,
                down: upload_matrix(&layer.down)?,
            });
        }

        // Only upload output projection if all layers fit or requested
        let output = if layers_to_upload == total_layers {
            Some(upload_matrix(&weights.output)?)
        } else {
            None
        };

        // Resident mode extras: norm weights and Q/K/V biases stay on the GPU
        // so a full token forward never touches host memory.
        let (attn_norms, ffn_norms, final_norm, qkv_biases) = if output.is_some() {
            let upload_f32 = |values: &[f32]| -> Result<usize, String> {
                if values.is_empty() {
                    return Ok(0);
                }
                let bytes =
                    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
                let d_ptr = device.alloc(bytes.len())?;
                device.copy_to_device(d_ptr, bytes)?;
                Ok(d_ptr)
            };

            let mut attn_norms = Vec::with_capacity(layers_to_upload);
            let mut ffn_norms = Vec::with_capacity(layers_to_upload);
            let mut qkv_biases = Vec::with_capacity(layers_to_upload);
            for layer in weights.layers.iter().take(layers_to_upload) {
                attn_norms.push(upload_f32(layer.attn_norm.values())?);
                ffn_norms.push(upload_f32(layer.ffn_norm.values())?);

                let mut bias = Vec::with_capacity(
                    layer.q.rows() + layer.k.rows() + layer.v.rows(),
                );
                let push_padded = |out: &mut Vec<f32>, bias: &crate::model::weights::F32Vector, rows: usize| {
                    let values = bias.values();
                    for i in 0..rows {
                        out.push(values.get(i).copied().unwrap_or(0.0));
                    }
                };
                push_padded(&mut bias, &layer.q_bias, layer.q.rows());
                push_padded(&mut bias, &layer.k_bias, layer.k.rows());
                push_padded(&mut bias, &layer.v_bias, layer.v.rows());
                qkv_biases.push(upload_f32(&bias)?);
            }
            let final_norm = upload_f32(weights.final_norm.values())?;
            (attn_norms, ffn_norms, final_norm, qkv_biases)
        } else {
            (Vec::new(), Vec::new(), 0, Vec::new())
        };

        device.synchronize()?;
        Ok(Self {
            device,
            output,
            layers,
            offloaded_layers: layers_to_upload,
            attn_norms,
            ffn_norms,
            final_norm,
            qkv_biases,
        })
    }
}

impl Drop for CudaModelWeights {
    fn drop(&mut self) {
        // Guard against freeing while the keepalive kernel spins (idempotent;
        // normally already stopped by CudaEngine::drop).
        self.device.stop_persistent_keepalive();
        if let Some(out) = &self.output {
            self.device.free(out.d_ptr);
        }
        for ptr in self
            .attn_norms
            .iter()
            .chain(&self.ffn_norms)
            .chain(&self.qkv_biases)
            .copied()
            .chain(std::iter::once(self.final_norm))
        {
            self.device.free(ptr);
        }
        for layer in &self.layers {
            self.device.free(layer.q.d_ptr);
            self.device.free(layer.k.d_ptr);
            self.device.free(layer.v.d_ptr);
            self.device.free(layer.o.d_ptr);
            self.device.free(layer.gate.d_ptr);
            self.device.free(layer.up.d_ptr);
            self.device.free(layer.down.d_ptr);
        }
    }
}

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
            let size_bytes = raw_bytes.len();
            let d_ptr = device.alloc(size_bytes)?;
            device.copy_to_device(d_ptr, raw_bytes)?;

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

        device.synchronize()?;
        Ok(Self {
            device,
            output,
            layers,
            offloaded_layers: layers_to_upload,
        })
    }
}

impl Drop for CudaModelWeights {
    fn drop(&mut self) {
        if let Some(out) = &self.output {
            self.device.free(out.d_ptr);
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

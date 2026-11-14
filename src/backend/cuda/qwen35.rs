use std::{
    cell::UnsafeCell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use super::driver::CudaDevice;
use crate::{
    kernel::{f16::f16_to_f32, q1_0::quantize_activations_q8_0},
    model::{
        config::ModelConfig,
        weights::{
            ModelWeights, QuantizedMatrix, Qwen35AttentionWeights, Qwen35LayerWeights,
        },
    },
};

#[derive(Debug)]
pub struct CudaQ1Matrix {
    pub d_ptr: usize,
    pub rows: usize,
    pub cols: usize,
    pub size_bytes: usize,
    pub u10: bool,
    pub scale_min: f32,
    pub scale_step: f32,
    pub tiled: bool,
}

#[derive(Debug)]
pub enum CudaQwen35Attention {
    Full {
        q: CudaQ1Matrix,
        k: CudaQ1Matrix,
        v: CudaQ1Matrix,
        output: CudaQ1Matrix,
        q_norm: usize,
        k_norm: usize,
    },
    Recurrent {
        qkv: CudaQ1Matrix,
        z: CudaQ1Matrix,
        alpha: CudaQ1Matrix,
        beta: CudaQ1Matrix,
        output: CudaQ1Matrix,
        conv1d: usize,
        dt_bias: usize,
        a: usize,
        state_norm: usize,
    },
}

#[derive(Debug)]
pub struct CudaQwen35Layer {
    pub attn_norm: usize,
    pub post_attention_norm: usize,
    pub attention: CudaQwen35Attention,
    pub gate: CudaQ1Matrix,
    pub up: CudaQ1Matrix,
    pub down: CudaQ1Matrix,
}

pub struct CudaQwen35Weights {
    device: Arc<CudaDevice>,
    arena_ptr: usize,
    pub layers: Vec<CudaQwen35Layer>,
    pub output: Option<CudaQ1Matrix>,
    pub final_norm: usize,
    pub offloaded_layers: usize,
}

impl CudaQwen35Weights {
    fn use_u10(device: &CudaDevice) -> bool {
        std::env::var_os("SPARK_Q1_U10").is_some()
            || (device.total_mem_bytes <= 4 * 1024 * 1024 * 1024
                && std::env::var_os("SPARK_Q1_EXACT_SCALES").is_none())
    }

    fn upload_f32(
        device: &CudaDevice,
        arena_ptr: usize,
        cursor: &mut usize,
        values: &[f32],
    ) -> Result<usize, String> {
        *cursor = (*cursor + 255) & !255;
        let d_ptr = arena_ptr + *cursor;
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
        };
        device.copy_to_device(d_ptr, bytes)?;
        *cursor += bytes.len();
        Ok(d_ptr)
    }

    fn upload_matrix(
        device: &CudaDevice,
        arena_ptr: usize,
        cursor: &mut usize,
        matrix: &QuantizedMatrix<'_>,
    ) -> Result<CudaQ1Matrix, String> {
        if !matrix.is_q1() {
            return Err("Qwen35 CUDA currently requires Q1_0 matrices".into());
        }
        let raw = matrix.raw_bytes();
        let use_u10 = Self::use_u10(device);
        let tiled = device.q1_tiled_enabled();
        let packed;
        let mut scale_min = 0.0f32;
        let mut scale_step = 0.0f32;
        let bytes = if use_u10 {
            let blocks_per_row = matrix.cols() / 128;
            if !blocks_per_row.is_multiple_of(4) {
                return Err("Q1 U10 rows require columns divisible by 512".into());
            }
            scale_min = f32::INFINITY;
            let mut scale_max = 0.0f32;
            for block in raw.as_chunks::<18>().0 {
                let value = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                scale_min = scale_min.min(value);
                scale_max = scale_max.max(value);
            }
            scale_step = (scale_max - scale_min) / 1023.0;
            let mut out = vec![0u8; Self::matrix_bytes(matrix, true)];
            if tiled {
                if !matrix.rows().is_multiple_of(8) {
                    return Err("Q1 U10 tiled matrices require rows divisible by 8".into());
                }
                for row_tile in 0..matrix.rows() / 8 {
                    for block in 0..blocks_per_row {
                        let dst = (row_tile * blocks_per_row + block) * 138;
                        for row in 0..8 {
                            let src = ((row_tile * 8 + row) * blocks_per_row + block) * 18;
                            let value = f16_to_f32(u16::from_le_bytes([raw[src], raw[src + 1]]));
                            let quant = if scale_step > 0.0 {
                                ((value - scale_min) / scale_step).round().clamp(0.0, 1023.0) as u16
                            } else {
                                0
                            };
                            let bit = row * 10;
                            let byte = bit / 8;
                            let shift = bit % 8;
                            let bits = (quant as u32) << shift;
                            out[dst + byte] |= bits as u8;
                            out[dst + byte + 1] |= (bits >> 8) as u8;
                            out[dst + 10 + row * 16..dst + 10 + (row + 1) * 16]
                                .copy_from_slice(&raw[src + 2..src + 18]);
                        }
                    }
                }
            } else {
                let row_bytes = (blocks_per_row / 4) * 69;
                let scale_bytes = blocks_per_row * 10 / 8;
                for row in 0..matrix.rows() {
                    let src_row = &raw[row * blocks_per_row * 18..(row + 1) * blocks_per_row * 18];
                    let dst_row = &mut out[row * row_bytes..(row + 1) * row_bytes];
                    for block in 0..blocks_per_row {
                        let src = block * 18;
                        let value = f16_to_f32(u16::from_le_bytes([src_row[src], src_row[src + 1]]));
                        let quant = if scale_step > 0.0 {
                            ((value - scale_min) / scale_step).round().clamp(0.0, 1023.0) as u16
                        } else {
                            0
                        };
                        let bit = block * 10;
                        let byte = bit / 8;
                        let shift = bit % 8;
                        let bits = (quant as u32) << shift;
                        dst_row[byte] |= bits as u8;
                        dst_row[byte + 1] |= (bits >> 8) as u8;
                        dst_row[scale_bytes + block * 16..scale_bytes + (block + 1) * 16]
                            .copy_from_slice(&src_row[src + 2..src + 18]);
                    }
                }
            }
            packed = out;
            packed.as_slice()
        } else if device.q1_tiled_enabled() {
            let blocks_per_row = matrix.cols() / 128;
            if !matrix.rows().is_multiple_of(8) {
                return Err("Q1 bit-plane matrices require rows divisible by 8".into());
            }
            let mut out = vec![0u8; raw.len()];
            for row_tile in 0..matrix.rows() / 8 {
                for block in 0..blocks_per_row {
                    let d = (row_tile * blocks_per_row + block) * 144;
                    for row in 0..8 {
                        let s = ((row_tile * 8 + row) * blocks_per_row + block) * 18;
                        out[d + row * 2..d + row * 2 + 2].copy_from_slice(&raw[s..s + 2]);
                        out[d + 16 + row * 16..d + 16 + (row + 1) * 16]
                            .copy_from_slice(&raw[s + 2..s + 18]);
                    }
                }
            }
            packed = out;
            packed.as_slice()
        } else {
            raw
        };
        *cursor = (*cursor + 255) & !255;
        let d_ptr = arena_ptr + *cursor;
        device.copy_to_device(d_ptr, bytes)?;
        *cursor += bytes.len();
        Ok(CudaQ1Matrix {
            d_ptr,
            rows: matrix.rows(),
            cols: matrix.cols(),
            size_bytes: bytes.len(),
            u10: use_u10,
            scale_min,
            scale_step,
            tiled,
        })
    }

    fn matrix_bytes(matrix: &QuantizedMatrix<'_>, u10: bool) -> usize {
        if u10 {
            matrix.rows() * (matrix.cols() / 128 / 4) * 69
        } else {
            matrix.raw_bytes().len()
        }
    }

    fn layer_bytes(layer: &Qwen35LayerWeights<'_>, u10: bool) -> usize {
        let attention = match layer.attention {
            Qwen35AttentionWeights::Full { q, k, v, output, .. } => {
                Self::matrix_bytes(&q, u10)
                    + Self::matrix_bytes(&k, u10)
                    + Self::matrix_bytes(&v, u10)
                    + Self::matrix_bytes(&output, u10)
            }
            Qwen35AttentionWeights::Recurrent { qkv, z, alpha, beta, output, .. } => {
                Self::matrix_bytes(&qkv, u10)
                    + Self::matrix_bytes(&z, u10)
                    + Self::matrix_bytes(&alpha, u10)
                    + Self::matrix_bytes(&beta, u10)
                    + Self::matrix_bytes(&output, u10)
            }
        };
        attention
            + Self::matrix_bytes(&layer.gate, u10)
            + Self::matrix_bytes(&layer.up, u10)
            + Self::matrix_bytes(&layer.down, u10)
    }

    fn layer_aux_bytes(layer: &Qwen35LayerWeights<'_>) -> usize {
        let attention = match layer.attention {
            Qwen35AttentionWeights::Full { q_norm, k_norm, .. } => {
                (q_norm.values().len() + k_norm.values().len()) * 4
            }
            Qwen35AttentionWeights::Recurrent {
                conv1d,
                dt_bias,
                a,
                state_norm,
                ..
            } => {
                (conv1d.values().len()
                    + dt_bias.values().len()
                    + a.values().len()
                    + state_norm.values().len())
                    * 4
            }
        };
        attention
            + (layer.attn_norm.values().len() + layer.post_attention_norm.values().len()) * 4
    }

    fn layer_aux_count(layer: &Qwen35LayerWeights<'_>) -> usize {
        match layer.attention {
            Qwen35AttentionWeights::Full { .. } => 4,
            Qwen35AttentionWeights::Recurrent { .. } => 6,
        }
    }

    fn upload_layer(
        device: &CudaDevice,
        arena_ptr: usize,
        cursor: &mut usize,
        layer: &Qwen35LayerWeights<'_>,
    ) -> Result<CudaQwen35Layer, String> {
        let attn_norm = Self::upload_f32(
            device,
            arena_ptr,
            cursor,
            layer.attn_norm.values(),
        )?;
        let post_attention_norm = Self::upload_f32(
            device,
            arena_ptr,
            cursor,
            layer.post_attention_norm.values(),
        )?;
        let attention = match layer.attention {
            Qwen35AttentionWeights::Full {
                q,
                k,
                v,
                output,
                q_norm,
                k_norm,
            } => CudaQwen35Attention::Full {
                q: Self::upload_matrix(device, arena_ptr, cursor, &q)?,
                k: Self::upload_matrix(device, arena_ptr, cursor, &k)?,
                v: Self::upload_matrix(device, arena_ptr, cursor, &v)?,
                output: Self::upload_matrix(device, arena_ptr, cursor, &output)?,
                q_norm: Self::upload_f32(device, arena_ptr, cursor, q_norm.values())?,
                k_norm: Self::upload_f32(device, arena_ptr, cursor, k_norm.values())?,
            },
            Qwen35AttentionWeights::Recurrent {
                qkv,
                z,
                alpha,
                beta,
                conv1d,
                dt_bias,
                a,
                state_norm,
                output,
            } => {
                CudaQwen35Attention::Recurrent {
                    qkv: Self::upload_matrix(device, arena_ptr, cursor, &qkv)?,
                    z: Self::upload_matrix(device, arena_ptr, cursor, &z)?,
                    alpha: Self::upload_matrix(device, arena_ptr, cursor, &alpha)?,
                    beta: Self::upload_matrix(device, arena_ptr, cursor, &beta)?,
                    output: Self::upload_matrix(device, arena_ptr, cursor, &output)?,
                    conv1d: Self::upload_f32(device, arena_ptr, cursor, conv1d.values())?,
                    dt_bias: Self::upload_f32(device, arena_ptr, cursor, dt_bias.values())?,
                    a: Self::upload_f32(device, arena_ptr, cursor, a.values())?,
                    state_norm: Self::upload_f32(
                        device,
                        arena_ptr,
                        cursor,
                        state_norm.values(),
                    )?,
                }
            }
        };
        Ok(CudaQwen35Layer {
            attn_norm,
            post_attention_norm,
            attention,
            gate: Self::upload_matrix(device, arena_ptr, cursor, &layer.gate)?,
            up: Self::upload_matrix(device, arena_ptr, cursor, &layer.up)?,
            down: Self::upload_matrix(device, arena_ptr, cursor, &layer.down)?,
        })
    }

    pub fn upload(
        device: Arc<CudaDevice>,
        weights: &ModelWeights<'_>,
        requested_layers: Option<usize>,
    ) -> Result<Self, String> {
        // Leave room for the desktop driver, GEMV scratch and CUDA context.
        let use_u10 = Self::use_u10(&device);
        let full_offload = std::env::var_os("SPARK_Q1_HYBRID").is_none()
            || std::env::var_os("SPARK_Q1_FULL_OFFLOAD").is_some()
            || use_u10;
        let layer_fraction = if full_offload { 0.90 } else { 0.82 };
        let budget = (device.total_mem_bytes as f64 * layer_fraction) as usize;
        let requested = requested_layers.unwrap_or(weights.qwen35_layers.len());
        let mut used = 0usize;
        let mut count = 0usize;
        for layer in weights.qwen35_layers.iter().take(requested) {
            let bytes = Self::layer_bytes(layer, use_u10);
            if used + bytes > budget {
                break;
            }
            used += bytes;
            count += 1;
        }

        println!(
            "\x1b[36m[spark] Qwen35 Q1_0 offload: {count}/{} layers ({:.1} MiB)\x1b[0m",
            weights.qwen35_layers.len(),
            used as f64 / (1024.0 * 1024.0)
        );

        // The LM head is disproportionately expensive on CPU. Permit it to
        // use a small reserve above the layer budget while retaining enough
        // VRAM for CUDA context and scratch buffers.
        let output_fraction = if full_offload { 0.95 } else { 0.86 };
        let output_budget = (device.total_mem_bytes as f64 * output_fraction) as usize;
        let output_bytes = Self::matrix_bytes(&weights.output, use_u10);
        let upload_output = used + output_bytes <= output_budget;
        let matrix_count: usize = weights
            .qwen35_layers
            .iter()
            .take(count)
            .map(|layer| match layer.attention {
                Qwen35AttentionWeights::Full { .. } => 7,
                Qwen35AttentionWeights::Recurrent { .. } => 8,
            })
            .sum::<usize>()
            + usize::from(upload_output);
        let aux_bytes: usize = weights
            .qwen35_layers
            .iter()
            .take(count)
            .map(Self::layer_aux_bytes)
            .sum::<usize>()
            + weights.final_norm.values().len() * 4;
        let aux_count: usize = weights
            .qwen35_layers
            .iter()
            .take(count)
            .map(Self::layer_aux_count)
            .sum::<usize>()
            + 1;
        let arena_bytes = used
            + if upload_output { output_bytes } else { 0 }
            + aux_bytes
            + (matrix_count + aux_count) * 255
            + 256;
        let arena_ptr = device.alloc(arena_bytes).map_err(|error| {
            format!(
                "Qwen35 weight arena allocation ({:.1} MiB) failed: {error}",
                arena_bytes as f64 / (1024.0 * 1024.0)
            )
        })?;
        let mut cursor = 0usize;
        let mut layers = Vec::with_capacity(count);
        for layer in weights.qwen35_layers.iter().take(count) {
            layers.push(Self::upload_layer(
                &device,
                arena_ptr,
                &mut cursor,
                layer,
            )?);
        }
        let output = if upload_output {
            Some(Self::upload_matrix(
                &device,
                arena_ptr,
                &mut cursor,
                &weights.output,
            )?)
        } else {
            None
        };
        let final_norm = Self::upload_f32(
            &device,
            arena_ptr,
            &mut cursor,
            weights.final_norm.values(),
        )?;
        device.synchronize()?;
        Ok(Self {
            device,
            arena_ptr,
            layers,
            output,
            final_norm,
            offloaded_layers: count,
        })
    }

}

impl Drop for CudaQwen35Weights {
    fn drop(&mut self) {
        self.device.free(self.arena_ptr);
    }
}

pub struct CudaQwen35Engine {
    pub device: Arc<CudaDevice>,
    pub weights: CudaQwen35Weights,
    pub resident: bool,
    pub context_capacity: usize,
    d_input: usize,
    d_output: usize,
    h_input: *mut u8,
    h_output: *mut f32,
    mapped_io: bool,
    input_capacity: usize,
    output_capacity: usize,
    staging: UnsafeCell<Vec<u8>>,
    host_output: UnsafeCell<Vec<f32>>,
    d_gdn_input: usize,
    d_gdn_output: usize,
    d_gdn_state: usize,
    gdn_state_bytes: usize,
    gdn_state_offsets: Vec<Option<usize>>,
    gdn_staging: UnsafeCell<Vec<f32>>,
    use_gpu_gdn: bool,
    use_bitplane: bool,
    d_resident_arena: usize,
    d_x: usize,
    d_norm: usize,
    d_q: usize,
    d_attn_gate: usize,
    d_context: usize,
    d_ffn_gate: usize,
    d_ffn_up: usize,
    d_residual_out: usize,
    d_scores: usize,
    d_cos_sin: usize,
    d_pos: usize,
    d_kv: usize,
    kv_one_side_bytes: usize,
    d_conv_state: usize,
    conv_state_bytes: usize,
    conv_state_offsets: Vec<Option<usize>>,
    attention_state_offsets: Vec<Option<usize>>,
    h_token_state: UnsafeCell<Vec<f32>>,
    h_token_out: *mut u32,
    d_token_out: usize,
    d_top64_tokens: usize,
    d_top64_logits: usize,
    h_top64_tokens: UnsafeCell<[u32; 64]>,
    h_top64_logits: UnsafeCell<[f32; 64]>,
    hidden_dim: usize,
    intermediate_dim: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_base: f32,
    rms_epsilon: f32,
    ssm_group_count: usize,
    ssm_value_heads: usize,
    ssm_head_dim: usize,
    ssm_inner_size: usize,
    ssm_conv_kernel: usize,
    graph_exec_argmax: AtomicUsize,
    graph_cap_argmax: AtomicUsize,
}

unsafe impl Send for CudaQwen35Engine {}
unsafe impl Sync for CudaQwen35Engine {}

impl CudaQwen35Engine {
    pub fn new(
        device: Arc<CudaDevice>,
        weights: CudaQwen35Weights,
        config: &ModelConfig,
    ) -> Result<Self, String> {
        let max_cols = weights
            .layers
            .iter()
            .flat_map(layer_matrices)
            .map(|m| m.cols)
            .max()
            .unwrap_or(128);
        let mut max_rows = weights.output.as_ref().map_or(1, |m| m.rows);
        for layer in &weights.layers {
            let attention_rows = match &layer.attention {
                CudaQwen35Attention::Full { q, k, v, .. } => q.rows + k.rows + v.rows,
                CudaQwen35Attention::Recurrent { qkv, z, alpha, beta, .. } => {
                    qkv.rows + z.rows + alpha.rows + beta.rows
                }
            };
            max_rows = max_rows
                .max(attention_rows)
                .max(layer.gate.rows + layer.up.rows)
                .max(layer.down.rows);
        }
        let use_u10 = weights
            .layers
            .first()
            .map(|layer| layer.gate.u10)
            .unwrap_or(false);
        let use_bitplane = device.q1_bitplane_enabled() && !use_u10;
        let input_group_bytes = if use_bitplane { 168 } else { 136 };
        let input_capacity = (max_cols / 128) * input_group_bytes;
        let output_capacity = max_rows * 4;
        let resident = weights.layers.len() == config.layers
            && weights.output.is_some()
            && !use_bitplane
            && std::env::var_os("SPARK_NO_QWEN35_RESIDENT").is_none();
        let mapped_io = !resident && std::env::var_os("SPARK_Q1_MAPPED_IO").is_some();
        let (h_input, d_input) = if mapped_io {
            device.alloc_mapped(input_capacity)?
        } else {
            (
                std::ptr::null_mut(),
                device
                    .alloc(input_capacity)
                    .map_err(|error| format!("Qwen35 input scratch allocation failed: {error}"))?,
            )
        };
        let (h_output_raw, d_output) = if mapped_io {
            device.alloc_mapped(output_capacity)?
        } else {
            (
                std::ptr::null_mut(),
                device
                    .alloc(output_capacity)
                    .map_err(|error| format!("Qwen35 output scratch allocation failed: {error}"))?,
            )
        };
        let use_gpu_gdn = resident || std::env::var_os("SPARK_GDN_CUDA").is_some();
        let mut gdn_state_offsets = vec![None; weights.layers.len()];
        let mut state_width = 0usize;
        let mut gdn_input_floats = 0usize;
        let mut gdn_output_floats = 0usize;
        let mut recurrent_count = 0usize;
        for (index, layer) in weights.layers.iter().enumerate() {
            if let CudaQwen35Attention::Recurrent { qkv, z, beta, .. } = &layer.attention {
                let value_heads = beta.rows;
                let head_dim = z.rows / value_heads;
                let key_dim = (qkv.rows - z.rows) / 2;
                state_width = value_heads * head_dim * head_dim;
                gdn_input_floats = 2 * key_dim + z.rows + 2 * value_heads;
                gdn_output_floats = z.rows;
                gdn_state_offsets[index] = Some(recurrent_count * state_width * 4);
                recurrent_count += 1;
            }
        }
        let gdn_state_bytes = if use_gpu_gdn { recurrent_count * state_width * 4 } else { 0 };
        let d_gdn_state = if gdn_state_bytes != 0 {
            let ptr = device.alloc(gdn_state_bytes).map_err(|error| {
                format!(
                    "Qwen35 recurrent state allocation ({:.1} MiB) failed: {error}",
                    gdn_state_bytes as f64 / (1024.0 * 1024.0)
                )
            })?;
            device.memset_zero(ptr, gdn_state_bytes)?;
            ptr
        } else {
            0
        };
        let d_gdn_input = if use_gpu_gdn { device.alloc(gdn_input_floats * 4)? } else { 0 };
        let d_gdn_output = if use_gpu_gdn { device.alloc(gdn_output_floats * 4)? } else { 0 };

        let requested_context = std::env::var("SPARK_QWEN35_GPU_CONTEXT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let default_context = || {
                if use_u10 {
                    1024
                } else if device.total_mem_bytes <= 4 * 1024 * 1024 * 1024 {
                    64
                } else {
                    config.context_len.min(2048)
                }
            };
        let mut context_capacity = if resident {
            requested_context
                .unwrap_or_else(default_context)
                .min(config.context_len)
                .max(1)
        } else {
            0
        };

        let mut attention_state_offsets = vec![None; weights.layers.len()];
        let mut conv_state_offsets = vec![None; weights.layers.len()];
        let mut attention_count = 0usize;
        let mut conv_count = 0usize;
        for (index, layer) in weights.layers.iter().enumerate() {
            match layer.attention {
                CudaQwen35Attention::Full { .. } => {
                    attention_count += 1;
                }
                CudaQwen35Attention::Recurrent { .. } => {
                    let width = (config.ssm_conv_kernel - 1)
                        * (config.ssm_inner_size
                            + 2 * config.ssm_group_count * config.ssm_state_size)
                        * 4;
                    conv_state_offsets[index] = Some(conv_count * width);
                    conv_count += 1;
                }
            }
        }

        let (d_kv, kv_one_side_bytes) = if resident {
            loop {
                let one_side = attention_count
                    * context_capacity
                    * config.kv_heads
                    * config.head_dim
                    * 4;
                match device.alloc(one_side * 2) {
                    Ok(ptr) => {
                        device.memset_zero(ptr, one_side * 2)?;
                        break (ptr, one_side);
                    }
                    Err(error) if requested_context.is_none() && context_capacity > 64 => {
                        context_capacity = (context_capacity / 2).max(64);
                        eprintln!(
                            "\x1b[33m[spark] reducing Qwen35 GPU context to {context_capacity} tokens ({error})\x1b[0m"
                        );
                    }
                    Err(error) => {
                        return Err(format!(
                            "Qwen35 KV cache allocation ({:.1} MiB) failed: {error}",
                            one_side as f64 * 2.0 / (1024.0 * 1024.0)
                        ));
                    }
                }
            }
        } else {
            (0, 0)
        };
        if resident {
            let width = context_capacity * config.kv_heads * config.head_dim * 4;
            let mut slot = 0usize;
            for (index, layer) in weights.layers.iter().enumerate() {
                if matches!(layer.attention, CudaQwen35Attention::Full { .. }) {
                    attention_state_offsets[index] = Some(slot * width);
                    slot += 1;
                }
            }
        }
        let conv_channels = config.ssm_inner_size
            + 2 * config.ssm_group_count * config.ssm_state_size;
        let conv_state_bytes = if resident {
            conv_count * (config.ssm_conv_kernel - 1) * conv_channels * 4
        } else {
            0
        };
        let d_conv_state = if conv_state_bytes != 0 {
            let ptr = device.alloc(conv_state_bytes).map_err(|error| {
                format!(
                    "Qwen35 convolution state allocation ({:.1} MiB) failed: {error}",
                    conv_state_bytes as f64 / (1024.0 * 1024.0)
                )
            })?;
            device.memset_zero(ptr, conv_state_bytes)?;
            ptr
        } else {
            0
        };

        let mut resident_bytes = 0usize;
        let mut reserve = |floats: usize| {
            resident_bytes = (resident_bytes + 255) & !255;
            let offset = resident_bytes;
            resident_bytes += floats * 4;
            offset
        };
        let x_offset = reserve(config.hidden_dim);
        let norm_offset = reserve(config.hidden_dim);
        let q_offset = reserve(config.heads * config.head_dim);
        let attn_gate_offset = reserve(config.heads * config.head_dim);
        let context_offset = reserve(config.ssm_inner_size.max(config.heads * config.head_dim));
        let ffn_gate_offset = reserve(config.intermediate_dim);
        let ffn_up_offset = reserve(config.intermediate_dim);
        let residual_out_offset = reserve(config.hidden_dim);
        let scores_offset = reserve(config.heads * context_capacity);
        let cos_sin_offset = reserve(config.rope_dimension_count);
        let pos_offset = reserve(1);
        let d_resident_arena = if resident {
            device.alloc(resident_bytes).map_err(|error| {
                format!(
                    "Qwen35 resident scratch allocation ({:.1} MiB) failed: {error}",
                    resident_bytes as f64 / (1024.0 * 1024.0)
                )
            })?
        } else {
            0
        };
        let resident_ptr = |offset: usize| {
            if resident { d_resident_arena + offset } else { 0 }
        };
        let d_x = resident_ptr(x_offset);
        let d_norm = resident_ptr(norm_offset);
        let d_q = resident_ptr(q_offset);
        let d_attn_gate = resident_ptr(attn_gate_offset);
        let d_context = resident_ptr(context_offset);
        let d_ffn_gate = resident_ptr(ffn_gate_offset);
        let d_ffn_up = resident_ptr(ffn_up_offset);
        let d_residual_out = resident_ptr(residual_out_offset);
        let d_scores = resident_ptr(scores_offset);
        let d_cos_sin = resident_ptr(cos_sin_offset);
        let d_pos = resident_ptr(pos_offset);
        let (h_token_out, d_token_out) = if resident {
            let (host, dev) = device.alloc_mapped(4)?;
            (host.cast::<u32>(), dev)
        } else {
            (std::ptr::null_mut(), 0)
        };
        let d_top64_tokens = if resident { device.alloc(64 * 4)? } else { 0 };
        let d_top64_logits = if resident { device.alloc(64 * 4)? } else { 0 };
        device.warmup(300);
        if resident {
            println!(
                "\x1b[32m[spark] Qwen35 resident CUDA decode active ({} token GPU cache)\x1b[0m",
                context_capacity
            );
        }
        let engine = Self {
            device,
            weights,
            resident,
            context_capacity,
            d_input,
            d_output,
            h_input,
            h_output: h_output_raw.cast::<f32>(),
            mapped_io,
            input_capacity,
            output_capacity,
            staging: UnsafeCell::new(vec![0u8; input_capacity]),
            host_output: UnsafeCell::new(vec![0.0; max_rows]),
            d_gdn_input,
            d_gdn_output,
            d_gdn_state,
            gdn_state_bytes,
            gdn_state_offsets,
            gdn_staging: UnsafeCell::new(vec![0.0; gdn_input_floats]),
            use_gpu_gdn,
            use_bitplane,
            d_resident_arena,
            d_x,
            d_norm,
            d_q,
            d_attn_gate,
            d_context,
            d_ffn_gate,
            d_ffn_up,
            d_residual_out,
            d_scores,
            d_cos_sin,
            d_pos,
            d_kv,
            kv_one_side_bytes,
            d_conv_state,
            conv_state_bytes,
            conv_state_offsets,
            attention_state_offsets,
            h_token_state: UnsafeCell::new(vec![
                0.0;
                config.hidden_dim + config.rope_dimension_count + 1
            ]),
            h_token_out,
            d_token_out,
            d_top64_tokens,
            d_top64_logits,
            h_top64_tokens: UnsafeCell::new([0; 64]),
            h_top64_logits: UnsafeCell::new([0.0; 64]),
            hidden_dim: config.hidden_dim,
            intermediate_dim: config.intermediate_dim,
            heads: config.heads,
            kv_heads: config.kv_heads,
            head_dim: config.head_dim,
            rope_dim: config.rope_dimension_count,
            rope_base: config.rope_base,
            rms_epsilon: config.rms_epsilon,
            ssm_group_count: config.ssm_group_count,
            ssm_value_heads: config.ssm_time_step_rank,
            ssm_head_dim: config.ssm_inner_size / config.ssm_time_step_rank.max(1),
            ssm_inner_size: config.ssm_inner_size,
            ssm_conv_kernel: config.ssm_conv_kernel,
            graph_exec_argmax: AtomicUsize::new(0),
            graph_cap_argmax: AtomicUsize::new(0),
        };
        if std::env::var_os("SPARK_Q1_KERNEL_BENCH").is_some() {
            engine.benchmark_q1_kernel()?;
        }
        Ok(engine)
    }

    fn launch_resident_matrix_quantized(
        &self,
        matrix: &CudaQ1Matrix,
        d_output: usize,
    ) -> Result<(), String> {
        if matrix.u10 {
            if matrix.tiled {
                self.device.launch_gemv_q1_0_u10_tiled(
                    matrix.d_ptr,
                    self.d_input,
                    d_output,
                    matrix.cols,
                    matrix.rows,
                    matrix.scale_min,
                    matrix.scale_step,
                )
            } else {
                self.device.launch_gemv_q1_0_u10(
                    matrix.d_ptr,
                    self.d_input,
                    d_output,
                    matrix.cols,
                    matrix.rows,
                    matrix.scale_min,
                    matrix.scale_step,
                )
            }
        } else {
            self.device.launch_gemv_q1_0(
                matrix.d_ptr,
                self.d_input,
                d_output,
                matrix.cols,
                matrix.rows,
            )
        }
    }

    fn launch_resident_matrices(
        &self,
        matrices: &[(&CudaQ1Matrix, usize)],
        d_input_f32: usize,
    ) -> Result<(), String> {
        let cols = matrices
            .first()
            .map(|(matrix, _)| matrix.cols)
            .ok_or_else(|| "empty resident GEMV batch".to_string())?;
        if matrices.iter().any(|(matrix, _)| matrix.cols != cols) {
            return Err("resident GEMV batch has mismatched input widths".into());
        }
        self.device
            .launch_quantize_q8_0_128(d_input_f32, self.d_input, cols)?;
        for &(matrix, output) in matrices {
            self.launch_resident_matrix_quantized(matrix, output)?;
        }
        Ok(())
    }

    fn upload_resident_token(&self, h_x: &[f32], pos: usize) -> Result<(), String> {
        if !self.resident {
            return Err("Qwen35 resident CUDA path is not active".into());
        }
        if pos >= self.context_capacity {
            return Err(format!(
                "Qwen35 GPU context limit reached ({})",
                self.context_capacity
            ));
        }
        if h_x.len() < self.hidden_dim {
            return Err("Qwen35 embedding vector is too short".into());
        }
        let state = unsafe { &mut *self.h_token_state.get() };
        state[..self.hidden_dim].copy_from_slice(&h_x[..self.hidden_dim]);
        let half = self.rope_dim / 2;
        let cos_sin = &mut state[self.hidden_dim..self.hidden_dim + self.rope_dim];
        for index in 0..half {
            let frequency = self
                .rope_base
                .powf(-(2.0 * index as f32 / self.rope_dim as f32));
            let (sin, cos) = (pos as f32 * frequency).sin_cos();
            cos_sin[index] = cos;
            cos_sin[half + index] = sin;
        }
        state[self.hidden_dim + self.rope_dim] = f32::from_bits(pos as u32);
        let x_bytes = unsafe {
            std::slice::from_raw_parts(state.as_ptr().cast::<u8>(), self.hidden_dim * 4)
        };
        let cs_bytes = unsafe {
            std::slice::from_raw_parts(
                state.as_ptr().add(self.hidden_dim).cast::<u8>(),
                self.rope_dim * 4,
            )
        };
        let pos_bytes = unsafe {
            std::slice::from_raw_parts(
                state
                    .as_ptr()
                    .add(self.hidden_dim + self.rope_dim)
                    .cast::<u8>(),
                4,
            )
        };
        self.device.copy_to_device(self.d_x, x_bytes)?;
        self.device.copy_to_device(self.d_cos_sin, cs_bytes)?;
        self.device.copy_to_device(self.d_pos, pos_bytes)
    }

    fn run_resident_layers_with_grid(
        &self,
        pos: usize,
        attention_grid_positions: usize,
    ) -> Result<usize, String> {
        let d = &self.device;
        let key_dim = self.ssm_group_count * self.ssm_head_dim;
        let conv_dim = self.ssm_inner_size + 2 * key_dim;
        let attn_dim = self.heads * self.head_dim;

        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            if layer_index == 0 {
                d.launch_rmsnorm_f32(
                    self.d_norm,
                    self.d_x,
                    layer.attn_norm,
                    self.hidden_dim,
                    self.rms_epsilon,
                )?;
            }

            match &layer.attention {
                CudaQwen35Attention::Full {
                    q,
                    k,
                    v,
                    output,
                    q_norm,
                    k_norm,
                } => {
                    let d_raw_q = self.d_output;
                    let d_raw_k = d_raw_q + q.rows * 4;
                    let d_raw_v = d_raw_k + k.rows * 4;
                    self.launch_resident_matrices(
                        &[(q, d_raw_q), (k, d_raw_k), (v, d_raw_v)],
                        self.d_norm,
                    )?;
                    d.launch_qwen35_q_gate_rope(
                        d_raw_q,
                        self.d_q,
                        self.d_attn_gate,
                        *q_norm,
                        self.d_cos_sin,
                        self.heads,
                        self.head_dim,
                        self.rope_dim,
                        self.rms_epsilon,
                    )?;
                    let cache_offset = self.attention_state_offsets[layer_index]
                        .ok_or_else(|| "missing Qwen35 attention cache offset".to_string())?;
                    let d_k_cache = self.d_kv + cache_offset;
                    let d_v_cache = self.d_kv + self.kv_one_side_bytes + cache_offset;
                    d.launch_qwen35_kv_rope_store(
                        d_raw_k,
                        d_raw_v,
                        d_k_cache,
                        d_v_cache,
                        *k_norm,
                        self.d_cos_sin,
                        self.d_pos,
                        self.kv_heads,
                        self.head_dim,
                        self.rope_dim,
                        self.rms_epsilon,
                    )?;
                    d.launch_attn_scores_f32(
                        self.d_q,
                        d_k_cache,
                        self.d_scores,
                        pos,
                        self.d_pos,
                        self.context_capacity,
                        self.heads,
                        self.kv_heads,
                        self.head_dim,
                        (self.head_dim as f32).sqrt().recip(),
                        attention_grid_positions,
                    )?;
                    d.launch_attn_weighted_f32(
                        self.d_context,
                        d_v_cache,
                        self.d_scores,
                        pos,
                        self.d_pos,
                        self.context_capacity,
                        self.heads,
                        self.kv_heads,
                        self.head_dim,
                    )?;
                    d.launch_qwen35_attention_gate(
                        self.d_context,
                        self.d_attn_gate,
                        attn_dim,
                    )?;
                    self.launch_resident_matrices(
                        &[(output, self.d_residual_out)],
                        self.d_context,
                    )?;
                }
                CudaQwen35Attention::Recurrent {
                    qkv,
                    z,
                    alpha,
                    beta,
                    output,
                    conv1d,
                    dt_bias,
                    a,
                    state_norm,
                } => {
                    let d_qkv = self.d_output;
                    let d_z = d_qkv + qkv.rows * 4;
                    let d_alpha = d_z + z.rows * 4;
                    let d_beta = d_alpha + alpha.rows * 4;
                    self.launch_resident_matrices(
                        &[
                            (qkv, d_qkv),
                            (z, d_z),
                            (alpha, d_alpha),
                            (beta, d_beta),
                        ],
                        self.d_norm,
                    )?;
                    let conv_offset = self.conv_state_offsets[layer_index]
                        .ok_or_else(|| "missing Qwen35 convolution cache offset".to_string())?;
                    d.launch_qwen35_conv_silu(
                        d_qkv,
                        *conv1d,
                        self.d_conv_state + conv_offset,
                        conv_dim,
                        self.ssm_conv_kernel,
                    )?;
                    let d_recurrent_q = d_qkv;
                    let d_recurrent_k = d_qkv + key_dim * 4;
                    let d_recurrent_v = d_recurrent_k + key_dim * 4;
                    d.launch_qwen35_qk_l2_norm(
                        d_recurrent_q,
                        d_recurrent_k,
                        self.ssm_group_count,
                        self.ssm_head_dim,
                        self.rms_epsilon,
                    )?;
                    d.launch_qwen35_decay_beta(d_alpha, d_beta, *dt_bias, *a, beta.rows)?;
                    let state_offset = self.gdn_state_offsets[layer_index]
                        .ok_or_else(|| "missing Qwen35 recurrent cache offset".to_string())?;
                    d.launch_gated_delta_f32(
                        self.d_gdn_state + state_offset,
                        d_recurrent_q,
                        d_recurrent_k,
                        d_recurrent_v,
                        d_alpha,
                        d_beta,
                        self.d_context,
                        self.ssm_group_count,
                        self.ssm_value_heads,
                        self.ssm_head_dim,
                    )?;
                    d.launch_qwen35_gated_rmsnorm(
                        self.d_context,
                        d_z,
                        *state_norm,
                        self.ssm_value_heads,
                        self.ssm_head_dim,
                        self.rms_epsilon,
                    )?;
                    self.launch_resident_matrices(
                        &[(output, self.d_residual_out)],
                        self.d_context,
                    )?;
                }
            }

            d.launch_add_rmsnorm_f32(
                self.d_norm,
                self.d_x,
                self.d_residual_out,
                layer.post_attention_norm,
                self.hidden_dim,
                self.rms_epsilon,
            )?;
            self.launch_resident_matrices(
                &[
                    (&layer.gate, self.d_ffn_gate),
                    (&layer.up, self.d_ffn_up),
                ],
                self.d_norm,
            )?;
            d.launch_swiglu_f32(self.d_ffn_gate, self.d_ffn_up, self.intermediate_dim)?;
            self.launch_resident_matrices(
                &[(&layer.down, self.d_residual_out)],
                self.d_ffn_gate,
            )?;
            let next_norm = self
                .weights
                .layers
                .get(layer_index + 1)
                .map_or(self.weights.final_norm, |next| next.attn_norm);
            d.launch_add_rmsnorm_f32(
                self.d_norm,
                self.d_x,
                self.d_residual_out,
                next_norm,
                self.hidden_dim,
                self.rms_epsilon,
            )?;
        }

        let output = self
            .weights
            .output
            .as_ref()
            .ok_or_else(|| "Qwen35 output projection is not resident".to_string())?;
        self.launch_resident_matrices(&[(output, self.d_output)], self.d_norm)?;
        Ok(output.rows)
    }

    fn run_resident_layers(&self, pos: usize) -> Result<usize, String> {
        self.run_resident_layers_with_grid(pos, pos + 1)
    }

    fn resident_argmax_graph(&self, pos: usize) -> Result<Option<usize>, String> {
        let current = self.graph_exec_argmax.load(Ordering::Relaxed);
        if current == usize::MAX {
            return Ok(None);
        }
        let current_cap = self.graph_cap_argmax.load(Ordering::Relaxed);
        if current != 0 && pos < current_cap {
            return Ok(Some(current));
        }
        if current != 0 {
            self.device.destroy_graph_exec(current);
            self.graph_exec_argmax.store(0, Ordering::Relaxed);
            self.graph_cap_argmax.store(0, Ordering::Relaxed);
        }

        let capacity = (pos + 1)
            .next_power_of_two()
            .max(32)
            .min(self.context_capacity);
        if let Err(error) = self.device.begin_capture() {
            self.graph_exec_argmax
                .store(usize::MAX, Ordering::Relaxed);
            eprintln!("\x1b[33m[spark] Qwen35 CUDA graph disabled: {error}\x1b[0m");
            return Ok(None);
        }
        let capture = self
            .run_resident_layers_with_grid(pos, capacity)
            .and_then(|rows| {
                self.device
                    .launch_argmax_f32(self.d_output, self.d_token_out, rows)
            });
        if let Err(error) = capture {
            let _ = self.device.end_capture();
            self.graph_exec_argmax
                .store(usize::MAX, Ordering::Relaxed);
            eprintln!("\x1b[33m[spark] Qwen35 CUDA graph disabled: {error}\x1b[0m");
            return Ok(None);
        }
        let graph = match self.device.end_capture() {
            Ok(graph) => graph,
            Err(error) => {
                self.graph_exec_argmax
                    .store(usize::MAX, Ordering::Relaxed);
                eprintln!("\x1b[33m[spark] Qwen35 CUDA graph disabled: {error}\x1b[0m");
                return Ok(None);
            }
        };
        let executable = match self.device.instantiate_graph(graph) {
            Ok(executable) => executable,
            Err(error) => {
                self.device.destroy_graph(graph);
                self.graph_exec_argmax
                    .store(usize::MAX, Ordering::Relaxed);
                eprintln!("\x1b[33m[spark] Qwen35 CUDA graph disabled: {error}\x1b[0m");
                return Ok(None);
            }
        };
        self.device.destroy_graph(graph);
        self.graph_exec_argmax
            .store(executable, Ordering::Relaxed);
        self.graph_cap_argmax.store(capacity, Ordering::Relaxed);
        if std::env::var_os("SPARK_GRAPH_DEBUG").is_some() {
            eprintln!("[spark] captured Qwen35 CUDA graph for {capacity} positions");
        }
        Ok(Some(executable))
    }

    pub fn resident_forward_logits(
        &self,
        h_x: &[f32],
        pos: usize,
        h_logits: &mut [f32],
    ) -> Result<(), String> {
        self.upload_resident_token(h_x, pos)?;
        let rows = self.run_resident_layers(pos)?;
        let copy_len = rows.min(h_logits.len());
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(h_logits.as_mut_ptr().cast::<u8>(), copy_len * 4)
        };
        self.device.copy_from_device(bytes, self.d_output)
    }

    pub fn resident_forward_sample(
        &self,
        h_x: &[f32],
        pos: usize,
        temperature: f32,
        top_p: f32,
        rng_seed: f32,
    ) -> Result<u32, String> {
        self.upload_resident_token(h_x, pos)?;
        if temperature <= 0.0 {
            let compact_scales = self
                .weights
                .layers
                .first()
                .is_some_and(|layer| layer.gate.u10);
            if compact_scales && std::env::var_os("SPARK_NO_GRAPH").is_none() {
                if let Some(executable) = self.resident_argmax_graph(pos)? {
                    self.device.graph_launch(executable)?;
                } else {
                    let rows = self.run_resident_layers(pos)?;
                    self.device
                        .launch_argmax_f32(self.d_output, self.d_token_out, rows)?;
                }
            } else {
                let rows = self.run_resident_layers(pos)?;
                self.device
                    .launch_argmax_f32(self.d_output, self.d_token_out, rows)?;
            }
            self.device.synchronize()?;
            Ok(unsafe { std::ptr::read_volatile(self.h_token_out) })
        } else {
            let rows = self.run_resident_layers(pos)?;
            self.device.launch_top64_sample(
                self.d_output,
                self.d_top64_tokens,
                self.d_top64_logits,
                rows,
                1.0 / temperature.max(f32::MIN_POSITIVE),
            )?;
            let tokens = unsafe { &mut *self.h_top64_tokens.get() };
            let logits = unsafe { &mut *self.h_top64_logits.get() };
            let token_bytes = unsafe {
                std::slice::from_raw_parts_mut(tokens.as_mut_ptr().cast::<u8>(), 64 * 4)
            };
            let logit_bytes = unsafe {
                std::slice::from_raw_parts_mut(logits.as_mut_ptr().cast::<u8>(), 64 * 4)
            };
            self.device
                .copy_from_device(token_bytes, self.d_top64_tokens)?;
            self.device
                .copy_from_device(logit_bytes, self.d_top64_logits)?;
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut normalizer = 0.0f32;
            let mut probabilities = [0.0f32; 64];
            for (probability, &logit) in probabilities.iter_mut().zip(logits.iter()) {
                *probability = (logit - max_logit).exp();
                normalizer += *probability;
            }
            if normalizer > 0.0 {
                for probability in &mut probabilities {
                    *probability /= normalizer;
                }
            }
            let mut order: [usize; 64] = std::array::from_fn(|index| index);
            order.sort_unstable_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]));
            let mut cumulative = 0.0f32;
            let mut selected = tokens[order[0]];
            for &index in &order {
                cumulative += probabilities[index];
                selected = tokens[index];
                if cumulative >= top_p.clamp(0.0, 1.0) || rng_seed <= cumulative {
                    break;
                }
            }
            Ok(selected)
        }
    }

    fn benchmark_q1_kernel(&self) -> Result<(), String> {
        let matrix = &self
            .weights
            .layers
            .first()
            .ok_or_else(|| "no Qwen35 CUDA layers".to_string())?
            .gate;
        if self.mapped_io {
            unsafe { std::ptr::write_bytes(self.h_input, 0, self.input_capacity) };
        } else {
            self.device.memset_zero(self.d_input, self.input_capacity)?;
        }
        let start = self.device.event_create()?;
        let end = self.device.event_create()?;
        self.device.event_record(start)?;
        const ITERATIONS: usize = 200;
        for _ in 0..ITERATIONS {
            if matrix.u10 {
                if matrix.tiled {
                    self.device.launch_gemv_q1_0_u10_tiled(
                        matrix.d_ptr,
                        self.d_input,
                        self.d_output,
                        matrix.cols,
                        matrix.rows,
                        matrix.scale_min,
                        matrix.scale_step,
                    )?;
                } else {
                    self.device.launch_gemv_q1_0_u10(
                        matrix.d_ptr,
                        self.d_input,
                        self.d_output,
                        matrix.cols,
                        matrix.rows,
                        matrix.scale_min,
                        matrix.scale_step,
                    )?;
                }
            } else {
                self.device.launch_gemv_q1_0(
                    matrix.d_ptr,
                    self.d_input,
                    self.d_output,
                    matrix.cols,
                    matrix.rows,
                )?;
            }
        }
        self.device.event_record(end)?;
        let elapsed_ms = self.device.event_elapsed_ms(start, end)?;
        self.device.event_destroy(start);
        self.device.event_destroy(end);
        let gib = matrix.size_bytes as f64 * ITERATIONS as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "[spark] Q1 kernel bench: {}x{} {:.2} MiB, {:.3} ms/launch, {:.1} GiB/s",
            matrix.rows,
            matrix.cols,
            matrix.size_bytes as f64 / (1024.0 * 1024.0),
            elapsed_ms as f64 / ITERATIONS as f64,
            gib / (elapsed_ms as f64 / 1000.0)
        );
        Ok(())
    }

    pub fn has_gpu_gdn(&self, layer: usize) -> bool {
        self.use_gpu_gdn && self.gdn_state_offsets.get(layer).is_some_and(Option::is_some)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta(
        &self,
        layer: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        decay: &[f32],
        beta: &[f32],
        output: &mut [f32],
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        let state_offset = self
            .gdn_state_offsets
            .get(layer)
            .copied()
            .flatten()
            .ok_or_else(|| "Qwen35 layer has no GPU recurrent state".to_string())?;
        let key_dim = key_heads * head_dim;
        let inner = value_heads * head_dim;
        if q.len() != key_dim
            || k.len() != key_dim
            || v.len() != inner
            || decay.len() != value_heads
            || beta.len() != value_heads
            || output.len() < inner
        {
            return Err("Qwen35 GPU GDN shape mismatch".into());
        }
        let staging = unsafe { &mut *self.gdn_staging.get() };
        let mut offset = 0usize;
        for values in [q, k, v] {
            staging[offset..offset + values.len()].copy_from_slice(values);
            offset += values.len();
        }
        for &value in decay {
            staging[offset] = value.exp();
            offset += 1;
        }
        staging[offset..offset + beta.len()].copy_from_slice(beta);
        offset += beta.len();
        let input_bytes = unsafe {
            std::slice::from_raw_parts(staging.as_ptr().cast::<u8>(), offset * 4)
        };
        self.device.copy_to_device(self.d_gdn_input, input_bytes)?;
        let d_q = self.d_gdn_input;
        let d_k = d_q + key_dim * 4;
        let d_v = d_k + key_dim * 4;
        let d_decay = d_v + inner * 4;
        let d_beta = d_decay + value_heads * 4;
        self.device.launch_gated_delta_f32(
            self.d_gdn_state + state_offset,
            d_q,
            d_k,
            d_v,
            d_decay,
            d_beta,
            self.d_gdn_output,
            key_heads,
            value_heads,
            head_dim,
        )?;
        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), inner * 4)
        };
        self.device.copy_from_device(output_bytes, self.d_gdn_output)
    }

    pub fn clear_recurrent_state(&self) -> Result<(), String> {
        if self.d_gdn_state != 0 {
            self.device.memset_zero(self.d_gdn_state, self.gdn_state_bytes)?;
        }
        if self.d_conv_state != 0 {
            self.device
                .memset_zero(self.d_conv_state, self.conv_state_bytes)?;
        }
        if self.d_kv != 0 {
            self.device
                .memset_zero(self.d_kv, self.kv_one_side_bytes * 2)?;
        }
        Ok(())
    }

    pub fn gemv(&self, matrix: &CudaQ1Matrix, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        let mut outputs = [output];
        self.gemv_many(&[matrix], input, &mut outputs)
    }

    pub fn gemv_many(
        &self,
        matrices: &[&CudaQ1Matrix],
        input: &[f32],
        outputs: &mut [&mut [f32]],
    ) -> Result<(), String> {
        if matrices.is_empty() || matrices.len() != outputs.len() {
            return Err("Qwen35 CUDA GEMV list mismatch".into());
        }
        if matrices
            .iter()
            .zip(outputs.iter())
            .any(|(matrix, output)| input.len() != matrix.cols || output.len() < matrix.rows)
        {
            return Err("Qwen35 CUDA GEMV shape mismatch".into());
        }
        let q8 = quantize_activations_q8_0(input);
        let input_group_bytes = if self.use_bitplane { 168 } else { 136 };
        let bytes_len = (input.len() / 128) * input_group_bytes;
        let total_rows: usize = matrices.iter().map(|m| m.rows).sum();
        if bytes_len > self.input_capacity || total_rows * 4 > self.output_capacity {
            return Err("Qwen35 CUDA GEMV scratch overflow".into());
        }
        let staging = if self.mapped_io {
            unsafe { std::slice::from_raw_parts_mut(self.h_input, self.input_capacity) }
        } else {
            unsafe { &mut *self.staging.get() }
        };
        for group in 0..input.len() / 128 {
            let dst = &mut staging
                [group * input_group_bytes..(group + 1) * input_group_bytes];
            for block in 0..4 {
                let src = &q8[group * 4 + block];
                dst[block * 2..block * 2 + 2].copy_from_slice(&src.d.to_le_bytes());
                if self.use_bitplane {
                    for bit in 0..8 {
                        let mut plane = 0u32;
                        for (index, &value) in src.qs.iter().enumerate() {
                            plane |= (((value as u8 >> bit) & 1) as u32) << index;
                        }
                        let p = 8 + block * 32 + bit * 4;
                        dst[p..p + 4].copy_from_slice(&plane.to_le_bytes());
                        dst[136 + block * 8 + bit] = plane.count_ones() as u8;
                    }
                } else {
                    let qs = unsafe {
                        std::slice::from_raw_parts(src.qs.as_ptr().cast::<u8>(), src.qs.len())
                    };
                    dst[8 + block * 32..8 + (block + 1) * 32].copy_from_slice(qs);
                }
            }
        }
        if !self.mapped_io {
            self.device.copy_to_device(self.d_input, &staging[..bytes_len])?;
        }
        let mut row_offset = 0usize;
        for matrix in matrices {
            if matrix.u10 {
                if matrix.tiled {
                    self.device.launch_gemv_q1_0_u10_tiled(
                        matrix.d_ptr,
                        self.d_input,
                        self.d_output + row_offset * 4,
                        matrix.cols,
                        matrix.rows,
                        matrix.scale_min,
                        matrix.scale_step,
                    )?;
                } else {
                    self.device.launch_gemv_q1_0_u10(
                        matrix.d_ptr,
                        self.d_input,
                        self.d_output + row_offset * 4,
                        matrix.cols,
                        matrix.rows,
                        matrix.scale_min,
                        matrix.scale_step,
                    )?;
                }
            } else {
                self.device.launch_gemv_q1_0(
                    matrix.d_ptr,
                    self.d_input,
                    self.d_output + row_offset * 4,
                    matrix.cols,
                    matrix.rows,
                )?;
            }
            row_offset += matrix.rows;
        }
        let host_output = if self.mapped_io {
            self.device.synchronize()?;
            unsafe { std::slice::from_raw_parts_mut(self.h_output, total_rows) }
        } else {
            let host = unsafe { &mut *self.host_output.get() };
            let output_bytes = unsafe {
                std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), total_rows * 4)
            };
            self.device.copy_from_device(output_bytes, self.d_output)?;
            host.as_mut_slice()
        };
        row_offset = 0;
        for (matrix, output) in matrices.iter().zip(outputs.iter_mut()) {
            output[..matrix.rows]
                .copy_from_slice(&host_output[row_offset..row_offset + matrix.rows]);
            row_offset += matrix.rows;
        }
        Ok(())
    }
}

impl Drop for CudaQwen35Engine {
    fn drop(&mut self) {
        self.device.stop_persistent_keepalive();
        let graph = self.graph_exec_argmax.load(Ordering::Relaxed);
        if graph != 0 && graph != usize::MAX {
            self.device.destroy_graph_exec(graph);
        }
        if self.mapped_io {
            self.device.free_mapped(self.h_input);
            self.device.free_mapped(self.h_output.cast::<u8>());
        } else {
            self.device.free(self.d_input);
            self.device.free(self.d_output);
        }
        self.device.free(self.d_gdn_input);
        self.device.free(self.d_gdn_output);
        self.device.free(self.d_gdn_state);
        self.device.free(self.d_resident_arena);
        self.device.free(self.d_kv);
        self.device.free(self.d_conv_state);
        self.device.free_mapped(self.h_token_out.cast::<u8>());
        self.device.free(self.d_top64_tokens);
        self.device.free(self.d_top64_logits);
    }
}

fn layer_matrices(layer: &CudaQwen35Layer) -> Vec<&CudaQ1Matrix> {
    let mut matrices = Vec::with_capacity(8);
    match &layer.attention {
        CudaQwen35Attention::Full { q, k, v, output, .. } => {
            matrices.extend([q, k, v, output]);
        }
        CudaQwen35Attention::Recurrent { qkv, z, alpha, beta, output, .. } => {
            matrices.extend([qkv, z, alpha, beta, output]);
        }
    }
    matrices.extend([&layer.gate, &layer.up, &layer.down]);
    matrices
}

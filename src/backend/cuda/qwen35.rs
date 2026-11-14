use std::{cell::UnsafeCell, sync::Arc};

use super::driver::CudaDevice;
use crate::{
    kernel::{f16::f16_to_f32, q1_0::quantize_activations_q8_0},
    model::weights::{
        ModelWeights, QuantizedMatrix, Qwen35AttentionWeights, Qwen35LayerWeights,
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
    },
    Recurrent {
        qkv: CudaQ1Matrix,
        z: CudaQ1Matrix,
        alpha: CudaQ1Matrix,
        beta: CudaQ1Matrix,
        output: CudaQ1Matrix,
    },
}

#[derive(Debug)]
pub struct CudaQwen35Layer {
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
    pub offloaded_layers: usize,
}

impl CudaQwen35Weights {
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
        let use_u10 = std::env::var_os("SPARK_Q1_U10").is_some();
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

    fn upload_layer(
        device: &CudaDevice,
        arena_ptr: usize,
        cursor: &mut usize,
        layer: &Qwen35LayerWeights<'_>,
    ) -> Result<CudaQwen35Layer, String> {
        let attention = match layer.attention {
            Qwen35AttentionWeights::Full { q, k, v, output, .. } => CudaQwen35Attention::Full {
                q: Self::upload_matrix(device, arena_ptr, cursor, &q)?,
                k: Self::upload_matrix(device, arena_ptr, cursor, &k)?,
                v: Self::upload_matrix(device, arena_ptr, cursor, &v)?,
                output: Self::upload_matrix(device, arena_ptr, cursor, &output)?,
            },
            Qwen35AttentionWeights::Recurrent { qkv, z, alpha, beta, output, .. } => {
                CudaQwen35Attention::Recurrent {
                    qkv: Self::upload_matrix(device, arena_ptr, cursor, &qkv)?,
                    z: Self::upload_matrix(device, arena_ptr, cursor, &z)?,
                    alpha: Self::upload_matrix(device, arena_ptr, cursor, &alpha)?,
                    beta: Self::upload_matrix(device, arena_ptr, cursor, &beta)?,
                    output: Self::upload_matrix(device, arena_ptr, cursor, &output)?,
                }
            }
        };
        Ok(CudaQwen35Layer {
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
        let use_u10 = std::env::var_os("SPARK_Q1_U10").is_some();
        let full_offload = std::env::var_os("SPARK_Q1_FULL_OFFLOAD").is_some() || use_u10;
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
        let arena_bytes = used
            + if upload_output { output_bytes } else { 0 }
            + matrix_count * 255
            + 256;
        let arena_ptr = device.alloc(arena_bytes)?;
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
        device.synchronize()?;
        Ok(Self {
            device,
            arena_ptr,
            layers,
            output,
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
}

unsafe impl Send for CudaQwen35Engine {}
unsafe impl Sync for CudaQwen35Engine {}

impl CudaQwen35Engine {
    pub fn new(device: Arc<CudaDevice>, weights: CudaQwen35Weights) -> Result<Self, String> {
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
        let use_u10 = std::env::var_os("SPARK_Q1_U10").is_some();
        let use_bitplane = device.q1_bitplane_enabled() && !use_u10;
        let input_group_bytes = if use_bitplane { 168 } else { 136 };
        let input_capacity = (max_cols / 128) * input_group_bytes;
        let output_capacity = max_rows * 4;
        let mapped_io = std::env::var_os("SPARK_Q1_MAPPED_IO").is_some();
        let (h_input, d_input) = if mapped_io {
            device.alloc_mapped(input_capacity)?
        } else {
            (std::ptr::null_mut(), device.alloc(input_capacity)?)
        };
        let (h_output_raw, d_output) = if mapped_io {
            device.alloc_mapped(output_capacity)?
        } else {
            (std::ptr::null_mut(), device.alloc(output_capacity)?)
        };
        let use_gpu_gdn = std::env::var_os("SPARK_GDN_CUDA").is_some();
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
            let ptr = device.alloc(gdn_state_bytes)?;
            device.memset_zero(ptr, gdn_state_bytes)?;
            ptr
        } else {
            0
        };
        let d_gdn_input = if use_gpu_gdn { device.alloc(gdn_input_floats * 4)? } else { 0 };
        let d_gdn_output = if use_gpu_gdn { device.alloc(gdn_output_floats * 4)? } else { 0 };
        device.warmup(300);
        if std::env::var_os("SPARK_LIGHT_KEEPALIVE").is_some() {
            device.start_persistent_keepalive();
        }
        let engine = Self {
            device,
            weights,
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
        };
        if std::env::var_os("SPARK_Q1_KERNEL_BENCH").is_some() {
            engine.benchmark_q1_kernel()?;
        }
        Ok(engine)
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
    }
}

fn layer_matrices(layer: &CudaQwen35Layer) -> Vec<&CudaQ1Matrix> {
    let mut matrices = Vec::with_capacity(8);
    match &layer.attention {
        CudaQwen35Attention::Full { q, k, v, output } => {
            matrices.extend([q, k, v, output]);
        }
        CudaQwen35Attention::Recurrent { qkv, z, alpha, beta, output } => {
            matrices.extend([qkv, z, alpha, beta, output]);
        }
    }
    matrices.extend([&layer.gate, &layer.up, &layer.down]);
    matrices
}

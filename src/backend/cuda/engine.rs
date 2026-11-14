use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use super::{driver::CudaDevice, weights::{CudaMatrix, CudaModelWeights}};
use crate::model::config::ModelConfig;
use crate::profile::ProfileStats;

fn record(
    profiler: &mut Option<&mut ProfileStats>,
    field: impl FnOnce(&mut ProfileStats) -> &mut std::time::Duration,
    started: Option<Instant>,
) {
    if let (Some(profiler), Some(started)) = (profiler.as_deref_mut(), started) {
        *field(profiler) += started.elapsed();
    }
}

/// When set, the resident forward path synchronizes the stream after every
/// operation group so the CPU-side profiler measures *actual* GPU execution
/// time instead of async launch overhead.
fn sync_profile_enabled() -> bool {
    std::env::var_os("SPARK_SYNC_PROFILE").is_some()
}

pub struct CudaEngine {
    pub device: Arc<CudaDevice>,
    pub weights: CudaModelWeights,
    pub d_norm: usize,
    pub d_qkv: usize,
    pub d_context: usize,
    pub d_ffn_out: usize,
    pub d_ffn_up: usize,
    pub d_out: usize,
    pub d_logits: usize,
    pub d_token_out: usize,
    pub h_token_out: *mut u32,
    pub d_top64_tokens: usize,
    pub d_top64_logits: usize,
    pub h_qkv_buf: std::cell::UnsafeCell<Vec<f32>>,
    pub h_top64_tokens: std::cell::UnsafeCell<[u32; 64]>,
    pub h_top64_logits: std::cell::UnsafeCell<[f32; 64]>,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub total_qkv: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    /// Fully device-resident forward: RoPE, attention, KV-cache and SwiGLU run
    /// on the GPU; host memory is touched only for the embedding row upload
    /// and the final token/logits download.
    pub resident: bool,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ctx_len: usize,
    pub rope_base: f32,
    pub rms_epsilon: f32,
    pub attn_scale: f32,
    pub d_x: usize,
    pub d_scores: usize,
    pub d_cos_sin: usize,
    pub d_pos: usize,
    pub d_kv_k: usize,
    pub d_kv_v: usize,
    /// Contiguous host staging for [x | RoPE cos/sin | position].
    pub h_token_state: std::cell::UnsafeCell<Vec<f32>>,
    /// CUDA graph decode state. 0 = not captured yet; usize::MAX = capture
    /// failed (permanent fallback to the eager path). Separate graphs for the
    /// argmax (temp==0) and top-p (temp>0) sampling flavors.
    pub graph_exec_argmax: AtomicUsize,
    pub graph_exec_top64: AtomicUsize,
    /// Number of sequence positions covered by each captured graph's QK grid.
    pub graph_cap_argmax: AtomicUsize,
    pub graph_cap_top64: AtomicUsize,
    /// True while the forward body is being captured into a graph: forces
    /// attn_scores_f32 to launch with the current power-of-two bucket grid.
    pub capturing: AtomicBool,
    pub capture_attn_positions: AtomicUsize,
}

impl CudaEngine {
    pub fn new(
        device: Arc<CudaDevice>,
        weights: CudaModelWeights,
        config: &ModelConfig,
    ) -> Result<Self, String> {
        let max_hidden = weights
            .layers
            .iter()
            .map(|l| l.o.rows.max(l.down.rows).max(l.q.cols))
            .max()
            .unwrap_or(config.hidden_dim)
            .max(config.hidden_dim);

        let max_qkv_rows = weights
            .layers
            .iter()
            .map(|l| l.q.rows.max(l.k.rows).max(l.v.rows))
            .max()
            .unwrap_or(0)
            .max(config.heads * config.head_dim + config.kv_heads * config.head_dim * 2);

        let max_ctx_dim = weights
            .layers
            .iter()
            .map(|l| l.o.cols)
            .max()
            .unwrap_or(0)
            .max(config.heads * config.head_dim);

        let max_intermediate = weights
            .layers
            .iter()
            .map(|l| l.gate.rows.max(l.up.rows).max(l.down.cols))
            .max()
            .unwrap_or(0)
            .max(config.intermediate_dim);

        let max_vocab = weights
            .output
            .as_ref()
            .map(|o| o.rows)
            .unwrap_or(config.vocab_size)
            .max(config.vocab_size);

        let d_norm = device.alloc(max_hidden * 4)?;
        let d_qkv = device.alloc(max_qkv_rows * 4)?;
        let d_context = device.alloc(max_ctx_dim * 4)?;
        let d_ffn_out = device.alloc(max_intermediate * 4)?;
        let d_ffn_up = device.alloc(max_intermediate * 4)?;
        let d_out = device.alloc(max_hidden * 4)?;
        let d_logits = device.alloc(max_vocab * 4)?;
        let (h_token_out, d_token_out) = device.alloc_mapped(4)?;
        let d_top64_tokens = device.alloc(64 * 4)?;
        let d_top64_logits = device.alloc(64 * 4)?;

        // Resident-mode buffers: only usable when every layer (plus the output
        // projection) lives in VRAM and the model needs no host-side ops.
        let resident =
            weights.output.is_some() && !config.has_hadamard && std::env::var_os("SPARK_NO_RESIDENT").is_none();
        let heads = config.heads;
        let kv_heads = config.kv_heads;
        let head_dim = config.head_dim;
        let (d_x, d_scores, d_cos_sin, d_kv_k, d_kv_v, d_pos) = if resident {
            // Keep per-token state contiguous so upload_token_state needs one
            // HtoD call instead of three synchronous driver calls.
            let d_state = device.alloc((max_hidden + head_dim + 1) * 4)?;
            (
                d_state,
                device.alloc(heads * config.context_len * 4)?,
                d_state + max_hidden * 4,
                device.alloc(weights.layers.len() * config.context_len * kv_heads * head_dim * 4)?,
                device.alloc(weights.layers.len() * config.context_len * kv_heads * head_dim * 4)?,
                d_state + (max_hidden + head_dim) * 4,
            )
        } else {
            (0, 0, 0, 0, 0, 0)
        };
        let head_dim_f = head_dim as f32;
        let attn_scale = 1.0 / head_dim_f.sqrt();

        // Raise GPU clocks before inference starts.
        device.warmup(300);

        // NOTE: the persistent keepalive spin kernel was measured to *destroy*
        // throughput (7.5 vs 166 tok/s on the RTX 3050 — the spinning blocks
        // starve the real work). Clocks stay usable without it; the env knob
        // remains for experiments.
        // device.start_persistent_keepalive();

        Ok(Self {
            device,
            weights,
            d_norm,
            d_qkv,
            d_context,
            d_ffn_out,
            d_ffn_up,
            d_out,
            d_logits,
            d_token_out,
            h_token_out: h_token_out.cast::<u32>(),
            d_top64_tokens,
            d_top64_logits,
            h_qkv_buf: std::cell::UnsafeCell::new(vec![0.0f32; max_qkv_rows]),
            h_top64_tokens: std::cell::UnsafeCell::new([0u32; 64]),
            h_top64_logits: std::cell::UnsafeCell::new([0.0f32; 64]),
            hidden_dim: max_hidden,
            q_dim: config.heads * config.head_dim,
            kv_dim: config.kv_heads * config.head_dim,
            total_qkv: max_qkv_rows,
            intermediate_dim: max_intermediate,
            vocab_size: max_vocab,
            resident,
            heads,
            kv_heads,
            head_dim,
            ctx_len: config.context_len,
            rope_base: config.rope_base,
            rms_epsilon: config.rms_epsilon,
            attn_scale,
            d_x,
            d_scores,
            d_cos_sin,
            d_pos,
            d_kv_k,
            d_kv_v,
            h_token_state: std::cell::UnsafeCell::new(vec![0.0f32; max_hidden + head_dim + 1]),
            graph_exec_argmax: AtomicUsize::new(0),
            graph_exec_top64: AtomicUsize::new(0),
            graph_cap_argmax: AtomicUsize::new(0),
            graph_cap_top64: AtomicUsize::new(0),
            capturing: AtomicBool::new(false),
            capture_attn_positions: AtomicUsize::new(0),
        })
    }

    /// Dispatch a single matrix GEMV on the device, picking the kernel that
    /// matches the matrix's block layout. Ternary matrices come in two flavors:
    /// PTQ1_0 (28-byte blocks of 128 weights) is handled by `gemv_ternary`,
    /// while Q2_0 (10-byte blocks of 32 weights) needs `gemv_q2_0`.
    fn launch_matrix(&self, m: &CudaMatrix, d_input: usize, d_output: usize) -> Result<(), String> {
        if m.is_ternary {
            if m.is_ptq1 {
                self.device.launch_gemv_ternary(m.d_ptr, d_input, d_output, m.cols, m.rows)
            } else {
                self.device.launch_gemv_q2_0(m.d_ptr, d_input, d_output, m.cols, m.rows)
            }
        } else if m.is_q4 {
            self.device.launch_gemv_q4(m.d_ptr, d_input, d_output, m.cols, m.rows)
        } else {
            self.device.launch_gemv_q8(m.d_ptr, d_input, d_output, m.cols, m.rows)
        }
    }

    pub fn gemv_qkv(
        &self,
        layer_idx: usize,
        h_norm: &[f32],
        h_q: &mut [f32],
        h_k: &mut [f32],
        h_v: &mut [f32],
    ) -> Result<(), String> {
        let layer = &self.weights.layers[layer_idx];
        let bytes_norm = unsafe {
            std::slice::from_raw_parts(h_norm.as_ptr().cast::<u8>(), h_norm.len() * 4)
        };
        self.device.copy_to_device(self.d_norm, bytes_norm)?;

        if layer.q.is_ternary {
            // Ternary path has no fused kernel: launch Q, K and V separately,
            // writing each result at its offset inside d_qkv.
            self.launch_matrix(&layer.q, self.d_norm, self.d_qkv)?;
            self.launch_matrix(&layer.k, self.d_norm, self.d_qkv + layer.q.rows * 4)?;
            self.launch_matrix(&layer.v, self.d_norm, self.d_qkv + (layer.q.rows + layer.k.rows) * 4)?;

            let total_rows = layer.q.rows + layer.k.rows + layer.v.rows;
            let host_qkv = unsafe { &mut *self.h_qkv_buf.get() };
            let copy_len = total_rows.min(host_qkv.len());
            let bytes_qkv = unsafe {
                std::slice::from_raw_parts_mut(host_qkv.as_mut_ptr().cast::<u8>(), copy_len * 4)
            };
            self.device.copy_from_device(bytes_qkv, self.d_qkv)?;

            let q_len = layer.q.rows.min(h_q.len());
            h_q[..q_len].copy_from_slice(&host_qkv[..q_len]);
            let k_len = layer.k.rows.min(h_k.len());
            h_k[..k_len].copy_from_slice(&host_qkv[layer.q.rows..layer.q.rows + k_len]);
            let v_len = layer.v.rows.min(h_v.len());
            h_v[..v_len]
                .copy_from_slice(&host_qkv[layer.q.rows + layer.k.rows..layer.q.rows + layer.k.rows + v_len]);
        } else if layer.q.is_q4 {
            self.device.launch_fused_qkv_q4(
                layer.q.d_ptr,
                layer.k.d_ptr,
                layer.v.d_ptr,
                self.d_norm,
                self.d_qkv,
                layer.q.cols,
                self.q_dim,
                self.kv_dim,
                self.kv_dim,
            )?;
            let host_qkv = unsafe { &mut *self.h_qkv_buf.get() };
            let copy_len = (self.q_dim + self.kv_dim * 2).min(host_qkv.len());
            let bytes_qkv = unsafe {
                std::slice::from_raw_parts_mut(host_qkv.as_mut_ptr().cast::<u8>(), copy_len * 4)
            };
            self.device.copy_from_device(bytes_qkv, self.d_qkv)?;

            h_q.copy_from_slice(&host_qkv[0..self.q_dim]);
            h_k.copy_from_slice(&host_qkv[self.q_dim..self.q_dim + self.kv_dim]);
            h_v.copy_from_slice(&host_qkv[self.q_dim + self.kv_dim..self.q_dim + self.kv_dim * 2]);
        } else {
            self.device.launch_fused_qkv_q8(
                layer.q.d_ptr,
                layer.k.d_ptr,
                layer.v.d_ptr,
                self.d_norm,
                self.d_qkv,
                layer.q.cols,
                self.q_dim,
                self.kv_dim,
                self.kv_dim,
                0,
            )?;
            let host_qkv = unsafe { &mut *self.h_qkv_buf.get() };
            let copy_len = (self.q_dim + self.kv_dim * 2).min(host_qkv.len());
            let bytes_qkv = unsafe {
                std::slice::from_raw_parts_mut(host_qkv.as_mut_ptr().cast::<u8>(), copy_len * 4)
            };
            self.device.copy_from_device(bytes_qkv, self.d_qkv)?;

            h_q.copy_from_slice(&host_qkv[0..self.q_dim]);
            h_k.copy_from_slice(&host_qkv[self.q_dim..self.q_dim + self.kv_dim]);
            h_v.copy_from_slice(&host_qkv[self.q_dim + self.kv_dim..self.q_dim + self.kv_dim * 2]);
        }

        Ok(())
    }

    pub fn gemv_attn_out(
        &self,
        layer_idx: usize,
        h_context: &[f32],
        h_out: &mut [f32],
    ) -> Result<(), String> {
        let layer = &self.weights.layers[layer_idx];
        let bytes_ctx = unsafe {
            std::slice::from_raw_parts(h_context.as_ptr().cast::<u8>(), h_context.len() * 4)
        };
        self.device.copy_to_device(self.d_context, bytes_ctx)?;

        if layer.o.is_ternary {
            self.launch_matrix(&layer.o, self.d_context, self.d_out)?;
        } else if layer.o.is_q4 {
            self.device.launch_gemv_q4(
                layer.o.d_ptr,
                self.d_context,
                self.d_out,
                layer.o.cols,
                layer.o.rows,
            )?;
        } else {
            self.device.launch_gemv_q8(
                layer.o.d_ptr,
                self.d_context,
                self.d_out,
                layer.o.cols,
                layer.o.rows,
            )?;
        }

        let copy_len = layer.o.rows.min(h_out.len());
        let bytes_out = unsafe {
            std::slice::from_raw_parts_mut(h_out.as_mut_ptr().cast::<u8>(), copy_len * 4)
        };
        self.device.copy_from_device(bytes_out, self.d_out)?;

        Ok(())
    }

    pub fn ffn_fused_down(
        &self,
        layer_idx: usize,
        h_norm: &[f32],
        h_out: &mut [f32],
    ) -> Result<(), String> {
        let layer = &self.weights.layers[layer_idx];
        let bytes_norm = unsafe {
            std::slice::from_raw_parts(h_norm.as_ptr().cast::<u8>(), h_norm.len() * 4)
        };
        self.device.copy_to_device(self.d_norm, bytes_norm)?;

        if layer.gate.is_ternary {
            // Ternary FFN: no fused SwiGLU kernel, so compute gate and up with
            // separate GEMVs, apply SwiGLU on the device, then run the down GEMV
            // on the activated activations.
            self.launch_matrix(&layer.gate, self.d_norm, self.d_ffn_out)?;
            self.launch_matrix(&layer.up, self.d_norm, self.d_ffn_up)?;
            self.device.launch_swiglu_f32(
                self.d_ffn_out,
                self.d_ffn_up,
                layer.gate.rows.min(layer.up.rows),
            )?;
            self.launch_matrix(&layer.down, self.d_ffn_out, self.d_out)?;
        } else if layer.gate.is_q4 {
            self.device.launch_fused_ffn_q4(
                layer.gate.d_ptr,
                layer.up.d_ptr,
                self.d_norm,
                self.d_ffn_out,
                layer.gate.cols,
                layer.gate.rows,
            )?;
            self.device.launch_gemv_q4(
                layer.down.d_ptr,
                self.d_ffn_out,
                self.d_out,
                layer.down.cols,
                layer.down.rows,
            )?;
        } else {
            self.device.launch_fused_ffn_q8(
                layer.gate.d_ptr,
                layer.up.d_ptr,
                self.d_norm,
                self.d_ffn_out,
                layer.gate.cols,
                layer.gate.rows,
            )?;
            self.device.launch_gemv_q8(
                layer.down.d_ptr,
                self.d_ffn_out,
                self.d_out,
                layer.down.cols,
                layer.down.rows,
            )?;
        }

        let copy_len = layer.down.rows.min(h_out.len());
        let bytes_out = unsafe {
            std::slice::from_raw_parts_mut(h_out.as_mut_ptr().cast::<u8>(), copy_len * 4)
        };
        self.device.copy_from_device(bytes_out, self.d_out)?;

        Ok(())
    }

    pub fn gemv_output(&self, h_norm: &[f32], h_logits: &mut [f32]) -> Result<(), String> {
        let output_weights = self.weights.output.as_ref().ok_or_else(|| "output weights on CPU".to_string())?;

        let bytes_norm = unsafe {
            std::slice::from_raw_parts(h_norm.as_ptr().cast::<u8>(), h_norm.len() * 4)
        };
        self.device.copy_to_device(self.d_norm, bytes_norm)?;

        if output_weights.is_ternary {
            self.launch_matrix(output_weights, self.d_norm, self.d_logits)?;
        } else if output_weights.is_q4 {
            self.device.launch_gemv_q4(
                output_weights.d_ptr,
                self.d_norm,
                self.d_logits,
                output_weights.cols,
                output_weights.rows,
            )?;
        } else {
            self.device.launch_gemv_q8(
                output_weights.d_ptr,
                self.d_norm,
                self.d_logits,
                output_weights.cols,
                output_weights.rows,
            )?;
        }

        let copy_len = output_weights.rows.min(h_logits.len());
        let bytes_logits = unsafe {
            std::slice::from_raw_parts_mut(h_logits.as_mut_ptr().cast::<u8>(), copy_len * 4)
        };
        self.device.copy_from_device(bytes_logits, self.d_logits)?;

        Ok(())
    }

    pub fn gemv_output_and_sample(
        &self,
        h_norm: &[f32],
        temperature: f32,
        top_p: f32,
        rng_seed: f32,
    ) -> Result<u32, String> {
        let output_weights = self.weights.output.as_ref().ok_or_else(|| "output weights on CPU".to_string())?;

        let bytes_norm = unsafe {
            std::slice::from_raw_parts(h_norm.as_ptr().cast::<u8>(), h_norm.len() * 4)
        };
        self.device.copy_to_device(self.d_norm, bytes_norm)?;

        if output_weights.is_ternary {
            self.launch_matrix(output_weights, self.d_norm, self.d_logits)?;
        } else if output_weights.is_q4 {
            self.device.launch_gemv_q4(
                output_weights.d_ptr,
                self.d_norm,
                self.d_logits,
                output_weights.cols,
                output_weights.rows,
            )?;
        } else {
            self.device.launch_gemv_q8(
                output_weights.d_ptr,
                self.d_norm,
                self.d_logits,
                output_weights.cols,
                output_weights.rows,
            )?;
        }

        if temperature <= 0.0 {
            self.device.launch_argmax_f32(self.d_logits, self.d_token_out, output_weights.rows)?;
            self.read_argmax_token()
        } else {
            let inv_temp = 1.0 / temperature.max(f32::MIN_POSITIVE);
            self.device.launch_top64_sample(
                self.d_logits,
                self.d_top64_tokens,
                self.d_top64_logits,
                output_weights.rows,
                inv_temp,
            )?;

            let tokens = unsafe { &mut *self.h_top64_tokens.get() };
            let logits = unsafe { &mut *self.h_top64_logits.get() };

            let bytes_tokens = unsafe {
                std::slice::from_raw_parts_mut(tokens.as_mut_ptr().cast::<u8>(), 64 * 4)
            };
            let bytes_logits = unsafe {
                std::slice::from_raw_parts_mut(logits.as_mut_ptr().cast::<u8>(), 64 * 4)
            };
            self.device.copy_from_device(bytes_tokens, self.d_top64_tokens)?;
            self.device.copy_from_device(bytes_logits, self.d_top64_logits)?;

            Ok(self.pick_top64(top_p, rng_seed))
        }
    }

    /// Host-side selection from the device-produced top-64 candidate list.
    fn pick_top64(&self, top_p: f32, rng_seed: f32) -> u32 {
        let tokens = unsafe { &mut *self.h_top64_tokens.get() };
        let logits = unsafe { &mut *self.h_top64_logits.get() };

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut normalizer = 0.0f32;
        let mut probs = [0.0f32; 64];
        for i in 0..64 {
            let p = (logits[i] - max_logit).exp();
            probs[i] = p;
            normalizer += p;
        }

        let inv_norm = if normalizer > 0.0 { normalizer.recip() } else { 1.0 };
        for p in &mut probs {
            *p *= inv_norm;
        }

        let mut order: [usize; 64] = std::array::from_fn(|i| i);
        order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

        let cutoff_p = top_p.clamp(0.0, 1.0);
        let mut cum_p = 0.0f32;
        let mut selected = tokens[order[0]];

        for &idx in &order {
            cum_p += probs[idx];
            selected = tokens[idx];
            if cum_p >= cutoff_p || rng_seed <= cum_p {
                break;
            }
        }

        selected
    }

    fn read_argmax_token(&self) -> Result<u32, String> {
        self.device.synchronize()?;
        Ok(unsafe { std::ptr::read_volatile(self.h_token_out) })
    }

    // ==================== GPU-resident forward ====================

    fn upload_token_state(&self, h_x: &[f32], pos: usize) -> Result<(), String> {
        let hidden = self.hidden_dim;
        if h_x.len() < hidden {
            return Err(format!(
                "resident forward: hidden vector too short ({}, need {hidden})",
                h_x.len()
            ));
        }
        let state = unsafe { &mut *self.h_token_state.get() };
        state[..hidden].copy_from_slice(&h_x[..hidden]);

        // RoPE cos/sin table for this position (same math as ops::rope).
        let cs = &mut state[hidden..hidden + self.head_dim];
        let half = self.head_dim / 2;
        for idx in 0..half {
            let frequency = self.rope_base.powf(-(2.0 * idx as f32 / self.head_dim as f32));
            let (sin, cos) = (pos as f32 * frequency).sin_cos();
            cs[idx] = cos;
            cs[half + idx] = sin;
        }
        if std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
            eprintln!(
                "\x1b[35m[dump] cos_sin pos={pos}: rope_base={} half={half} c0={:.6} s0={:.6} c1={:.6} s1={:.6}\x1b[0m",
                self.rope_base, cs[0], cs[half], cs[1], cs[half + 1]
            );
        }
        state[hidden + self.head_dim] = f32::from_bits(pos as u32);
        let bytes = unsafe {
            std::slice::from_raw_parts(state.as_ptr().cast::<u8>(), state.len() * 4)
        };
        self.device.copy_to_device(self.d_x, bytes)
    }

    fn dump_dev(&self, d_ptr: usize, len: usize, name: &str, pos: usize, layer: usize) {
        if std::env::var_os("SPARK_DUMP_LAYERS").is_none() || layer != 0 {
            return;
        }
        let mut h = vec![0.0f32; len];
        let bytes = unsafe { std::slice::from_raw_parts_mut(h.as_mut_ptr().cast::<u8>(), len * 4) };
        if self.device.copy_from_device(bytes, d_ptr).is_err() {
            return;
        }
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &v in &h {
            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);
        }
        eprintln!(
            "\x1b[35m[dump] {name} L{layer} pos={pos}: sum={sum:.6} sum_sq={sum_sq:.6} len={len} head8={:?} mid32={:?}\x1b[0m",
            &h[..8.min(len)],
            if len > 40 { &h[32..40] } else { &[] }
        );
    }

    fn run_layers(&self, pos: usize, mut profiler: Option<&mut ProfileStats>) -> Result<(), String> {
        let d = &self.device;
        let fused_resnorm = std::env::var_os("SPARK_NO_FUSED_RESNORM").is_none();
        let hidden = self.hidden_dim;
        let q_dim = self.q_dim;
        let kv_dim = self.kv_dim;
        let kv_stride = self.kv_heads * self.head_dim;
        let kv_layer_stride = self.ctx_len * kv_stride;
        let total_qkv = q_dim + 2 * kv_dim;

        // Optional CUDA-event timing (ground truth per kernel, no sync bias).
        let evp = std::env::var_os("SPARK_EVENT_PROFILE").is_some();
        let ea = if evp { Some(d.event_create()?) } else { None };
        let eb = if evp { Some(d.event_create()?) } else { None };
        let e_span_a = if evp { Some(d.event_create()?) } else { None };
        let e_span_b = if evp { Some(d.event_create()?) } else { None };
        if let Some(ev) = e_span_a {
            d.event_record(ev)?;
        }
        let mut t_norm = 0.0f64;
        let mut t_qkv = 0.0f64;
        let mut t_rope = 0.0f64;
        let mut t_scores = 0.0f64;
        let mut t_o = 0.0f64;
        let mut t_ffn = 0.0f64;
        let mut t_down = 0.0f64;
        let evrec = |e: &Option<usize>| -> Result<(), String> {
            if let Some(ev) = e {
                d.event_record(*ev)?;
            }
            Ok(())
        };

        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            let k_base = self.d_kv_k + layer_index * kv_layer_stride * 4;
            let v_base = self.d_kv_v + layer_index * kv_layer_stride * 4;

            let started = profiler.as_ref().and_then(|p| p.start());
            evrec(&ea)?;
            // With fused residual normalization, every layer after the first
            // receives d_norm from the preceding layer's down-residual kernel.
            if !fused_resnorm || layer_index == 0 {
                d.launch_rmsnorm_f32(
                    self.d_norm,
                    self.d_x,
                    self.weights.attn_norms[layer_index],
                    hidden,
                    self.rms_epsilon,
                )?;
            }
            evrec(&eb)?;
            if evp {
                t_norm += d.event_elapsed_ms(ea.unwrap(), eb.unwrap())? as f64;
            }
            evrec(&ea)?;
            if layer.q.is_ternary {
                self.launch_matrix(&layer.q, self.d_norm, self.d_qkv)?;
                self.launch_matrix(&layer.k, self.d_norm, self.d_qkv + q_dim * 4)?;
                self.launch_matrix(&layer.v, self.d_norm, self.d_qkv + (q_dim + kv_dim) * 4)?;
            } else if layer.q.is_q4 {
                d.launch_fused_qkv_q4(
                    layer.q.d_ptr,
                    layer.k.d_ptr,
                    layer.v.d_ptr,
                    self.d_norm,
                    self.d_qkv,
                    layer.q.cols,
                    q_dim,
                    kv_dim,
                    kv_dim,
                )?;
            } else {
                d.launch_fused_qkv_q8(
                    layer.q.d_ptr,
                    layer.k.d_ptr,
                    layer.v.d_ptr,
                    self.d_norm,
                    self.d_qkv,
                    layer.q.cols,
                    q_dim,
                    kv_dim,
                    kv_dim,
                    self.weights.qkv_biases[layer_index],
                )?;
            }
            if layer.q.is_ternary || layer.q.is_q4 {
                d.launch_add_bias_f32(
                    self.d_qkv,
                    self.weights.qkv_biases[layer_index],
                    total_qkv,
                )?;
            }
            evrec(&eb)?;
            if evp {
                t_qkv += d.event_elapsed_ms(ea.unwrap(), eb.unwrap())? as f64;
            }
            self.dump_dev(self.d_qkv, total_qkv, "qkv_bias", pos, layer_index);

            evrec(&ea)?;
            d.launch_rope_store_f32(
                self.d_qkv,
                self.d_qkv + q_dim * 4,
                self.d_qkv + (q_dim + kv_dim) * 4,
                k_base,
                v_base,
                self.d_cos_sin,
                pos,
                self.d_pos,
                self.heads * self.head_dim / 2,
                self.kv_heads * self.head_dim / 2,
                self.head_dim,
                kv_stride,
            )?;
            evrec(&eb)?;
            if evp {
                t_rope += d.event_elapsed_ms(ea.unwrap(), eb.unwrap())? as f64;
            }
            self.dump_dev(self.d_qkv, q_dim, "q_rope", pos, layer_index);
            self.dump_dev(k_base, (pos + 1) * kv_stride, "kcache", pos, layer_index);
            self.dump_dev(v_base, (pos + 1) * kv_stride, "vcache", pos, layer_index);
            if sync_profile_enabled() {
                d.synchronize()?;
            }
            record(&mut profiler, |p| &mut p.qkv, started);

            let started = profiler.as_ref().and_then(|p| p.start());
            evrec(&ea)?;
            d.launch_attn_scores_f32(
                self.d_qkv,
                k_base,
                self.d_scores,
                pos,
                self.d_pos,
                self.ctx_len,
                self.heads,
                self.kv_heads,
                self.head_dim,
                self.attn_scale,
                if self.capturing.load(Ordering::Relaxed) {
                    self.capture_attn_positions.load(Ordering::Relaxed)
                } else {
                    pos + 1
                },
            )?;
            self.dump_dev(self.d_scores, self.heads * (pos + 1), "scores", pos, layer_index);
            d.launch_attn_weighted_f32(
                self.d_context,
                v_base,
                self.d_scores,
                pos,
                self.d_pos,
                self.ctx_len,
                self.heads,
                self.kv_heads,
                self.head_dim,
            )?;
            evrec(&eb)?;
            if evp {
                t_scores += d.event_elapsed_ms(ea.unwrap(), eb.unwrap())? as f64;
            }
            self.dump_dev(self.d_context, q_dim, "context", pos, layer_index);
            if sync_profile_enabled() {
                d.synchronize()?;
            }
            record(&mut profiler, |p| &mut p.attention, started);

            let started = profiler.as_ref().and_then(|p| p.start());
            evrec(&ea)?;
            self.launch_matrix(&layer.o, self.d_context, self.d_out)?;
            if fused_resnorm {
                d.launch_add_rmsnorm_f32(
                    self.d_norm,
                    self.d_x,
                    self.d_out,
                    self.weights.ffn_norms[layer_index],
                    hidden,
                    self.rms_epsilon,
                )?;
            } else {
                d.launch_add_f32(self.d_x, self.d_out, hidden)?;
            }
            evrec(&eb)?;
            if evp {
                t_o += d.event_elapsed_ms(ea.unwrap(), eb.unwrap())? as f64;
            }
            if sync_profile_enabled() {
                d.synchronize()?;
            }
            record(&mut profiler, |p| &mut p.attention_output, started);

            let started = profiler.as_ref().and_then(|p| p.start());
            evrec(&ea)?;
            if !fused_resnorm {
                d.launch_rmsnorm_f32(
                    self.d_norm,
                    self.d_x,
                    self.weights.ffn_norms[layer_index],
                    hidden,
                    self.rms_epsilon,
                )?;
            }

            if layer.gate.is_ternary {
                self.launch_matrix(&layer.gate, self.d_norm, self.d_ffn_out)?;
                self.launch_matrix(&layer.up, self.d_norm, self.d_ffn_up)?;
                d.launch_swiglu_f32(
                    self.d_ffn_out,
                    self.d_ffn_up,
                    layer.gate.rows.min(layer.up.rows),
                )?;
            } else if layer.gate.is_q4 {
                d.launch_fused_ffn_q4(
                    layer.gate.d_ptr,
                    layer.up.d_ptr,
                    self.d_norm,
                    self.d_ffn_out,
                    layer.gate.cols,
                    layer.gate.rows,
                )?;
            } else {
                d.launch_fused_ffn_q8(
                    layer.gate.d_ptr,
                    layer.up.d_ptr,
                    self.d_norm,
                    self.d_ffn_out,
                    layer.gate.cols,
                    layer.gate.rows,
                )?;
            }
            evrec(&eb)?;
            if evp {
                t_ffn += d.event_elapsed_ms(ea.unwrap(), eb.unwrap())? as f64;
            }
            if sync_profile_enabled() {
                d.synchronize()?;
            }
            record(&mut profiler, |p| &mut p.ffn_gate_up, started);

            let started = profiler.as_ref().and_then(|p| p.start());
            evrec(&ea)?;
            self.launch_matrix(&layer.down, self.d_ffn_out, self.d_out)?;
            if fused_resnorm {
                let next_weight = self
                    .weights
                    .attn_norms
                    .get(layer_index + 1)
                    .copied()
                    .unwrap_or(self.weights.final_norm);
                d.launch_add_rmsnorm_f32(
                    self.d_norm,
                    self.d_x,
                    self.d_out,
                    next_weight,
                    hidden,
                    self.rms_epsilon,
                )?;
            } else {
                d.launch_add_f32(self.d_x, self.d_out, hidden)?;
            }
            evrec(&eb)?;
            if evp {
                t_down += d.event_elapsed_ms(ea.unwrap(), eb.unwrap())? as f64;
            }
            if sync_profile_enabled() {
                d.synchronize()?;
            }
            if std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                let mut h_x = vec![0.0f32; hidden];
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(h_x.as_mut_ptr().cast::<u8>(), hidden * 4)
                };
                d.copy_from_device(bytes, self.d_x)?;
                let mut sum = 0.0f64;
                let mut sum_sq = 0.0f64;
                for &v in &h_x {
                    sum_sq += (v as f64) * (v as f64);
                    sum += v as f64;
                }
                eprintln!(
                    "\x1b[35m[dump] layer_{layer_index}: sum={sum:.6} sum_sq={sum_sq:.6} len={hidden}\x1b[0m"
                );
            }
            record(&mut profiler, |p| &mut p.ffn_down, started);
        }

        if evp {
            if let Some(ev) = e_span_b {
                d.event_record(ev)?;
            }
            let span = d.event_elapsed_ms(e_span_a.unwrap(), e_span_b.unwrap())?;
            eprintln!(
                "\x1b[33m[ev] pos={pos}: norm={:.1}us qkv={:.1}us rope={:.1}us attn={:.1}us o={:.1}us ffn={:.1}us down={:.1}us (sum={:.1}us span={:.1}us gaps={:.1}us)\x1b[0m",
                t_norm * 1000.0,
                t_qkv * 1000.0,
                t_rope * 1000.0,
                t_scores * 1000.0,
                t_o * 1000.0,
                t_ffn * 1000.0,
                t_down * 1000.0,
                (t_norm + t_qkv + t_rope + t_scores + t_o + t_ffn + t_down) * 1000.0,
                span * 1000.0,
                (span as f64 - (t_norm + t_qkv + t_rope + t_scores + t_o + t_ffn + t_down)) * 1000.0
            );
            d.event_destroy(ea.unwrap());
            d.event_destroy(eb.unwrap());
            d.event_destroy(e_span_a.unwrap());
            d.event_destroy(e_span_b.unwrap());
        }

        Ok(())
    }

    fn final_norm_output(&self) -> Result<usize, String> {
        let output = self
            .weights
            .output
            .as_ref()
            .ok_or_else(|| "output weights not resident".to_string())?;
        // The last fused down-residual already produced final_norm in d_norm.
        if std::env::var_os("SPARK_NO_FUSED_RESNORM").is_some()
            || self.weights.layers.is_empty()
        {
            self.device.launch_rmsnorm_f32(
                self.d_norm,
                self.d_x,
                self.weights.final_norm,
                self.hidden_dim,
                self.rms_epsilon,
            )?;
        }
        self.launch_matrix(output, self.d_norm, self.d_logits)?;
        Ok(output.rows)
    }

    /// Resident prefill: run one token fully on the GPU and download logits.
    pub fn resident_forward_logits(
        &self,
        h_x: &[f32],
        pos: usize,
        h_logits: &mut [f32],
        mut profiler: Option<&mut ProfileStats>,
    ) -> Result<(), String> {
        if !self.resident {
            return Err("resident forward not available for this model".into());
        }
        self.upload_token_state(h_x, pos)?;
        self.run_layers(pos, profiler.as_deref_mut())?;
        let started = profiler.as_ref().and_then(|p| p.start());
        let rows = self.final_norm_output()?;
        let copy_len = rows.min(h_logits.len());
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(h_logits.as_mut_ptr().cast::<u8>(), copy_len * 4)
        };
        self.device.copy_from_device(bytes, self.d_logits)?;
        if std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
            let mut max_idx = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &v) in h_logits.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }
            eprintln!(
                "\x1b[35m[dump] gpu logits pos={}: argmax={} val={:.6} first8={:?}\x1b[0m",
                pos,
                max_idx,
                max_val,
                &h_logits[..8.min(h_logits.len())]
            );
        }
        record(&mut profiler, |p| &mut p.final_output, started);
        Ok(())
    }

    /// Resident decode: run one token fully on the GPU and sample on device.
    pub fn resident_forward_sample(
        &self,
        h_x: &[f32],
        pos: usize,
        temperature: f32,
        top_p: f32,
        rng_seed: f32,
        mut profiler: Option<&mut ProfileStats>,
    ) -> Result<u32, String> {
        if !self.resident {
            return Err("resident forward not available for this model".into());
        }
        self.upload_token_state(h_x, pos)?;

        if self.graphs_enabled() && profiler.is_none() {
            if let Some(exec) = self.decode_graph_exec(pos, temperature)? {
                // Replay (or first execute) the captured decode step.
                self.device.graph_launch(exec)?;
                return self.read_sampled_token(temperature, top_p, rng_seed);
            }
            // capture failed permanently -> eager path below
        }

        self.run_layers(pos, profiler.as_deref_mut())?;
        let started = profiler.as_ref().and_then(|p| p.start());
        let rows = self.final_norm_output()?;
        if temperature <= 0.0 {
            self.device.launch_argmax_f32(self.d_logits, self.d_token_out, rows)?;
            let token = self.read_argmax_token()?;
            record(&mut profiler, |p| &mut p.final_output, started);
            Ok(token)
        } else {
            let inv_temp = 1.0 / temperature.max(f32::MIN_POSITIVE);
            self.device.launch_top64_sample(
                self.d_logits,
                self.d_top64_tokens,
                self.d_top64_logits,
                rows,
                inv_temp,
            )?;
            let tokens = unsafe { &mut *self.h_top64_tokens.get() };
            let logits = unsafe { &mut *self.h_top64_logits.get() };
            let bytes_tokens = unsafe {
                std::slice::from_raw_parts_mut(tokens.as_mut_ptr().cast::<u8>(), 64 * 4)
            };
            let bytes_logits = unsafe {
                std::slice::from_raw_parts_mut(logits.as_mut_ptr().cast::<u8>(), 64 * 4)
            };
            self.device.copy_from_device(bytes_tokens, self.d_top64_tokens)?;
            self.device.copy_from_device(bytes_logits, self.d_top64_logits)?;
            record(&mut profiler, |p| &mut p.final_output, started);
            Ok(self.pick_top64(top_p, rng_seed))
        }
    }

    /// Whether the decode CUDA graph may be used. Debug instrumentation paths
    /// that copy device memory mid-forward or sync the stream are incompatible
    /// with capture and disable it.
    fn graphs_enabled(&self) -> bool {
        std::env::var_os("SPARK_NO_GRAPH").is_none()
            && std::env::var_os("SPARK_EVENT_PROFILE").is_none()
            && std::env::var_os("SPARK_SYNC_PROFILE").is_none()
            && std::env::var_os("SPARK_DUMP_LAYERS").is_none()
    }

    /// Return the captured decode graph for the given sampling flavor,
    /// capturing it on first use. `Ok(None)` means capture failed and the
    /// caller must use the eager path.
    fn decode_graph_exec(&self, pos: usize, temperature: f32) -> Result<Option<usize>, String> {
        let (key, cap_key) = if temperature <= 0.0 {
            (&self.graph_exec_argmax, &self.graph_cap_argmax)
        } else {
            (&self.graph_exec_top64, &self.graph_cap_top64)
        };
        let mut cur = key.load(Ordering::Relaxed);
        if cur != 0 {
            if cur == usize::MAX {
                return Ok(None);
            }
            let cap = cap_key.load(Ordering::Relaxed);
            if pos < cap {
                return Ok(Some(cur));
            }
            // The sequence crossed this graph's QK grid bucket. Replace it
            // with the next power-of-two graph instead of using max_ctx from
            // the first token onward.
            self.device.destroy_graph_exec(cur);
            key.store(0, Ordering::Relaxed);
            cap_key.store(0, Ordering::Relaxed);
            cur = 0;
        }
        debug_assert_eq!(cur, 0);
        let capture_positions = (pos + 1)
            .next_power_of_two()
            .max(64)
            .min(self.ctx_len);
        self.capture_attn_positions
            .store(capture_positions, Ordering::Relaxed);
        self.capturing.store(true, Ordering::Relaxed);
        let capture_started = Instant::now();
        let res = (|| -> Result<usize, String> {
            self.device.begin_capture()?;
            self.run_layers(pos, None)?;
            let rows = self.final_norm_output()?;
            if temperature <= 0.0 {
                self.device
                    .launch_argmax_f32(self.d_logits, self.d_token_out, rows)?;
            } else {
                let inv_temp = 1.0 / temperature.max(f32::MIN_POSITIVE);
                self.device.launch_top64_sample(
                    self.d_logits,
                    self.d_top64_tokens,
                    self.d_top64_logits,
                    rows,
                    inv_temp,
                )?;
            }
            let graph = self.device.end_capture()?;
            let exec = self.device.instantiate_graph(graph)?;
            self.device.destroy_graph(graph);
            Ok(exec)
        })();
        self.capturing.store(false, Ordering::Relaxed);
        match res {
            Ok(exec) => {
                key.store(exec, Ordering::Relaxed);
                cap_key.store(capture_positions, Ordering::Relaxed);
                if std::env::var_os("SPARK_GRAPH_DEBUG").is_some() {
                    eprintln!(
                        "[spark] captured decode graph: positions={capture_positions}, {:.2} ms",
                        capture_started.elapsed().as_secs_f64() * 1e3
                    );
                }
                Ok(Some(exec))
            }
            Err(e) => {
                key.store(usize::MAX, Ordering::Relaxed);
                eprintln!("\x1b[33m[spark] CUDA graph capture failed ({e}); using eager launches\x1b[0m");
                Ok(None)
            }
        }
    }

    /// Read the sampling result produced by the (replayed) decode graph.
    fn read_sampled_token(
        &self,
        temperature: f32,
        top_p: f32,
        rng_seed: f32,
    ) -> Result<u32, String> {
        if temperature <= 0.0 {
            self.read_argmax_token()
        } else {
            let tokens = unsafe { &mut *self.h_top64_tokens.get() };
            let logits = unsafe { &mut *self.h_top64_logits.get() };
            let bytes_tokens = unsafe {
                std::slice::from_raw_parts_mut(tokens.as_mut_ptr().cast::<u8>(), 64 * 4)
            };
            let bytes_logits = unsafe {
                std::slice::from_raw_parts_mut(logits.as_mut_ptr().cast::<u8>(), 64 * 4)
            };
            self.device.copy_from_device(bytes_tokens, self.d_top64_tokens)?;
            self.device.copy_from_device(bytes_logits, self.d_top64_logits)?;
            Ok(self.pick_top64(top_p, rng_seed))
        }
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        // Stop the spinning keepalive kernel first: cuMemFree may implicitly
        // synchronize with the whole context and would deadlock on it.
        self.device.stop_persistent_keepalive();
        for key in [&self.graph_exec_argmax, &self.graph_exec_top64] {
            let v = key.load(Ordering::Relaxed);
            if v != 0 && v != usize::MAX {
                self.device.destroy_graph_exec(v);
            }
        }
        self.device.free(self.d_norm);
        self.device.free(self.d_qkv);
        self.device.free(self.d_context);
        self.device.free(self.d_ffn_out);
        self.device.free(self.d_ffn_up);
        self.device.free(self.d_out);
        self.device.free(self.d_logits);
        self.device.free_mapped(self.h_token_out.cast::<u8>());
        self.device.free(self.d_top64_tokens);
        self.device.free(self.d_top64_logits);
        self.device.free(self.d_x);
        self.device.free(self.d_scores);
        self.device.free(self.d_kv_k);
        self.device.free(self.d_kv_v);
    }
}

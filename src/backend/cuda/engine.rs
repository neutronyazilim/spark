use std::sync::Arc;

use super::{driver::CudaDevice, weights::{CudaMatrix, CudaModelWeights}};
use crate::model::config::ModelConfig;
use crate::model::ops;

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
    pub d_top64_tokens: usize,
    pub d_top64_logits: usize,
    pub h_qkv_buf: std::cell::UnsafeCell<Vec<f32>>,
    pub h_ffn_gate: std::cell::UnsafeCell<Vec<f32>>,
    pub h_ffn_up: std::cell::UnsafeCell<Vec<f32>>,
    pub h_top64_tokens: std::cell::UnsafeCell<[u32; 64]>,
    pub h_top64_logits: std::cell::UnsafeCell<[f32; 64]>,
    pub hidden_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub total_qkv: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
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
        let d_token_out = device.alloc(4)?;
        let d_top64_tokens = device.alloc(64 * 4)?;
        let d_top64_logits = device.alloc(64 * 4)?;

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
            d_top64_tokens,
            d_top64_logits,
            h_qkv_buf: std::cell::UnsafeCell::new(vec![0.0f32; max_qkv_rows]),
            h_ffn_gate: std::cell::UnsafeCell::new(vec![0.0f32; max_intermediate]),
            h_ffn_up: std::cell::UnsafeCell::new(vec![0.0f32; max_intermediate]),
            h_top64_tokens: std::cell::UnsafeCell::new([0u32; 64]),
            h_top64_logits: std::cell::UnsafeCell::new([0.0f32; 64]),
            hidden_dim: max_hidden,
            q_dim: config.heads * config.head_dim,
            kv_dim: config.kv_heads * config.head_dim,
            total_qkv: max_qkv_rows,
            intermediate_dim: max_intermediate,
            vocab_size: max_vocab,
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
            // separate GEMVs, apply SwiGLU on the host, then run the down GEMV
            // on the activated activations.
            self.launch_matrix(&layer.gate, self.d_norm, self.d_ffn_out)?;
            self.launch_matrix(&layer.up, self.d_norm, self.d_ffn_up)?;

            let host_gate = unsafe { &mut *self.h_ffn_gate.get() };
            let host_up = unsafe { &mut *self.h_ffn_up.get() };

            let gate_rows = layer.gate.rows.min(host_gate.len());
            let bytes_gate = unsafe {
                std::slice::from_raw_parts_mut(host_gate.as_mut_ptr().cast::<u8>(), gate_rows * 4)
            };
            self.device.copy_from_device(bytes_gate, self.d_ffn_out)?;

            let up_rows = layer.up.rows.min(host_up.len());
            let bytes_up = unsafe {
                std::slice::from_raw_parts_mut(host_up.as_mut_ptr().cast::<u8>(), up_rows * 4)
            };
            self.device.copy_from_device(bytes_up, self.d_ffn_up)?;

            let act_rows = gate_rows.min(up_rows);
            ops::swiglu(&mut host_gate[..act_rows], &host_up[..act_rows]);

            let bytes_gate = unsafe {
                std::slice::from_raw_parts(host_gate.as_ptr().cast::<u8>(), gate_rows * 4)
            };
            self.device.copy_to_device(self.d_ffn_out, bytes_gate)?;
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
            let mut token_bytes = [0u8; 4];
            self.device.copy_from_device(&mut token_bytes, self.d_token_out)?;
            Ok(u32::from_le_bytes(token_bytes))
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

            Ok(selected)
        }
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        self.device.free(self.d_norm);
        self.device.free(self.d_qkv);
        self.device.free(self.d_context);
        self.device.free(self.d_ffn_out);
        self.device.free(self.d_ffn_up);
        self.device.free(self.d_out);
        self.device.free(self.d_logits);
        self.device.free(self.d_token_out);
        self.device.free(self.d_top64_tokens);
        self.device.free(self.d_top64_logits);
    }
}

use super::{
    config::ModelConfig,
    kv_cache::KvCache,
    ops::{hadamard_transform_1024, rmsnorm, rope, softmax},
    weights::{ModelWeights, QuantizedMatrix, Qwen35AttentionWeights},
};
use crate::backend::{CudaEngine, CudaQ1Matrix, CudaQwen35Attention, CudaQwen35Engine};
use crate::kernel::pool::ThreadPool;
use crate::memory::arena::Arena;
use crate::profile::ProfileStats;

pub struct Transformer<'a> {
    pub config: ModelConfig,
    pub weights: ModelWeights<'a>,
    pub pool: ThreadPool,
    pub cuda: Option<CudaEngine>,
    pub qwen35_cuda: Option<CudaQwen35Engine>,
}

impl<'a> Transformer<'a> {
    pub fn forward_token<'b>(
        &self,
        token: u32,
        pos: usize,
        cache: &mut KvCache,
        arena: &'b mut Arena,
        mut profiler: Option<&mut ProfileStats>,
    ) -> Result<&'b [f32], String> {
        let forward_started = profiler.as_ref().and_then(|p| p.start());
        if pos >= self.config.context_len {
            return Err("position exceeds configured context".into());
        }

        if self.config.is_qwen35 {
            if let Some(cuda) = self.qwen35_cuda.as_ref().filter(|cuda| cuda.resident) {
                let hidden = self.config.hidden_dim;
                let max_vocab = self.weights.output.rows().max(self.config.vocab_size);
                arena.reset();
                let scratch = arena
                    .alloc_slice::<f32>(hidden + max_vocab)
                    .ok_or_else(|| "scratch arena is too small".to_string())?;
                let (x, rest) = scratch.split_at_mut(hidden);
                let logits = &mut rest[..max_vocab];
                embedding(&self.weights.token_embedding, token as usize, x)?;
                cuda.resident_forward_logits(x, pos, logits)?;
                record(
                    &mut profiler,
                    |profile| &mut profile.resident_cuda,
                    forward_started,
                );
                return Ok(logits);
            }
            return self.forward_token_qwen35(
                token,
                pos,
                cache,
                arena,
                profiler.as_deref_mut(),
                forward_started,
            );
        }

        if let Some(cuda) = self.cuda.as_ref().filter(|c| c.resident) {
            // Fully device-resident path: only the embedding row goes up and
            // logits come back.
            let hidden = self.config.hidden_dim;
            let max_vocab = self.weights.output.rows().max(self.config.vocab_size);
            arena.reset();
            let scratch = arena
                .alloc_slice::<f32>(hidden + max_vocab)
                .ok_or_else(|| "scratch arena is too small".to_string())?;
            let (x, rest) = scratch.split_at_mut(hidden);
            let logits = &mut rest[..max_vocab];
            embedding(&self.weights.token_embedding, token as usize, x)?;
            debug_hidden("embedding", x);
            cuda.resident_forward_logits(x, pos, logits, profiler.as_deref_mut())?;
            record(&mut profiler, |p| &mut p.forward, forward_started);
            return Ok(logits);
        }

        let hidden = self.config.hidden_dim;
        let kv_dim = self.config.kv_heads * self.config.head_dim;
        let q_dim = self.config.heads * self.config.head_dim;
        let max_qkv = self.weights.layers.iter().map(|l| l.q.rows()).max().unwrap_or(q_dim).max(q_dim);
        let max_ctx = self.weights.layers.iter().map(|l| l.o.cols()).max().unwrap_or(q_dim).max(q_dim);
        let max_inter = self.weights.layers.iter().map(|l| l.gate.rows().max(l.down.cols())).max().unwrap_or(self.config.intermediate_dim).max(self.config.intermediate_dim);
        let max_vocab = self.weights.output.rows().max(self.config.vocab_size);

        let total = hidden * 4
            + max_qkv * 2
            + max_ctx * 2
            + kv_dim * 2
            + self.config.context_len
            + max_inter * 2
            + max_vocab;
        arena.reset();
        let scratch = arena
            .alloc_slice::<f32>(total)
            .ok_or_else(|| "scratch arena is too small".to_string())?;
        let (x, rest) = scratch.split_at_mut(hidden);
        let (norm, rest) = rest.split_at_mut(hidden);
        let (q, rest) = rest.split_at_mut(max_qkv);
        let (k, rest) = rest.split_at_mut(kv_dim);
        let (v, rest) = rest.split_at_mut(kv_dim);
        let (scores, rest) = rest.split_at_mut(self.config.context_len);
        let (context, rest) = rest.split_at_mut(max_ctx);
        let (gate, rest) = rest.split_at_mut(max_inter);
        // logits must be exactly the vocabulary length: gemv validation and
        // CPU-side sampling must not see leftover scratch slack.
        let logits = &mut rest[..max_vocab];

        embedding(&self.weights.token_embedding, token as usize, x)?;
        debug_hidden("embedding", x);
        let offloaded_gpu_layers = self.cuda.as_ref().map_or(0, |c| c.weights.offloaded_layers);
        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            let on_gpu = layer_index < offloaded_gpu_layers;

            let started = profiler.as_ref().and_then(|p| p.start());
            rmsnorm(norm, x, layer.attn_norm.values(), self.config.rms_epsilon);
            if self.config.has_hadamard {
                hadamard_transform_1024(norm);
            }
            record(&mut profiler, |p| &mut p.rmsnorm, started);
            debug_activation("attn_norm", layer_index, pos, norm);

            let started = profiler.as_ref().and_then(|p| p.start());
            let overhead_before = profiler
                .as_ref()
                .map_or(std::time::Duration::ZERO, |p| p.pool_overhead);

            if on_gpu {
                if let Some(cuda) = &self.cuda {
                    cuda.gemv_qkv(layer_index, norm, q, k, v)?;
                }
            } else {
                layer
                    .gemv_qkv_parallel(norm, q, k, v, &self.pool, profiler.as_deref_mut())
                    .map_err(|error| error.to_string())?;
            }
            record_gemv(&mut profiler, |p| &mut p.qkv, started, overhead_before);

            add_bias(q, layer.q_bias.values());
            add_bias(k, layer.k_bias.values());
            add_bias(v, layer.v_bias.values());
            if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                debug_hidden(&format!("qkv_bias pos={pos}"), &q[..q_dim]);
                debug_hidden(&format!("k_bias pos={pos}"), &k[..kv_dim]);
                debug_hidden(&format!("v_bias pos={pos}"), &v[..kv_dim]);
            }

            let started = profiler.as_ref().and_then(|p| p.start());
            rope(q, pos, self.config.head_dim, self.config.rope_base);
            rope(k, pos, self.config.head_dim, self.config.rope_base);
            if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                debug_hidden(&format!("q_rope pos={pos}"), &q[..q_dim]);
                debug_hidden(&format!("k_rope pos={pos}"), &k[..kv_dim]);
            }
            cache
                .store(layer_index, pos, k, v)
                .map_err(str::to_string)?;
            attention(
                q,
                context,
                scores,
                cache,
                layer_index,
                pos,
                self.config.heads,
                self.config.kv_heads,
                self.config.head_dim,
            );
            if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                debug_hidden(&format!("context pos={pos}"), &context[..q_dim]);
            }
            record(&mut profiler, |p| &mut p.attention, started);

            let started = profiler.as_ref().and_then(|p| p.start());
            let overhead_before = profiler
                .as_ref()
                .map_or(std::time::Duration::ZERO, |p| p.pool_overhead);

            if on_gpu {
                if let Some(cuda) = &self.cuda {
                    cuda.gemv_attn_out(layer_index, context, q)?;
                }
            } else {
                layer
                    .o
                    .gemv_parallel(context, q, &self.pool, profiler.as_deref_mut())
                    .map_err(|error| error.to_string())?;
            }
            record_gemv(
                &mut profiler,
                |p| &mut p.attention_output,
                started,
                overhead_before,
            );
            debug_activation("attention", layer_index, pos, q);
            add_in_place(x, q);

            let started = profiler.as_ref().and_then(|p| p.start());
            rmsnorm(norm, x, layer.ffn_norm.values(), self.config.rms_epsilon);
            if self.config.has_hadamard {
                hadamard_transform_1024(norm);
            }
            record(&mut profiler, |p| &mut p.rmsnorm, started);

            let started = profiler.as_ref().and_then(|p| p.start());
            let overhead_before = profiler
                .as_ref()
                .map_or(std::time::Duration::ZERO, |p| p.pool_overhead);

            if on_gpu {
                if let Some(cuda) = &self.cuda {
                    cuda.ffn_fused_down(layer_index, norm, q)?;
                }
            } else {
                layer
                    .gemv_fused_ffn_parallel(norm, gate, &self.pool, profiler.as_deref_mut())
                    .map_err(|error| error.to_string())?;
                layer
                    .down
                    .gemv_parallel(gate, q, &self.pool, profiler.as_deref_mut())
                    .map_err(|error| error.to_string())?;
            }
            record_gemv(
                &mut profiler,
                |p| &mut p.ffn_gate_up,
                started,
                overhead_before,
            );
            debug_activation("ffn", layer_index, pos, q);
            add_in_place(x, q);
            debug_hidden(&format!("layer_{layer_index}"), x);
        }

        let started = profiler.as_ref().and_then(|p| p.start());
        rmsnorm(
            norm,
            x,
            self.weights.final_norm.values(),
            self.config.rms_epsilon,
        );
        if self.config.has_hadamard {
            hadamard_transform_1024(norm);
        }
        record(&mut profiler, |p| &mut p.rmsnorm, started);

        let started = profiler.as_ref().and_then(|p| p.start());
        let overhead_before = profiler
            .as_ref()
            .map_or(std::time::Duration::ZERO, |p| p.pool_overhead);

        let has_gpu_output = self.cuda.as_ref().is_some_and(|c| c.weights.output.is_some());
        if has_gpu_output {
            if let Some(cuda) = &self.cuda {
                cuda.gemv_output(norm, logits)?;
            }
        } else {
            self.weights
                .output
                .gemv_parallel(norm, logits, &self.pool, profiler.as_deref_mut())
                .map_err(|error| error.to_string())?;
        }
        record_gemv(
            &mut profiler,
            |p| &mut p.final_output,
            started,
            overhead_before,
        );
        if std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
            let mut max_idx = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }
            eprintln!(
                "\x1b[35m[dump] cpu logits pos={pos}: argmax={max_idx} val={max_val:.6} first8={:?}\x1b[0m",
                &logits[..8.min(logits.len())]
            );
        }

        record(&mut profiler, |p| &mut p.forward, forward_started);
        debug_activation("final_logits", 0, pos, logits);
        Ok(logits)
    }

    fn forward_token_qwen35<'b>(
        &self,
        token: u32,
        pos: usize,
        cache: &mut KvCache,
        arena: &'b mut Arena,
        mut profiler: Option<&mut ProfileStats>,
        forward_started: Option<std::time::Instant>,
    ) -> Result<&'b [f32], String> {
        let cfg = &self.config;
        let use_avx_gdn = (self.qwen35_cuda.is_some()
            && std::env::var_os("SPARK_NO_GDN_AVX").is_none())
            || std::env::var_os("SPARK_GDN_AVX").is_some();
        let hidden = cfg.hidden_dim;
        let attn_dim = cfg.heads * cfg.head_dim;
        let qg_dim = attn_dim * 2;
        let full_kv_dim = cfg.kv_heads * cfg.head_dim;
        let key_dim = cfg.ssm_group_count * cfg.ssm_state_size;
        let conv_dim = cfg.ssm_inner_size + 2 * key_dim;
        let value_head_dim = cfg.ssm_inner_size / cfg.ssm_time_step_rank;
        let proj_dim = qg_dim.max(conv_dim);
        let key_buf_dim = key_dim.max(full_kv_dim);
        let vocab = self.weights.output.rows().max(cfg.vocab_size);
        let total = hidden * 3
            + proj_dim * 2
            + attn_dim * 2
            + key_buf_dim
            + cfg.ssm_inner_size * 3
            + cfg.ssm_time_step_rank * 2
            + cfg.intermediate_dim * 2
            + cfg.context_len
            + vocab;

        arena.reset();
        let scratch = arena
            .alloc_slice::<f32>(total)
            .ok_or_else(|| "scratch arena is too small for Qwen35".to_string())?;
        let (x, rest) = scratch.split_at_mut(hidden);
        let (norm, rest) = rest.split_at_mut(hidden);
        let (out, rest) = rest.split_at_mut(hidden);
        let (projection, rest) = rest.split_at_mut(proj_dim);
        let (conv_output, rest) = rest.split_at_mut(proj_dim);
        let (q, rest) = rest.split_at_mut(attn_dim);
        let (attn_gate, rest) = rest.split_at_mut(attn_dim);
        let (k, rest) = rest.split_at_mut(key_buf_dim);
        let (v, rest) = rest.split_at_mut(cfg.ssm_inner_size);
        let (context, rest) = rest.split_at_mut(cfg.ssm_inner_size);
        let (z, rest) = rest.split_at_mut(cfg.ssm_inner_size);
        let (alpha, rest) = rest.split_at_mut(cfg.ssm_time_step_rank);
        let (beta, rest) = rest.split_at_mut(cfg.ssm_time_step_rank);
        let (ffn_gate, rest) = rest.split_at_mut(cfg.intermediate_dim);
        let (ffn_up, rest) = rest.split_at_mut(cfg.intermediate_dim);
        let (scores, rest) = rest.split_at_mut(cfg.context_len);
        let logits = &mut rest[..vocab];

        embedding(&self.weights.token_embedding, token as usize, x)?;
        debug_hidden("qwen35_embedding", x);

        for (layer_index, layer) in self.weights.qwen35_layers.iter().enumerate() {
            let cuda_layer = self
                .qwen35_cuda
                .as_ref()
                .and_then(|cuda| cuda.weights.layers.get(layer_index));
            let started = profiler.as_ref().and_then(|p| p.start());
            rmsnorm(norm, x, layer.attn_norm.values(), cfg.rms_epsilon);
            record(&mut profiler, |p| &mut p.rmsnorm, started);

            match layer.attention {
                Qwen35AttentionWeights::Full {
                    q: q_weight,
                    k: k_weight,
                    v: v_weight,
                    output,
                    q_norm,
                    k_norm,
                } => {
                    let started = profiler.as_ref().and_then(|p| p.start());
                    let gpu = match cuda_layer.map(|layer| &layer.attention) {
                        Some(CudaQwen35Attention::Full { q, k, v, output, .. }) => {
                            (Some(q), Some(k), Some(v), Some(output))
                        }
                        _ => (None, None, None, None),
                    };
                    if let (Some(cuda), Some(gq), Some(gk), Some(gv)) =
                        (self.qwen35_cuda.as_ref(), gpu.0, gpu.1, gpu.2)
                    {
                        let mut outputs: [&mut [f32]; 3] = [
                            &mut projection[..qg_dim],
                            &mut k[..full_kv_dim],
                            &mut v[..full_kv_dim],
                        ];
                        cuda.gemv_many(&[gq, gk, gv], norm, &mut outputs)?;
                    } else {
                        qwen35_gemv(&self.pool, None, &q_weight, None, norm, &mut projection[..qg_dim])?;
                        qwen35_gemv(&self.pool, None, &k_weight, None, norm, &mut k[..full_kv_dim])?;
                        qwen35_gemv(&self.pool, None, &v_weight, None, norm, &mut v[..full_kv_dim])?;
                    }
                    record(&mut profiler, |p| &mut p.qkv, started);

                    let started = profiler.as_ref().and_then(|p| p.start());
                    for head in 0..cfg.heads {
                        let src = &projection[head * cfg.head_dim * 2..(head + 1) * cfg.head_dim * 2];
                        q[head * cfg.head_dim..(head + 1) * cfg.head_dim]
                            .copy_from_slice(&src[..cfg.head_dim]);
                        attn_gate[head * cfg.head_dim..(head + 1) * cfg.head_dim]
                            .copy_from_slice(&src[cfg.head_dim..]);
                    }
                    rmsnorm_heads_in_place(
                        &mut q[..attn_dim],
                        q_norm.values(),
                        cfg.head_dim,
                        cfg.rms_epsilon,
                    );
                    rmsnorm_heads_in_place(
                        &mut k[..full_kv_dim],
                        k_norm.values(),
                        cfg.head_dim,
                        cfg.rms_epsilon,
                    );
                    rope_partial(
                        &mut q[..attn_dim],
                        pos,
                        cfg.head_dim,
                        cfg.rope_dimension_count,
                        cfg.rope_base,
                    );
                    rope_partial(
                        &mut k[..full_kv_dim],
                        pos,
                        cfg.head_dim,
                        cfg.rope_dimension_count,
                        cfg.rope_base,
                    );
                    cache
                        .store(layer_index, pos, &k[..full_kv_dim], &v[..full_kv_dim])
                        .map_err(str::to_string)?;
                    attention(
                        &q[..attn_dim],
                        &mut context[..attn_dim],
                        scores,
                        cache,
                        layer_index,
                        pos,
                        cfg.heads,
                        cfg.kv_heads,
                        cfg.head_dim,
                    );
                    for (value, &gate) in context[..attn_dim]
                        .iter_mut()
                        .zip(&attn_gate[..attn_dim])
                    {
                        *value *= sigmoid(gate);
                    }
                    record(&mut profiler, |p| &mut p.attention, started);
                    let started = profiler.as_ref().and_then(|p| p.start());
                    qwen35_gemv(
                        &self.pool,
                        self.qwen35_cuda.as_ref(),
                        &output,
                        gpu.3,
                        &context[..attn_dim],
                        out,
                    )?;
                    record(&mut profiler, |p| &mut p.attention_output, started);
                }
                Qwen35AttentionWeights::Recurrent {
                    qkv,
                    z: z_weight,
                    alpha: alpha_weight,
                    beta: beta_weight,
                    conv1d,
                    dt_bias,
                    a,
                    state_norm,
                    output,
                } => {
                    let started = profiler.as_ref().and_then(|p| p.start());
                    let gpu = match cuda_layer.map(|layer| &layer.attention) {
                        Some(CudaQwen35Attention::Recurrent { qkv, z, alpha, beta, output, .. }) => {
                            (Some(qkv), Some(z), Some(alpha), Some(beta), Some(output))
                        }
                        _ => (None, None, None, None, None),
                    };
                    if let (Some(cuda), Some(gqkv), Some(gz), Some(ga), Some(gb)) =
                        (self.qwen35_cuda.as_ref(), gpu.0, gpu.1, gpu.2, gpu.3)
                    {
                        let mut outputs: [&mut [f32]; 4] = [
                            &mut projection[..conv_dim],
                            &mut z[..cfg.ssm_inner_size],
                            alpha,
                            beta,
                        ];
                        cuda.gemv_many(&[gqkv, gz, ga, gb], norm, &mut outputs)?;
                    } else {
                        qwen35_gemv(&self.pool, None, &qkv, None, norm, &mut projection[..conv_dim])?;
                        qwen35_gemv(&self.pool, None, &z_weight, None, norm, &mut z[..cfg.ssm_inner_size])?;
                        qwen35_gemv(&self.pool, None, &alpha_weight, None, norm, alpha)?;
                        qwen35_gemv(&self.pool, None, &beta_weight, None, norm, beta)?;
                    }
                    record(&mut profiler, |p| &mut p.qkv, started);

                    if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                        debug_hidden("qwen35_l0_attn_norm", norm);
                        debug_hidden("qwen35_l0_qkv", &projection[..conv_dim]);
                        debug_hidden("qwen35_l0_z", &z[..cfg.ssm_inner_size]);
                        debug_hidden("qwen35_l0_alpha_raw", alpha);
                        debug_hidden("qwen35_l0_beta_raw", beta);
                    }

                    let started = profiler.as_ref().and_then(|p| p.start());
                    for value in beta.iter_mut() {
                        *value = sigmoid(*value);
                    }
                    for ((value, &bias), &decay) in alpha
                        .iter_mut()
                        .zip(dt_bias.values())
                        .zip(a.values())
                    {
                        *value = softplus(*value + bias) * decay;
                    }

                    let (conv_state, recurrent_state) = cache
                        .recurrent_state_mut(layer_index)
                        .map_err(str::to_string)?;
                    causal_conv_silu(
                        &projection[..conv_dim],
                        conv1d.values(),
                        conv_state,
                        &mut conv_output[..conv_dim],
                        cfg.ssm_conv_kernel,
                    );
                    if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                        debug_hidden("qwen35_l0_conv_silu", &conv_output[..conv_dim]);
                    }
                    q[..key_dim].copy_from_slice(&conv_output[..key_dim]);
                    k[..key_dim].copy_from_slice(&conv_output[key_dim..2 * key_dim]);
                    v[..cfg.ssm_inner_size]
                        .copy_from_slice(&conv_output[2 * key_dim..conv_dim]);
                    l2_norm_heads_in_place(
                        &mut q[..key_dim],
                        cfg.ssm_state_size,
                        cfg.rms_epsilon,
                    );
                    l2_norm_heads_in_place(
                        &mut k[..key_dim],
                        cfg.ssm_state_size,
                        cfg.rms_epsilon,
                    );
                    if self
                        .qwen35_cuda
                        .as_ref()
                        .is_some_and(|cuda| cuda.has_gpu_gdn(layer_index))
                    {
                        self.qwen35_cuda.as_ref().unwrap().gated_delta(
                            layer_index,
                            &q[..key_dim],
                            &k[..key_dim],
                            &v[..cfg.ssm_inner_size],
                            alpha,
                            beta,
                            &mut context[..cfg.ssm_inner_size],
                            cfg.ssm_group_count,
                            cfg.ssm_time_step_rank,
                            value_head_dim,
                        )?;
                    } else {
                        self.pool.gated_delta_net_step(
                            &q[..key_dim],
                            &k[..key_dim],
                            &v[..cfg.ssm_inner_size],
                            alpha,
                            beta,
                            recurrent_state,
                            &mut context[..cfg.ssm_inner_size],
                            cfg.ssm_group_count,
                            cfg.ssm_time_step_rank,
                            value_head_dim,
                            use_avx_gdn,
                        );
                    }
                    if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                        debug_hidden("qwen35_l0_delta_out", &context[..cfg.ssm_inner_size]);
                    }
                    gated_head_rmsnorm(
                        &mut context[..cfg.ssm_inner_size],
                        &z[..cfg.ssm_inner_size],
                        state_norm.values(),
                        value_head_dim,
                        cfg.rms_epsilon,
                    );
                    if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                        debug_hidden("qwen35_l0_final_output", &context[..cfg.ssm_inner_size]);
                    }
                    record(&mut profiler, |p| &mut p.attention, started);
                    let started = profiler.as_ref().and_then(|p| p.start());
                    qwen35_gemv(
                        &self.pool,
                        self.qwen35_cuda.as_ref(),
                        &output,
                        gpu.4,
                        &context[..cfg.ssm_inner_size],
                        out,
                    )?;
                    record(&mut profiler, |p| &mut p.attention_output, started);
                }
            }

            if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                debug_hidden("qwen35_l0_attn_out", out);
            }

            add_in_place(x, out);
            if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                debug_hidden("qwen35_l0_attn_residual", x);
            }
            let started = profiler.as_ref().and_then(|p| p.start());
            rmsnorm(norm, x, layer.post_attention_norm.values(), cfg.rms_epsilon);
            record(&mut profiler, |p| &mut p.rmsnorm, started);
            let started = profiler.as_ref().and_then(|p| p.start());
            if let (Some(cuda), Some(gpu_layer)) = (self.qwen35_cuda.as_ref(), cuda_layer) {
                let mut outputs: [&mut [f32]; 2] = [ffn_gate, ffn_up];
                cuda.gemv_many(&[&gpu_layer.gate, &gpu_layer.up], norm, &mut outputs)?;
            } else {
                qwen35_gemv(&self.pool, None, &layer.gate, None, norm, ffn_gate)?;
                qwen35_gemv(&self.pool, None, &layer.up, None, norm, ffn_up)?;
            }
            super::ops::swiglu(ffn_gate, ffn_up);
            record(&mut profiler, |p| &mut p.ffn_gate_up, started);
            let started = profiler.as_ref().and_then(|p| p.start());
            qwen35_gemv(
                &self.pool,
                self.qwen35_cuda.as_ref(),
                &layer.down,
                cuda_layer.map(|layer| &layer.down),
                ffn_gate,
                out,
            )?;
            record(&mut profiler, |p| &mut p.ffn_down, started);
            if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                debug_hidden("qwen35_l0_ffn_out", out);
            }
            add_in_place(x, out);

            if std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                debug_hidden(&format!("qwen35_layer_{layer_index}"), x);
            }
        }

        let started = profiler.as_ref().and_then(|p| p.start());
        rmsnorm(norm, x, self.weights.final_norm.values(), cfg.rms_epsilon);
        record(&mut profiler, |p| &mut p.rmsnorm, started);
        let started = profiler.as_ref().and_then(|p| p.start());
        qwen35_gemv(
            &self.pool,
            self.qwen35_cuda.as_ref(),
            &self.weights.output,
            self.qwen35_cuda.as_ref().and_then(|cuda| cuda.weights.output.as_ref()),
            norm,
            logits,
        )?;
        record(&mut profiler, |p| &mut p.final_output, started);
        if std::env::var_os("SPARK_DEBUG_LOGITS").is_some() {
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.select_nth_unstable_by(10, |&a, &b| logits[b].total_cmp(&logits[a]));
            order[..10].sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            eprintln!(
                "[qwen35 logits pos={pos}] {:?}",
                order[..10]
                    .iter()
                    .map(|&id| (id, logits[id]))
                    .collect::<Vec<_>>()
            );
        }
        record(&mut profiler, |p| &mut p.forward, forward_started);
        Ok(logits)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_token_and_sample(
        &self,
        token: u32,
        pos: usize,
        cache: &mut KvCache,
        arena: &mut Arena,
        temperature: f32,
        top_p: f32,
        rng_seed: f32,
        mut profiler: Option<&mut ProfileStats>,
    ) -> Result<u32, String> {
        let forward_started = profiler.as_ref().and_then(|p| p.start());
        if pos >= self.config.context_len {
            return Err("position exceeds configured context".into());
        }

        if self.config.is_qwen35 {
            if let Some(cuda) = self.qwen35_cuda.as_ref().filter(|cuda| cuda.resident) {
                let hidden = self.config.hidden_dim;
                arena.reset();
                let x = arena
                    .alloc_slice::<f32>(hidden)
                    .ok_or_else(|| "scratch arena is too small".to_string())?;
                embedding(&self.weights.token_embedding, token as usize, x)?;
                let selected = cuda.resident_forward_sample(
                    x,
                    pos,
                    temperature,
                    top_p,
                    rng_seed,
                )?;
                record(
                    &mut profiler,
                    |profile| &mut profile.resident_cuda,
                    forward_started,
                );
                return Ok(selected);
            }
        }

        if let Some(cuda) = self.cuda.as_ref().filter(|c| c.resident) {
            let hidden = self.config.hidden_dim;
            arena.reset();
            let scratch = arena
                .alloc_slice::<f32>(hidden)
                .ok_or_else(|| "scratch arena is too small".to_string())?;
            embedding(&self.weights.token_embedding, token as usize, scratch)?;
            let selected = cuda.resident_forward_sample(
                scratch,
                pos,
                temperature,
                top_p,
                rng_seed,
                profiler.as_deref_mut(),
            )?;
            record(&mut profiler, |p| &mut p.forward, forward_started);
            return Ok(selected);
        }

        let cuda = self.cuda.as_ref().ok_or_else(|| "CUDA engine not initialized".to_string())?;
        // This path only supports the fully resident model: every layer *and*
        // the output projection must live in VRAM. Partial offload and CPU-only
        // models go through forward_token + host sampling instead.
        if cuda.weights.offloaded_layers < self.weights.layers.len()
            || cuda.weights.output.is_none()
        {
            return Err(
            "forward_token_and_sample requires the fully resident model; hybrid offload must use forward_token + host sampler"
                .into(),
        );
        }
        let hidden = self.config.hidden_dim;
        let kv_dim = self.config.kv_heads * self.config.head_dim;
        let q_dim = self.config.heads * self.config.head_dim;
        let max_qkv = self.weights.layers.iter().map(|l| l.q.rows()).max().unwrap_or(q_dim).max(q_dim);
        let max_ctx = self.weights.layers.iter().map(|l| l.o.cols()).max().unwrap_or(q_dim).max(q_dim);
        let max_inter = self.weights.layers.iter().map(|l| l.gate.rows().max(l.down.cols())).max().unwrap_or(self.config.intermediate_dim).max(self.config.intermediate_dim);

        let total = hidden * 4
            + max_qkv * 2
            + max_ctx * 2
            + kv_dim * 2
            + self.config.context_len
            + max_inter * 2;
        arena.reset();
        let scratch = arena
            .alloc_slice::<f32>(total)
            .ok_or_else(|| "scratch arena is too small".to_string())?;
        let (x, rest) = scratch.split_at_mut(hidden);
        let (norm, rest) = rest.split_at_mut(hidden);
        let (q, rest) = rest.split_at_mut(max_qkv);
        let (k, rest) = rest.split_at_mut(kv_dim);
        let (v, rest) = rest.split_at_mut(kv_dim);
        let (scores, rest) = rest.split_at_mut(self.config.context_len);
        let (context, _rest) = rest.split_at_mut(max_ctx);

        embedding(&self.weights.token_embedding, token as usize, x)?;
        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            let started = profiler.as_ref().and_then(|p| p.start());
            rmsnorm(norm, x, layer.attn_norm.values(), self.config.rms_epsilon);
            if self.config.has_hadamard {
                hadamard_transform_1024(norm);
            }
            record(&mut profiler, |p| &mut p.rmsnorm, started);

            let started = profiler.as_ref().and_then(|p| p.start());
            cuda.gemv_qkv(layer_index, norm, q, k, v)?;
            record(&mut profiler, |p| &mut p.qkv, started);

            add_bias(q, layer.q_bias.values());
            add_bias(k, layer.k_bias.values());
            add_bias(v, layer.v_bias.values());
            if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                debug_hidden(&format!("qkv_bias pos={pos}"), &q[..q_dim]);
                debug_hidden(&format!("k_bias pos={pos}"), &k[..kv_dim]);
                debug_hidden(&format!("v_bias pos={pos}"), &v[..kv_dim]);
            }

            let started = profiler.as_ref().and_then(|p| p.start());
            rope(q, pos, self.config.head_dim, self.config.rope_base);
            rope(k, pos, self.config.head_dim, self.config.rope_base);
            if layer_index == 0 && std::env::var_os("SPARK_DUMP_LAYERS").is_some() {
                debug_hidden(&format!("q_rope pos={pos}"), &q[..q_dim]);
                debug_hidden(&format!("k_rope pos={pos}"), &k[..kv_dim]);
            }
            cache
                .store(layer_index, pos, k, v)
                .map_err(str::to_string)?;
            attention(
                q,
                context,
                scores,
                cache,
                layer_index,
                pos,
                self.config.heads,
                self.config.kv_heads,
                self.config.head_dim,
            );
            record(&mut profiler, |p| &mut p.attention, started);

            let started = profiler.as_ref().and_then(|p| p.start());
            cuda.gemv_attn_out(layer_index, context, q)?;
            record(&mut profiler, |p| &mut p.attention_output, started);
            add_in_place(x, q);

            let started = profiler.as_ref().and_then(|p| p.start());
            rmsnorm(norm, x, layer.ffn_norm.values(), self.config.rms_epsilon);
            if self.config.has_hadamard {
                hadamard_transform_1024(norm);
            }
            record(&mut profiler, |p| &mut p.rmsnorm, started);

            let started = profiler.as_ref().and_then(|p| p.start());
            cuda.ffn_fused_down(layer_index, norm, q)?;
            record(&mut profiler, |p| &mut p.ffn_gate_up, started);
            add_in_place(x, q);
        }

        let started = profiler.as_ref().and_then(|p| p.start());
        rmsnorm(
            norm,
            x,
            self.weights.final_norm.values(),
            self.config.rms_epsilon,
        );
        if self.config.has_hadamard {
            hadamard_transform_1024(norm);
        }
        record(&mut profiler, |p| &mut p.rmsnorm, started);

        let started = profiler.as_ref().and_then(|p| p.start());
        let selected_token = cuda.gemv_output_and_sample(norm, temperature, top_p, rng_seed)?;
        record(&mut profiler, |p| &mut p.final_output, started);
        record(&mut profiler, |p| &mut p.forward, forward_started);

        Ok(selected_token)
    }
}

fn qwen35_gemv(
    pool: &ThreadPool,
    cuda: Option<&CudaQwen35Engine>,
    cpu_matrix: &QuantizedMatrix<'_>,
    gpu_matrix: Option<&CudaQ1Matrix>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    if let (Some(cuda), Some(matrix)) = (cuda, gpu_matrix) {
        cuda.gemv(matrix, input, output)
    } else {
        cpu_matrix
            .gemv_parallel(input, output, pool, None)
            .map_err(|error| error.to_string())
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let e = value.exp();
        e / (1.0 + e)
    }
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

fn rmsnorm_heads_in_place(values: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
    for head in values.chunks_exact_mut(head_dim) {
        let mean = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv = (mean + eps).sqrt().recip();
        for (value, &scale) in head.iter_mut().zip(weight) {
            *value *= inv * scale;
        }
    }
}

fn l2_norm_heads_in_place(values: &mut [f32], head_dim: usize, eps: f32) {
    for head in values.chunks_exact_mut(head_dim) {
        let sum = head.iter().map(|v| v * v).sum::<f32>();
        let inv = (sum + eps).sqrt().recip();
        for value in head {
            *value *= inv;
        }
    }
}

fn rope_partial(values: &mut [f32], pos: usize, head_dim: usize, n_rot: usize, base: f32) {
    let n_rot = n_rot.min(head_dim) & !1;
    let half = n_rot / 2;
    for head in values.chunks_exact_mut(head_dim) {
        for index in 0..half {
            let frequency = base.powf(-(2.0 * index as f32 / n_rot as f32));
            let (sin, cos) = (pos as f32 * frequency).sin_cos();
            let a = head[index];
            let b = head[index + half];
            head[index] = a * cos - b * sin;
            head[index + half] = b * cos + a * sin;
        }
    }
}

fn causal_conv_silu(
    current: &[f32],
    weights: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    kernel: usize,
) {
    let history = kernel - 1;
    for channel in 0..current.len() {
        let s = &mut state[channel * history..(channel + 1) * history];
        let w = &weights[channel * kernel..(channel + 1) * kernel];
        let mut value = current[channel] * w[history];
        for i in 0..history {
            value = s[i].mul_add(w[i], value);
        }
        if history > 1 {
            s.copy_within(1..history, 0);
        }
        s[history - 1] = current[channel];
        output[channel] = value * sigmoid(value);
    }
}

fn gated_head_rmsnorm(
    values: &mut [f32],
    gate: &[f32],
    weight: &[f32],
    head_dim: usize,
    eps: f32,
) {
    for (head_index, head) in values.chunks_exact_mut(head_dim).enumerate() {
        let mean = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv = (mean + eps).sqrt().recip();
        let z = &gate[head_index * head_dim..(head_index + 1) * head_dim];
        for i in 0..head_dim {
            head[i] = head[i] * inv * weight[i] * (z[i] * sigmoid(z[i]));
        }
    }
}

fn record(
    profiler: &mut Option<&mut ProfileStats>,
    field: impl FnOnce(&mut ProfileStats) -> &mut std::time::Duration,
    started: Option<std::time::Instant>,
) {
    if let (Some(profiler), Some(started)) = (profiler.as_deref_mut(), started) {
        *field(profiler) += started.elapsed();
    }
}

fn record_gemv(
    profiler: &mut Option<&mut ProfileStats>,
    field: impl FnOnce(&mut ProfileStats) -> &mut std::time::Duration,
    started: Option<std::time::Instant>,
    overhead_before: std::time::Duration,
) {
    if let (Some(profiler), Some(started)) = (profiler.as_deref_mut(), started) {
        let overhead_added = profiler.pool_overhead.saturating_sub(overhead_before);
        let elapsed = started.elapsed();
        *field(profiler) += elapsed.saturating_sub(overhead_added);
    }
}

fn debug_activation(name: &str, layer: usize, pos: usize, values: &[f32]) {
    if std::env::var_os("SPARK_DEBUG_ACTIVATIONS").is_none() || pos != 0 {
        return;
    }
    let mut sum = 0.0f64;
    let mut abs_max = 0.0f32;
    let mut nan_cnt = 0;
    let mut inf_cnt = 0;
    let mut large_cnt = 0;
    for &v in values {
        if v.is_nan() {
            nan_cnt += 1;
        } else if v.is_infinite() {
            inf_cnt += 1;
        } else {
            sum += v as f64;
            abs_max = abs_max.max(v.abs());
            if v.abs() > 1000.0 {
                large_cnt += 1;
            }
        }
    }
    let mean = sum / values.len().max(1) as f64;
    println!(
        "\x1b[35m[debug] layer={layer} {name}: mean={mean:.4} abs_max={abs_max:.4} NaN={nan_cnt} Inf={inf_cnt} >1k={large_cnt}\x1b[0m"
    );
}

fn embedding(matrix: &QuantizedMatrix<'_>, token: usize, output: &mut [f32]) -> Result<(), String> {
    if token >= matrix.rows() {
        return Err(format!("token {token} exceeds vocabulary {}", matrix.rows()));
    }
    let cols = matrix.cols();
    let num_blocks = cols / 32;
    let start = token * num_blocks;

    match matrix {
        QuantizedMatrix::Q1(q1) => {
            let blocks_per_row = cols / 128;
            let start = token * blocks_per_row;
            let blocks = &q1.blocks()[start..start + blocks_per_row];
            for (index, destination) in output.iter_mut().enumerate() {
                *destination = blocks[index / 128].value(index % 128);
            }
        }
        QuantizedMatrix::Q8(q8) => {
            let blocks = &q8.blocks()[start..start + num_blocks];
            for (index, destination) in output.iter_mut().enumerate() {
                *destination = blocks[index / 32].value(index % 32);
            }
        }
        QuantizedMatrix::Q4(q4) => {
            let blocks = &q4.blocks()[start..start + num_blocks];
            for (index, destination) in output.iter_mut().enumerate() {
                *destination = blocks[index / 32].value(index % 32);
            }
        }
        QuantizedMatrix::Ternary(ternary) => {
            let blocks = &ternary.blocks()[start..start + num_blocks];
            for (index, destination) in output.iter_mut().enumerate() {
                *destination = blocks[index / 32].scale() * blocks[index / 32].weight(index % 32);
            }
        }
        QuantizedMatrix::PTQ1(ptq1) => {
            let blocks = &ptq1.blocks()[start..start + num_blocks];
            for (index, destination) in output.iter_mut().enumerate() {
                *destination = blocks[index / 32].weight(index % 32);
            }
        }
    }
    Ok(())
}

fn add_bias(output: &mut [f32], bias: &[f32]) {
    for (value, &offset) in output.iter_mut().zip(bias) {
        *value += offset;
    }
}

fn add_in_place(output: &mut [f32], input: &[f32]) {
    for (value, &addition) in output.iter_mut().zip(input) {
        *value += addition;
    }
}

/// Debug helper: prints a stable checksum of a hidden state slice so the CPU
/// and GPU forward paths can be compared layer by layer.
fn debug_hidden(name: &str, values: &[f32]) {
    if std::env::var_os("SPARK_DUMP_LAYERS").is_none() {
        return;
    }
    let mut sum_sq = 0.0f64;
    let mut sum = 0.0f64;
    for &v in values {
        sum_sq += (v as f64) * (v as f64);
        sum += v as f64;
    }
    eprintln!(
        "\x1b[35m[dump] {name}: sum={sum:.6} sum_sq={sum_sq:.6} len={} head8={:?} mid32={:?}\x1b[0m",
        values.len(),
        &values[..8.min(values.len())],
        if values.len() > 40 { &values[32..40] } else { &[] }
    );
}

#[allow(clippy::too_many_arguments)]
fn attention(
    q: &[f32],
    context: &mut [f32],
    scores: &mut [f32],
    cache: &KvCache,
    layer: usize,
    pos: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) {
    let group = heads / kv_heads;
    for query_head in 0..heads {
        let query = &q[query_head * head_dim..(query_head + 1) * head_dim];
        for (position, score) in scores.iter_mut().enumerate().take(pos + 1) {
            let key = cache.key(layer, position, query_head / group);
            *score = query
                .iter()
                .zip(key)
                .map(|(&left, &right)| left * right)
                .sum::<f32>()
                / (head_dim as f32).sqrt();
        }
        softmax(&mut scores[..pos + 1]);
        let destination = &mut context[query_head * head_dim..(query_head + 1) * head_dim];
        destination.fill(0.0);
        for (position, &score) in scores.iter().enumerate().take(pos + 1) {
            for (value, &cached) in
                destination
                    .iter_mut()
                    .zip(cache.value(layer, position, query_head / group))
            {
                *value += score * cached;
            }
        }
    }
}

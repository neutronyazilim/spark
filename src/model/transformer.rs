use super::{
    config::ModelConfig,
    kv_cache::KvCache,
    ops::{hadamard_transform_1024, rmsnorm, rope, softmax},
    weights::{ModelWeights, QuantizedMatrix},
};
use crate::backend::CudaEngine;
use crate::kernel::pool::ThreadPool;
use crate::memory::arena::Arena;
use crate::profile::ProfileStats;

pub struct Transformer<'a> {
    pub config: ModelConfig,
    pub weights: ModelWeights<'a>,
    pub pool: ThreadPool,
    pub cuda: Option<CudaEngine>,
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

            let started = profiler.as_ref().and_then(|p| p.start());
            rope(q, pos, self.config.head_dim, self.config.rope_base);
            rope(k, pos, self.config.head_dim, self.config.rope_base);
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

        record(&mut profiler, |p| &mut p.forward, forward_started);
        debug_activation("final_logits", 0, pos, logits);
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
        let cuda = self.cuda.as_ref().ok_or_else(|| "CUDA engine not initialized".to_string())?;
        let forward_started = profiler.as_ref().and_then(|p| p.start());
        if pos >= self.config.context_len {
            return Err("position exceeds configured context".into());
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

            let started = profiler.as_ref().and_then(|p| p.start());
            rope(q, pos, self.config.head_dim, self.config.rope_base);
            rope(k, pos, self.config.head_dim, self.config.rope_base);
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

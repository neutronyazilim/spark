use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    backend::{cuda::CudaModelWeights, CudaDevice, CudaEngine},
    kernel::pool::ThreadPool,
    loader::gguf::{GgufError, GgufModel},
    memory::arena::Arena,
    model::{
        config::ModelConfig, kv_cache::KvCache, transformer::Transformer, weights::ModelWeights,
    },
    profile::ProfileStats,
    sampler::Sampler,
    tokenizer::bpe::BpeTokenizer,
};

pub struct InferenceSession<'a> {
    pub tokenizer: BpeTokenizer,
    pub transformer: Transformer<'a>,
    pub cache: KvCache,
    pub arena: Arena,
    pub position: usize,
    pub logits: Option<Vec<f32>>,
    pub last_token: Option<u32>,
    pub profile: Option<ProfileStats>,
}

impl<'a> InferenceSession<'a> {
    pub fn from_model(
        model: &'a GgufModel<'a>,
        threads: usize,
        gpu_layers: Option<usize>,
    ) -> Result<Self, GgufError> {
        let config = ModelConfig::from_gguf(model)?;
        let tokenizer = BpeTokenizer::from_gguf(model)?;
        let weights = ModelWeights::from_gguf(model, config.layers)?;
        let pool =
            ThreadPool::new(threads).map_err(|message| GgufError::InvalidFormat(message.into()))?;
        let cache = KvCache::new(
            config.layers,
            config.context_len,
            config.kv_heads,
            config.head_dim,
        )
        .map_err(|message| GgufError::InvalidFormat(message.into()))?;
        let max_qkv = weights.layers.iter().map(|l| l.q.rows()).max().unwrap_or(config.heads * config.head_dim);
        let max_ctx = weights.layers.iter().map(|l| l.o.cols()).max().unwrap_or(config.heads * config.head_dim);
        let max_inter = weights.layers.iter().map(|l| l.gate.rows().max(l.down.cols())).max().unwrap_or(config.intermediate_dim);
        let max_vocab = weights.output.rows().max(config.vocab_size);

        let scratch_floats = max_vocab
            + config.hidden_dim * 4
            + max_qkv * 2
            + max_ctx * 2
            + config.kv_heads * config.head_dim * 2
            + config.context_len
            + max_inter * 2;

        let cuda = if let Some(device) = CudaDevice::init() {
            let dev_name = device.name.clone();
            let dev_arc = Arc::new(device);
            match CudaModelWeights::upload(Arc::clone(&dev_arc), &weights, &config, gpu_layers) {
                Ok(cuda_weights) => match CudaEngine::new(dev_arc, cuda_weights, &config) {
                    Ok(engine) => {
                        println!("\x1b[32m[spark] GPU Acceleration Active: {}\x1b[0m", dev_name);
                        Some(engine)
                    }
                    Err(e) => {
                        eprintln!("\x1b[33m[spark] CUDA Engine init failed ({}), falling back to CPU\x1b[0m", e);
                        None
                    }
                },
                Err(e) => {
                    eprintln!("\x1b[33m[spark] CUDA Weight upload failed ({}), falling back to CPU\x1b[0m", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            tokenizer,
            transformer: Transformer {
                config,
                weights,
                pool,
                cuda,
            },
            cache,
            arena: Arena::new(scratch_floats * std::mem::size_of::<f32>()),
            position: 0,
            logits: None,
            last_token: None,
            profile: None,
        })
    }

    pub fn context_len(&self) -> usize {
        self.transformer.config.context_len
    }

    pub fn thread_count(&self) -> usize {
        self.transformer.pool.thread_count()
    }

    pub fn reset(&mut self) {
        self.cache.clear();
        self.position = 0;
        self.logits = None;
        self.last_token = None;
    }

    pub fn enable_profile(&mut self) {
        self.profile = Some(ProfileStats::new(true));
    }

    pub fn print_profile(&self, tokens: usize) {
        if let Some(profile) = &self.profile {
            profile.print(tokens);
        }
    }

    pub fn push_tokens(&mut self, tokens: &[u32]) -> Result<Duration, GgufError> {
        let end = self
            .position
            .checked_add(tokens.len())
            .ok_or_else(|| GgufError::InvalidArgument("context position overflow".into()))?;
        if end > self.context_len() {
            return Err(GgufError::InvalidArgument(format!(
                "context limit reached ({} tokens)",
                self.context_len()
            )));
        }
        let started = Instant::now();
        for &token in tokens {
            let profiler = self.profile.as_mut();
            let logits = self
                .transformer
                .forward_token(
                    token,
                    self.position,
                    &mut self.cache,
                    &mut self.arena,
                    profiler,
                )
                .map_err(GgufError::InvalidFormat)?;

            if let Some(existing) = &mut self.logits {
                existing.copy_from_slice(logits);
            } else {
                self.logits = Some(logits.to_vec());
            }
            self.position += 1;
        }
        self.last_token = tokens.last().copied();
        Ok(started.elapsed())
    }

    pub fn sample_and_forward(
        &mut self,
        sampler: &mut Sampler,
    ) -> Result<(u32, bool, Duration), GgufError> {
        let rng_seed = sampler.next_random();
        let temperature = sampler.temperature;
        let top_p = sampler.top_p;

        let token = if let Some(logits) = &self.logits {
            let sample_start = self.profile.as_ref().and_then(|p| p.start());
            let t = sampler.sample(logits) as u32;
            if let (Some(p), Some(st)) = (self.profile.as_mut(), sample_start) {
                p.sampling += st.elapsed();
            }
            self.logits = None; // consumed initial prefill logits
            t
        } else if let Some(prev) = self.last_token {
            if self.transformer.cuda.is_some() {
                let profiler = self.profile.as_mut();
                self.transformer
                    .forward_token_and_sample(
                        prev,
                        self.position,
                        &mut self.cache,
                        &mut self.arena,
                        temperature,
                        top_p,
                        rng_seed,
                        profiler,
                    )
                    .map_err(GgufError::InvalidFormat)?
            } else {
                let profiler = self.profile.as_mut();
                let logits = self
                    .transformer
                    .forward_token(
                        prev,
                        self.position,
                        &mut self.cache,
                        &mut self.arena,
                        profiler,
                    )
                    .map_err(GgufError::InvalidFormat)?;

                let sample_start = self.profile.as_ref().and_then(|p| p.start());
                let t = sampler.sample(logits) as u32;
                if let (Some(p), Some(st)) = (self.profile.as_mut(), sample_start) {
                    p.sampling += st.elapsed();
                }
                t
            }
        } else {
            return Err(GgufError::InvalidArgument("no context or tokens available".into()));
        };

        self.last_token = Some(token);
        self.position += 1;
        let stop = self.tokenizer.is_stop_token(token);
        Ok((token, stop, Duration::ZERO))
    }

    pub fn print_stats(&self, generated: usize, elapsed: Duration, prefix: &str) -> io::Result<()> {
        let seconds = elapsed.as_secs_f64().max(1e-9);
        println!(
            "\x1b[90m[{}{} tokens | {:.1} tok/s | {:.2}s]\x1b[0m",
            prefix,
            generated,
            generated as f64 / seconds,
            elapsed.as_secs_f64()
        );
        io::stdout().flush()
    }
}

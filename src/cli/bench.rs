use std::time::Instant;

use super::runtime::InferenceSession;
use crate::{loader::gguf::GgufError, sampler::Sampler};

pub fn run(
    session: &mut InferenceSession<'_>,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    max_tokens: usize,
    profile: bool,
) -> Result<(), GgufError> {
    let prompt = "<|im_start|>user\nBenchmark a short response.<|im_end|>\n<|im_start|>assistant\n";
    let tokens = session
        .tokenizer
        .encode(prompt, session.tokenizer.should_add_bos());
    let prefill_started = Instant::now();
    session.push_tokens(&tokens)?;
    if profile {
        session.enable_profile();
    }
    let prefill = prefill_started.elapsed();
    let mut sampler = Sampler::new(temperature, top_p, top_k, repetition_penalty);
    let decode_started = Instant::now();
    let mut generated = 0usize;
    let limit = max_tokens;
    while generated < limit && session.position < session.context_len() {
        let (_, stop, _) = session.sample_and_forward(&mut sampler)?;
        generated += 1;
        if stop {
            break;
        }
    }
    let decode = decode_started.elapsed();
    println!("threads: {}", session.thread_count());
    println!(
        "prefill: {} tokens, {:.2} tok/s, {:.2}s",
        tokens.len(),
        tokens.len() as f64 / prefill.as_secs_f64().max(1e-9),
        prefill.as_secs_f64()
    );
    println!(
        "decode: {} tokens, {:.2} tok/s, {:.2}s",
        generated,
        generated as f64 / decode.as_secs_f64().max(1e-9),
        decode.as_secs_f64()
    );
    if profile {
        session.print_profile(generated);
    }
    Ok(())
}

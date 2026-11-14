use std::{
    env,
    fs::File,
    io::{self, Write},
    process::ExitCode,
    time::Instant,
};

use memmap2::Mmap;

use spark::{
    cli::{bench, chat, runtime::InferenceSession},
    loader::gguf::{GgufError, GgufModel},
    sampler::Sampler,
};

enum Command {
    Run { prompt: String },
    Chat { max_tokens: usize },
    Bench,
}

struct Arguments {
    command: Command,
    model: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    threads: usize,
    gpu_layers: Option<usize>,
    raw: bool,
    profile: bool,
}

fn parse_arguments() -> Result<Arguments, GgufError> {
    let mut args = env::args().skip(1);
    let command = match args.next().as_deref() {
        Some("run") => Command::Run {
            prompt: String::new(),
        },
        Some("chat") => Command::Chat { max_tokens: 128 },
        Some("bench") => Command::Bench,
        _ => return Err(usage()),
    };
    let model = args
        .next()
        .ok_or_else(|| GgufError::InvalidArgument("missing model path".into()))?;
    let mut prompt = None;
    let mut max_tokens = 128;
    let mut temperature = 0.7;
    let mut top_p = 0.9;
    let mut top_k = 40;
    let mut repetition_penalty = 1.1;
    let mut threads = default_threads();
    let mut gpu_layers = None;
    let mut raw = false;
    let mut profile = false;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "-p" | "--prompt" => {
                prompt = Some(next_value(&mut args, "prompt")?);
            }
            "--raw" => raw = true,
            "-n" | "--max-tokens" => {
                let value = next_value(&mut args, "token count")?
                    .parse()
                    .map_err(|_| invalid("invalid max token count"))?;
                max_tokens = value;
            }
            "--temp" | "--temperature" => {
                temperature = next_value(&mut args, "temperature")?
                    .parse()
                    .map_err(|_| invalid("invalid temperature"))?;
            }
            "--top-p" => {
                top_p = next_value(&mut args, "top-p")?
                    .parse()
                    .map_err(|_| invalid("invalid top-p"))?;
            }
            "--top-k" => {
                top_k = next_value(&mut args, "top-k")?
                    .parse()
                    .map_err(|_| invalid("invalid top-k"))?;
            }
            "--repeat-penalty" => {
                repetition_penalty = next_value(&mut args, "repetition penalty")?
                    .parse()
                    .map_err(|_| invalid("invalid repetition penalty"))?;
            }
            "-t" | "--threads" => {
                threads = next_value(&mut args, "thread count")?
                    .parse()
                    .map_err(|_| invalid("invalid thread count"))?;
            }
            "-ngl" | "--gpu-layers" => {
                let value = next_value(&mut args, "gpu layers")?
                    .parse()
                    .map_err(|_| invalid("invalid gpu layer count"))?;
                gpu_layers = Some(value);
            }
            "--profile" => profile = true,
            "--help" | "-h" => return Err(usage()),
            unknown => return Err(invalid(format!("unknown argument {unknown}"))),
        }
    }

    let command = match command {
        Command::Run { .. } => Command::Run {
            prompt: prompt.ok_or_else(|| invalid("missing prompt"))?,
        },
        Command::Chat { .. } => Command::Chat { max_tokens },
        command => command,
    };
    Ok(Arguments {
        command,
        model,
        max_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        threads,
        gpu_layers,
        raw,
        profile,
    })
}

fn default_threads() -> usize {
    let available = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    if available >= 8 {
        // On hybrid Intel (e.g. 4P + 4E cores) and SMT architectures,
        // pinning to physical Performance cores without SMT port sharing
        // yields the highest AVX2 execution port utilization and L1 bandwidth.
        4
    } else {
        available
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, GgufError> {
    args.next()
        .ok_or_else(|| invalid(format!("missing {name} value")))
}

fn invalid(message: impl Into<String>) -> GgufError {
    GgufError::InvalidArgument(message.into())
}

fn usage() -> GgufError {
    invalid(
        "usage: spark <run|chat|bench> <model.gguf> [options]\n\
         run:   -p \"prompt\" [-n 128] [--temp 0.7] [--top-p 0.9] [--top-k 40]\n\
         chat:  [-n 128] [--temp 0.7] [--top-p 0.9] [--top-k 40]\n\
         bench: [-n 128] [--temp 0.7] [--top-p 0.9] [--top-k 40]\n\
         common: -t, --threads N [-ngl, --gpu-layers N] [--profile]",
    )
}

fn run(arguments: Arguments) -> Result<(), GgufError> {
    let file = File::open(&arguments.model).map_err(|source| GgufError::Io {
        context: format!("cannot open {}", arguments.model),
        source,
    })?;
    // SAFETY: the file descriptor remains alive for the lifetime of the mapping.
    let mmap = unsafe {
        Mmap::map(&file).map_err(|source| GgufError::Io {
            context: format!("cannot mmap {}", arguments.model),
            source,
        })?
    };
    let model = GgufModel::parse(&mmap)?;
    let mut session = InferenceSession::from_model(&model, arguments.threads, arguments.gpu_layers)?;
    println!("threads: {}", session.thread_count());

    match arguments.command {
        Command::Chat { max_tokens } => chat::run(
            &mut session,
            arguments.temperature,
            arguments.top_p,
            arguments.top_k,
            arguments.repetition_penalty,
            max_tokens,
        ),
        Command::Bench => bench::run(
            &mut session,
            arguments.temperature,
            arguments.top_p,
            arguments.top_k,
            arguments.repetition_penalty,
            arguments.max_tokens,
            arguments.profile,
        ),
        Command::Run { prompt } => run_once(
            &mut session,
            &prompt,
            arguments.max_tokens,
            arguments.temperature,
            arguments.top_p,
            arguments.top_k,
            arguments.repetition_penalty,
            arguments.raw,
            arguments.profile,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_once(
    session: &mut InferenceSession<'_>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    raw: bool,
    profile: bool,
) -> Result<(), GgufError> {
    let formatted = if raw {
        prompt.to_string()
    } else if session.transformer.config.is_qwen35 {
        format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
    } else {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    };
    let tokens = session
        .tokenizer
        .encode(&formatted, session.tokenizer.should_add_bos() && !raw);
    if std::env::var_os("SPARK_DEBUG_TOKENS").is_some() {
        eprintln!("FORMATTED: {formatted:?}");
        eprintln!("TOKENS: {tokens:?}");
        for (i, t) in tokens.iter().enumerate() {
            eprintln!("  [{i}] id={t} -> {:?}", session.tokenizer.decode_token(*t));
        }
    }
    if tokens.is_empty() {
        return Err(invalid("prompt produced no tokens"));
    }
    let prefill_started = Instant::now();
    session.push_tokens(&tokens)?;
    if profile {
        session.enable_profile();
    }
    let prefill = prefill_started.elapsed();
    let mut sampler = Sampler::new(temperature, top_p, top_k, repetition_penalty);
    let mut pending = Vec::new();
    let mut generated = 0usize;
    let generation_started = Instant::now();
    print!("{prompt}");
    io::stdout().flush().map_err(|source| GgufError::Io {
        context: "flush prompt".into(),
        source,
    })?;

    let limit = max_tokens;
    while generated < limit && session.position < session.context_len() {
        let (token, stop, _) = session.sample_and_forward(&mut sampler)?;
        if stop {
            break;
        }
        let detok_start = session.profile.as_ref().and_then(|p| p.start());
        let text = session.tokenizer.decode_stream(token, &mut pending);
        print!("{text}");
        io::stdout().flush().map_err(|source| GgufError::Io {
            context: "flush generated text".into(),
            source,
        })?;
        if let (Some(p), Some(st)) = (session.profile.as_mut(), detok_start) {
            p.detokenize += st.elapsed();
        }
        generated += 1;
    }
    let generation = generation_started.elapsed();
    println!();
    println!(
        "\x1b[90m[prefill: {} tokens, {:.2} tok/s | generation: {} tokens, {:.2} tok/s]\x1b[0m",
        tokens.len(),
        tokens.len() as f64 / prefill.as_secs_f64().max(1e-9),
        generated,
        generated as f64 / generation.as_secs_f64().max(1e-9)
    );
    if profile {
        session.print_profile(generated);
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse_arguments().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

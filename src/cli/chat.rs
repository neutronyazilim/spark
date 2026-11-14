use std::io::{self, Write};
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
) -> Result<(), GgufError> {
    println!(
        "\x1b[1;36mspark chat\x1b[0m  \x1b[90m{} threads, context {} tokens\x1b[0m",
        session.thread_count(),
        session.context_len()
    );
    println!("\x1b[90mType exit or quit to leave.\x1b[0m");
    let mut sampler = Sampler::new(temperature, top_p, top_k, repetition_penalty);

    loop {
        print!("\x1b[1;32mtux > \x1b[0m");
        io::stdout().flush().map_err(|source| GgufError::Io {
            context: "flush chat prompt".into(),
            source,
        })?;
        let mut input = String::new();
        if io::stdin()
            .read_line(&mut input)
            .map_err(|source| GgufError::Io {
                context: "read chat input".into(),
                source,
            })?
            == 0
        {
            println!();
            break;
        }
        let input = input.trim_end();
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }
        if input.is_empty() {
            continue;
        }

        let prompt = format!("<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n");
        let tokens = session.tokenizer.encode(
            &prompt,
            session.position == 0 && session.tokenizer.should_add_bos(),
        );
        if session.position + tokens.len() > session.context_len() {
            eprintln!("\x1b[33mcontext limit reached; clearing conversation history\x1b[0m");
            session.reset();
        }
        if let Err(error) = session.push_tokens(&tokens) {
            eprintln!("\x1b[31m{error}\x1b[0m");
            continue;
        }

        print!("\x1b[1;34mspark > \x1b[0m");
        io::stdout().flush().map_err(|source| GgufError::Io {
            context: "flush model prompt".into(),
            source,
        })?;
        let started = Instant::now();
        let mut generated = 0usize;
        let mut pending = Vec::new();
        let mut generated_tokens = Vec::new();
        loop {
            if generated >= max_tokens || session.position >= session.context_len() {
                break;
            }
            let (token, stop, _) = session.sample_and_forward(&mut sampler)?;
            let text = if stop {
                String::new()
            } else {
                session.tokenizer.decode_stream(token, &mut pending)
            };
            if !text.is_empty() {
                print!("{text}");
                io::stdout().flush().map_err(|source| GgufError::Io {
                    context: "flush chat response".into(),
                    source,
                })?;
            }
            if stop {
                let closing = if session.tokenizer.is_im_end_token(token) {
                    "\n"
                } else {
                    "<|im_end|>\n"
                };
                let closing_tokens = session.tokenizer.encode(closing, false);
                let closing_end = session.position.checked_add(closing_tokens.len());
                if closing_end.is_some_and(|end| end <= session.context_len()) {
                    session.push_tokens(&closing_tokens)?;
                } else {
                    eprintln!(
                        "\x1b[33mcontext limit reached; response closing was not cached\x1b[0m"
                    );
                }
                break;
            }
            generated_tokens.push(token);
            generated += 1;
            if has_repetition_loop(&generated_tokens) {
                eprintln!("\x1b[33mrepetition guard stopped generation\x1b[0m");
                break;
            }
        }
        println!();
        session
            .print_stats(generated, started.elapsed(), "")
            .map_err(|source| GgufError::Io {
                context: "write chat statistics".into(),
                source,
            })?;
    }
    Ok(())
}

fn has_repetition_loop(tokens: &[u32]) -> bool {
    if tokens.len() >= 8
        && tokens[tokens.len() - 8..]
            .windows(2)
            .all(|pair| pair[0] == pair[1])
    {
        return true;
    }
    if tokens.len() < 12 {
        return false;
    }
    let tail = &tokens[tokens.len() - 12..];
    tail[..4] == tail[4..8] && tail[4..8] == tail[8..12]
}

#[cfg(test)]
mod tests {
    use super::has_repetition_loop;

    #[test]
    fn repetition_guard_detects_constant_and_cycle_loops() {
        assert!(has_repetition_loop(&[7; 8]));
        assert!(has_repetition_loop(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]));
        assert!(!has_repetition_loop(&[1, 2, 3, 4, 5, 6, 7, 8]));
    }
}

# spark

[Türkçe README](README_tr.md)

`spark` is a Rust GGUF LLM inference engine with automatic CUDA acceleration and CPU fallback, implemented without heavyweight external ML libraries.

The project currently supports GGUF model loading, Q8_0/Q4_0/Q1_0 and ternary GEMV kernels, CUDA GPU acceleration, Qwen2.5 and hybrid-attention Qwen3.5 forward passes, KV/recurrent caches, BPE tokenization from GGUF metadata, and autoregressive text generation.

## Features

- GGUF v2/v3 header, metadata, and tensor parser
- Zero-copy model-weight access through `mmap`
- On-the-fly Q8_0 and Q4_0 dequantization
- Ternary (BitNet-style) weight support:
  - Q2_0 / 2-bit ternary block format (32 weights / 10 bytes)
  - Prism ML / BitNet PTQ1_0 format (128 weights / 28 bytes, 1.75 bpw)
- GGML Q1_0 bipolar format (128 weights / 18 bytes, 1.125 bpw)
- Scalar, AVX, and AVX2/FMA GEMV kernels
- Runtime CPU feature dispatch
- CUDA backend (CUDA Driver API loaded through `dlopen`, with no external libraries):
  - Handwritten PTX kernels in `kernels/gemv.ptx` are loaded at runtime
  - Q8_0/Q4_0/Q1_0/PTQ1_0 GEMV, fused QKV, and fused FFN kernels
  - Q8 activation + `dp4a`, multi-projection batching, and two-row/warp kernel for Q1_0
  - GPU-based argmax and top-64 sampling
  - Hybrid layer offloading (`-ngl/--gpu-layers`): the number of layers that fit in VRAM is calculated automatically, and the remaining layers run on the CPU
  - Can be disabled with `SPARK_NO_CUDA=1`; automatically falls back to the CPU if CUDA initialization fails
- Arena-based scratch buffer
- Qwen2/Qwen2.5 transformer blocks
- Qwen3.5/Qwen35 hybrid architecture:
  - 4-tap causal convolution + Gated DeltaNet recurrent layers
  - Gated full attention, head-wise Q/K RMSNorm, and text MRoPE
  - Separate KV cache and recurrent-state cache
- Attention with Q/K/V bias support
- Grouped Query Attention (GQA)
- RoPE, RMSNorm, Softmax, and SwiGLU
- Preallocated KV cache
- Minimal BPE tokenizer using GGUF tokenizer metadata
- Temperature, Top-K, Top-P, and repetition-penalty sampling
- Streaming terminal output
- OpenAI-compatible HTTP API: Chat Completions, SSE streaming, bearer authentication, and tool calls
- Static GEMV row partitioning with a persistent worker thread pool
- ANSI terminal interface and interactive multi-turn chat mode

## Requirements

- Stable Rust toolchain
- An x86-64 CPU is recommended
- SIMD kernels are selected automatically on CPUs with AVX/AVX2 support
- A Qwen2/Qwen2.5 or Qwen3.5/Qwen35 model in GGUF format
- For CUDA support: an NVIDIA GPU and installed driver (`libcuda.so.1`); the CUDA Toolkit is not required (PTX is JIT-compiled at runtime)

## Building

```bash
cargo build --release
```

## Usage

Run a prompt with a Qwen2.5 GGUF model:

```bash
cargo run --release -- run ./qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Hello, how are you?" \
  -n 128 \
  --temp 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --repeat-penalty 1.1 \
  --threads 8
```

With a CUDA GPU (hybrid layer offloading is automatic):

```bash
./target/release/spark run ./model.gguf -p "Hello" -n 128 --threads 8
```

- Layers that fit in VRAM are loaded onto the GPU; the remaining layers run on the CPU.
- Use `-ngl/--gpu-layers N` to specify the number of layers to offload manually.
- `SPARK_NO_CUDA=1` disables CUDA completely.

Qwen3.5 27B Q1_0 example:

```bash
./target/release/spark run ~/Downloads/Bonsai-27B-Q1_0.gguf \
  -p "Hello" -n 64 --temp 0 --threads 8
```

For Qwen3.5 Q1_0 models, the layer budget, scale format, and LM head are selected
automatically according to available VRAM. On 4 GB GPUs, all 64 layers, the LM
head, Gated DeltaNet state, and attention cache for up to 1024 tokens reside in
VRAM. Activation quantization, causal convolution, recurrent updates, attention,
FFN, and sampling run on the same CUDA stream; there are no per-layer CPU/GPU
copies during decoding.

Example result measured on an RTX 3050 Laptop 4 GB with the
`Bonsai-27B-Q1_0.gguf` model in this repository (32 decode tokens, greedy):

```text
prefill: 15 tokens, 19.34 tok/s
decode: 32 tokens, 19.47 tok/s
```

On 4 GB GPUs, the default resident mode compresses Q1 block scales to 10 bits
per matrix. This allows the 1024-token GPU cache to fit. If there is not enough
free VRAM, the cache size is automatically halved. For higher speed while
preserving the original FP16 scales exactly, use a shorter cache:

```bash
SPARK_Q1_EXACT_SCALES=1 SPARK_QWEN35_GPU_CONTEXT=64 \
  ./target/release/spark bench ~/Downloads/Bonsai-27B-Q1_0.gguf \
  -n 32 --temp 0 --threads 8
```

This mode delivers approximately `26.5 tok/s` decoding on the same GPU.
`SPARK_QWEN35_GPU_CONTEXT` can be increased according to available VRAM; lower
the value if an excessively large setting causes a CUDA allocation error.
`SPARK_NO_QWEN35_RESIDENT=1` forces the old hybrid path, while
`SPARK_Q1_HYBRID=1` forces conservative partial layer offloading.

Using the compiled binary:

```bash
./target/release/spark run ./qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "What is Rust?" \
  --max-tokens 64 \
  --temperature 0.0
```

Interactive chat:

```bash
./target/release/spark chat ./qwen2.5-0.5b-instruct-q8_0.gguf \
  --temp 0.7 \
  --threads 8
```

Exit the chat with `exit`, `quit`, or `Ctrl+D`. Previous turns are retained in
the same KV cache; when the context limit is reached, the engine displays a
warning and starts a new chat cache.

### OpenAI-compatible HTTP API

`serve` or `--serve` loads the model once and starts an OpenAI
Chat Completions-compatible HTTP server:

```bash
SPARK_API_KEY=local-dev-key \
  ./target/release/spark --serve ~/Downloads/Bonsai-27B-Q1_0.gguf \
  --host 127.0.0.1 --port 8080 --model-id bonsai-27b \
  --max-tokens 256 --threads 8
```

Authentication is disabled if neither `SPARK_API_KEY` nor `--api-key` is
provided. The server clears its GPU recurrent/KV state at the start of every
stateless API request and processes inference requests sequentially. Supported
endpoints and features:

- `GET /health` and `GET /v1/health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- Regular JSON and `stream: true` SSE responses
- OpenAI `tools` / `tool_calls`; XML calls from the Qwen GGUF tool template are
  converted into OpenAI function-call objects

In API mode, the Qwen thinking block is disabled by default, allowing short
responses and tool calls to complete without consuming the output budget. Send
`"enable_thinking": true` in a request to enable it again.

Quick test:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Authorization: Bearer local-dev-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bonsai-27b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false,
    "max_tokens": 64
  }'
```

Project or global `opencode.jsonc` configuration for OpenCode V2:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "spark/bonsai-27b",
  "providers": {
    "spark": {
      "name": "Spark Local",
      "env": ["SPARK_API_KEY"],
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "{env:SPARK_API_KEY}"
      },
      "models": {
        "bonsai-27b": {
          "name": "Bonsai 27B Q1_0",
          "body": {
            "enable_thinking": false
          },
          "capabilities": {
            "tools": true,
            "input": ["text"],
            "output": ["text"]
          },
          "compatibility": {
            "reasoningField": "reasoning_content",
            "maxTokensField": "max_tokens",
            "streamUsage": true
          },
          "limit": {
            "context": 1024,
            "output": 256
          }
        }
      }
    }
  }
}
```

Start OpenCode with the same key as the server:
`SPARK_API_KEY=local-dev-key opencode`. If you do not want authentication,
remove the key from the server as well as the `apiKey` and `env` lines from the
configuration. If OpenCode's large system prompt and tool schemas exceed the
1024-token context limit on a 4 GB GPU, Spark automatically compacts the prompt
while retaining a concise coding-agent instruction, essential tools, and the
most recent messages. Use `SPARK_NO_PROMPT_COMPACTION=1` to disable this
behavior. If more VRAM is available, increase `SPARK_QWEN35_GPU_CONTEXT`.

Pure generation benchmark:

```bash
./target/release/spark bench ./qwen2.5-0.5b-instruct-q8_0.gguf \
  --threads 8
```

For the `run` and `bench` commands, `--profile` produces an operator-level
microprofiling table for the first 10 decoded tokens:

```bash
./target/release/spark run ./qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Tell me a short joke." -n 10 --temp 0 --threads 8 --profile

./target/release/spark bench ./qwen2.5-0.5b-instruct-q8_0.gguf \
  --threads 8 --profile
```

The table shows RMSNorm, QKV/FFN/output GEMV, RoPE-attention, SwiGLU, and
thread-pool dispatch wait times. Profiling is disabled by default, so no
additional timing measurements are performed during normal execution.

### CLI options

| Option | Description | Default |
|---|---|---:|
| `run <model.gguf>` | Specifies the path to the GGUF model | Required |
| `serve`, `--serve <model.gguf>` | Starts the OpenAI-compatible HTTP API | — |
| `-p`, `--prompt` | Input prompt | Required |
| `-n`, `--max-tokens` | Maximum number of tokens to generate | `128` |
| `--temp`, `--temperature` | Temperature; `0.0` uses greedy argmax | `0.7` |
| `--top-k` | Size of the sampling candidate pool | `40` |
| `--top-p` | Nucleus sampling threshold | `0.9` |
| `--repeat-penalty` | Repetition penalty for the last 64 generated tokens | `1.1` |
| `-t`, `--threads` | Number of GEMV worker threads | CPU parallelism level |
| `-ngl`, `--gpu-layers` | Number of layers loaded onto the GPU (when CUDA is active) | Automatic based on VRAM |
| `--profile` | Operator profile for the first 10 decoded tokens (`run`, `bench`) | Disabled |
| `--host`, `--port` | HTTP bind address and port (`serve`) | `127.0.0.1:8080` |
| `--model-id` | Model identifier exposed by the API | `spark-local` |
| `--api-key` | Bearer API key; `SPARK_API_KEY` can also be used | Disabled |

The `-t`/`--threads` and `--temp` options are shared by the `run`, `chat`,
`bench`, and `serve` commands.

Example temperature/top-p usage:

```bash
cargo run --release -- run model.gguf \
  -p "Write a short poem." \
  -n 64 \
  --temp 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --repeat-penalty 1.1
```

Prefill and generation speeds are printed when generation completes:

```text
[prefill: 4 tokens, 8.30 tok/s | generation: 32 tokens, 18.06 tok/s]
```

## Architecture

```text
GGUF file
   |
   v
mmap + GGUF parser
   |
   +--> mmap-backed quantized weights
   +--> tokenizer metadata --> BPE tokenizer
   |
   v
CUDA backend?                    CPU backend
   |                                |
   +--> PTX kernels (Q8/Q4/         +--> scalar/AVX/AVX2 GEMV
   |    PTQ1 GEMV, fused QKV,       |    + thread pool
   |    fused FFN, argmax,          |
   |    top-64 sampling)            |
   |                                |
   v                                v
Qwen2 or Qwen35 transformer (hybrid: GPU layers + CPU/recurrent operations)
   |
   +--> RMSNorm
   +--> Q/K/V projections
   +--> RoPE + GQA attention
   +--> KV cache
   +--> SwiGLU FFN
   +--> output projection
   |
   v
logits --> sampler --> token decode --> streaming stdout
```

During `forward_token`, intermediate activations are allocated from the arena.
Quantized weights are not expanded into an FP32 buffer ahead of time;
dequantization occurs during the dot product. For GPU-loaded layers, weights are
copied to VRAM in their quantized form, and dequantization occurs inside the
CUDA kernel.

## Tests and benchmarks

Run the tests with:

```bash
cargo test
```

Lint:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

Build or run the GEMV benchmark with:

```bash
cargo bench --bench gemv
```

## Current scope and limitations

- The Qwen2/Qwen2.5 Q8_0 and Qwen35 Q1_0 paths have been validated end to end; the Q2_0/PTQ1_0 CUDA GEMV kernels have synthetic CPU reference tests.
- The tokenizer uses vocabulary and merge metadata from the GGUF file.
- For Qwen2.5 Instruct, the prompt is automatically wrapped in ChatML format:
  `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.
- For Qwen35, the model template's generation suffix (`<think>\n`) and GGUF
  control-token types are applied automatically.
- A `spark chat` session retains its KV cache between turns and streams
  responses with ANSI colors.
- The forward pass uses the arena; tokenizer encoding/decoding and temperature sampling still use regular Rust `Vec` allocations.
- Metadata, tensor definitions, and layer-weight binding tables are created while the model is loaded.
- CUDA requires an NVIDIA GPU and driver (`libcuda.so.1`), but not the CUDA Toolkit.
  The PTX module is loaded at runtime; initialization failures silently fall back to the CPU.
- The Qwen35 Q1_0 offload path batches projections that share the same activation
  into a single H2D/D2H round trip. By default, recurrent state is maintained on
  the bit-stable CPU path; experimental GPU Gated DeltaNet can be enabled with
  `SPARK_GDN_CUDA=1`.
- GEMV worker threads are created once during model initialization; no new
  threads are spawned during token generation.
- There is no Metal or external ONNX/PyTorch/Candle integration.

## Packaging

For a Debian package, install `cargo-deb` and run:

```bash
cargo build --release
cargo deb
```

For Arch Linux/AUR, the root `PKGBUILD` installs the `spark-llm-git` package as
`/usr/bin/spark`.

## License

This project is provided under the [MIT License](LICENSE). Check model files'
licenses and distribution terms separately.

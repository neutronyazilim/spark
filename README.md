# spark

`spark`, harici ağır ML kütüphaneleri kullanmadan yazılan, CPU-first + CUDA destekli Rust LLM inference motorudur.

Proje şu anda GGUF model yükleme, Q8_0/Q4_0/Q1_0 ve ternary GEMV kernel'leri, CUDA GPU acceleration, Qwen2.5 ile hibrit-attention Qwen3.5 forward pass'leri, KV/recurrent cache, GGUF içinden BPE tokenizer ve autoregressive text generation desteğine sahiptir.

## Özellikler

- GGUF v2/v3 header, metadata ve tensor parser
- `mmap` üzerinden zero-copy model ağırlığı erişimi
- Q8_0 ve Q4_0 on-the-fly dequantization
- Ternary (BitNet tarzı) ağırlık desteği:
  - Q2_0 / 2-bit ternary blok formatı (32 ağırlık / 10 bayt)
  - Prism ML / BitNet PTQ1_0 formatı (128 ağırlık / 28 bayt, 1.75 bpw)
- GGML Q1_0 bipolar formatı (128 ağırlık / 18 bayt, 1.125 bpw)
- Scalar, AVX ve AVX2/FMA GEMV kernel'leri
- Runtime CPU feature dispatch
- CUDA backend (harici kütüphane olmadan, `dlopen` ile CUDA Driver API):
  - `kernels/gemv.ptx` içindeki elle yazılmış PTX kernel'ları runtime'da yüklenir
  - Q8_0/Q4_0/Q1_0/PTQ1_0 GEMV, fused QKV ve fused FFN kernel'ları
  - Q1_0 için Q8 aktivasyon + `dp4a`, çoklu projection batching ve iki-row/warp kernel
  - GPU üzerinde argmax ve top-64 sampling
  - Hibrit katman offloading (`-ngl/--gpu-layers`): VRAM'e sığan katman sayısı otomatik hesaplanır, kalan katmanlar CPU'da çalışır
  - `SPARK_NO_CUDA=1` ile devre dışı bırakılabilir; CUDA başlatılamazsa otomatik CPU'ya düşer
- Arena tabanlı scratch buffer
- Qwen2/Qwen2.5 transformer blokları
- Qwen3.5/Qwen35 hibrit mimarisi:
  - 4-tap causal convolution + Gated DeltaNet recurrent katmanları
  - Gated full attention, head-wise Q/K RMSNorm ve text MRoPE
  - Ayrı KV-cache ve recurrent-state cache
- Q/K/V bias destekli attention
- Grouped Query Attention (GQA)
- RoPE, RMSNorm, Softmax ve SwiGLU
- Önceden tahsis edilen KV-cache
- GGUF tokenizer metadata'sından minimal BPE tokenizer
- Temperature, Top-K, Top-P ve repetition penalty sampling
- Streaming terminal çıktısı
- Kalıcı worker thread pool ile statik GEMV satır bölüşümü
- ANSI terminal arayüzü ve interaktif çok turlu sohbet modu

## Gereksinimler

- Rust stable toolchain
- x86-64 CPU önerilir
- AVX/AVX2 destekli CPU'larda SIMD kernel'leri otomatik seçilir
- GGUF formatında Qwen2/Qwen2.5 veya Qwen3.5/Qwen35 modeli
- CUDA desteği için: NVIDIA GPU + kurulu sürücü (`libcuda.so.1`); ayrıca CUDA Toolkit gerekmez (PTX runtime'da JIT derlenir)

## Derleme

```bash
cargo build --release
```

## Kullanım

Qwen2.5 GGUF modeliyle prompt çalıştırma:

```bash
cargo run --release -- run ./qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Merhaba, nasılsın?" \
  -n 128 \
  --temp 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --repeat-penalty 1.1 \
  --threads 8
```

CUDA GPU ile (hibrit katman offloading otomatik):

```bash
./target/release/spark run ./model.gguf -p "Merhaba" -n 128 --threads 8
```

- VRAM'e sığan katmanlar GPU'ya yüklenir, kalanlar CPU'da çalışır.
- `-ngl/--gpu-layers N` ile offload edilecek katman sayısı elle belirtilebilir.
- `SPARK_NO_CUDA=1` CUDA'yı tamamen devre dışı bırakır.

Qwen3.5 27B Q1_0 örneği:

```bash
./target/release/spark run ~/Downloads/Bonsai-27B-Q1_0.gguf \
  -p "Merhaba" -n 64 --temp 0 --threads 8
```

Qwen3.5 Q1_0 modelde katman bütçesi, ölçek biçimi ve LM head VRAM'e göre
otomatik seçilir. 4 GB GPU'larda 64 katmanın tamamı, LM head, Gated DeltaNet
durumu ve 1024 token'a kadar attention cache VRAM'de tutulur. Aktivasyon
quantization, causal convolution, recurrent update, attention, FFN ve sampling
aynı CUDA stream'i üzerinde çalışır; decode sırasında katman başına CPU/GPU
kopyası yapılmaz.

Bu depodaki `Bonsai-27B-Q1_0.gguf` ile RTX 3050 Laptop 4 GB üzerinde ölçülen
örnek sonuç (32 decode token, greedy):

```text
prefill: 15 tokens, 19.34 tok/s
decode: 32 tokens, 19.47 tok/s
```

4 GB kartlarda varsayılan resident mod, Q1 blok ölçeklerini matris başına 10
bit'e sıkıştırır. Bu, 1024 token GPU cache'in sığmasını sağlar. Yeterli boş VRAM
yoksa cache boyutu otomatik olarak yarıya indirilir. Daha yüksek hız
ve orijinal FP16 ölçekleri birebir korumak için daha kısa cache ile:

```bash
SPARK_Q1_EXACT_SCALES=1 SPARK_QWEN35_GPU_CONTEXT=64 \
  ./target/release/spark bench ~/Downloads/Bonsai-27B-Q1_0.gguf \
  -n 32 --temp 0 --threads 8
```

Bu mod aynı kartta yaklaşık `26.5 tok/s` decode üretir. Kullanılabilir VRAM'e
göre `SPARK_QWEN35_GPU_CONTEXT` artırılabilir; fazla büyük değer CUDA allocation
hatasına yol açarsa değeri düşürün. `SPARK_NO_QWEN35_RESIDENT=1` eski hibrit
yolu, `SPARK_Q1_HYBRID=1` ise konservatif kısmi katman offload'unu zorlar.

Derlenmiş binary ile:

```bash
./target/release/spark run ./qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "Rust nedir?" \
  --max-tokens 64 \
  --temperature 0.0
```

İnteraktif sohbet:

```bash
./target/release/spark chat ./qwen2.5-0.5b-instruct-q8_0.gguf \
  --temp 0.7 \
  --threads 8
```

Sohbetten `exit`, `quit` veya `Ctrl+D` ile çıkılır. Önceki turlar aynı KV-cache
üzerinde tutulur; context sınırı dolduğunda motor uyarı verip yeni bir sohbet
cache'i başlatır.

Saf generation benchmark'ı:

```bash
./target/release/spark bench ./qwen2.5-0.5b-instruct-q8_0.gguf \
  --threads 8
```

`run` ve `bench` komutlarında `--profile`, ilk 10 decode token'ı için
operatör bazlı mikro-profilleme tablosu üretir:

```bash
./target/release/spark run ./qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Kısa bir şaka anlat." -n 10 --temp 0 --threads 8 --profile

./target/release/spark bench ./qwen2.5-0.5b-instruct-q8_0.gguf \
  --threads 8 --profile
```

Tablo RMSNorm, QKV/FFN/output GEMV, RoPE-attention, SwiGLU ve thread-pool
dispatch bekleme sürelerini gösterir. Profilleme varsayılan olarak kapalıdır;
normal çalıştırmada ek zaman ölçümü yapılmaz.

### CLI seçenekleri

| Seçenek | Açıklama | Varsayılan |
|---|---|---:|
| `run <model.gguf>` | GGUF model yolunu belirtir | Zorunlu |
| `-p`, `--prompt` | Girdi prompt'u | Zorunlu |
| `-n`, `--max-tokens` | Üretilecek maksimum token sayısı | `128` |
| `--temp`, `--temperature` | Temperature; `0.0` greedy argmax kullanır | `0.7` |
| `--top-k` | Sampling aday havuzunun boyutu | `40` |
| `--top-p` | Nucleus sampling eşiği | `0.9` |
| `--repeat-penalty` | Son 64 üretilen token için tekrar cezası | `1.1` |
| `-t`, `--threads` | GEMV worker thread sayısı | CPU paralellik seviyesi |
| `-ngl`, `--gpu-layers` | GPU'ya yüklenecek katman sayısı (CUDA aktifse) | VRAM'e göre otomatik |
| `--profile` | İlk 10 decode token'ı için operatör profili (`run`, `bench`) | Kapalı |

`-t`/`--threads` ve `--temp` seçenekleri `run`, `chat` ve `bench`
komutlarında ortaktır.

Örnek temperature/top-p kullanımı:

```bash
cargo run --release -- run model.gguf \
  -p "Kısa bir şiir yaz." \
  -n 64 \
  --temp 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --repeat-penalty 1.1
```

Generation tamamlandığında prefill ve generation hızları yazdırılır:

```text
[prefill: 4 tokens, 8.30 tok/s | generation: 32 tokens, 18.06 tok/s]
```

## Mimari

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
   +--> PTX kernel'lar (Q8/Q4/      +--> scalar/AVX/AVX2 GEMV
   |    PTQ1 GEMV, fused QKV,       |    + thread pool
   |    fused FFN, argmax,          |
   |    top-64 sampling)            |
   |                                |
   v                                v
Qwen2 veya Qwen35 transformer (hibrit: GPU katmanları + CPU/recurrent işlemler)
   |
   +--> RMSNorm
   +--> Q/K/V projections
   +--> RoPE + GQA attention
   +--> KV-cache
   +--> SwiGLU FFN
   +--> output projection
   |
   v
logits --> sampler --> token decode --> streaming stdout
```

`forward_token` sırasında ara aktivasyonlar arena üzerinden alınır. Quantized ağırlıklar önceden FP32 buffer'ına açılmaz; dequantization dot-product sırasında yapılır. GPU'ya yüklenen katmanlarda ağırlıklar quantize halleriyle VRAM'e kopyalanır, dequantization CUDA kernel'ı içinde yapılır.

## Test ve benchmark

Testleri çalıştırmak için:

```bash
cargo test
```

Lint:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

GEMV benchmark'ını derlemek veya çalıştırmak için:

```bash
cargo bench --bench gemv
```

## Mevcut kapsam ve sınırlamalar

- Qwen2/Qwen2.5 Q8_0 ve Qwen35 Q1_0 yolları end-to-end doğrulanmıştır; Q2_0/PTQ1_0 CUDA GEMV'leri sentetik CPU referans testlerine sahiptir.
- Tokenizer, GGUF içindeki vocab ve merge metadata'sını kullanır.
- Qwen2.5 Instruct için prompt otomatik olarak ChatML formatına sarılır:
  `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.
- Qwen35 için model template'inin generation eki (`<think>\n`) ve GGUF control
  token tipleri otomatik uygulanır.
- `spark chat` oturumu KV-cache'i turlar arasında korur ve yanıtları ANSI
  renkleriyle akış halinde yazdırır.
- Forward pass arena kullanır; tokenizer encode/decode ve temperature sampling tarafında normal Rust `Vec` kullanımı devam eder.
- Model yükleme sırasında metadata, tensor tanımları ve layer weight binding tabloları oluşturulur.
- CUDA: sürücü (`libcuda.so.1`) + NVIDIA GPU gerekir, CUDA Toolkit gerekmez.
  PTX modülü runtime'da yüklenir; başlatma başarısız olursa sessizce CPU'ya düşülür.
- Qwen35 Q1_0 offload yolu aynı aktivasyonu kullanan projection'ları tek H2D/D2H
  turunda batch eder. Recurrent state varsayılan olarak bit-kararlı CPU yolunda
  tutulur; deneysel GPU Gated DeltaNet `SPARK_GDN_CUDA=1` ile açılabilir.
- GEMV worker thread'leri model başlangıcında bir kez oluşturulur; token
  üretimi sırasında yeni thread spawn edilmez.
- Metal ve harici ONNX/PyTorch/Candle entegrasyonu yoktur.

## Paketleme

Debian paketi için `cargo-deb` kurup:

```bash
cargo build --release
cargo deb
```

komutlarını çalıştırın. Arch Linux/AUR için kökteki `PKGBUILD` dosyası
`spark-llm-git` paketini `/usr/bin/spark` olarak kurar.

## Lisans

Bu depo için lisans belirtilmemiştir. Model dosyalarının lisansını ve dağıtım koşullarını ayrıca kontrol edin.

# hipfire

LLM inference engine for AMD RDNA GPUs. Written in Rust. Faster than llama.cpp at generation on every model tested.

## What it does

Takes a quantized language model and runs it on your AMD GPU. Generates text at **59 tok/s for Qwen3-8B** and **256 tok/s for Qwen3-0.6B** on an RX 5700 XT ($200 GPU from 2019).

No Python runtime. No ROCm link-time dependency. Loads `libamdhip64.so` via `dlopen` at runtime — works across ROCm versions without recompilation.

## Performance

Measured on AMD RX 5700 XT (gfx1010, RDNA1, 8GB GDDR6).

| Benchmark | hipfire | llama.cpp | Ratio |
|-----------|---------|-----------|-------|
| **Qwen3-8B generation** | **59.3 tok/s** | 44.3 tok/s | **1.34x** |
| **Qwen3-8B long gen (1000+ tokens)** | **52.7 tok/s** | 42.8 tok/s | **1.23x** |
| Qwen3-8B prompt processing | 108 tok/s | 189.2 tok/s | 0.57x |
| **Qwen3-0.6B generation** | **256.3 tok/s** | 193.6 tok/s | **1.32x** |
| Qwen3-0.6B prompt processing | 1053 tok/s | 1534 tok/s | 0.69x |

hipfire wins all generation benchmarks. llama.cpp wins prompt processing (prefill) due to rocBLAS GEMM. Full methodology and analysis in [docs/PERF_COMPARISON.md](docs/PERF_COMPARISON.md).

## Quick Start

```bash
# Build
cd hipfire
cargo build --release

# Quantize a model from HuggingFace
# Downloads safetensors + tokenizer, quantizes weights to 4-bit, embeds tokenizer
cargo run --release -p hipfire-quantize -- \
  --input ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ \
  --output models/qwen3-8b.hfq \
  --format hfq4

# Run inference
cargo run --release --example infer_hfq -- models/qwen3-8b.hfq "Hello"
```

### Requirements

- AMD GPU with ROCm (tested on RDNA1 gfx1010, should work on RDNA2+)
- `hipcc` in PATH (from ROCm installation)
- Rust 1.75+

## How it works

### Weight quantization: HFQ4

Weights are stored in HFQ4 (HipFire Quantized 4-bit) format — a custom format designed for high GPU occupancy on RDNA.

Each block of 256 weights: `[f32 scale][f32 zero][128 packed nibble bytes]` = 136 bytes. The GEMV kernel that multiplies these against activation vectors uses **18 VGPRs** — half what llama.cpp's Q4_K uses (39 VGPRs). Lower register usage means more concurrent wavefronts, which means better memory latency hiding, which means higher effective bandwidth.

The quantizer (`hipfire-quantize`) reads HuggingFace safetensors directly. It auto-selects G256 (group size 256) for models with dim >= 4096 and G128 for smaller models. The tokenizer from `tokenizer.json` is embedded in the `.hfq` file — no external tokenizer dependency at inference time.

### KV cache quantization: Q8_0

During generation, the key-value cache is quantized to Q8_0 format (int8 values with f16 scale, 34 bytes per 32 elements). This reduces KV cache bandwidth by 3.76x compared to FP32, with negligible quality impact. The biggest effect is on long generation: **+39% tok/s** at 1000+ tokens where attention bandwidth dominates.

### Batched prefill

Prompt tokens are processed in parallel:
- **Batched GEMM**: all prompt tokens' projections computed with one weight load
- **Batched RoPE**: all positions rotated in one kernel
- **Batched causal attention**: all query positions attend to their causal context in one kernel
- **Batched KV cache write**: all positions quantized and written in one launch

### Runtime kernel compilation

HIP kernels are embedded as C++ string constants in the Rust source. On first use, each kernel is compiled to `.hsaco` via `hipcc --genco` and cached to `/tmp/hipfire_kernels/`. A source hash ensures stale caches are recompiled. First inference run compiles ~15 kernels (takes a few seconds), subsequent runs use cache.

## Architecture

```
hipfire/
├── crates/
│   ├── hip-bridge/          # Safe Rust FFI to libamdhip64.so via dlopen
│   ├── rdna-compute/        # HIP kernel compilation, dispatch, GPU tensor ops
│   ├── engine/              # Model loading (GGUF + HFQ), forward pass, tokenizer
│   └── hipfire-quantize/    # Quantizer: HuggingFace safetensors -> .hfq
├── bench/                   # Profiling scripts (run_profile.sh, compile_results.py)
└── docs/
    ├── PERF_COMPARISON.md   # Detailed benchmark comparison vs llama.cpp
    └── HFQ_FAMILY.md        # HFQ quantization format family spec (Q2-Q8)
```

## Supported Models

Any LLaMA-architecture or Qwen3 model from HuggingFace. Tested:

| Model | Weight format | VRAM | Generation tok/s |
|-------|-------------|------|-----------------|
| Qwen3-8B | HFQ4-G256 | ~4.4 GB | 59.3 |
| Qwen3-0.6B | HFQ4-G128 | ~0.4 GB | 256.3 |

GGUF models also supported via the `infer` example (Q4_K_M, Q8_0).

## License

MIT

# hipfire

LLM inference engine for AMD RDNA GPUs. Written from scratch in Rust + HIP. **9x faster than llama.cpp** on Qwen3.5 DeltaNet models.

## Quickstart

```bash
# Install (Linux, requires AMD GPU)
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/alpha-builds/scripts/install.sh | bash

# Or build from source
cargo build --release --features deltanet --example daemon --example infer --example infer_hfq -p engine

# Run
hipfire run models/qwen3.5-4b.q4.hfq "What is the capital of France?"
# Or directly:
./target/release/examples/infer models/qwen3.5-4b.q4.hfq "What is the capital of France?"
```

## Performance (RX 5700 XT, 8GB)

| Model | Quant | tok/s | Notes |
|-------|-------|-------|-------|
| Qwen3.5-0.8B | HFQ4 | **222** | DeltaNet, tiled LDS GDN |
| Qwen3.5-0.8B | HFQ6 | **210** | Higher quality, ~600MB |
| Qwen3.5-2B | HFQ4 | **141** | |
| Qwen3.5-4B | HFQ4 | **63** | Best balance of speed + quality |
| Qwen3.5-4B | HFQ6 | **53** | |
| Qwen3.5-9B | HFQ4 | **45** | Best quality, fits 8GB |
| Qwen3.5-9B | HFQ6 | **37** | Near-FP16 quality |
| Qwen3-8B | HFQ4 | **59.9** | Standard attention |
| ollama Qwen3.5-9B | — | 4.93 | llama.cpp + ROCm (same GPU) |

Recommended picks:
- **Speed**: 0.8B HFQ4 (222 tok/s) — fast drafting, coding assistants
- **Balance**: 4B HFQ4 (63 tok/s) — best quality-per-token for 8GB
- **Quality**: 9B HFQ4 (45 tok/s) — strongest reasoning, still real-time

Full benchmarks: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)

## TurboQuant KV Cache

Compress the KV cache with FWHT + quantization for longer context in less VRAM:

| KV Mode | Compression | Speed (4B) | Quality |
|---------|-------------|------------|---------|
| Q8 (default) | 3.8x | 63 tok/s | baseline |
| Turbo4 | **7.8x** | 61 tok/s | minimal loss |
| Turbo2 | **15.5x** | 59 tok/s | good on ≤4B |

```bash
# Use turbo4 for 2x longer context
./target/release/examples/infer models/qwen3.5-4b.q4.hfq --turbo4 "Your prompt"
```

## Features

- **Qwen3.5 DeltaNet**: Gated linear attention with tiled LDS kernel (32 VGPRs, 20 waves)
- **HFQ4 + HFQ6**: Two weight quantization levels, both with GPU GEMV kernels
- **Vision-Language (VL)**: GPU vision encoder, `hipfire run model --image img.png "Describe this"`
- **TurboQuant KV**: Symmetric turbo4/turbo2 with 256-dim FWHT, up to 15.5x compression
- **Thinking mode**: `<think>` reasoning with n-gram loop prevention
- **Pre-compiled kernels**: Ship .hsaco blobs, no ROCm SDK needed at runtime
- **4 GPU arches**: gfx1010 (5700 XT), gfx1030 (6800 XT), gfx1100 (7900 XTX), gfx1200 (9070)
- **Zero VRAM leak**: Explicit GPU free + pool drain for model eviction
- **OpenAI-compatible API**: `hipfire serve` → `/v1/chat/completions` with SSE streaming

## Supported Models

| Family | Sizes | Arch | Quants |
|--------|-------|------|--------|
| Qwen3.5 | 0.8B, 2B, 4B, 9B | DeltaNet hybrid | HFQ4, HFQ6 |
| Qwen3.5-VL | 0.8B, 4B, 9B | DeltaNet + ViT | HFQ4 + F16 vision |
| Qwen3 | 0.6B, 8B | LLaMA attention | HFQ4 |

## CLI

```bash
hipfire serve [port]                        # OpenAI-compatible HTTP server
hipfire run <model> [prompt]                # Interactive generation
hipfire run <model> --image img.png [prompt] # Vision-language
hipfire list                                # Show local models
```

## API

```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

## Architecture

```
Bun CLI (hipfire serve/run)
  └→ Rust daemon (JSON lines IPC)
       └→ GPU kernels (pre-compiled .hsaco, 100+ kernels per arch)
            ├→ HFQ4/HFQ6 GEMV (18 VGPRs, max occupancy)
            ├→ Tiled LDS GDN (32 VGPRs, warp shuffle)
            ├→ TurboQuant KV (turbo4/turbo2, 128+256-dim FWHT)
            └→ Vision encoder (GEMM, LayerNorm, ViT attention)
```

## License

MIT

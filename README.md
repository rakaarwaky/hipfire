# hipfire

LLM inference engine for AMD RDNA GPUs. Rust + HIP. No ROCm runtime needed for deployment.

## Quickstart

```bash
# Install (Linux, requires AMD GPU + HIP SDK)
curl -L https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.sh | bash

# Windows (PowerShell)
irm https://raw.githubusercontent.com/Kaden-Schutt/hipfire/master/scripts/install.ps1 | iex

# Pull a model and chat
hipfire pull qwen3.5:9b
hipfire run qwen3.5:9b
```

## Interactive Chat

```
>>> What is the capital of France?
The capital of France is Paris.
(12 tokens, 34 tok/s)

>>> What about Germany?
Berlin is the capital of Germany.
(15 tokens, 33 tok/s)

>>> /stats
Position: 342/4096 tokens used
Total generated: 297 tokens

>>> /reset
Conversation reset.
```

Commands: `/reset`, `/stats`, `/quit`, `/help`

## Performance

**RX 5700 XT (8GB, gfx1010):**

| Model | Quant | tok/s | Notes |
|-------|-------|-------|-------|
| Qwen3.5-0.8B | HF4 | **190** | DeltaNet |
| Qwen3.5-4B | HF4 | **61** | Best balance of speed + quality |
| Qwen3.5-9B | HF4 | **43** | Best quality on 8GB |
| Qwen3.5-9B | HF6 | **34** | Near-FP16 quality |
| Qwen3-8B | HF4 | **60** | Standard attention |
| ollama Qwen3.5-9B | — | 4.93 | llama.cpp + ROCm (same GPU) |

**RX 7900 XTX (24GB, gfx1100):**

| Model | Quant | tok/s |
|-------|-------|-------|
| Qwen3.5-9B | HF4 | **62** |
| Qwen3.5-27B | HF4 | **25-27** |
| Qwen3.5-27B | HF6 | **16-20** |

## Supported Hardware

Any AMD GPU with HIP SDK support. Kernels JIT-compile for the detected arch:

| Generation | Cards | Status |
|-----------|-------|--------|
| RDNA 1 | RX 5500/5600/5700 | Tested, stable |
| RDNA 2 | RX 6600/6700/6800/6900 | Supported |
| RDNA 3 | RX 7600/7800/7900 | Tested (7900 XTX) |
| RDNA 3.5 | Strix Halo / Strix Point APUs | Supported (JIT) |
| RDNA 4 | RX 9070 | Supported (JIT) |
| Datacenter | BC-250, MI-series | Supported (JIT) |

## Features

- **Qwen3.5 DeltaNet**: Gated linear attention with tiled LDS kernel, stochastic-rounded Q8 state
- **Multi-turn conversation**: Cumulative KV cache + DeltaNet state across turns
- **System prompts**: ChatML format, persists across turns
- **HF4/HF6 weight formats**: Hipfire-native quantization optimized for RDNA GEMV
- **TurboQuant KV**: FWHT + polynomial centroid dequant, boundary layer protection (LA-V7)
- **Asymmetric KV**: Q8 keys + turbo4 values — 9B at 8K+ context on 8GB VRAM
- **Vision-Language**: GPU vision encoder for Qwen3.5-VL models
- **Thinking mode**: `<think>` reasoning with n-gram loop prevention
- **JIT kernels**: hipcc compiles for any GPU arch at first run — no pre-compiled blobs
- **OpenAI-compatible API**: `hipfire serve` → `/v1/chat/completions` with SSE streaming
- **Interactive REPL**: `hipfire run` with `/reset`, `/stats`, system prompts

## Supported Models

| Family | Sizes | Arch | Quants |
|--------|-------|------|--------|
| Qwen3.5 | 0.8B, 2B, 4B, 9B, 27B | DeltaNet hybrid | HF4, HF6 |
| Qwen3.5-VL | 0.8B, 4B, 9B | DeltaNet + ViT | HF4 + F16 vision |
| Qwen3 | 0.6B, 8B | LLaMA attention | HF4 |

## CLI

```bash
hipfire pull qwen3.5:9b                           # Download model
hipfire run qwen3.5:9b                             # Interactive chat
hipfire run qwen3.5:9b "What is 2+2?"             # Single prompt
hipfire run qwen3.5:9b --image img.png "Describe"  # Vision
hipfire run qwen3.5:9b --system "Be concise"       # System prompt
hipfire serve [port]                                # HTTP server
hipfire list -r                                     # Show models
hipfire update                                      # Pull latest + rebuild
hipfire diag                                        # Diagnostics
```

## API (Server Mode)

```bash
hipfire serve                          # Default port 11435

curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Works with Open WebUI, SillyTavern, and any OpenAI-compatible frontend.

## Advanced: TurboQuant KV Cache

Compress KV cache for longer context. Recommended on RDNA2+ (6800 XT and newer):

```bash
# Asymmetric: Q8 keys + turbo4 values (5.1x compression)
hipfire run qwen3.5:9b --asym --boundary 2

# Symmetric turbo4 (7.8x compression)  
hipfire run qwen3.5:4b --turbo 4
```

| Mode | Compression | Best for |
|------|-------------|----------|
| Q8 (default) | 3.8x | RDNA1 (5700 XT) — fastest decode |
| Asym + boundary | 5.1x | RDNA2+ — fits larger models in VRAM |
| Turbo4 | 7.8x | RDNA2+ — maximum context length |

## Architecture

```
Bun CLI (hipfire run/serve/pull)
  └→ Rust daemon (JSON lines IPC)
       └→ GPU kernels (JIT compiled via hipcc, 100+ kernels)
            ├→ HF4/HF6 GEMV (18 VGPRs, max occupancy)
            ├→ DeltaNet GDN (stochastic Q8 state, warp shuffle FWHT)
            ├→ TurboQuant KV (polynomial dequant, boundary layer protection)
            └→ Vision encoder (GEMM, LayerNorm, ViT attention)
```

## Redline (experimental)

Direct-KMD GPU compute that bypasses HIP entirely. Talks to `libdrm_amdgpu.so` (55KB).

- 30µs dispatch latency, 0.5ms startup, 2.8MB RSS
- Dispatches real inference kernels (GEMM, SiLU, RMSNorm)
- Working compute barriers (RELEASE_MEM + WAIT_REG_MEM)
- See `benchmarks/redline_vs_hip.md` for numbers

## Contributing

Technical deep-dive: [docs/DELTANET.md](docs/DELTANET.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, benchmarking, and quantizing models.

**Benchmarks wanted**: if you have a 6800 XT, 7900 XTX, 9070, or Strix Halo — we need your numbers!

## License

MIT

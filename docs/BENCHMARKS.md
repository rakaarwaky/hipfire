# hipfire Benchmarks

Hardware: AMD Radeon RX 5700 XT (8GB VRAM, RDNA1 gfx1010, 448 GB/s peak)
Date: 2026-04-02

## Qwen3.5 DeltaNet — Full Model Matrix

All models, Q8 KV (default), `--no-think`, averaged across 4 prompts:

| Model | Quant | tok/s | Quality | VRAM |
|-------|-------|-------|---------|------|
| Qwen3.5-0.8B | HFQ4 | **222** | OK | ~600MB |
| Qwen3.5-0.8B | HFQ6 | **210** | OK | ~750MB |
| Qwen3.5-2B | HFQ4 | **141** | OK | ~1.5GB |
| Qwen3.5-2B | HFQ6 | **127** | OK | ~2GB |
| Qwen3.5-4B | HFQ4 | **63** | OK | ~2.5GB |
| Qwen3.5-4B | HFQ6 | **53** | OK | ~3.5GB |
| Qwen3.5-9B | HFQ4 | **45** | OK | ~5.5GB |
| Qwen3.5-9B | HFQ6 | **37** | OK | ~7.5GB |

Notes:
- HFQ4 = 4-bit (0.53 B/w). Best speed, good quality.
- HFQ6 = 6-bit (0.78 B/w). Better quality, ~15% slower.
- All models: DeltaNet hybrid attention (linear + full attention layers).
- 9B HFQ6 barely fits in 8GB — usable but tight.

## TurboQuant KV Cache — Qwen3.5 (head_dim=256)

Tested on "Compare TCP/UDP protocols" prompt:

| Model | KV Mode | tok/s | Compression | Quality |
|-------|---------|-------|-------------|---------|
| 4B-Q4 | Q8 (default) | **62.6** | 3.8x | OK |
| 4B-Q4 | Turbo4 | **60.9** | 7.8x | OK |
| 4B-Q4 | Turbo2 | **58.5** | 15.5x | OK |
| 9B-Q4 | Q8 (default) | **43.4** | 3.8x | OK |
| 9B-Q4 | Turbo4 | **42.3** | 7.8x | OK |
| 9B-Q4 | Turbo2 | **35.9** | 15.5x | degraded |

Notes:
- Turbo4: FWHT-256 + 4-bit quantize. 7.8x compression, minimal quality loss. Recommended.
- Turbo2: FWHT-256 + 2-bit quantize. 15.5x compression. Works on 4B, degrades on 9B.
- New 256-dim kernels: `attention_turbo4_kv_256.hip`, `kv_cache_write_turbo4_256.hip` (and turbo2 variants).
- 32 threads (one RDNA wavefront) × 8 dims/thread. FWHT-256 via warp shuffle. ~28 VGPRs.

## Long Context Stress Test

~400 token input, 512 token generation:

| Model | KV Mode | tok/s | Quality |
|-------|---------|-------|---------|
| 4B-Q4 | Q8 | **59.8** | OK |
| 4B-Q4 | Turbo4 | **50.4** | OK |
| 4B-Q4 | Turbo2 | **48.0** | OK |
| 9B-Q4 | Turbo4 | **38.0** | OK |

Turbo4 enables ~2x longer context than Q8 in the same VRAM budget.

## TurboQuant KV Cache — Qwen3-8B (head_dim=128)

From earlier benchmarks (2026-03-27):

| Config | Short (91 tok) | Hard (128 tok) | KV Compression |
|--------|---------------|----------------|----------------|
| Q8 KV (baseline) | **59.9 tok/s** | **58.8 tok/s** | 3.88x |
| FP32 KV | 57.0 | — | 1.0x |
| turbo2 (2-bit) | 54.1 | 51.8 | **14.2x** |
| turbo3 (3-bit) | 50.8 | 44.5 | 9.85x |
| turbo4 (4-bit) | 53.6 | 51.0 | 7.5x |

## Comparison vs llama.cpp + ROCm

| Setup | Model | tok/s |
|-------|-------|-------|
| **hipfire** | Qwen3.5-9B HFQ4 | **45** |
| **hipfire** | Qwen3.5-4B HFQ4 | **63** |
| **hipfire** | Qwen3.5-0.8B HFQ4 | **222** |
| ollama (llama.cpp) | Qwen3.5-9B Q4 | 4.93 |

hipfire is **9x faster** than llama.cpp+ROCm on the same hardware for Qwen3.5 models.

## Notes

- All benchmarks use ChatML prompting, greedy decoding (temp=0.3), repeat penalty 1.3.
- Kernel cache warm (first-run compilation excluded).
- Turbo KV supports head_dim=128 (Qwen3) and head_dim=256 (Qwen3.5).
- Quality rated by automated coherence check: repetition frequency + output length.

# Phase 1 Research Synthesis

**Date:** 2026-03-20

## Summary

3 research agents dispatched in parallel. Key findings below.

## HIP Runtime API

- Library: `libamdhip64.so.6` → `libamdhip64.so.6.3.60304` (23MB)
- ROCm 6.x ABI change: `hipGetDeviceProperties` is macro'd to `hipGetDevicePropertiesR0600`
- Our dlopen approach avoids this since we don't use device properties
- All function signatures verified against headers

## GGUF Format

- Our parser is correct (magic 0x46554747, version 3, alignment 32)
- Q4_K = 144 bytes/block, Q6_K = 210 bytes/block, Q8_0 = 34 bytes/block
- No existing Rust GGUF crate has ROCm/HIP support — we are first
- candle-rs has production GGUF loader but no HIP backend

## Rust HIP Ecosystem

- ncdrone/rustane does NOT exist publicly (the CLAUDE.md reference was hallucinated)
- Real crates: cubecl-hip-sys (tracel-ai, actively maintained), hip-rs (smedegaard)
- Our hip-bridge independently follows the same dlopen + RAII + safe wrapper pattern
- cudarc (CUDA) is the gold standard for the 3-tier FFI pattern

## Qwen3 vs Qwen3.5

CRITICAL: Qwen3.5 is NOT LLaMA-compatible. Uses Gated DeltaNet (linear attention).

| Feature | Qwen3 dense | Qwen3.5 |
|---------|-------------|---------|
| Arch | LLaMA-like + QK norm | DeltaNet hybrid |
| Engine change | Minimal | Major (new operator) |
| 8GB VRAM fit | 0.6B-4B (F32) | 4B Q4_K_M (2.9GB), 9B Q4_K_M (5.9GB) |

"Configurable" = thinking mode toggle (/think, /no_think), not architecture.

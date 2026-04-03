# hipfire-tester: Agent Guide

You are a Claude Code agent helping an alpha tester run benchmarks, quantize models,
and report results for the hipfire inference engine. hipfire runs LLMs on AMD RDNA GPUs
via a Rust-native HIP path — no Python in the hot path.

## Supported GPUs

| Card | Arch | VRAM | Status |
|------|------|------|--------|
| RX 5700 XT | gfx1010 | 8GB | Primary dev target |
| RX 6800 XT | gfx1030 | 16GB | Alpha target |
| RX 7900 XTX | gfx1100 | 24GB | Alpha target |
| RX 9060 | gfx1200 | 16GB | Alpha target |
| RX 9070 XT | gfx1201 | 16GB | Alpha target, RDNA4-optimized kernels |

## Quantizing Models

### Build the quantizer

```bash
cargo build --release -p hipfire-quantize
```

### Download and quantize from HuggingFace

```bash
# HFQ4 — fastest, 0.53 B/w
target/release/hipfire-quantize \
  --input Qwen/Qwen3.5-27B \
  --output models/qwen3.5-27b.q4.hfq \
  --format hfq4

# HFQ6 — best quality, 0.78 B/w
target/release/hipfire-quantize \
  --input Qwen/Qwen3.5-27B \
  --output models/qwen3.5-27b.hfq6.hfq \
  --format hfq6
```

### Available formats

| Format | B/weight | Speed | Quality | Use case |
|--------|----------|-------|---------|----------|
| `hfq4` | 0.53 | Fastest | Good | Daily use, 8GB cards |
| `hfq6` | 0.78 | Fast | Better | Quality-focused, 16GB+ |
| `q8hfq` | 1.06 | Moderate | Reference | Correctness testing |

### Model size rules of thumb

- HFQ4: params × 0.53 bytes (27B → ~14.3GB)
- HFQ6: params × 0.78 bytes (27B → ~21.1GB)
- Add ~1–2GB overhead for KV cache at 2K context

### Which model fits which GPU

| Model | HFQ4 size | HFQ6 size | Fits 8GB | Fits 16GB | Fits 24GB |
|-------|-----------|-----------|----------|-----------|-----------|
| 0.8B | ~430MB | ~625MB | Yes | Yes | Yes |
| 2B | ~1.1GB | ~1.6GB | Yes | Yes | Yes |
| 4B | ~2.1GB | ~3.1GB | Yes | Yes | Yes |
| 9B | ~4.8GB | ~7.0GB | Yes | Yes | Yes |
| 27B | ~14.3GB | ~21.1GB | No | Yes* | Yes |

*27B HFQ6 may not fit 16GB with full KV cache — use `--turbo4` to reduce KV VRAM.

## Running Inference

### Build the inference binaries

```bash
cargo build --release --features deltanet --example infer --example daemon -p engine
```

### Basic run

```bash
target/release/examples/infer models/qwen3.5-9b.q4.hfq "Your prompt here"
```

### Options

```bash
# Disable thinking mode (faster, no <think> block)
target/release/examples/infer models/qwen3.5-9b.q4.hfq --no-think "Your prompt"

# TurboQuant KV — 7.8x KV compression, minimal quality loss
target/release/examples/infer models/qwen3.5-9b.q4.hfq --turbo4 "Your prompt"

# TurboQuant KV — 15.5x KV compression, good quality on <=4B
target/release/examples/infer models/qwen3.5-9b.q4.hfq --turbo2 "Your prompt"

# Vision-language mode
target/release/examples/infer models/qwen3.5-4b.q4.hfq --image photo.png "Describe this"
```

### Via the CLI (ollama-style)

```bash
hipfire pull qwen3.5:9b         # Download model
hipfire run qwen3.5:9b "Hello"  # Run (auto-pulls if needed)
hipfire serve                   # OpenAI-compatible API on port 11435
hipfire run qwen3.5:4b --image img.png "Describe this"
hipfire list -r                 # Show local + available models
```

### KV mode summary

| Flag | Compression | Relative speed | Quality | When to use |
|------|-------------|----------------|---------|-------------|
| (none) | 3.8x | baseline | best | Default |
| `--turbo4` | 7.8x | ~97% | minimal loss | Long context, 8–16GB |
| `--turbo2` | 15.5x | ~94% | good on <=4B | Very long context |

## Running Benchmarks

### Automated megabench (all models, all KV modes)

```bash
bash scripts/megabench-q35.sh
```

This runs every combination of model, quant, and KV mode. Takes 20–60 minutes depending
on your GPU. Output goes to stdout — pipe to a file if you want to save it:

```bash
bash scripts/megabench-q35.sh 2>&1 | tee bench-results.txt
```

### Quick single-model bench

```bash
timeout 60 target/release/examples/infer models/MODEL.hfq --no-think \
  "Explain the three laws of thermodynamics" 2>&1 | grep "Done"
```

Look for the `=== Done` line — it contains tok/s and total tokens.

### What to capture

Run at least two prompts per model to get stable numbers:
1. Short prompt (~10 tokens input): `"Hello, who are you?"`
2. Long prompt (~400 tokens input): use a multi-paragraph passage or the megabench script

## Submitting Benchmark Results

Paste the following template into a GitHub issue titled "Benchmarks: [GPU name]":

```
GPU: [card name] ([gfx ID], [VRAM]GB)
OS: [Linux distro + kernel / Windows version]
ROCm version: [output of: dpkg -l | grep rocm-core | awk '{print $3}']
Model: [model name and quant, e.g. Qwen3.5-9B HFQ4]
tok/s: [number from === Done line]
Coherence: [OK / LOOP / REPET / SHORT]
KV mode: [Q8 / turbo4 / turbo2]
Context: [short (~10 tok input) / long (~400 tok input)]
Notes: [anything unusual: thermal throttle, OOM, etc.]
```

### Coherence codes

| Code | Meaning |
|------|---------|
| OK | Output is coherent and on-topic |
| LOOP | Model repeats the same phrase in a loop |
| REPET | Output degrades into repetition after N tokens |
| SHORT | Output stops abnormally short (< 20 tokens) |

If coherence is not OK, paste the first 100 characters of the output in Notes.

## Compiling Kernels for New Arches

If your GPU arch is not in `kernels/compiled/`, you need to compile kernels yourself.
This requires the ROCm SDK (hipcc).

```bash
# Compile for a new arch
scripts/compile-kernels.sh gfxNNNN

# Verify — should print 102+ kernels
ls kernels/compiled/gfxNNNN/*.hsaco | wc -l
```

If you don't have the ROCm SDK installed:

```bash
# Ubuntu/Debian
sudo apt install rocm-hip-sdk

# Check hipcc works
hipcc --version
```

After compiling, share the `kernels/compiled/gfxNNNN/` directory in your GitHub issue
so the maintainers can include it in the next release.

## Troubleshooting

For GPU detection, driver, and ROCm runtime issues, use the hipfire-diag skill:
- Run `.skills/hipfire-diag/run-diagnostics.sh` and share the JSON output

### Common tester issues

**OOM (out of memory)**
Model is too large for available VRAM. Options:
1. Use a smaller model (e.g., 4B instead of 9B)
2. Use `--turbo4` to reduce KV cache VRAM usage
3. Close other GPU-using applications

**"daemon not found" error**
The daemon binary wasn't built. Run:
```bash
cargo build --release --features deltanet --example daemon -p engine
```

**No kernels for my GPU**
The installer should have copied pre-compiled kernels to `~/.hipfire/bin/kernels/compiled/{arch}/`.
Check:
```bash
ls ~/.hipfire/bin/kernels/compiled/
# Should show one directory per supported arch, e.g.: gfx1010  gfx1030  gfx1100  gfx1200
```

If your arch is missing, see "Compiling Kernels for New Arches" above.

**Windows: runtime not found**
Ensure `amdhip64.dll` is in `%USERPROFILE%\.hipfire\runtime\`. The PowerShell installer
should place it there automatically. If not, copy it from your ROCm installation:
```
C:\Program Files\AMD\ROCm\bin\amdhip64.dll
```

**Low tok/s (well below expected)**
Possible causes:
- VRAM contention: check `radeontop` or Task Manager for GPU usage by other apps
- Thermal throttling: check GPU temperature with `radeontop` or `sensors`
- Wrong kernel arch: run diag to confirm the loaded arch matches your GPU
- Cold start: first run compiles kernels (slow); subsequent runs use cache

**Thinking mode loop / repetition**
Add `--no-think` to disable the `<think>` block. This is faster and avoids loop issues
on some prompt styles. If the issue persists without `--no-think`, report it with the
Coherence code LOOP or REPET in your benchmark submission.

## Asking for Help

Open a GitHub issue at https://github.com/Kaden-Schutt/hipfire with:
- The output of `.skills/hipfire-diag/run-diagnostics.sh`
- Your GPU model and OS
- The exact command you ran
- The full error output (not just the last line)

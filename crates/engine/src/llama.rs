//! LLaMA model implementation using RDNA GPU compute.
//! Supports loading from GGUF files and running inference.

use crate::gguf::{GgmlType, GgufFile, TensorInfo};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;

/// Model architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Qwen3,
}

/// Model configuration, read from GGUF metadata.
/// Supports LLaMA-family and Qwen3 architectures.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub arch: ModelArch,
    pub dim: usize,        // model dimension (embedding size)
    pub hidden_dim: usize, // FFN hidden dimension
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize, // for GQA
    pub vocab_size: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub bos_token: u32,
    pub eos_token: u32,
    pub has_qk_norm: bool, // Qwen3 feature
}

impl LlamaConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Option<Self> {
        let arch_str = gguf.meta_str("general.architecture")?;

        // Determine architecture and metadata prefix
        let (arch, prefix) = match arch_str {
            "llama" => (ModelArch::Llama, "llama"),
            "qwen3" => (ModelArch::Qwen3, "qwen3"),
            other => {
                eprintln!("Warning: unknown architecture '{other}', attempting LLaMA-compatible");
                (ModelArch::Llama, other)
            }
        };

        let dim = gguf.meta_u32(&format!("{prefix}.embedding_length"))? as usize;
        let n_layers = gguf.meta_u32(&format!("{prefix}.block_count"))? as usize;
        let n_heads = gguf.meta_u32(&format!("{prefix}.attention.head_count"))? as usize;
        let n_kv_heads = gguf
            .meta_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;
        let hidden_dim = gguf.meta_u32(&format!("{prefix}.feed_forward_length"))? as usize;
        let vocab_size = gguf
            .meta_u32(&format!("{prefix}.vocab_size"))
            .or_else(|| {
                gguf.find_tensor("token_embd.weight")
                    .map(|t| t.shape[1] as u32)
            })?
            as usize;
        let head_dim = gguf
            .meta_u32(&format!("{prefix}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(dim / n_heads);
        let norm_eps = gguf.meta_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
        let max_seq_len = gguf
            .meta_u32(&format!("{prefix}.context_length"))
            .unwrap_or(2048) as usize;
        let rope_freq_base = gguf
            .meta_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(10000.0);
        let bos_token = gguf.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_token = gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        // Check for QK normalization (Qwen3 feature)
        let has_qk_norm = gguf.find_tensor("blk.0.attn_q_norm.weight").is_some();

        Some(LlamaConfig {
            arch,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            head_dim,
            norm_eps,
            max_seq_len,
            rope_freq_base,
            bos_token,
            eos_token,
            has_qk_norm,
        })
    }
}

/// Dequantize Q4_0 data to f32.
/// Q4_0 block: 2 bytes (f16 scale) + 16 bytes (32 x 4-bit values)
pub fn dequantize_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 32;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let block_offset = b * 18; // 2 + 16 bytes per block
        if block_offset + 18 > data.len() {
            break;
        }
        let scale_bytes = [data[block_offset], data[block_offset + 1]];
        let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));

        for j in 0..16 {
            let byte = data[block_offset + 2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;

            let idx = b * block_size + j * 2;
            if idx < n {
                out[idx] = lo as f32 * scale;
            }
            if idx + 1 < n {
                out[idx + 1] = hi as f32 * scale;
            }
        }
    }
    out
}

/// Dequantize Q8_0 data to f32.
/// Q8_0 block: 2 bytes (f16 scale) + 32 bytes (32 x int8)
pub fn dequantize_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 32;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let block_offset = b * 34; // 2 + 32 bytes per block
        if block_offset + 34 > data.len() {
            break;
        }
        let scale_bytes = [data[block_offset], data[block_offset + 1]];
        let scale = f16_to_f32(u16::from_le_bytes(scale_bytes));

        for j in 0..32 {
            let idx = b * block_size + j;
            if idx < n {
                let val = data[block_offset + 2 + j] as i8;
                out[idx] = val as f32 * scale;
            }
        }
    }
    out
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormalized
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    let exp32 = exp + 127 - 15;
    f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
}

/// Dequantize Q4_K data to f32.
/// Q4_K super-block: 256 elements
///   2 bytes: f16 d (super-block scale)
///   2 bytes: f16 dmin (super-block min)
///   12 bytes: scales/mins for 8 sub-blocks (6 bits each, packed)
///   128 bytes: 256 x 4-bit quantized values
pub fn dequantize_q4_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 144; // 2+2+12+128
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }

        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));

        // Unpack scales and mins from 12 bytes (at off+4)
        let sc_data = &data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        // First 4 sub-blocks: lower 6 bits from bytes 0-3 (scales) and 4-7 (mins)
        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        // Next 4 sub-blocks: lower 4 bits from bytes 8-11, upper 2 bits from bytes 0-7
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        // Dequantize 256 values from 128 bytes of 4-bit data.
        // GGML layout: 4 groups of 64 elements. Each group has 2 sub-blocks
        // sharing 32 bytes: lower nibble → even sub-block, upper nibble → odd.
        let qdata = &data[off + 16..off + 16 + 128];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;
            let sc_even = d * scales[sb_even] as f32;
            let m_even = dmin * mins[sb_even] as f32;
            let sc_odd = d * scales[sb_odd] as f32;
            let m_odd = dmin * mins[sb_odd] as f32;

            for l in 0..32 {
                let byte = qdata[group * 32 + l];
                let idx_even = b * block_size + group * 64 + l;
                let idx_odd = idx_even + 32;
                if idx_even < n {
                    out[idx_even] = (byte & 0x0F) as f32 * sc_even - m_even;
                }
                if idx_odd < n {
                    out[idx_odd] = ((byte >> 4) & 0x0F) as f32 * sc_odd - m_odd;
                }
            }
        }
    }
    out
}

/// Dequantize Q6_K data to f32 (matches GGML reference exactly).
/// Q6_K super-block: 256 elements = 2 groups of 128
///   ql[128]: lower 4 bits (shared between lo/hi nibble pairs)
///   qh[64]: upper 2 bits (packed 4 per byte)
///   scales[16]: int8 scales for sub-groups of 16 elements
///   d: f16 super-block scale
pub fn dequantize_q6_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 210; // 128 + 64 + 16 + 2
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];

    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }

        let mut ql = &data[off..off + 128];
        let mut qh = &data[off + 128..off + 192];
        let mut sc = &data[off + 192..off + 208];
        let d = f16_to_f32(u16::from_le_bytes([data[off + 208], data[off + 209]]));

        let base = b * block_size;

        // Process 2 groups of 128 elements each
        for group in 0..2 {
            let y_off = base + group * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;

                let idx0 = y_off + l;
                let idx1 = y_off + l + 32;
                let idx2 = y_off + l + 64;
                let idx3 = y_off + l + 96;

                if idx0 < n { out[idx0] = d * sc[is] as i8 as f32 * q1 as f32; }
                if idx1 < n { out[idx1] = d * sc[is + 2] as i8 as f32 * q2 as f32; }
                if idx2 < n { out[idx2] = d * sc[is + 4] as i8 as f32 * q3 as f32; }
                if idx3 < n { out[idx3] = d * sc[is + 6] as i8 as f32 * q4 as f32; }
            }
            // Advance pointers for next group
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }
    out
}

/// Load tensor from GGUF as f32, dequantizing if needed.
fn load_tensor_f32(gguf: &GgufFile, info: &TensorInfo) -> Vec<f32> {
    let data = gguf.tensor_data(info);
    let n = info.numel();

    match info.dtype {
        GgmlType::F32 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in data.chunks_exact(4).enumerate().take(n) {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            out
        }
        GgmlType::F16 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in data.chunks_exact(2).enumerate().take(n) {
                out[i] = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            out
        }
        GgmlType::Q4_0 => dequantize_q4_0(data, n),
        GgmlType::Q8_0 => dequantize_q8_0(data, n),
        GgmlType::Q4K => dequantize_q4_k(data, n),
        GgmlType::Q6K => dequantize_q6_k(data, n),
        other => panic!("unsupported tensor type: {:?}", other),
    }
}

/// A weight matrix on GPU — may be quantized or F32.
pub struct WeightTensor {
    pub buf: GpuTensor,
    pub dtype: GgmlType,
    pub m: usize, // output dim (rows)
    pub k: usize, // input dim (cols)
}

/// GPU-resident LLaMA model weights.
pub struct LlamaWeights {
    pub token_embd: GpuTensor, // always F32 (for embedding lookup)
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
}

pub struct LayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub q_norm: Option<GpuTensor>, // Qwen3: per-head Q normalization
    pub k_norm: Option<GpuTensor>, // Qwen3: per-head K normalization
    pub ffn_norm: GpuTensor,
    pub w_gate: WeightTensor,
    pub w_up: WeightTensor,
    pub w_down: WeightTensor,
}

/// Dispatch GEMV for a weight tensor (quantized or F32).
/// y = W * x where W is the weight tensor, x is F32 input, y is F32 output.
pub fn weight_gemv(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
) -> HipResult<()> {
    match w.dtype {
        GgmlType::F32 => gpu.gemv_f32(&w.buf, x, y),
        GgmlType::Q4K => gpu.gemv_q4k(&w.buf, x, y, w.m, w.k),
        GgmlType::Q6K => gpu.gemv_q6k(&w.buf, x, y, w.m, w.k),
        GgmlType::Q8_0 => gpu.gemv_q8_0(&w.buf, x, y, w.m, w.k),
        other => {
            // Fallback: dequantize to F32 on CPU, upload, GEMV
            eprintln!("WARNING: no GPU kernel for {:?}, falling back to CPU dequant", other);
            Err(hip_bridge::HipError::new(0, &format!("unsupported quant type {:?}", other)))
        }
    }
}

/// Load LLaMA weights from GGUF onto GPU.
/// Quantized weights stay quantized (Q4_K, Q6_K, Q8_0).
/// Only norm weights and embeddings are dequantized to F32.
pub fn load_weights(
    gguf: &GgufFile,
    config: &LlamaConfig,
    gpu: &mut Gpu,
) -> HipResult<LlamaWeights> {
    // Helper: upload F32 tensor
    fn up_f32(gguf: &GgufFile, gpu: &mut Gpu, name: &str, shape: &[usize]) -> HipResult<GpuTensor> {
        let info = gguf.find_tensor(name).unwrap_or_else(|| panic!("tensor not found: {name}"));
        let data = load_tensor_f32(gguf, info);
        gpu.upload_f32(&data, shape)
    }
    // Helper: upload quantized weight
    fn up_weight(gguf: &GgufFile, gpu: &Gpu, name: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
        let info = gguf.find_tensor(name).unwrap_or_else(|| panic!("tensor not found: {name}"));
        let raw_data = gguf.tensor_data(info);
        let buf = gpu.upload_raw(raw_data, &[raw_data.len()])?;
        Ok(WeightTensor { buf, dtype: info.dtype, m, k })
    }

    eprintln!("  loading token_embd...");
    let token_embd = up_f32(gguf, gpu, "token_embd.weight", &[config.vocab_size, config.dim])?;
    eprintln!("  loading output_norm...");
    let output_norm = up_f32(gguf, gpu, "output_norm.weight", &[config.dim])?;

    eprintln!("  loading output...");
    let output = if gguf.find_tensor("output.weight").is_some() {
        up_weight(gguf, gpu, "output.weight", config.vocab_size, config.dim)?
    } else {
        let info = gguf.find_tensor("token_embd.weight").unwrap();
        let data = load_tensor_f32(gguf, info);
        let buf = gpu.upload_f32(&data, &[config.vocab_size, config.dim])?;
        WeightTensor { buf, dtype: GgmlType::F32, m: config.vocab_size, k: config.dim }
    };

    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!("  loading layer {i}/{} ...", config.n_layers);
        let p = format!("blk.{i}");
        let kv_dim = config.n_kv_heads * config.head_dim;

        let q_out_dim = config.n_heads * config.head_dim;
        let _k_out_dim = config.n_kv_heads * config.head_dim;

        let layer = LayerWeights {
            attn_norm: up_f32(gguf, gpu, &format!("{p}.attn_norm.weight"), &[config.dim])?,
            wq: up_weight(gguf, gpu, &format!("{p}.attn_q.weight"), q_out_dim, config.dim)?,
            wk: up_weight(gguf, gpu, &format!("{p}.attn_k.weight"), kv_dim, config.dim)?,
            wv: up_weight(gguf, gpu, &format!("{p}.attn_v.weight"), kv_dim, config.dim)?,
            wo: up_weight(gguf, gpu, &format!("{p}.attn_output.weight"), config.dim, q_out_dim)?,
            q_norm: if config.has_qk_norm {
                Some(up_f32(gguf, gpu, &format!("{p}.attn_q_norm.weight"), &[config.head_dim])?)
            } else {
                None
            },
            k_norm: if config.has_qk_norm {
                Some(up_f32(gguf, gpu, &format!("{p}.attn_k_norm.weight"), &[config.head_dim])?)
            } else {
                None
            },
            ffn_norm: up_f32(gguf, gpu, &format!("{p}.ffn_norm.weight"), &[config.dim])?,
            w_gate: up_weight(gguf, gpu, &format!("{p}.ffn_gate.weight"), config.hidden_dim, config.dim)?,
            w_up: up_weight(gguf, gpu, &format!("{p}.ffn_up.weight"), config.hidden_dim, config.dim)?,
            w_down: up_weight(gguf, gpu, &format!("{p}.ffn_down.weight"), config.dim, config.hidden_dim)?,
        };
        layers.push(layer);
    }

    Ok(LlamaWeights {
        token_embd,
        output_norm,
        output,
        layers,
    })
}

/// Run a single forward pass for one token (decode step).
/// Returns logits over vocab.
pub fn forward(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    token: u32,
    pos: usize,
    kv_cache: &mut KvCache,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;
    let head_dim = config.head_dim;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let kv_dim = n_kv_heads * head_dim;

    // Embedding lookup — GPU-side D2D copy of one row (8KB vs 262MB download)
    let mut x = gpu.alloc_tensor(&[dim], DType::F32)?;
    gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?;

    let tmp = gpu.alloc_tensor(&[dim], DType::F32)?;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];

        // RMSNorm before attention
        gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

        // Q, K, V projections
        let q_dim = n_heads * head_dim;
        let q = gpu.alloc_tensor(&[q_dim], DType::F32)?;
        let k = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
        let v = gpu.alloc_tensor(&[kv_dim], DType::F32)?;

        weight_gemv(gpu, &layer.wq, &tmp, &q)?;
        weight_gemv(gpu, &layer.wk, &tmp, &k)?;
        weight_gemv(gpu, &layer.wv, &tmp, &v)?;

        // QK normalization (Qwen3: still CPU-side for now)
        if config.has_qk_norm {
            let mut q_data = gpu.download_f32(&q)?;
            let mut k_data = gpu.download_f32(&k)?;
            if let Some(ref qn) = layer.q_norm {
                let qn_w = gpu.download_f32(qn)?;
                per_head_rmsnorm_cpu(&mut q_data, &qn_w, n_heads, head_dim, config.norm_eps);
            }
            if let Some(ref kn) = layer.k_norm {
                let kn_w = gpu.download_f32(kn)?;
                per_head_rmsnorm_cpu(&mut k_data, &kn_w, n_kv_heads, head_dim, config.norm_eps);
            }
            // Re-upload normalized Q and K
            let q2 = gpu.upload_f32(&q_data, &[q_dim])?;
            let k2 = gpu.upload_f32(&k_data, &[kv_dim])?;
            // Copy back into q and k buffers using D2D
            gpu.hip.memcpy_dtod(&q.buf, &q2.buf, q_dim * 4)?;
            gpu.hip.memcpy_dtod(&k.buf, &k2.buf, kv_dim * 4)?;
            gpu.free_tensor(q2)?;
            gpu.free_tensor(k2)?;
        }

        // RoPE — GPU-side, no CPU round-trip
        gpu.rope_f32(&q, &k, pos, n_heads, n_kv_heads, head_dim, config.rope_freq_base)?;

        // Store K, V in GPU cache — pure D2D copy, no CPU involvement
        let cache_byte_offset = pos * kv_dim * 4;
        gpu.hip.memcpy_dtod_at(
            &kv_cache.k_gpu[layer_idx].buf, cache_byte_offset,
            &k.buf, 0,
            kv_dim * 4,
        )?;
        gpu.hip.memcpy_dtod_at(
            &kv_cache.v_gpu[layer_idx].buf, cache_byte_offset,
            &v.buf, 0,
            kv_dim * 4,
        )?;
        gpu.free_tensor(k)?;
        gpu.free_tensor(v)?;

        // GPU-side attention
        let attn_gpu = gpu.alloc_tensor(&[q_dim], DType::F32)?;
        gpu.attention_f32(
            &q,
            &kv_cache.k_gpu[layer_idx],
            &kv_cache.v_gpu[layer_idx],
            &attn_gpu,
            pos + 1,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_cache.max_seq,
        )?;
        gpu.free_tensor(q)?;

        // Output projection: o = Wo * attn_out
        let o = gpu.alloc_tensor(&[dim], DType::F32)?;
        weight_gemv(gpu, &layer.wo, &attn_gpu, &o)?;
        gpu.free_tensor(attn_gpu)?;

        // Residual: x = x + o
        let x_new = gpu.alloc_tensor(&[dim], DType::F32)?;
        gpu.add_f32(&x, &o, &x_new)?;
        gpu.free_tensor(x)?;
        gpu.free_tensor(o)?;
        x = x_new;

        // FFN
        gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;

        let gate = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
        let up = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
        weight_gemv(gpu, &layer.w_gate, &tmp, &gate)?;
        weight_gemv(gpu, &layer.w_up, &tmp, &up)?;

        // Fused SiLU(gate) * up — saves one kernel launch + intermediate buffer
        let ffn_hidden = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
        gpu.silu_mul_f32(&gate, &up, &ffn_hidden)?;
        gpu.free_tensor(gate)?;
        gpu.free_tensor(up)?;

        // Down projection
        let ffn_out = gpu.alloc_tensor(&[dim], DType::F32)?;
        weight_gemv(gpu, &layer.w_down, &ffn_hidden, &ffn_out)?;
        gpu.free_tensor(ffn_hidden)?;

        // Residual
        let x_new = gpu.alloc_tensor(&[dim], DType::F32)?;
        gpu.add_f32(&x, &ffn_out, &x_new)?;
        gpu.free_tensor(x)?;
        gpu.free_tensor(ffn_out)?;
        x = x_new;
    }

    // Final norm
    gpu.rmsnorm_f32(&x, &weights.output_norm, &tmp, config.norm_eps)?;

    // Logits: output = output_weight * x
    let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
    weight_gemv(gpu, &weights.output, &tmp, &logits)?;

    let logits_data = gpu.download_f32(&logits)?;
    gpu.free_tensor(x)?;
    gpu.free_tensor(tmp)?;
    gpu.free_tensor(logits)?;

    Ok(logits_data)
}

pub fn apply_rope_cpu_pub(data: &mut [f32], n_heads: usize, head_dim: usize, pos: usize) {
    apply_rope_cpu(data, n_heads, head_dim, pos, 10000.0);
}

fn apply_rope_cpu(data: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, freq_base: f32) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = 1.0 / (freq_base.powf((2 * i) as f32 / head_dim as f32));
            let val = pos as f32 * freq;
            let cos_val = val.cos();
            let sin_val = val.sin();
            let v0 = data[base + i];
            let v1 = data[base + i + half];
            data[base + i] = v0 * cos_val - v1 * sin_val;
            data[base + i + half] = v0 * sin_val + v1 * cos_val;
        }
    }
}

/// Apply per-head RMSNorm (for Qwen3 QK normalization).
fn per_head_rmsnorm_cpu(data: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let base = h * head_dim;
        let head = &data[base..base + head_dim];
        let ss: f32 = head.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
        let rms = 1.0 / (ss + eps).sqrt();
        for d in 0..head_dim {
            data[base + d] = data[base + d] * rms * weight[d];
        }
    }
}

/// GPU-resident KV cache for autoregressive generation.
/// Pre-allocates [max_seq * n_kv_heads * head_dim] per layer on GPU.
pub struct KvCache {
    pub k_gpu: Vec<GpuTensor>,   // [n_layers] each [max_seq * kv_dim]
    pub v_gpu: Vec<GpuTensor>,   // [n_layers] each [max_seq * kv_dim]
    pub kv_dim: usize,
    pub max_seq: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCache {
    pub fn new_gpu(
        gpu: &mut Gpu,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> HipResult<Self> {
        let kv_dim = n_kv_heads * head_dim;
        let cache_size = max_seq_len * kv_dim;
        let mut k_gpu = Vec::with_capacity(n_layers);
        let mut v_gpu = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_gpu.push(gpu.zeros(&[cache_size], DType::F32)?);
            v_gpu.push(gpu.zeros(&[cache_size], DType::F32)?);
        }
        Ok(Self { k_gpu, v_gpu, kv_dim, max_seq: max_seq_len, n_kv_heads, head_dim })
    }

    /// Store K, V at position `pos` in layer cache (CPU → GPU copy into cache slot).
    pub fn store_kv_pub(&mut self, gpu: &Gpu, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> HipResult<()> {
        self.store_kv(gpu, layer, pos, k, v)
    }

    fn store_kv(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        pos: usize,
        k_data: &[f32],
        v_data: &[f32],
    ) -> HipResult<()> {
        let byte_offset = pos * self.kv_dim * 4; // float = 4 bytes
        let k_bytes = unsafe {
            std::slice::from_raw_parts(k_data.as_ptr() as *const u8, k_data.len() * 4)
        };
        let v_bytes = unsafe {
            std::slice::from_raw_parts(v_data.as_ptr() as *const u8, v_data.len() * 4)
        };
        gpu.hip.memcpy_htod_offset(&self.k_gpu[layer].buf, byte_offset, k_bytes)?;
        gpu.hip.memcpy_htod_offset(&self.v_gpu[layer].buf, byte_offset, v_bytes)?;
        Ok(())
    }
}

// attention_cpu removed — GPU attention is now used

/// Sample the next token from logits using argmax (greedy).
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

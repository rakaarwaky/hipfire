//! Qwen3.5 model: hybrid DeltaNet (linear attention) + standard attention.
//! Feature-gated behind `deltanet`.

use crate::hfq::HfqFile;
use crate::llama::{self, f16_to_f32, EmbeddingFormat, WeightTensor, weight_gemv};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    LinearAttention,  // DeltaNet
    FullAttention,    // Standard MHA with gated output
}

#[derive(Debug)]
pub struct Qwen35Config {
    pub dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub eos_token: u32,

    // Full attention params
    pub n_heads: usize,        // 8
    pub n_kv_heads: usize,     // 2
    pub head_dim: usize,       // 256
    pub rope_theta: f32,
    pub partial_rotary_factor: f32, // 0.25 — only 64/256 dims get RoPE

    // DeltaNet params
    pub linear_num_key_heads: usize,   // 16
    pub linear_num_value_heads: usize, // 16
    pub linear_key_head_dim: usize,    // 128
    pub linear_value_head_dim: usize,  // 128
    pub conv_kernel_dim: usize,        // 4

    // FFN
    pub hidden_dim: usize,     // 3584

    // Per-layer type dispatch
    pub layer_types: Vec<LayerType>,
}

pub fn config_from_hfq(hfq: &HfqFile) -> Option<Qwen35Config> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    let tc = config.get("text_config").unwrap_or(config);

    let dim = tc.get("hidden_size")?.as_u64()? as usize;
    let n_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let n_heads = tc.get("num_attention_heads")?.as_u64()? as usize;
    let n_kv_heads = tc.get("num_key_value_heads").and_then(|v| v.as_u64()).unwrap_or(n_heads as u64) as usize;
    let head_dim = tc.get("head_dim").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(dim / n_heads);
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    let hidden_dim = tc.get("intermediate_size")?.as_u64()? as usize;
    let norm_eps = tc.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;

    let rope_params = tc.get("rope_parameters");
    let rope_theta = rope_params.and_then(|r| r.get("rope_theta")).and_then(|v| v.as_f64()).unwrap_or(10_000_000.0) as f32;
    let partial_rotary_factor = tc.get("partial_rotary_factor")
        .or_else(|| rope_params.and_then(|r| r.get("partial_rotary_factor")))
        .and_then(|v| v.as_f64()).unwrap_or(0.25) as f32;

    let eos_token = tc.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(248044) as u32;

    let linear_num_key_heads = tc.get("linear_num_key_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let linear_num_value_heads = tc.get("linear_num_value_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let linear_key_head_dim = tc.get("linear_key_head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
    let linear_value_head_dim = tc.get("linear_value_head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
    let conv_kernel_dim = tc.get("linear_conv_kernel_dim").and_then(|v| v.as_u64()).unwrap_or(4) as usize;

    let layer_types: Vec<LayerType> = tc.get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(|v| {
            match v.as_str().unwrap_or("full_attention") {
                "linear_attention" => LayerType::LinearAttention,
                _ => LayerType::FullAttention,
            }
        }).collect())
        .unwrap_or_else(|| vec![LayerType::FullAttention; n_layers]);

    Some(Qwen35Config {
        dim, n_layers, vocab_size, norm_eps, eos_token,
        n_heads, n_kv_heads, head_dim, rope_theta, partial_rotary_factor,
        linear_num_key_heads, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim, conv_kernel_dim,
        hidden_dim, layer_types,
    })
}

// ─── Weight structs ─────────────────────────────────────────────────────

/// Weights for a DeltaNet (linear attention) layer.
pub struct DeltaNetLayerWeights {
    pub attn_norm: GpuTensor,       // input_layernorm [dim]
    pub wqkv: WeightTensor,         // in_proj_qkv [6144, dim] → Q+K+V concat
    pub wz: WeightTensor,           // in_proj_z [2048, dim] → gate Z
    pub w_alpha: WeightTensor,      // in_proj_a [n_heads, dim] → decay
    pub w_beta: WeightTensor,       // in_proj_b [n_heads, dim] → update
    pub a_log: GpuTensor,           // A_log [n_heads] — learnable log-decay
    pub dt_bias: GpuTensor,         // dt_bias [n_heads]
    pub conv_weight: GpuTensor,     // conv1d.weight [conv_channels, 1, 4] → F32
    pub norm_weight: GpuTensor,     // norm.weight [head_dim] — gated output norm
    pub wo: WeightTensor,           // out_proj [dim, d_inner]
    pub ffn_norm: GpuTensor,        // post_attention_layernorm [dim]
    pub w_gate: WeightTensor,       // mlp.gate_proj
    pub w_up: WeightTensor,         // mlp.up_proj
    pub w_down: WeightTensor,       // mlp.down_proj
}

/// Weights for a full attention (gated) layer — similar to Qwen3 but with q+gate split.
pub struct FullAttnLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,           // q_proj [4096, dim] — 2x wide (query + gate)
    pub wk: WeightTensor,           // k_proj
    pub wv: WeightTensor,           // v_proj
    pub wo: WeightTensor,           // o_proj
    pub q_norm: GpuTensor,          // q_norm [head_dim]
    pub k_norm: GpuTensor,          // k_norm [head_dim]
    pub ffn_norm: GpuTensor,
    pub w_gate: WeightTensor,
    pub w_up: WeightTensor,
    pub w_down: WeightTensor,
}

pub enum LayerWeights {
    DeltaNet(DeltaNetLayerWeights),
    FullAttn(FullAttnLayerWeights),
}

pub struct Qwen35Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
}

// ─── State ──────────────────────────────────────────────────────────────

/// Persistent state for DeltaNet layers across tokens.
/// State quantization mode for DeltaNet S matrix.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StateQuant {
    FP32,
    Q8,
    Q4,
}

pub struct DeltaNetState {
    /// S matrix storage — FP32 or Q8 depending on quant mode
    pub s_matrices: Vec<GpuTensor>,
    /// Per-head scale factors (only used for Q8 mode)
    pub s_scales: Vec<GpuTensor>,
    /// Conv ring buffer: [n_deltanet_layers × conv_channels × (kernel_size-1)] FP32
    pub conv_states: Vec<GpuTensor>,
    /// Current quantization mode
    pub quant: StateQuant,
}

impl DeltaNetState {
    pub fn new(gpu: &mut Gpu, config: &Qwen35Config) -> HipResult<Self> {
        Self::new_with_quant(gpu, config, StateQuant::Q8)
    }

    pub fn new_with_quant(gpu: &mut Gpu, config: &Qwen35Config, quant: StateQuant) -> HipResult<Self> {
        let n_delta_layers = config.layer_types.iter().filter(|t| **t == LayerType::LinearAttention).count();
        let s_dim = config.linear_key_head_dim; // 128
        let n_heads = config.linear_num_value_heads; // 16
        let s_size = n_heads * s_dim * s_dim; // 16 * 128 * 128 = 262144

        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
                          + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = conv_channels * (config.conv_kernel_dim - 1);

        let mut s_matrices = Vec::with_capacity(n_delta_layers);
        let mut s_scales = Vec::with_capacity(n_delta_layers);
        let mut conv_states = Vec::with_capacity(n_delta_layers);
        for _ in 0..n_delta_layers {
            match quant {
                StateQuant::FP32 => {
                    s_matrices.push(gpu.zeros(&[s_size], DType::F32)?);
                    s_scales.push(gpu.zeros(&[n_heads], DType::F32)?);
                }
                StateQuant::Q8 => {
                    // int8 state: s_size bytes (1 byte each), per-row scales
                    let buf = gpu.hip.malloc(s_size)?;
                    gpu.hip.memset(&buf, 0, s_size)?;
                    s_matrices.push(GpuTensor { buf, shape: vec![s_size], dtype: DType::F32 });
                    s_scales.push(gpu.zeros(&[n_heads * s_dim], DType::F32)?);
                }
                StateQuant::Q4 => {
                    // 4-bit nibble-packed: s_size/2 bytes, per-row scales
                    let buf = gpu.hip.malloc(s_size / 2)?;
                    gpu.hip.memset(&buf, 0, s_size / 2)?;
                    s_matrices.push(GpuTensor { buf, shape: vec![s_size / 2], dtype: DType::F32 });
                    s_scales.push(gpu.zeros(&[n_heads * s_dim], DType::F32)?);
                }
            }
            conv_states.push(gpu.zeros(&[conv_state_size], DType::F32)?);
        }
        Ok(Self { s_matrices, s_scales, conv_states, quant })
    }
}

// ─── Weight loading ─────────────────────────────────────────────────────

/// Load norm weight for Qwen3.5: stored as offset from 1.0 (output = x * (1 + weight))
fn load_norm_weight(hfq: &HfqFile, gpu: &mut Gpu, name: &str, shape: &[usize]) -> HipResult<GpuTensor> {
    let full_name = format!("model.language_model.{name}");
    let (info, data) = hfq.tensor_data(&full_name)
        .or_else(|| hfq.tensor_data(name))
        .unwrap_or_else(|| panic!("tensor not found: {name} or {full_name}"));

    let mut f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        _ => panic!("expected F16/F32 for {name}, got qt={}", info.quant_type),
    };
    // Qwen3.5 RMSNorm: output = x * rsqrt(var+eps) * (1 + weight)
    for v in &mut f32_data { *v += 1.0; }
    gpu.upload_f32(&f32_data, shape)
}

fn load_weight_tensor(hfq: &HfqFile, gpu: &Gpu, name: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
    let full_name = format!("model.language_model.{name}");
    let (info, data) = hfq.tensor_data(&full_name)
        .or_else(|| hfq.tensor_data(name))
        .unwrap_or_else(|| panic!("tensor not found: {name} or {full_name}"));

    match info.quant_type {
        6 => { // HFQ4-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G256, m, k, row_stride: 0 })
        }
        7 => { // HFQ4-G128
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::HFQ4G128, m, k, row_stride: 0 })
        }
        1 => { // F16 → F32
            let f32_data: Vec<f32> = data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
        }
        _ => panic!("unsupported quant_type {} for {name}", info.quant_type),
    }
}

/// Load a tensor as F32 on GPU, handling any quant type by dequanting on CPU.
fn load_any_as_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let full_name = format!("model.language_model.{name}");
    let (info, data) = hfq.tensor_data(&full_name)
        .or_else(|| hfq.tensor_data(name))
        .unwrap_or_else(|| panic!("tensor not found: {name} or {full_name}"));

    let f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        6 | 7 => {
            // HFQ4-G256 or G128 — CPU dequant
            let group_size: usize = if info.quant_type == 6 { 256 } else { 128 };
            let bytes_per_group = 8 + group_size / 2;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale = f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
                let zero = f32::from_le_bytes([data[off+4], data[off+5], data[off+6], data[off+7]]);
                for i in 0..group_size {
                    let byte_idx = i / 2;
                    let byte_val = data[off + 8 + byte_idx];
                    let nibble = if i % 2 == 0 { byte_val & 0xF } else { byte_val >> 4 };
                    out.push(scale * nibble as f32 + zero);
                }
            }
            out
        }
        _ => panic!("unsupported quant_type {} for {name}", info.quant_type),
    };
    gpu.upload_f32(&f32_data[..n], &[n])
}

/// Alias for load_any_as_f32.
fn load_raw_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    load_any_as_f32(hfq, gpu, name, n)
}

pub fn load_weights(hfq: &HfqFile, config: &Qwen35Config, gpu: &mut Gpu) -> HipResult<Qwen35Weights> {
    eprintln!("  loading token_embd...");
    let embd_info = hfq.tensor_data("model.language_model.embed_tokens.weight")
        .expect("embed_tokens not found");
    let (token_embd, embd_fmt) = if embd_info.0.quant_type == 6 {
        eprintln!("    (HFQ4-G256 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?, EmbeddingFormat::HFQ4G256)
    } else if embd_info.0.quant_type == 7 {
        eprintln!("    (HFQ4-G128 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?, EmbeddingFormat::HFQ4G128)
    } else if embd_info.0.quant_type == 3 {
        // Q8_0: [f16 scale][32 × int8] per block — upload raw, use Q8 embedding lookup
        eprintln!("    (Q8_0 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?, EmbeddingFormat::Q8_0)
    } else {
        let f32_data: Vec<f32> = embd_info.1.chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        (gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?, EmbeddingFormat::F32)
    };

    eprintln!("  loading output_norm...");
    let output_norm = load_norm_weight(hfq, gpu, "norm.weight", &[config.dim])?;

    eprintln!("  loading output (tied embeddings)...");
    // Qwen3.5 uses tied embeddings — output = embed_tokens
    let embd_data = hfq.tensor_data("model.language_model.embed_tokens.weight").unwrap().1;
    let output = if embd_info.0.quant_type == 6 || embd_info.0.quant_type == 7 {
        let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
        let dtype = if embd_info.0.quant_type == 6 { DType::HFQ4G256 } else { DType::HFQ4G128 };
        WeightTensor { buf, gpu_dtype: dtype, m: config.vocab_size, k: config.dim, row_stride: 0 }
    } else if embd_info.0.quant_type == 3 {
        // Q8_0 embedding — also used for output GEMV
        let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
        WeightTensor { buf, gpu_dtype: DType::Q8_0, m: config.vocab_size, k: config.dim, row_stride: 0 }
    } else {
        let f32_data: Vec<f32> = embd_data.chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
        };
        let buf = gpu.upload_raw(bytes, &[config.vocab_size, config.dim])?;
        WeightTensor { buf, gpu_dtype: DType::F32, m: config.vocab_size, k: config.dim, row_stride: 0 }
    };

    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!("  loading layer {i}/{} ({:?})...", config.n_layers, config.layer_types[i]);
        let p = format!("layers.{i}");

        match config.layer_types[i] {
            LayerType::LinearAttention => {
                let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                            + config.linear_num_value_heads * config.linear_value_head_dim;
                let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;

                layers.push(LayerWeights::DeltaNet(DeltaNetLayerWeights {
                    attn_norm: load_norm_weight(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[config.dim])?,
                    wqkv: load_weight_tensor(hfq, gpu, &format!("{p}.linear_attn.in_proj_qkv.weight"), qkv_dim, config.dim)?,
                    wz: load_weight_tensor(hfq, gpu, &format!("{p}.linear_attn.in_proj_z.weight"), d_inner, config.dim)?,
                    w_alpha: load_weight_tensor(hfq, gpu, &format!("{p}.linear_attn.in_proj_a.weight"),
                        config.linear_num_value_heads, config.dim)?,
                    w_beta: load_weight_tensor(hfq, gpu, &format!("{p}.linear_attn.in_proj_b.weight"),
                        config.linear_num_value_heads, config.dim)?,
                    a_log: load_raw_f32(hfq, gpu, &format!("{p}.linear_attn.A_log"), config.linear_num_value_heads)?,
                    dt_bias: load_raw_f32(hfq, gpu, &format!("{p}.linear_attn.dt_bias"), config.linear_num_value_heads)?,
                    conv_weight: load_any_as_f32(hfq, gpu, &format!("{p}.linear_attn.conv1d.weight"),
                        qkv_dim * config.conv_kernel_dim)?,  // flatten [channels, 1, kernel] → [channels * kernel]
                    norm_weight: load_any_as_f32(hfq, gpu, &format!("{p}.linear_attn.norm.weight"), config.linear_value_head_dim)?,
                    wo: load_weight_tensor(hfq, gpu, &format!("{p}.linear_attn.out_proj.weight"), config.dim, d_inner)?,
                    ffn_norm: load_norm_weight(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"), &[config.dim])?,
                    w_gate: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.gate_proj.weight"), config.hidden_dim, config.dim)?,
                    w_up: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.up_proj.weight"), config.hidden_dim, config.dim)?,
                    w_down: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.down_proj.weight"), config.dim, config.hidden_dim)?,
                }));
            }
            LayerType::FullAttention => {
                let q_out_dim = config.n_heads * config.head_dim * 2; // 2x for query + gate
                let kv_dim = config.n_kv_heads * config.head_dim;

                layers.push(LayerWeights::FullAttn(FullAttnLayerWeights {
                    attn_norm: load_norm_weight(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[config.dim])?,
                    wq: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), q_out_dim, config.dim)?,
                    wk: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.k_proj.weight"), kv_dim, config.dim)?,
                    wv: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.v_proj.weight"), kv_dim, config.dim)?,
                    wo: load_weight_tensor(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"), config.dim, config.n_heads * config.head_dim)?,
                    q_norm: load_norm_weight(hfq, gpu, &format!("{p}.self_attn.q_norm.weight"), &[config.head_dim])?,
                    k_norm: load_norm_weight(hfq, gpu, &format!("{p}.self_attn.k_norm.weight"), &[config.head_dim])?,
                    ffn_norm: load_norm_weight(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"), &[config.dim])?,
                    w_gate: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.gate_proj.weight"), config.hidden_dim, config.dim)?,
                    w_up: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.up_proj.weight"), config.hidden_dim, config.dim)?,
                    w_down: load_weight_tensor(hfq, gpu, &format!("{p}.mlp.down_proj.weight"), config.dim, config.hidden_dim)?,
                }));
            }
        }
    }

    Ok(Qwen35Weights { token_embd, embd_format: embd_fmt, output_norm, output, layers })
}

// ─── Forward pass (decode, one token at a time) ─────────────────────────

/// Run one token through the Qwen3.5 model. Returns logits.
/// For DeltaNet layers, updates state in-place (S matrix + conv ring buffer).
/// For full attention layers, uses KV cache like standard transformer.
pub fn forward(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    token: u32,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;

    // Embedding lookup
    let x = gpu.alloc_tensor(&[dim], DType::F32)?;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x, token, dim)?,
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x, token, dim)?,
        _ => panic!("unsupported embedding format"),
    }

    let tmp = gpu.alloc_tensor(&[dim], DType::F32)?;
    let pos_buf = gpu.hip.malloc(4)?;
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes())?;

    let mut delta_layer_idx = 0usize;
    let debug_layers = std::env::var("DEBUG_LAYERS").is_ok();
    let dump_norms = std::env::var("DUMP_NORMS").is_ok();

    if debug_layers && pos == 0 {
        let hid = gpu.download_f32(&x)?;
        let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("EMB: first4=[{:.6},{:.6},{:.6},{:.6}] norm={norm:.4}", hid[0], hid[1], hid[2], hid[3]);
    }

    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // ── DeltaNet layer ──
                gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

                // QKV projection
                let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                             + config.linear_num_value_heads * config.linear_value_head_dim;
                let qkv = gpu.alloc_tensor(&[qkv_dim], DType::F32)?;
                weight_gemv(gpu, &layer.wqkv, &tmp, &qkv)?;

                // Z (gate) projection
                let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
                let z = gpu.alloc_tensor(&[d_inner], DType::F32)?;
                weight_gemv(gpu, &layer.wz, &tmp, &z)?;

                // Beta projection + sigmoid
                let n_v_heads = config.linear_num_value_heads;
                let beta_out = gpu.alloc_tensor(&[n_v_heads], DType::F32)?;
                weight_gemv(gpu, &layer.w_beta, &tmp, &beta_out)?;
                gpu.sigmoid_f32(&beta_out)?;

                // Alpha projection + gate compute (all on GPU, no CPU roundtrip)
                let alpha_out = gpu.alloc_tensor(&[n_v_heads], DType::F32)?;
                weight_gemv(gpu, &layer.w_alpha, &tmp, &alpha_out)?;
                gpu.alpha_gate_f32(&alpha_out, &layer.dt_bias, &layer.a_log, n_v_heads)?;

                // Fused conv1d + SiLU (one kernel instead of two)
                let conv_out = gpu.alloc_tensor(&[qkv_dim], DType::F32)?;
                gpu.conv1d_silu_f32(
                    &conv_out, &qkv, &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx], qkv_dim,
                )?;

                // Split conv output into Q, K, V
                let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
                let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
                let q_part = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                let k_part = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                let v_part = gpu.alloc_tensor(&[v_dim], DType::F32)?;
                gpu.hip.memcpy_dtod_at(&q_part.buf, 0, &conv_out.buf, 0, k_dim * 4)?;
                gpu.hip.memcpy_dtod_at(&k_part.buf, 0, &conv_out.buf, k_dim * 4, k_dim * 4)?;
                gpu.hip.memcpy_dtod_at(&v_part.buf, 0, &conv_out.buf, k_dim * 2 * 4, v_dim * 4)?;

                // L2 normalize Q and K
                gpu.l2_norm_f32(&q_part, config.linear_num_key_heads, config.linear_key_head_dim, config.norm_eps)?;
                gpu.l2_norm_f32(&k_part, config.linear_num_key_heads, config.linear_key_head_dim, config.norm_eps)?;

                // Scale Q by 1/sqrt(S_k) before recurrence (all on GPU)
                gpu.scale_f32(&q_part, 1.0 / (config.linear_key_head_dim as f32).sqrt())?;

                // Repeat Q/K heads if num_k_heads < num_v_heads (GQA-style)
                let (q_gdn, k_gdn) = if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    let expanded_dim = n_v_heads * config.linear_key_head_dim;
                    let q_exp = gpu.alloc_tensor(&[expanded_dim], DType::F32)?;
                    let k_exp = gpu.alloc_tensor(&[expanded_dim], DType::F32)?;
                    let hd = config.linear_key_head_dim;
                    for kh in 0..config.linear_num_key_heads {
                        for r in 0..ratio {
                            let dst = (kh * ratio + r) * hd * 4;
                            let src = kh * hd * 4;
                            gpu.hip.memcpy_dtod_at(&q_exp.buf, dst, &q_part.buf, src, hd * 4)?;
                            gpu.hip.memcpy_dtod_at(&k_exp.buf, dst, &k_part.buf, src, hd * 4)?;
                        }
                    }
                    (q_exp, k_exp)
                } else {
                    // Same number of heads — no repeat needed, reuse buffers directly
                    // (we'll skip freeing these in the cleanup below)
                    let q_ref = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                    let k_ref = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                    gpu.hip.memcpy_dtod_at(&q_ref.buf, 0, &q_part.buf, 0, k_dim * 4)?;
                    gpu.hip.memcpy_dtod_at(&k_ref.buf, 0, &k_part.buf, 0, k_dim * 4)?;
                    (q_ref, k_ref)
                };

                // Gated Delta Net recurrence
                let attn_out = gpu.alloc_tensor(&[v_dim], DType::F32)?;
                match dn_state.quant {
                    StateQuant::FP32 => gpu.gated_delta_net_f32(
                        &q_gdn, &k_gdn, &v_part, &alpha_out, &beta_out,
                        &dn_state.s_matrices[delta_layer_idx], &attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                    StateQuant::Q8 => gpu.gated_delta_net_q8(
                        &q_gdn, &k_gdn, &v_part, &alpha_out, &beta_out,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx], &attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                    StateQuant::Q4 => gpu.gated_delta_net_q4(
                        &q_gdn, &k_gdn, &v_part, &alpha_out, &beta_out,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx], &attn_out,
                        1, n_v_heads, config.linear_value_head_dim,
                    )?,
                }

                // Q-only scaling. llama.cpp also scales output by 1/sqrt(S_v)
                // in the kernel, but that makes L00 too small (0.175 vs ref 0.501).
                // Q-only gives L00 = 0.489 vs ref 0.501. Keeping Q-only for now.

                // Gated norm: rmsnorm(attn_out) * silu(z)
                let normed_out = gpu.alloc_tensor(&[v_dim], DType::F32)?;
                gpu.gated_norm_f32(&attn_out, &z, &layer.norm_weight, &normed_out,
                    n_v_heads, config.linear_value_head_dim, config.norm_eps)?;

                // Output projection
                let o = gpu.alloc_tensor(&[dim], DType::F32)?;
                weight_gemv(gpu, &layer.wo, &normed_out, &o)?;

                // Residual
                gpu.add_inplace_f32(&x, &o)?;

                // FFN
                gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;
                let gate = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                let up = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                weight_gemv(gpu, &layer.w_gate, &tmp, &gate)?;
                weight_gemv(gpu, &layer.w_up, &tmp, &up)?;
                let ffn_hidden = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                gpu.silu_mul_f32(&gate, &up, &ffn_hidden)?;
                let ffn_out = gpu.alloc_tensor(&[dim], DType::F32)?;
                weight_gemv(gpu, &layer.w_down, &ffn_hidden, &ffn_out)?;
                gpu.add_inplace_f32(&x, &ffn_out)?;

                if dump_norms {
                    let hid = gpu.download_f32(&x)?;
                    let norm: f32 = hid.iter().map(|v| v*v).sum::<f32>().sqrt();
                    eprintln!("[pos={pos}] L{layer_idx:02} DN  x_norm={norm:.4}");
                }

                // Free temporaries
                for t in [qkv, z, beta_out, alpha_out, conv_out, q_part, k_part, v_part, q_gdn, k_gdn, attn_out, normed_out, o, gate, up, ffn_hidden, ffn_out] {
                    gpu.free_tensor(t)?;
                }
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                // ── Full attention layer (gated) ──
                gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps)?;

                // Q projection (2x wide → split into query + gate)
                let q_full_dim = config.n_heads * config.head_dim * 2;
                let q_full = gpu.alloc_tensor(&[q_full_dim], DType::F32)?;
                weight_gemv(gpu, &layer.wq, &tmp, &q_full)?;

                // Split Q into query and gate — interleaved per head:
                // [Q_h0(256), Gate_h0(256), Q_h1(256), Gate_h1(256), ...]
                let q_dim = config.n_heads * config.head_dim;
                let q = gpu.alloc_tensor(&[q_dim], DType::F32)?;
                let gate_vec = gpu.alloc_tensor(&[q_dim], DType::F32)?;
                // Extract interleaved: for each head, copy head_dim floats with stride 2*head_dim
                for h in 0..config.n_heads {
                    let src_q_off = h * config.head_dim * 2 * 4;
                    let src_g_off = (h * config.head_dim * 2 + config.head_dim) * 4;
                    let dst_off = h * config.head_dim * 4;
                    gpu.hip.memcpy_dtod_at(&q.buf, dst_off, &q_full.buf, src_q_off, config.head_dim * 4)?;
                    gpu.hip.memcpy_dtod_at(&gate_vec.buf, dst_off, &q_full.buf, src_g_off, config.head_dim * 4)?;
                }

                // Q norm
                gpu.rmsnorm_batched(&q, &layer.q_norm, &q, config.n_heads, config.head_dim, config.norm_eps)?;

                // K, V projections
                let kv_dim = config.n_kv_heads * config.head_dim;
                let k = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
                let v = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
                weight_gemv(gpu, &layer.wk, &tmp, &k)?;
                weight_gemv(gpu, &layer.wv, &tmp, &v)?;

                // K norm
                gpu.rmsnorm_batched(&k, &layer.k_norm, &k, config.n_kv_heads, config.head_dim, config.norm_eps)?;

                // Partial interleaved RoPE: rotate first n_rot dims, pairs (d0,d1),(d2,d3),...
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize; // 64
                gpu.rope_partial_interleaved_f32(&q, &k, pos as i32,
                    config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;

                // KV cache write + attention
                gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k, &pos_buf, kv_dim)?;
                gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v, &pos_buf, kv_dim)?;
                let attn_out = gpu.alloc_tensor(&[q_dim], DType::F32)?;
                gpu.attention_f32(
                    &q, &kv_cache.k_gpu[layer_idx], &kv_cache.v_gpu[layer_idx],
                    &attn_out, &pos_buf, pos + 1, config.n_heads, config.n_kv_heads, config.head_dim, kv_cache.max_seq,
                )?;

                // Sigmoid gate
                gpu.sigmoid_f32(&gate_vec)?;
                // attn_out *= gate
                gpu.mul_f32(&attn_out, &gate_vec, &attn_out)?;

                // Output projection
                let o = gpu.alloc_tensor(&[dim], DType::F32)?;
                weight_gemv(gpu, &layer.wo, &attn_out, &o)?;

                // Residual
                gpu.add_inplace_f32(&x, &o)?;

                // FFN
                gpu.rmsnorm_f32(&x, &layer.ffn_norm, &tmp, config.norm_eps)?;
                let gate_ffn = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                let up = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                weight_gemv(gpu, &layer.w_gate, &tmp, &gate_ffn)?;
                weight_gemv(gpu, &layer.w_up, &tmp, &up)?;
                let ffn_hidden = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                gpu.silu_mul_f32(&gate_ffn, &up, &ffn_hidden)?;
                let ffn_out = gpu.alloc_tensor(&[dim], DType::F32)?;
                weight_gemv(gpu, &layer.w_down, &ffn_hidden, &ffn_out)?;
                gpu.add_inplace_f32(&x, &ffn_out)?;

                if dump_norms {
                    let hid = gpu.download_f32(&x)?;
                    let norm: f32 = hid.iter().map(|v| v*v).sum::<f32>().sqrt();
                    eprintln!("[pos={pos}] L{layer_idx:02} FA  x_norm={norm:.4}");
                }

                for t in [q_full, q, gate_vec, k, v, attn_out, o, gate_ffn, up, ffn_hidden, ffn_out] {
                    gpu.free_tensor(t)?;
                }
            }

            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }

        if debug_layers && pos == 0 {
            let hid = gpu.download_f32(&x)?;
            let norm: f32 = hid.iter().map(|v| v * v).sum::<f32>().sqrt();
            let lt = match config.layer_types[layer_idx] { LayerType::LinearAttention => "D", LayerType::FullAttention => "F" };
            eprintln!("L{layer_idx:02}({lt}): first4=[{:.4},{:.4},{:.4},{:.4}] norm={norm:.2}", hid[0], hid[1], hid[2], hid[3]);
        }
    }

    // Final norm + output projection
    gpu.rmsnorm_f32(&x, &weights.output_norm, &tmp, config.norm_eps)?;
    let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
    weight_gemv(gpu, &weights.output, &tmp, &logits)?;

    let logits_data = gpu.download_f32(&logits)?;

    gpu.free_tensor(x)?;
    gpu.free_tensor(tmp)?;
    gpu.free_tensor(logits)?;
    gpu.hip.free(pos_buf)?;

    Ok(logits_data)
}

/// Batched prefill: process all prompt tokens at once.
/// DeltaNet layers still sequential per token (recurrent state), but full attention
/// layers and FFN use batched GEMM + batched causal attention.
/// Returns logits for the LAST position only.
pub fn prefill_forward(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    kv_cache: &mut llama::KvCache,
    dn_state: &mut DeltaNetState,
) -> HipResult<Vec<f32>> {
    let batch = tokens.len();
    if batch <= 1 {
        // Single token: use regular forward
        return forward(gpu, weights, config, tokens[0], 0, kv_cache, dn_state);
    }
    let dim = config.dim;

    // Allocate batched hidden state: [batch × dim]
    let x_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;

    // Embedding: lookup each token
    let x_single = gpu.alloc_tensor(&[dim], DType::F32)?;
    for (i, &token) in tokens.iter().enumerate() {
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.token_embd, &x_single, token, dim)?,
            EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.token_embd, &x_single, token, dim)?,
            EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, &x_single, token, dim)?,
            EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, &x_single, token, dim)?,
            _ => panic!("unsupported embedding format"),
        }
        gpu.hip.memcpy_dtod_at(&x_batch.buf, i * dim * 4, &x_single.buf, 0, dim * 4)?;
    }
    gpu.free_tensor(x_single)?;

    let tmp_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;

    // Scratch for per-token ops
    let x_tok = gpu.alloc_tensor(&[dim], DType::F32)?;
    let tmp_tok = gpu.alloc_tensor(&[dim], DType::F32)?;
    let pos_buf = gpu.hip.malloc(4)?;

    let mut delta_layer_idx = 0usize;

    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // DeltaNet: process each token sequentially (state depends on previous)
                for t in 0..batch {
                    // Extract token t from batch
                    gpu.hip.memcpy_dtod_at(&x_tok.buf, 0, &x_batch.buf, t * dim * 4, dim * 4)?;

                    gpu.rmsnorm_f32(&x_tok, &layer.attn_norm, &tmp_tok, config.norm_eps)?;

                    let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                                 + config.linear_num_value_heads * config.linear_value_head_dim;
                    let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
                    let n_v_heads = config.linear_num_value_heads;
                    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
                    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;

                    let qkv = gpu.alloc_tensor(&[qkv_dim], DType::F32)?;
                    weight_gemv(gpu, &layer.wqkv, &tmp_tok, &qkv)?;
                    let z = gpu.alloc_tensor(&[d_inner], DType::F32)?;
                    weight_gemv(gpu, &layer.wz, &tmp_tok, &z)?;
                    let beta_out = gpu.alloc_tensor(&[n_v_heads], DType::F32)?;
                    weight_gemv(gpu, &layer.w_beta, &tmp_tok, &beta_out)?;
                    gpu.sigmoid_f32(&beta_out)?;
                    let alpha_out = gpu.alloc_tensor(&[n_v_heads], DType::F32)?;
                    weight_gemv(gpu, &layer.w_alpha, &tmp_tok, &alpha_out)?;
                    gpu.alpha_gate_f32(&alpha_out, &layer.dt_bias, &layer.a_log, n_v_heads)?;

                    let conv_out = gpu.alloc_tensor(&[qkv_dim], DType::F32)?;
                    gpu.conv1d_silu_f32(&conv_out, &qkv, &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx], qkv_dim)?;

                    let q_part = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                    let k_part = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                    let v_part = gpu.alloc_tensor(&[v_dim], DType::F32)?;
                    gpu.hip.memcpy_dtod_at(&q_part.buf, 0, &conv_out.buf, 0, k_dim * 4)?;
                    gpu.hip.memcpy_dtod_at(&k_part.buf, 0, &conv_out.buf, k_dim * 4, k_dim * 4)?;
                    gpu.hip.memcpy_dtod_at(&v_part.buf, 0, &conv_out.buf, k_dim * 2 * 4, v_dim * 4)?;

                    gpu.l2_norm_f32(&q_part, config.linear_num_key_heads, config.linear_key_head_dim, config.norm_eps)?;
                    gpu.l2_norm_f32(&k_part, config.linear_num_key_heads, config.linear_key_head_dim, config.norm_eps)?;
                    gpu.scale_f32(&q_part, 1.0 / (config.linear_key_head_dim as f32).sqrt())?;

                    // Repeat Q/K heads if needed
                    let (q_gdn, k_gdn) = if config.linear_num_key_heads < n_v_heads {
                        let ratio = n_v_heads / config.linear_num_key_heads;
                        let expanded_dim = n_v_heads * config.linear_key_head_dim;
                        let q_exp = gpu.alloc_tensor(&[expanded_dim], DType::F32)?;
                        let k_exp = gpu.alloc_tensor(&[expanded_dim], DType::F32)?;
                        let hd = config.linear_key_head_dim;
                        for kh in 0..config.linear_num_key_heads {
                            for r in 0..ratio {
                                let dst = (kh * ratio + r) * hd * 4;
                                let src = kh * hd * 4;
                                gpu.hip.memcpy_dtod_at(&q_exp.buf, dst, &q_part.buf, src, hd * 4)?;
                                gpu.hip.memcpy_dtod_at(&k_exp.buf, dst, &k_part.buf, src, hd * 4)?;
                            }
                        }
                        (q_exp, k_exp)
                    } else {
                        let q_ref = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                        let k_ref = gpu.alloc_tensor(&[k_dim], DType::F32)?;
                        gpu.hip.memcpy_dtod_at(&q_ref.buf, 0, &q_part.buf, 0, k_dim * 4)?;
                        gpu.hip.memcpy_dtod_at(&k_ref.buf, 0, &k_part.buf, 0, k_dim * 4)?;
                        (q_ref, k_ref)
                    };

                    let attn_out = gpu.alloc_tensor(&[v_dim], DType::F32)?;
                    match dn_state.quant {
                        StateQuant::FP32 => gpu.gated_delta_net_f32(
                            &q_gdn, &k_gdn, &v_part, &alpha_out, &beta_out,
                            &dn_state.s_matrices[delta_layer_idx], &attn_out,
                            1, n_v_heads, config.linear_value_head_dim)?,
                        StateQuant::Q8 => gpu.gated_delta_net_q8(
                            &q_gdn, &k_gdn, &v_part, &alpha_out, &beta_out,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx], &attn_out,
                            1, n_v_heads, config.linear_value_head_dim)?,
                        StateQuant::Q4 => gpu.gated_delta_net_q4(
                            &q_gdn, &k_gdn, &v_part, &alpha_out, &beta_out,
                            &dn_state.s_matrices[delta_layer_idx],
                            &dn_state.s_scales[delta_layer_idx], &attn_out,
                            1, n_v_heads, config.linear_value_head_dim)?,
                    }

                    let normed_out = gpu.alloc_tensor(&[v_dim], DType::F32)?;
                    gpu.gated_norm_f32(&attn_out, &z, &layer.norm_weight, &normed_out,
                        n_v_heads, config.linear_value_head_dim, config.norm_eps)?;

                    let o = gpu.alloc_tensor(&[dim], DType::F32)?;
                    weight_gemv(gpu, &layer.wo, &normed_out, &o)?;
                    gpu.add_inplace_f32(&x_tok, &o)?;

                    // FFN
                    gpu.rmsnorm_f32(&x_tok, &layer.ffn_norm, &tmp_tok, config.norm_eps)?;
                    let gate = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                    let up = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                    weight_gemv(gpu, &layer.w_gate, &tmp_tok, &gate)?;
                    weight_gemv(gpu, &layer.w_up, &tmp_tok, &up)?;
                    let ffn_hidden = gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?;
                    gpu.silu_mul_f32(&gate, &up, &ffn_hidden)?;
                    let ffn_out = gpu.alloc_tensor(&[dim], DType::F32)?;
                    weight_gemv(gpu, &layer.w_down, &ffn_hidden, &ffn_out)?;
                    gpu.add_inplace_f32(&x_tok, &ffn_out)?;

                    // Write back to batch
                    gpu.hip.memcpy_dtod_at(&x_batch.buf, t * dim * 4, &x_tok.buf, 0, dim * 4)?;

                    for tensor in [qkv, z, beta_out, alpha_out, conv_out, q_part, k_part, v_part, q_gdn, k_gdn, attn_out, normed_out, o, gate, up, ffn_hidden, ffn_out] {
                        gpu.free_tensor(tensor)?;
                    }
                }
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                // Full attention: batched processing for all tokens at once
                gpu.rmsnorm_batched(&x_batch, &layer.attn_norm, &tmp_batch, batch, dim, config.norm_eps)?;

                // Batched Q projection (2x wide)
                let q_full_dim = config.n_heads * config.head_dim * 2;
                let q_full_batch = gpu.alloc_tensor(&[batch, q_full_dim], DType::F32)?;
                llama::weight_gemm(gpu, &layer.wq, &tmp_batch, &q_full_batch, batch)?;

                // Split Q/gate for each token and head
                let q_dim = config.n_heads * config.head_dim;
                let q_batch_buf = gpu.alloc_tensor(&[batch, q_dim], DType::F32)?;
                let gate_batch_buf = gpu.alloc_tensor(&[batch, q_dim], DType::F32)?;
                for t in 0..batch {
                    for h in 0..config.n_heads {
                        let src_q_off = t * q_full_dim * 4 + h * config.head_dim * 2 * 4;
                        let src_g_off = t * q_full_dim * 4 + (h * config.head_dim * 2 + config.head_dim) * 4;
                        let dst_off = t * q_dim * 4 + h * config.head_dim * 4;
                        gpu.hip.memcpy_dtod_at(&q_batch_buf.buf, dst_off, &q_full_batch.buf, src_q_off, config.head_dim * 4)?;
                        gpu.hip.memcpy_dtod_at(&gate_batch_buf.buf, dst_off, &q_full_batch.buf, src_g_off, config.head_dim * 4)?;
                    }
                }
                gpu.free_tensor(q_full_batch)?;

                // Batched Q norm (batch * n_heads instances)
                gpu.rmsnorm_batched(&q_batch_buf, &layer.q_norm, &q_batch_buf,
                    batch * config.n_heads, config.head_dim, config.norm_eps)?;

                // Batched K, V projections
                let kv_dim = config.n_kv_heads * config.head_dim;
                let k_batch_buf = gpu.alloc_tensor(&[batch, kv_dim], DType::F32)?;
                let v_batch_buf = gpu.alloc_tensor(&[batch, kv_dim], DType::F32)?;
                llama::weight_gemm(gpu, &layer.wk, &tmp_batch, &k_batch_buf, batch)?;
                llama::weight_gemm(gpu, &layer.wv, &tmp_batch, &v_batch_buf, batch)?;

                // Batched K norm
                gpu.rmsnorm_batched(&k_batch_buf, &layer.k_norm, &k_batch_buf,
                    batch * config.n_kv_heads, config.head_dim, config.norm_eps)?;

                // Per-token RoPE + KV cache write (partial interleaved RoPE not batched yet)
                let q_tok = gpu.alloc_tensor(&[q_dim], DType::F32)?;
                let k_tok = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                for t in 0..batch {
                    gpu.hip.memcpy_dtod_at(&q_tok.buf, 0, &q_batch_buf.buf, t * q_dim * 4, q_dim * 4)?;
                    gpu.hip.memcpy_dtod_at(&k_tok.buf, 0, &k_batch_buf.buf, t * kv_dim * 4, kv_dim * 4)?;
                    gpu.rope_partial_interleaved_f32(&q_tok, &k_tok, t as i32,
                        config.n_heads, config.n_kv_heads, config.head_dim, n_rot, config.rope_theta)?;
                    gpu.hip.memcpy_dtod_at(&q_batch_buf.buf, t * q_dim * 4, &q_tok.buf, 0, q_dim * 4)?;
                    gpu.hip.memcpy_dtod_at(&k_batch_buf.buf, t * kv_dim * 4, &k_tok.buf, 0, kv_dim * 4)?;

                    let pos_i32 = t as i32;
                    gpu.hip.memcpy_htod(&pos_buf, &pos_i32.to_ne_bytes())?;
                    gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &k_tok, &pos_buf, kv_dim)?;
                    let v_tok_tmp = gpu.alloc_tensor(&[kv_dim], DType::F32)?;
                    gpu.hip.memcpy_dtod_at(&v_tok_tmp.buf, 0, &v_batch_buf.buf, t * kv_dim * 4, kv_dim * 4)?;
                    gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &v_tok_tmp, &pos_buf, kv_dim)?;
                    gpu.free_tensor(v_tok_tmp)?;
                }
                gpu.free_tensor(q_tok)?;
                gpu.free_tensor(k_tok)?;

                // Batched causal attention
                let attn_out_batch = gpu.alloc_tensor(&[batch, q_dim], DType::F32)?;
                gpu.attention_causal_batched(
                    &q_batch_buf, &k_batch_buf, &v_batch_buf, &attn_out_batch,
                    batch, config.n_heads, config.n_kv_heads, config.head_dim)?;

                // Batched sigmoid gate
                gpu.sigmoid_f32(&gate_batch_buf)?;
                gpu.mul_f32(&attn_out_batch, &gate_batch_buf, &attn_out_batch)?;

                // Batched output projection
                let o_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;
                llama::weight_gemm(gpu, &layer.wo, &attn_out_batch, &o_batch, batch)?;
                gpu.add_inplace_f32(&x_batch, &o_batch)?;

                // Batched FFN
                gpu.rmsnorm_batched(&x_batch, &layer.ffn_norm, &tmp_batch, batch, dim, config.norm_eps)?;
                let gate_ffn_batch = gpu.alloc_tensor(&[batch, config.hidden_dim], DType::F32)?;
                let up_batch = gpu.alloc_tensor(&[batch, config.hidden_dim], DType::F32)?;
                llama::weight_gemm(gpu, &layer.w_gate, &tmp_batch, &gate_ffn_batch, batch)?;
                llama::weight_gemm(gpu, &layer.w_up, &tmp_batch, &up_batch, batch)?;
                let ffn_hidden_batch = gpu.alloc_tensor(&[batch, config.hidden_dim], DType::F32)?;
                gpu.silu_mul_f32(&gate_ffn_batch, &up_batch, &ffn_hidden_batch)?;
                let ffn_out_batch = gpu.alloc_tensor(&[batch, dim], DType::F32)?;
                llama::weight_gemm(gpu, &layer.w_down, &ffn_hidden_batch, &ffn_out_batch, batch)?;
                gpu.add_inplace_f32(&x_batch, &ffn_out_batch)?;

                for t in [q_batch_buf, gate_batch_buf, k_batch_buf, v_batch_buf, attn_out_batch, o_batch, gate_ffn_batch, up_batch, ffn_hidden_batch, ffn_out_batch] {
                    gpu.free_tensor(t)?;
                }
            }

            _ => panic!("layer type mismatch"),
        }
    }

    // Final norm + output for last position only
    let last_off = (batch - 1) * dim * 4;
    let x_last = gpu.alloc_tensor(&[dim], DType::F32)?;
    gpu.hip.memcpy_dtod_at(&x_last.buf, 0, &x_batch.buf, last_off, dim * 4)?;
    let tmp_last = gpu.alloc_tensor(&[dim], DType::F32)?;
    gpu.rmsnorm_f32(&x_last, &weights.output_norm, &tmp_last, config.norm_eps)?;
    let logits = gpu.alloc_tensor(&[config.vocab_size], DType::F32)?;
    weight_gemv(gpu, &weights.output, &tmp_last, &logits)?;
    let logits_data = gpu.download_f32(&logits)?;

    gpu.free_tensor(x_batch)?;
    gpu.free_tensor(tmp_batch)?;
    gpu.free_tensor(x_tok)?;
    gpu.free_tensor(tmp_tok)?;
    gpu.free_tensor(x_last)?;
    gpu.free_tensor(tmp_last)?;
    gpu.free_tensor(logits)?;
    gpu.hip.free(pos_buf)?;

    Ok(logits_data)
}

// ─── Chunked DeltaNet (CPU reference) ────────────────────────────────────

/// CPU: sequential DeltaNet recurrence for one head.
/// S [hd×hd] row-major. Q/K/V [n_tokens×hd]. alpha/beta [n_tokens].
/// alpha = raw gate (negative); exp(alpha) = decay.
pub fn deltanet_sequential_cpu(
    q: &[f32], k: &[f32], v: &[f32],
    alpha: &[f32], beta: &[f32],
    s_in: &[f32], n_tokens: usize, hd: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut s = s_in.to_vec();
    let mut output = vec![0.0f32; n_tokens * hd];
    for t in 0..n_tokens {
        let qt = &q[t*hd..(t+1)*hd];
        let kt = &k[t*hd..(t+1)*hd];
        let vt = &v[t*hd..(t+1)*hd];
        let a = alpha[t].exp();
        let b = beta[t];
        // kv = S^T @ k
        let mut kv = vec![0.0f32; hd];
        for r in 0..hd { for c in 0..hd { kv[r] += s[c*hd+r] * kt[c]; } }
        // delta = (v - a*kv) * beta
        let mut delta = vec![0.0f32; hd];
        for i in 0..hd { delta[i] = (vt[i] - a * kv[i]) * b; }
        // S = a*S + k ⊗ delta
        for r in 0..hd { for c in 0..hd { s[r*hd+c] = a * s[r*hd+c] + kt[c] * delta[r]; } }
        // output = S @ q
        for r in 0..hd {
            let mut sum = 0.0f32;
            for c in 0..hd { sum += s[r*hd+c] * qt[c]; }
            output[t*hd+r] = sum;
        }
    }
    (output, s)
}

/// CPU: chunked DeltaNet for one head. Computes output via intra-chunk attention
/// matrix (parallel) + cross-chunk state. Mathematically identical to sequential.
pub fn deltanet_chunked_cpu(
    q: &[f32], k: &[f32], v: &[f32],
    alpha: &[f32], beta: &[f32],
    s_in: &[f32], n_tokens: usize, hd: usize,
) -> (Vec<f32>, Vec<f32>) {
    let c = n_tokens;
    // Decay mask D[i][j] = product(exp(alpha[k]) for k=j+1..=i), causal
    let mut d = vec![0.0f32; c * c];
    for i in 0..c {
        d[i*c+i] = 1.0;
        for j in (0..i).rev() { d[i*c+j] = d[i*c+(j+1)] * alpha[j+1].exp(); }
    }
    // Exactly follow HF's torch_chunk_gated_delta_rule algorithm:
    // 1. v_beta = V * beta, k_beta = K * beta
    let mut v_beta = vec![0.0f32; c * hd];
    let mut k_beta = vec![0.0f32; c * hd];
    for t in 0..c { for dd in 0..hd {
        v_beta[t*hd+dd] = v[t*hd+dd] * beta[t];
        k_beta[t*hd+dd] = k[t*hd+dd] * beta[t];
    }}

    // 2. g = cumsum(alpha) — HF does g = g.cumsum(dim=-1) BEFORE decay_mask
    let mut g_cum = vec![0.0f32; c];
    g_cum[0] = alpha[0];
    for i in 1..c { g_cum[i] = g_cum[i-1] + alpha[i]; }

    // 3. decay_mask[i][j] = exp(g[i] - g[j]) if j <= i, 0 otherwise (lower triangular)
    //    Note: HF does .tril() after exp, and the formula is (g[i] - g[j]).tril().exp().tril()
    let mut decay = vec![0.0f32; c * c];
    for i in 0..c { for j in 0..=i {
        decay[i*c+j] = (g_cum[i] - g_cum[j]).exp();
    }}

    // 4. attn = -((k_beta @ key^T) * decay_mask), zero on upper triangle AND diagonal
    let mut attn = vec![0.0f32; c * c];
    for i in 0..c { for j in 0..i { // j < i strictly (zero diagonal)
        let mut dot = 0.0f32;
        for dd in 0..hd { dot += k_beta[i*hd+dd] * k[j*hd+dd]; }
        attn[i*c+j] = -dot * decay[i*c+j];
    }}

    // 5. Forward substitution (resolve recursive S dependency)
    for i in 1..c {
        let row: Vec<f32> = (0..i).map(|j| attn[i*c+j]).collect();
        for j in 0..i {
            // correction = sum_m row[m] * attn[m][j] for m in 0..i
            let mut corr = 0.0f32;
            for m in 0..i { corr += row[m] * attn[m*c+j]; }
            attn[i*c+j] = row[j] + corr;
        }
    }

    // 6. attn += I
    for i in 0..c { attn[i*c+i] = 1.0; }

    // 7. value_corrected = attn @ v_beta
    let mut v_corr = vec![0.0f32; c * hd];
    for i in 0..c { for j in 0..=i {
        let aij = attn[i*c+j];
        for dd in 0..hd { v_corr[i*hd+dd] += aij * v_beta[j*hd+dd]; }
    }}

    // 8. k_cumdecay = attn @ (k_beta * exp(g))
    let mut kb_g = vec![0.0f32; c * hd];
    for t in 0..c { for dd in 0..hd { kb_g[t*hd+dd] = k_beta[t*hd+dd] * g_cum[t].exp(); } }
    let mut k_cumdecay = vec![0.0f32; c * hd];
    for i in 0..c { for j in 0..=i {
        let aij = attn[i*c+j];
        for dd in 0..hd { k_cumdecay[i*hd+dd] += aij * kb_g[j*hd+dd]; }
    }}

    // 9. Per-chunk processing (single chunk = entire prompt)
    let mut s_state = s_in.to_vec();
    let mut output = vec![0.0f32; c * hd];

    // QK attention: (Q @ K^T) * decay_mask, causal (upper triangle masked)
    // Note: HF masks upper triangle with 1 (not 0) using mask = triu(ones, diagonal=1)
    // and then masked_fill_(mask, 0) — so upper strictly above diagonal is 0
    let mut qk_decay = vec![0.0f32; c * c];
    for i in 0..c { for j in 0..=i {
        let mut dot = 0.0f32;
        for dd in 0..hd { dot += q[i*hd+dd] * k[j*hd+dd]; }
        qk_decay[i*c+j] = dot * decay[i*c+j];
    }}

    // v_prime = k_cumdecay @ S_state
    let mut v_prime = vec![0.0f32; c * hd];
    for t in 0..c { for r in 0..hd {
        let mut sum = 0.0f32;
        for cc in 0..hd { sum += k_cumdecay[t*hd+cc] * s_state[cc*hd+r]; }
        v_prime[t*hd+r] = sum;
    }}

    // v_new = v_corr - v_prime
    let mut v_new = vec![0.0f32; c * hd];
    for i in 0..c*hd { v_new[i] = v_corr[i] - v_prime[i]; }

    // attn_inter = (Q * exp(g)) @ S_state
    let mut o_inter = vec![0.0f32; c * hd];
    for i in 0..c {
        let decay_i = g_cum[i].exp();
        for r in 0..hd {
            let mut sum = 0.0f32;
            for cc in 0..hd { sum += q[i*hd+cc] * decay_i * s_state[cc*hd+r]; }
            o_inter[i*hd+r] = sum;
        }
    }

    // O = attn_inter + qk_decay @ v_new
    for i in 0..c*hd { output[i] = o_inter[i]; }
    for i in 0..c { for j in 0..=i {
        let w = qk_decay[i*c+j];
        for dd in 0..hd { output[i*hd+dd] += w * v_new[j*hd+dd]; }
    }}

    // State update: S = S * exp(g[-1]) + K_decayed^T @ v_new
    let total_decay = g_cum[c-1].exp();
    // HF: last_recurrent_state = last_recurrent_state * exp(g[:,:,i,-1,...])
    //   + (k * exp(g[-1] - g).T) @ v_new
    for r in 0..hd { for cc in 0..hd { s_state[r*hd+cc] *= total_decay; } }
    for t in 0..c {
        let decay_to_end = (g_cum[c-1] - g_cum[t]).exp();
        for r in 0..hd { for cc in 0..hd {
            // k[t] outer v_new[t], with k transposed: k[t][cc] * v_new[t][r]
            // HF: (k * decay).transpose(-1,-2) @ v_new → matrix [hd × hd]
            s_state[cc*hd+r] += k[t*hd+cc] * decay_to_end * v_new[t*hd+r];
        }}
    }
    (output, s_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_matches_sequential_zero_state() {
        let hd = 4;
        let n = 5;
        let q: Vec<f32> = (0..n*hd).map(|i| ((i as f32*0.37+1.1).sin()*0.5)).collect();
        let k: Vec<f32> = (0..n*hd).map(|i| ((i as f32*0.53+2.3).sin()*0.5)).collect();
        let v: Vec<f32> = (0..n*hd).map(|i| ((i as f32*0.71+0.7).sin()*0.5)).collect();
        let alpha: Vec<f32> = (0..n).map(|i| -0.5 - (i as f32 * 0.1)).collect();
        let beta: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.05)).collect();
        let s_in = vec![0.0f32; hd*hd];
        let (os, ss) = deltanet_sequential_cpu(&q, &k, &v, &alpha, &beta, &s_in, n, hd);
        let (oc, sc) = deltanet_chunked_cpu(&q, &k, &v, &alpha, &beta, &s_in, n, hd);
        // Per-token output comparison
        for t in 0..n {
            let seq_norm: f32 = os[t*hd..(t+1)*hd].iter().map(|x| x*x).sum::<f32>().sqrt();
            let ch_norm: f32 = oc[t*hd..(t+1)*hd].iter().map(|x| x*x).sum::<f32>().sqrt();
            let err: f32 = os[t*hd..(t+1)*hd].iter().zip(&oc[t*hd..(t+1)*hd]).map(|(a,b)|(a-b).abs()).fold(0.0, f32::max);
            eprintln!("  t={t}: seq_norm={seq_norm:.6} ch_norm={ch_norm:.6} max_err={err:.2e}");
        }
        let mo: f32 = os.iter().zip(&oc).map(|(a,b)|(a-b).abs()).fold(0.0, f32::max);
        let ms: f32 = ss.iter().zip(&sc).map(|(a,b)|(a-b).abs()).fold(0.0, f32::max);
        eprintln!("zero: out={mo:.2e} state={ms:.2e}");
        assert!(mo < 1e-5, "out err {mo}"); assert!(ms < 1e-5, "state err {ms}");
    }

    #[test]
    fn chunked_matches_sequential_nonzero_state() {
        let hd = 4; let n = 3;
        let q: Vec<f32> = (0..n*hd).map(|i| ((i as f32*0.41+0.3).cos()*0.3)).collect();
        let k: Vec<f32> = (0..n*hd).map(|i| ((i as f32*0.67+1.5).sin()*0.4)).collect();
        let v: Vec<f32> = (0..n*hd).map(|i| ((i as f32*0.29+2.1).cos()*0.6)).collect();
        let alpha = vec![-0.3f32, -0.7, -0.5];
        let beta = vec![0.6f32, 0.8, 0.5];
        let s_in: Vec<f32> = (0..hd*hd).map(|i| ((i as f32*0.13+0.9).sin()*0.1)).collect();
        let (os, ss) = deltanet_sequential_cpu(&q, &k, &v, &alpha, &beta, &s_in, n, hd);
        let (oc, sc) = deltanet_chunked_cpu(&q, &k, &v, &alpha, &beta, &s_in, n, hd);
        let mo: f32 = os.iter().zip(&oc).map(|(a,b)|(a-b).abs()).fold(0.0, f32::max);
        let ms: f32 = ss.iter().zip(&sc).map(|(a,b)|(a-b).abs()).fold(0.0, f32::max);
        eprintln!("nonzero: out={mo:.2e} state={ms:.2e}");
        assert!(mo < 1e-5, "out err {mo}"); assert!(ms < 1e-5, "state err {ms}");
    }

    #[test]
    fn chunked_larger_head_dim() {
        let hd = 16; let n = 8;
        let q: Vec<f32> = (0..n*hd).map(|i| ((i as f32*0.19+3.7).sin()*0.3)).collect();
        let k: Vec<f32> = (0..n*hd).map(|i| ((i as f32*0.31+1.3).cos()*0.4)).collect();
        let v: Vec<f32> = (0..n*hd).map(|i| ((i as f32*0.43+2.9).sin()*0.5)).collect();
        let alpha: Vec<f32> = (0..n).map(|i| -0.2 - (i as f32 * 0.15)).collect();
        let beta: Vec<f32> = (0..n).map(|i| 0.4 + (i as f32 * 0.07)).collect();
        let s_in: Vec<f32> = (0..hd*hd).map(|i| ((i as f32*0.07+0.5).cos()*0.05)).collect();
        let (os, ss) = deltanet_sequential_cpu(&q, &k, &v, &alpha, &beta, &s_in, n, hd);
        let (oc, sc) = deltanet_chunked_cpu(&q, &k, &v, &alpha, &beta, &s_in, n, hd);
        let mo: f32 = os.iter().zip(&oc).map(|(a,b)|(a-b).abs()).fold(0.0, f32::max);
        let ms: f32 = ss.iter().zip(&sc).map(|(a,b)|(a-b).abs()).fold(0.0, f32::max);
        eprintln!("hd16 n8: out={mo:.2e} state={ms:.2e}");
        assert!(mo < 1e-4, "out err {mo}"); assert!(ms < 1e-4, "state err {ms}");
    }
}

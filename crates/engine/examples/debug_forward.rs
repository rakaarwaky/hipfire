//! Debug the forward pass step by step to find where outputs diverge.

use engine::gguf::GgufFile;
use engine::llama::{self, LlamaConfig, KvCache};
use std::path::Path;

fn stats(name: &str, v: &[f32]) {
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
    eprintln!("  {name}: min={min:.6} max={max:.6} mean={mean:.6} std={:.6} [{}..{}]",
        var.sqrt(), format_slice(&v[..4.min(v.len())]), format_slice(&v[v.len()-2..]));
}

fn format_slice(v: &[f32]) -> String {
    v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(", ")
}

fn main() {
    let path = "/home/kaden/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    let gguf = GgufFile::open(Path::new(path)).unwrap();
    let config = LlamaConfig::from_gguf(&gguf).unwrap();

    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("Loading weights...");
    let weights = llama::load_weights(&gguf, &config, &gpu).unwrap();

    let dim = config.dim;
    let kv_dim = config.n_kv_heads * config.head_dim;

    // Step through BOS token (token=1)
    let token: u32 = 1;
    let pos: usize = 0;
    eprintln!("\n=== Forward pass debug: token={token}, pos={pos} ===\n");

    // 1. Embedding lookup
    let embd_data = gpu.download_f32(&weights.token_embd).unwrap();
    let start = (token as usize) * dim;
    let x_data = embd_data[start..start + dim].to_vec();
    stats("embedding", &x_data);

    let x = gpu.upload_f32(&x_data, &[dim]).unwrap();
    let tmp = gpu.zeros(&[dim], rdna_compute::DType::F32).unwrap();

    // 2. Layer 0: RMSNorm
    let layer = &weights.layers[0];
    gpu.rmsnorm_f32(&x, &layer.attn_norm, &tmp, config.norm_eps).unwrap();
    let norm_out = gpu.download_f32(&tmp).unwrap();
    stats("after_rmsnorm_L0", &norm_out);

    // Check norm weight
    let norm_w = gpu.download_f32(&layer.attn_norm).unwrap();
    stats("norm_weight_L0", &norm_w);

    // 3. Q projection
    let q = gpu.zeros(&[dim], rdna_compute::DType::F32).unwrap();
    gpu.gemv_f32(&layer.wq, &tmp, &q).unwrap();
    let q_out = gpu.download_f32(&q).unwrap();
    stats("q_proj_L0", &q_out);

    // 4. K projection
    let k = gpu.zeros(&[kv_dim], rdna_compute::DType::F32).unwrap();
    gpu.gemv_f32(&layer.wk, &tmp, &k).unwrap();
    let k_out = gpu.download_f32(&k).unwrap();
    stats("k_proj_L0", &k_out);

    // 5. RoPE
    let mut q_rope = q_out.clone();
    let mut k_rope = k_out.clone();
    llama::apply_rope_cpu_pub(&mut q_rope, config.n_heads, config.head_dim, pos);
    llama::apply_rope_cpu_pub(&mut k_rope, config.n_kv_heads, config.head_dim, pos);
    stats("q_rope_L0", &q_rope);
    stats("k_rope_L0", &k_rope);

    // 6. Full forward pass result
    let mut kv_cache = KvCache::new(config.n_layers, kv_dim, config.max_seq_len);
    let logits = llama::forward(&mut gpu, &weights, &config, 1, 0, &mut kv_cache).unwrap();

    let top5: Vec<(usize, f32)> = {
        let mut indexed: Vec<_> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.into_iter().take(5).collect()
    };
    eprintln!("\n  logits top5: {:?}", top5);
    stats("logits", &logits);

    gpu.free_tensor(x).unwrap();
    gpu.free_tensor(tmp).unwrap();
    gpu.free_tensor(q).unwrap();
    gpu.free_tensor(k).unwrap();
}

//! Run inference on Qwen3.5 via DeltaNet.
//! Usage: cargo run --release --features deltanet --example infer_qwen35 -- <model.hfq> [prompt]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("Build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::qwen35;
    use std::io::Write;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("Usage: infer_qwen35 <model.hfq> [--temp T] [--top-p P] [prompt]");
    let mut temperature = 0.6f32;
    let mut top_p = 0.9f32;
    let mut prompt_parts = Vec::new();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--temp" => { temperature = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(0.6); i += 2; }
            "--top-p" => { top_p = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(0.9); i += 2; }
            _ => { prompt_parts.push(args[i].clone()); i += 1; }
        }
    }
    let prompt_text = if prompt_parts.is_empty() { "Hello".to_string() } else { prompt_parts.join(" ") };
    eprintln!("Sampling: temp={temperature}, top_p={top_p}");

    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to open HFQ");
    let config = qwen35::config_from_hfq(&hfq).expect("failed to parse config");
    eprintln!("Model: {model_path}");
    eprintln!("Config: dim={}, layers={} ({} DeltaNet + {} FullAttn), heads={}, vocab={}",
        config.dim, config.n_layers,
        config.layer_types.iter().filter(|t| **t == qwen35::LayerType::LinearAttention).count(),
        config.layer_types.iter().filter(|t| **t == qwen35::LayerType::FullAttention).count(),
        config.n_heads, config.vocab_size);

    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("no tokenizer in HFQ");
    eprintln!("Tokenizer: {} tokens", tokenizer.vocab_size());

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");

    eprintln!("Loading weights...");
    let t0 = Instant::now();
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("failed to load weights");
    eprintln!("  Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // KV cache: env KV_SEQ_LEN overrides, default 2048
    let kv_seq_len: usize = std::env::var("KV_SEQ_LEN")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let mut kv_cache = engine::llama::KvCache::new_gpu(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len,
    ).unwrap();

    // DeltaNet state (check for --q8-state flag)
    let state_quant = if std::env::var("Q4_STATE").is_ok() {
        qwen35::StateQuant::Q4
    } else if std::env::var("FP32_STATE").is_ok() {
        qwen35::StateQuant::FP32
    } else {
        qwen35::StateQuant::Q8  // default: validated, 4x compression, no quality loss
    };
    let mut dn_state = qwen35::DeltaNetState::new_with_quant(&mut gpu, &config, state_quant).unwrap();
    eprintln!("DeltaNet state: {} S matrices ({:?}), {} conv states",
        dn_state.s_matrices.len(), state_quant, dn_state.conv_states.len());

    // Tokenize with ChatML (skip if NO_CHATML env var set)
    let mut prompt_tokens = tokenizer.encode(&prompt_text);
    let use_chatml = std::env::var("CHATML").is_ok();
    if use_chatml {
        // ChatML mode — for instruct-tuned models only (no forced <think>)
        let template = format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n",
            prompt_text
        );
        prompt_tokens = tokenizer.encode(&template);
    }
    eprintln!("Prompt: \"{}\" → {} tokens: {:?}", prompt_text, prompt_tokens.len(), &prompt_tokens[..prompt_tokens.len().min(10)]);

    if prompt_tokens.len() > kv_seq_len {
        eprintln!("ERROR: prompt ({} tokens) exceeds KV cache capacity ({kv_seq_len}). Set KV_SEQ_LEN=N to increase.",
            prompt_tokens.len());
        std::process::exit(1);
    }

    // Process prompt
    let t1 = Instant::now();
    let mut logits = Vec::new();
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        logits = qwen35::forward(&mut gpu, &weights, &config, token, pos, &mut kv_cache, &mut dn_state)
            .expect("forward failed");
        if pos % 5 == 0 { eprint!("."); }
    }
    let prompt_ms = t1.elapsed().as_millis();
    eprintln!("\nPrompt: {}ms ({} tokens, {:.0} tok/s)",
        prompt_ms, prompt_tokens.len(),
        prompt_tokens.len() as f64 / (prompt_ms as f64 / 1000.0));

    // Debug: check logit distribution
    let top5: Vec<(usize, f32)> = {
        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(5).collect()
    };
    eprintln!("Top 5 logits after prompt:");
    for (id, val) in &top5 {
        let text = tokenizer.decode(&[*id as u32]);
        eprintln!("  id={id} logit={val:.4} text={text:?}");
    }
    // Check specific tokens
    eprintln!("logit[248068] (<think>): {:.4}", logits.get(248068).unwrap_or(&f32::NAN));
    eprintln!("logit[248069] (</think>): {:.4}", logits.get(248069).unwrap_or(&f32::NAN));
    eprintln!("logit[248045] (<|im_start|>): {:.4}", logits.get(248045).unwrap_or(&f32::NAN));
    let has_nan = logits.iter().any(|v| v.is_nan());
    let has_inf = logits.iter().any(|v| v.is_infinite());
    eprintln!("NaN: {has_nan}, Inf: {has_inf}, min: {:.4}, max: {:.4}",
        logits.iter().cloned().fold(f32::INFINITY, f32::min),
        logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Generate — cap to stay within KV cache
    let max_gen = 128.min(kv_seq_len.saturating_sub(prompt_tokens.len() + 1));
    if max_gen == 0 {
        eprintln!("ERROR: prompt ({} tokens) exceeds KV cache capacity ({kv_seq_len}). Set KV_SEQ_LEN=N to increase.",
            prompt_tokens.len());
        std::process::exit(1);
    }
    let t2 = Instant::now();
    let mut next_token = engine::llama::sample_top_p(&logits, temperature, top_p);
    let mut generated = Vec::new();

    for gi in 0..max_gen {
        generated.push(next_token);
        let text = tokenizer.decode(&[next_token]);
        print!("{text}");
        std::io::stdout().flush().ok();
        if gi < 5 { eprintln!("[gen {gi}: id={next_token} text={text:?}]"); }

        // Stop on EOS or turn-end tokens
        if next_token == config.eos_token { break; }
        if use_chatml {
            let im_end_id = tokenizer.encode("<|im_end|>");
            if im_end_id.len() == 1 && next_token == im_end_id[0] { break; }
        }

        let pos = prompt_tokens.len() + generated.len() - 1;
        logits = qwen35::forward(&mut gpu, &weights, &config, next_token, pos, &mut kv_cache, &mut dn_state)
            .expect("forward failed");
        next_token = engine::llama::sample_top_p(&logits, temperature, top_p);
    }

    let gen_ms = t2.elapsed().as_millis();
    let tok_s = if gen_ms > 0 { generated.len() as f64 / (gen_ms as f64 / 1000.0) } else { 0.0 };
    eprintln!("\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===", generated.len(), gen_ms, tok_s);
}

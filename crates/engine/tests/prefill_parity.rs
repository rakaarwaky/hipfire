//! Integration test: sequential prefill must match batched prefill.
//! Requires deltanet feature and a model file at QWEN35_TEST_MODEL env var.
//! Run: QWEN35_TEST_MODEL=models/qwen3.5-0.8b.q4.hfq cargo test --release --features deltanet -p engine --test prefill_parity

#[cfg(feature = "deltanet")]
#[test]
fn sequential_matches_batched_prefill() {
    use engine::hfq::HfqFile;
    use engine::qwen35;

    let model_path = match std::env::var("QWEN35_TEST_MODEL") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("QWEN35_TEST_MODEL not set, skipping prefill parity test");
            return;
        }
    };

    let hfq = HfqFile::open(std::path::Path::new(&model_path)).expect("failed to open HFQ");
    let config = qwen35::config_from_hfq(&hfq).expect("failed to parse config");
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights");

    let prompt = "The quick brown fox jumps over the lazy dog and then";
    let tokens = tokenizer.encode(prompt);
    assert!(tokens.len() >= 4, "prompt too short: {} tokens", tokens.len());

    let kv_seq_len = 2048;

    // Sequential path
    let mut kv_seq = engine::llama::KvCache::new_gpu(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len,
    ).unwrap();
    let mut dn_seq = qwen35::DeltaNetState::new_with_quant(
        &mut gpu, &config, qwen35::StateQuant::FP32,
    ).unwrap();

    let mut logits_seq = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        logits_seq = qwen35::forward(&mut gpu, &weights, &config, tok, pos, &mut kv_seq, &mut dn_seq)
            .expect("sequential forward failed");
    }

    // Batched path
    let mut kv_batch = engine::llama::KvCache::new_gpu(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq_len,
    ).unwrap();
    let mut dn_batch = qwen35::DeltaNetState::new_with_quant(
        &mut gpu, &config, qwen35::StateQuant::FP32,
    ).unwrap();

    let logits_batch = qwen35::prefill_forward(
        &mut gpu, &weights, &config, &tokens, &mut kv_batch, &mut dn_batch,
    ).expect("batched prefill failed");

    // Compare
    assert_eq!(logits_seq.len(), logits_batch.len(), "logit length mismatch");

    let max_abs_err: f32 = logits_seq.iter().zip(&logits_batch)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);

    assert!(max_abs_err < 1e-4,
        "prefill parity FAILED: max logit error {max_abs_err:.2e} exceeds 1e-4");
}

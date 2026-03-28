# EAGLE-3 Speculative Decoding Plan for hipfire

## Expected Speedup: 2.7× (59.8 → ~162 tok/s on Qwen3-8B)

## Why it works on RDNA1

Verification of batch=7 draft tokens costs nearly the same as batch=1 because
the bottleneck is weight bandwidth (4.4 GB read per forward pass), not FLOPs.
Reading the same weights for 7 inputs adds minimal extra cost.

## VRAM Budget (fits in 8GB)

| Component | Size |
|-----------|------|
| Qwen3-8B Q4 weights | 4.4 GB |
| EAGLE draft head (1 transformer block) | 0.5 GB |
| KV cache (2048 ctx) | 1.8 GB |
| Activations + scratch | 0.2 GB |
| **Total** | **6.9 GB** |

## Implementation Steps

### Step 1: Batched GEMV kernel (1 day)
Copy existing GEMV_Q4K_SRC, change 2 lines:
- blockIdx.x → row = blockIdx.x % M, batch_idx = blockIdx.x / M
- Input/output pointer offsets += batch_idx * stride
Grid: [M * batch, 1, 1]. Same __launch_bounds__(32, 20). Same VGPRs.

### Step 2: forward_batched() in llama.rs (1 day)
Takes &[u32] token_ids (batch), dispatches batched GEMVs per layer.
Attention: separate forward for each batch element (they have different positions).

### Step 3: EAGLE head loading (0.5 day)
Load as LlamaWeights with n_layers=1. Share output tensor with base model.

### Step 4: Speculative loop (1 day)
```
loop {
    // Draft k=6 tokens using EAGLE head (~3.6ms)
    for i in 0..k:
        draft_tokens[i] = eagle_forward(hidden_state)

    // Verify batch of k+1 tokens through base model (~18ms)
    verify_logits = base_forward_batched(draft_tokens)

    // Accept longest valid prefix (~3.5 tokens average at 80% acceptance)
    n_accepted = verify(verify_logits, draft_tokens)
    emit tokens
}
```

### Step 5: Acquire EAGLE-3 weights
Need pretrained EAGLE head for Qwen3-8B from HuggingFace.

## Speedup Math

- Draft k=6: 6 × 0.6ms = 3.6ms
- Verify batch=7: ~18ms (same weights, ~same bandwidth)
- Cycle: 21.6ms
- Accepted tokens per cycle: ~3.5 (at 80% acceptance rate)
- Effective: 3.5 / 0.0216 = **162 tok/s**

## Alternative: Lookahead Decoding (1.7× speedup, no draft model needed)
Jacobi iteration on same model. Needs batched GEMV + multi-query attention.
Lower speedup but zero extra VRAM and no pretrained draft head.

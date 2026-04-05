#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Asymmetric attention: Q8 K cache + turbo4 V cache, head_dim=256.
// Structurally matches attention_turbo4_kv_256.hip exactly — only the K dot
// product differs (Q8 dequant instead of turbo centroid lookup, Q not rotated).
// Grid: [n_heads]. Block: [32]. Shared: scores[seq_len].
extern "C" __launch_bounds__(32, 8)
__global__ void attention_q8k_turbo4v_256(
    const float* __restrict__ q,           // [n_heads * 256]
    const unsigned char* __restrict__ k_q8, // Q8_0 format K cache
    const unsigned char* __restrict__ v_t4, // turbo4 format V cache
    float* __restrict__ out,               // [n_heads * 256]
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,      // [256]
    const float* __restrict__ signs2,      // [256]
    int seq_len_hint,
    int n_heads, int n_kv_heads, int head_dim, int max_seq,
    float scale_attn
) {
    const int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int d0 = tid * 8;

    float* scores = sdata;

    // Q8 K cache layout: [max_seq * n_kv_heads * (blocks_per_head * 34)]
    // Each Q8_0 block: [f16 scale (2B)] [32 × int8 (32B)] = 34 bytes per 32 elements
    const int q8_blocks_per_head = head_dim / 32;  // 8
    const int q8_bytes_per_head = q8_blocks_per_head * 34;  // 272
    const int q8_bytes_per_pos = n_kv_heads * q8_bytes_per_head;

    // Turbo4 V cache layout: [max_seq * n_kv_heads * (4 + head_dim/2)]
    const int t4_bytes_per_head = 4 + head_dim / 2;  // 132
    const int t4_bytes_per_pos = n_kv_heads * t4_bytes_per_head;

    // Load raw Q (NOT FWHT-rotated — K is stored as raw Q8, not turbo)
    const float* q_head = q + h * head_dim;

    // ── Phase 1: Q @ K^T with Q8 dequant ──
    // Warp-cooperative: all 32 threads on each position, shuffle-reduce.
    // Each thread handles 8 dims of the 256-dim dot product.
    for (int t = 0; t < seq_len; t++) {
        const unsigned char* kb = k_q8 + (size_t)t * q8_bytes_per_pos + kv_h * q8_bytes_per_head;

        // Each thread dequants its 8 dims from Q8 blocks and dots with Q
        float partial = 0.0f;
        // tid*8 .. tid*8+7 spans across Q8 blocks: block = d/32, offset = d%32
        for (int i = 0; i < 8; i++) {
            int d = d0 + i;
            int blk = d / 32;
            int off = d % 32;
            const unsigned char* block = kb + blk * 34;
            float bscale = (float)(*((const _Float16*)block));
            float k_val = bscale * (float)((const signed char*)(block + 2))[off];
            partial += q_head[d] * k_val;
        }

        // Warp shuffle reduce (32 → 1)
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0) scores[t] = partial * scale_attn;
    }
    __syncthreads();

    // ── Softmax (warp shuffle, matches symmetric kernel exactly) ──
    float lmx = -1e30f;
    for (int t = tid; t < seq_len; t += 32) lmx = fmaxf(lmx, scores[t]);
    for (int off = 16; off > 0; off >>= 1) lmx = fmaxf(lmx, __shfl_xor(lmx, off));
    float mx = lmx;
    float lsm = 0.0f;
    for (int t = tid; t < seq_len; t += 32) {
        float e = expf(scores[t] - mx); scores[t] = e; lsm += e;
    }
    for (int off = 16; off > 0; off >>= 1) lsm += __shfl_xor(lsm, off);
    __syncthreads();
    float inv_sum = 1.0f / lsm;
    for (int t = tid; t < seq_len; t += 32) scores[t] *= inv_sum;
    __syncthreads();

    // ── Phase 2: V weighted sum (turbo4, identical to symmetric kernel) ──
    float mv[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int t = 0; t < seq_len; t++) {
        float w = scores[t];
        if (w < 1e-8f) continue;  // sparse V skip
        const unsigned char* vb = v_t4 + (size_t)t * t4_bytes_per_pos + kv_h * t4_bytes_per_head;
        float wn = w * *(const float*)vb;
        unsigned char b0 = vb[4 + tid * 4];
        unsigned char b1 = vb[4 + tid * 4 + 1];
        unsigned char b2 = vb[4 + tid * 4 + 2];
        unsigned char b3 = vb[4 + tid * 4 + 3];
        mv[0] += wn * TURBO_C4_256[b0 & 0xF];
        mv[1] += wn * TURBO_C4_256[b0 >> 4];
        mv[2] += wn * TURBO_C4_256[b1 & 0xF];
        mv[3] += wn * TURBO_C4_256[b1 >> 4];
        mv[4] += wn * TURBO_C4_256[b2 & 0xF];
        mv[5] += wn * TURBO_C4_256[b2 >> 4];
        mv[6] += wn * TURBO_C4_256[b3 & 0xF];
        mv[7] += wn * TURBO_C4_256[b3 >> 4];
    }

    // ── Inverse FWHT-256 on V (undo turbo4 rotation) ──
    fwht_shfl_inverse_256(mv[0], mv[1], mv[2], mv[3], mv[4], mv[5], mv[6], mv[7],
                          signs1, signs2, tid);

    float* oh = out + h * head_dim;
    oh[d0] = mv[0]; oh[d0+1] = mv[1]; oh[d0+2] = mv[2]; oh[d0+3] = mv[3];
    oh[d0+4] = mv[4]; oh[d0+5] = mv[5]; oh[d0+6] = mv[6]; oh[d0+7] = mv[7];
}

// Flash attention tile kernel for givens4 KV (block-diagonal Givens rotation).
//
// Same architecture as flash_turbo4_tile but replaces FWHT with register-local
// Givens rotation on Q. K/V are read in the same packed nibble format as turbo4.
// V accumulation stays in Givens-rotated space — reduce kernel applies inverse.
//
// Key flash-compatibility property: Givens rotation errors are confined to
// 2-element blocks. Flash tiling reorder cannot propagate errors across the
// full 128-dim head — the fundamental fix over FWHT.
//
// Grid: [n_heads, n_tiles, 1]. Block: [32, 1, 1] (WAVE32).
// LDS: tile_size * 4 bytes for scores[] only — Q stays in registers.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_givens4_tile(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ pos_buf,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size
) {
    const int seq_len = pos_buf[0] + 1;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tile_id = blockIdx.y;
    const int tile_start = tile_id * tile_size;
    const int tile_end = min(tile_start + tile_size, seq_len);
    if (tile_start >= seq_len) return;
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;

    // givens4 geometry (same byte layout as turbo4)
    const int bytes_per_head = 4 + head_dim / 2;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const int kv_head_off = kv_h * bytes_per_head;

    // LDS: scores[tile_size] only — Q stays in registers
    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q and Givens-forward rotate (register-local, 4 FMAs per thread)
    const float* q_head = q + h * head_dim;
    float mq0 = q_head[tid * 4 + 0];
    float mq1 = q_head[tid * 4 + 1];
    float mq2 = q_head[tid * 4 + 2];
    float mq3 = q_head[tid * 4 + 3];
    givens_forward(mq0, mq1, mq2, mq3, cos_theta, sin_theta, tid);

    // Phase A: QK dot products (same read pattern as turbo4)
    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        const unsigned char* kb = k_cache + (size_t)t * bytes_per_pos + kv_head_off;
        float cnorm = *(const float*)kb;
        unsigned short pk = *(const unsigned short*)(kb + 4 + tid * 2);
        float partial = mq0 * (cnorm * TURBO_C4[(pk)       & 0xF])
                      + mq1 * (cnorm * TURBO_C4[(pk >> 4)  & 0xF])
                      + mq2 * (cnorm * TURBO_C4[(pk >> 8)  & 0xF])
                      + mq3 * (cnorm * TURBO_C4[(pk >> 12) & 0xF]);
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0)
            scores[t_local] = partial * scale_attn;
    }
    __syncthreads();

    // Phase B: tile max
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    // Phase C: exp + sum
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    float tile_sum = local_sum;
    __syncthreads();

    // Phase D: V accumulate (Givens-rotated space)
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const unsigned char* vb = v_cache + (size_t)t * bytes_per_pos + kv_head_off;
        float vs = *(const float*)vb;
        unsigned short pk = *(const unsigned short*)(vb + 4 + tid * 2);
        float wv = w * vs;
        out0 += wv * TURBO_C4[(pk)       & 0xF];
        out1 += wv * TURBO_C4[(pk >> 4)  & 0xF];
        out2 += wv * TURBO_C4[(pk >> 8)  & 0xF];
        out3 += wv * TURBO_C4[(pk >> 12) & 0xF];
    }

    // Write tile partials (still in Givens-rotated space)
    const int n_tiles = (seq_len + tile_size - 1) / tile_size;
    float* p = partials + (h * n_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    p[2 + tid * 4 + 0] = out0;
    p[2 + tid * 4 + 1] = out1;
    p[2 + tid * 4 + 2] = out2;
    p[2 + tid * 4 + 3] = out3;
}

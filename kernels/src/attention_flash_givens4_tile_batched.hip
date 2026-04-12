// Batched flash attention tile kernel for givens4 KV cache.
// Processes multiple batch positions in one launch via blockIdx.z.
// Each batch position has its own seq_len from positions[batch_idx] + 1.
//
// Grid: [n_heads, max_tiles, sub_batch_size]. Block: [32, 1, 1] (WAVE32).
// LDS: tile_size * 4 bytes (scores only).
// Partials: [sub_batch_size, n_heads, max_tiles, 2 + head_dim].
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_givens4_tile_batched(
    const float* __restrict__ q,             // [sub_batch × n_heads × head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,            // [sub_batch × n_heads × max_tiles × stride]
    const int* __restrict__ positions,       // [total_batch] absolute positions
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles,                           // pre-computed (max_ctx_len + tile_size - 1) / tile_size
    int batch_offset                         // offset into positions[] for this sub-batch
) {
    const int h = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int local_bid = blockIdx.z;        // index within sub-batch
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const int global_bid = batch_offset + local_bid;
    const int seq_len = positions[global_bid] + 1;
    const int tile_start = tile_id * tile_size;
    if (tile_start >= seq_len) return;
    const int tile_end = min(tile_start + tile_size, seq_len);
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;

    const int bytes_per_head = 4 + head_dim / 2;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const int kv_head_off = kv_h * bytes_per_head;

    extern __shared__ float sdata[];
    float* scores = sdata;

    // Load Q for this batch position and Givens-forward rotate
    const float* q_head = q + local_bid * q_dim + h * head_dim;
    float mq0 = q_head[tid * 4 + 0];
    float mq1 = q_head[tid * 4 + 1];
    float mq2 = q_head[tid * 4 + 2];
    float mq3 = q_head[tid * 4 + 3];
    givens_forward(mq0, mq1, mq2, mq3, cos_theta, sin_theta, tid);

    // Phase A: QK dot products
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

    // Write partials: [local_bid, h, tile_id, ...]
    const int stride = 2 + head_dim;
    const int n_tiles_this = (seq_len + tile_size - 1) / tile_size;
    float* p = partials + ((long long)local_bid * n_heads + h) * max_tiles * stride + tile_id * stride;
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    p[2 + tid * 4 + 0] = out0;
    p[2 + tid * 4 + 1] = out1;
    p[2 + tid * 4 + 2] = out2;
    p[2 + tid * 4 + 3] = out3;
}

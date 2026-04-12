// Batched flash attention reduce for givens4 KV cache.
// Processes multiple batch positions in one launch via blockIdx.y.
//
// Grid: [n_heads, sub_batch_size, 1]. Block: [32, 1, 1] (WAVE32). Zero LDS.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_givens4_reduce_batched(
    const float* __restrict__ partials,      // [sub_batch × n_heads × max_tiles × stride]
    float* __restrict__ out,                 // [sub_batch × n_heads × head_dim]
    const int* __restrict__ positions,       // [total_batch]
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_heads,
    int head_dim,
    int tile_size,
    int max_tiles,
    int batch_offset
) {
    const int h = blockIdx.x;
    const int local_bid = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int d_base = tid * 4;
    const int stride = 2 + head_dim;

    const int global_bid = batch_offset + local_bid;
    const int seq_len = positions[global_bid] + 1;
    const int n_tiles = (seq_len + tile_size - 1) / tile_size;

    const float* my_partials = partials + ((long long)local_bid * n_heads + h) * max_tiles * stride;

    // Pass 1: find global max
    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        if (p[1] > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // Pass 2: accumulate
    float global_sum = 0.0f;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;

    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * stride;
        float tile_max = p[0];
        float tile_sum = p[1];
        if (tile_sum <= 0.0f) continue;

        float corr = expf(tile_max - global_max);
        g0 += p[2 + d_base + 0] * corr;
        g1 += p[2 + d_base + 1] * corr;
        g2 += p[2 + d_base + 2] * corr;
        g3 += p[2 + d_base + 3] * corr;
        global_sum += tile_sum * corr;
    }

    // Normalize
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    g0 *= inv_sum;
    g1 *= inv_sum;
    g2 *= inv_sum;
    g3 *= inv_sum;

    // Givens inverse
    givens_inverse(g0, g1, g2, g3, cos_theta, sin_theta, tid);

    // Write output
    const int q_dim = n_heads * head_dim;
    float* my_out = out + local_bid * q_dim + h * head_dim;
    my_out[d_base + 0] = g0;
    my_out[d_base + 1] = g1;
    my_out[d_base + 2] = g2;
    my_out[d_base + 3] = g3;
}

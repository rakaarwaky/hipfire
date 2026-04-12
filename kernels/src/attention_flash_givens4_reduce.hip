// Flash attention reduce for givens4 KV — 2-pass combine + Givens inverse.
//
// Same 2-pass reduction as Q8 flash reduce, but after normalization applies
// givens_inverse to convert from Givens-rotated space back to real space.
// Register-local inverse — no wave shuffles, no LDS.
//
// Grid: [n_heads, 1, 1]. Block: [32, 1, 1] (WAVE32). Zero LDS.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_givens4_reduce(
    const float* __restrict__ partials,
    float* __restrict__ out,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_heads,
    int head_dim,
    int n_tiles
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int d_base = tid * 4;
    const int stride = 2 + head_dim;

    // Pass 1: find global max
    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = partials + (h * n_tiles + tile) * stride;
        if (p[1] > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // Pass 2: accumulate
    float global_sum = 0.0f;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;

    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = partials + (h * n_tiles + tile) * stride;
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

    // Normalize (still in Givens-rotated space)
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    g0 *= inv_sum;
    g1 *= inv_sum;
    g2 *= inv_sum;
    g3 *= inv_sum;

    // Givens inverse: rotated space -> real space (register-local, 4 FMAs)
    givens_inverse(g0, g1, g2, g3, cos_theta, sin_theta, tid);

    // Write real-space output
    out[h * head_dim + d_base + 0] = g0;
    out[h * head_dim + d_base + 1] = g1;
    out[h * head_dim + d_base + 2] = g2;
    out[h * head_dim + d_base + 3] = g3;
}

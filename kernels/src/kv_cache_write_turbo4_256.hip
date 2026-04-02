#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Fused K+V write for turbo4 at head_dim=256. 32 threads × 8 dims each.
// FWHT-256 via fwht_shfl_forward_256. bytes_per_head = 4 + 128 = 132.
extern "C" __global__ void kv_cache_write_turbo4_256(
    unsigned char* __restrict__ k_dst,
    unsigned char* __restrict__ v_dst,
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads, int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];

    const int bytes_per_head = 4 + head_dim / 2;  // 132 for hd=256
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;            // [256]
    float* scratch = smem + head_dim;  // [32]

    for (int kv = 0; kv < 2; kv++) {
        const float* src = (kv == 0) ? (k_src + h * head_dim) : (v_src + h * head_dim);
        unsigned char* out = ((kv == 0) ? k_dst : v_dst) + (size_t)pos * bytes_per_pos + h * bytes_per_head;

        // Load vector into shared memory
        for (int d = tid; d < head_dim; d += 32) x[d] = src[d];
        __syncthreads();

        // L2 norm via shared memory reduction
        float lsq = 0.0f;
        for (int d = tid; d < head_dim; d += 32) lsq += x[d] * x[d];
        scratch[tid] = lsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
        float orig_norm = sqrtf(scratch[0]);
        float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

        // Normalize
        for (int d = tid; d < head_dim; d += 32) x[d] *= inv_norm;
        __syncthreads();

        // Load 8 elements into registers for FWHT-256
        const int d0 = tid * 8;
        float v0 = x[d0], v1 = x[d0+1], v2 = x[d0+2], v3 = x[d0+3];
        float v4 = x[d0+4], v5 = x[d0+5], v6 = x[d0+6], v7 = x[d0+7];

        // FWHT-256 forward
        fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, tid);

        // 4-bit quantize + pack in two batches to reduce live VGPRs
        float local_rsq = 0.0f;
        int ia, ib;

        // Batch 1: v0-v3
        ia = turbo_quantize_4bit_256(v0); local_rsq += TURBO_C4_256[ia] * TURBO_C4_256[ia];
        ib = turbo_quantize_4bit_256(v1); local_rsq += TURBO_C4_256[ib] * TURBO_C4_256[ib];
        unsigned char p0 = (unsigned char)((ib << 4) | (ia & 0xF));
        ia = turbo_quantize_4bit_256(v2); local_rsq += TURBO_C4_256[ia] * TURBO_C4_256[ia];
        ib = turbo_quantize_4bit_256(v3); local_rsq += TURBO_C4_256[ib] * TURBO_C4_256[ib];
        unsigned char p1 = (unsigned char)((ib << 4) | (ia & 0xF));

        // Batch 2: v4-v7
        ia = turbo_quantize_4bit_256(v4); local_rsq += TURBO_C4_256[ia] * TURBO_C4_256[ia];
        ib = turbo_quantize_4bit_256(v5); local_rsq += TURBO_C4_256[ib] * TURBO_C4_256[ib];
        unsigned char p2 = (unsigned char)((ib << 4) | (ia & 0xF));
        ia = turbo_quantize_4bit_256(v6); local_rsq += TURBO_C4_256[ia] * TURBO_C4_256[ia];
        ib = turbo_quantize_4bit_256(v7); local_rsq += TURBO_C4_256[ib] * TURBO_C4_256[ib];
        unsigned char p3 = (unsigned char)((ib << 4) | (ia & 0xF));

        // Store packed nibbles (4 bytes per thread, 128 bytes total per head)
        out[4 + tid * 4]     = p0;
        out[4 + tid * 4 + 1] = p1;
        out[4 + tid * 4 + 2] = p2;
        out[4 + tid * 4 + 3] = p3;

        // Norm correction via warp reduction
        scratch[tid] = local_rsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
        float recon_norm = sqrtf(scratch[0]);
        float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
        if (tid == 0) *(float*)out = cnorm;
        __syncthreads();
    }
}

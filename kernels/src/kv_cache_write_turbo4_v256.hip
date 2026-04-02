#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Turbo4 V-only write for head_dim=256. K stays Q8 (asymmetric).
// 32 threads × 8 elements each = 256. FWHT-256 via fwht_shfl_forward_256.
// V compression is free (TurboQuant+ finding) — all quality loss is from K.
extern "C" __global__ void kv_cache_write_turbo4_v256(
    unsigned char* __restrict__ v_dst,
    const float* __restrict__ v_src,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,   // [256]
    const float* __restrict__ signs2,   // [256]
    int n_kv_heads, int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];

    // turbo4 at hd=256: 4 bytes (norm) + 128 bytes (256 nibbles) = 132 bytes/head
    const int bytes_per_head = 4 + head_dim / 2;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;           // [256]
    float* scratch = smem + head_dim;  // [32]

    const float* src = v_src + h * head_dim;
    unsigned char* out = v_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    // Load V vector into shared memory
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

    // FWHT-256 forward (register-only + warp shuffle)
    fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, tid);

    // 4-bit quantize + pack in two batches of 4 to reduce live VGPRs
    float local_rsq = 0.0f;
    int ia, ib;

    // Batch 1: v0-v3
    ia = turbo_quantize_4bit(v0); local_rsq += TURBO_C4[ia] * TURBO_C4[ia];
    ib = turbo_quantize_4bit(v1); local_rsq += TURBO_C4[ib] * TURBO_C4[ib];
    unsigned char p0 = (unsigned char)((ib << 4) | (ia & 0xF));
    ia = turbo_quantize_4bit(v2); local_rsq += TURBO_C4[ia] * TURBO_C4[ia];
    ib = turbo_quantize_4bit(v3); local_rsq += TURBO_C4[ib] * TURBO_C4[ib];
    unsigned char p1 = (unsigned char)((ib << 4) | (ia & 0xF));

    // Batch 2: v4-v7
    ia = turbo_quantize_4bit(v4); local_rsq += TURBO_C4[ia] * TURBO_C4[ia];
    ib = turbo_quantize_4bit(v5); local_rsq += TURBO_C4[ib] * TURBO_C4[ib];
    unsigned char p2 = (unsigned char)((ib << 4) | (ia & 0xF));
    ia = turbo_quantize_4bit(v6); local_rsq += TURBO_C4[ia] * TURBO_C4[ia];
    ib = turbo_quantize_4bit(v7); local_rsq += TURBO_C4[ib] * TURBO_C4[ib];
    unsigned char p3 = (unsigned char)((ib << 4) | (ia & 0xF));

    // Store packed nibbles immediately (frees p0-p3 before norm reduction)
    out[4 + tid * 4]     = p0;
    out[4 + tid * 4 + 1] = p1;
    out[4 + tid * 4 + 2] = p2;
    out[4 + tid * 4 + 3] = p3;

    // Norm correction via warp reduction (p0-p3 now dead)
    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm;
}

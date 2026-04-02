#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Fused K+V write for turbo2 at head_dim=256. 32 threads × 8 dims each.
// 2-bit: 8 dims → 2 bytes per thread. bytes_per_head = 4 + 64 = 68.
extern "C" __global__ void kv_cache_write_turbo2_256(
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

    const int bytes_per_head = 4 + head_dim / 4;  // 68 for hd=256
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;            // [256]
    float* scratch = smem + head_dim;  // [32]

    for (int kv = 0; kv < 2; kv++) {
        const float* src = (kv == 0) ? (k_src + h * head_dim) : (v_src + h * head_dim);
        unsigned char* out = ((kv == 0) ? k_dst : v_dst) + (size_t)pos * bytes_per_pos + h * bytes_per_head;

        for (int d = tid; d < head_dim; d += 32) x[d] = src[d];
        __syncthreads();

        // L2 norm
        float lsq = 0.0f;
        for (int d = tid; d < head_dim; d += 32) lsq += x[d] * x[d];
        scratch[tid] = lsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
        float orig_norm = sqrtf(scratch[0]);
        float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

        for (int d = tid; d < head_dim; d += 32) x[d] *= inv_norm;
        __syncthreads();

        // Load 8 elements for FWHT-256
        const int d0 = tid * 8;
        float v0 = x[d0], v1 = x[d0+1], v2 = x[d0+2], v3 = x[d0+3];
        float v4 = x[d0+4], v5 = x[d0+5], v6 = x[d0+6], v7 = x[d0+7];

        fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7, signs1, signs2, tid);

        // 2-bit quantize: 8 values → 2 bytes per thread (byte-aligned)
        float local_rsq = 0.0f;
        float vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
        unsigned char qs0 = 0, qs1 = 0;
        // First byte: dims 0-3
        for (int i = 0; i < 4; i++) {
            int idx = turbo_quantize_2bit_256(vals[i]);
            local_rsq += TURBO_C2_256[idx] * TURBO_C2_256[idx];
            qs0 |= ((idx & 0x3) << (i * 2));
        }
        // Second byte: dims 4-7
        for (int i = 0; i < 4; i++) {
            int idx = turbo_quantize_2bit_256(vals[4 + i]);
            local_rsq += TURBO_C2_256[idx] * TURBO_C2_256[idx];
            qs1 |= ((idx & 0x3) << (i * 2));
        }

        // Store packed 2-bit values (2 bytes per thread, byte-aligned)
        out[4 + tid * 2]     = qs0;
        out[4 + tid * 2 + 1] = qs1;

        // Norm correction
        scratch[tid] = local_rsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
        float recon_norm = sqrtf(scratch[0]);
        float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
        if (tid == 0) *(float*)out = cnorm;
        __syncthreads();
    }
}

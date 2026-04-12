// Fused K+V write for givens4: block-diagonal Givens rotation + 4-bit quantize.
// Same output format as turbo4: [4B f32 scale][head_dim/2 packed nibbles].
// Replaces FWHT with 64 independent 2x2 Givens rotations (register-local).
//
// Grid: [n_kv_heads, 1, 1]. Block: [32, 1, 1] (WAVE32).
// LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

extern "C" __global__ void kv_cache_write_givens4(
    unsigned char* __restrict__ k_dst,
    unsigned char* __restrict__ v_dst,
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    const int* __restrict__ pos_buf,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_kv_heads, int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int pos = pos_buf[0];

    const int bytes_per_head = 4 + head_dim / 2;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    for (int kv = 0; kv < 2; kv++) {
        const float* src = (kv == 0) ? (k_src + h * head_dim) : (v_src + h * head_dim);
        unsigned char* out = ((kv == 0) ? k_dst : v_dst) + (size_t)pos * bytes_per_pos + h * bytes_per_head;

        for (int d = tid; d < head_dim; d += nthreads) x[d] = src[d];
        __syncthreads();

        // Parallel L2 norm
        float lsq = 0.0f;
        for (int d = tid; d < head_dim; d += nthreads) lsq += x[d] * x[d];
        scratch[tid] = lsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
        float orig_norm = sqrtf(scratch[0]);
        float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

        for (int d = tid; d < head_dim; d += nthreads) x[d] *= inv_norm;
        __syncthreads();

        // Block-diagonal Givens rotation (register-local, no wave shuffles)
        const int d0 = tid * 4;
        float fa = x[d0], fb = x[d0+1], fc = x[d0+2], fd = x[d0+3];
        givens_forward(fa, fb, fc, fd, cos_theta, sin_theta, tid);

        // 4-bit quantize from registers (same centroids as turbo4)
        float local_rsq = 0.0f;
        int idx0, idx1, idx2, idx3;
        idx0 = turbo_quantize_4bit(fa); local_rsq += TURBO_C4[idx0] * TURBO_C4[idx0];
        idx1 = turbo_quantize_4bit(fb); local_rsq += TURBO_C4[idx1] * TURBO_C4[idx1];
        idx2 = turbo_quantize_4bit(fc); local_rsq += TURBO_C4[idx2] * TURBO_C4[idx2];
        idx3 = turbo_quantize_4bit(fd); local_rsq += TURBO_C4[idx3] * TURBO_C4[idx3];

        scratch[tid] = local_rsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
        float recon_norm = sqrtf(scratch[0]);
        float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
        if (tid == 0) *(float*)out = cnorm;
        // Pack 4 nibbles into 2 bytes (same packing as turbo4)
        out[4 + tid * 2]     = (unsigned char)((idx1 << 4) | (idx0 & 0xF));
        out[4 + tid * 2 + 1] = (unsigned char)((idx3 << 4) | (idx2 & 0xF));
        __syncthreads();
    }
}

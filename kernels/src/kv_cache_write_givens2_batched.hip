// Batched givens2 KV write — N positions in one launch.
// Format: [4B f32 scale][head_dim/4 packed bytes] = 36 bytes/head (hd=128).
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

extern "C" __global__ void kv_cache_write_givens2_batched(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,
    const int* __restrict__ positions,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_kv_heads, int head_dim, int batch_size
) {
    const int h = blockIdx.x;
    const int bid = blockIdx.y;
    if (h >= n_kv_heads || bid >= batch_size) return;
    const int tid = threadIdx.x;
    const int pos = positions[bid];

    const int kv_dim = n_kv_heads * head_dim;
    const int bytes_per_head = 4 + head_dim / 4;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    const float* head_src = src + bid * kv_dim + h * head_dim;
    for (int d = tid; d < head_dim; d += 32) x[d] = head_src[d];
    __syncthreads();

    float lsq = 0.0f;
    for (int d = tid; d < head_dim; d += 32) lsq += x[d] * x[d];
    scratch[tid] = lsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;
    for (int d = tid; d < head_dim; d += 32) x[d] *= inv_norm;
    __syncthreads();

    const int d0 = tid * 4;
    float fa = x[d0], fb = x[d0+1], fc = x[d0+2], fd = x[d0+3];
    givens_forward(fa, fb, fc, fd, cos_theta, sin_theta, tid);

    float local_rsq = 0.0f;
    int i0 = turbo_quantize_2bit(fa); local_rsq += TURBO_C2[i0] * TURBO_C2[i0];
    int i1 = turbo_quantize_2bit(fb); local_rsq += TURBO_C2[i1] * TURBO_C2[i1];
    int i2 = turbo_quantize_2bit(fc); local_rsq += TURBO_C2[i2] * TURBO_C2[i2];
    int i3 = turbo_quantize_2bit(fd); local_rsq += TURBO_C2[i3] * TURBO_C2[i3];

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;

    unsigned char* out = dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;
    if (tid == 0) *(float*)out = cnorm;
    out[4 + tid] = (unsigned char)((i0 & 0x3) | ((i1 & 0x3) << 2) | ((i2 & 0x3) << 4) | ((i3 & 0x3) << 6));
}

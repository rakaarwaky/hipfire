// Batched givens4 KV write — processes N positions in one launch.
// Each wavefront handles one KV head for one batch position.
// Same output format as kv_cache_write_givens4: [4B f32 scale][head_dim/2 nibbles].
//
// Grid: [n_kv_heads, batch_size, 1]. Block: [32, 1, 1] (WAVE32).
// LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

extern "C" __global__ void kv_cache_write_givens4_batched(
    unsigned char* __restrict__ dst,         // givens4 KV cache
    const float* __restrict__ src,           // [batch_size × kv_dim]
    const int* __restrict__ positions,       // [batch_size] absolute positions
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_kv_heads,
    int head_dim,
    int batch_size
) {
    const int h = blockIdx.x;
    const int bid = blockIdx.y;
    if (h >= n_kv_heads || bid >= batch_size) return;
    const int tid = threadIdx.x;
    const int pos = positions[bid];

    const int kv_dim = n_kv_heads * head_dim;
    const int bytes_per_head = 4 + head_dim / 2;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    // Load this head's data from batched src
    const float* head_src = src + bid * kv_dim + h * head_dim;
    for (int d = tid; d < head_dim; d += 32) x[d] = head_src[d];
    __syncthreads();

    // L2 normalize
    float lsq = 0.0f;
    for (int d = tid; d < head_dim; d += 32) lsq += x[d] * x[d];
    scratch[tid] = lsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;
    for (int d = tid; d < head_dim; d += 32) x[d] *= inv_norm;
    __syncthreads();

    // Givens forward rotation (register-local)
    const int d0 = tid * 4;
    float fa = x[d0], fb = x[d0+1], fc = x[d0+2], fd = x[d0+3];
    givens_forward(fa, fb, fc, fd, cos_theta, sin_theta, tid);

    // 4-bit quantize
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

    // Write givens4 format at the correct position
    unsigned char* out = dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;
    if (tid == 0) *(float*)out = cnorm;
    out[4 + tid * 2]     = (unsigned char)((idx1 << 4) | (idx0 & 0xF));
    out[4 + tid * 2 + 1] = (unsigned char)((idx3 << 4) | (idx2 & 0xF));
}

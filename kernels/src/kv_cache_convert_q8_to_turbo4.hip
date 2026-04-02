#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Convert Q8_0 KV cache → turbo4 format for all positions up to convert_len.
// Grid: [n_kv_heads, convert_len]. Block: [32].
// Reads Q8_0: [f16 scale][32 × int8] per block, 4 blocks per head (hd=128).
// Writes turbo4: [f32 norm_correction][64 nibble-packed bytes] per head.
extern "C" __global__ void kv_cache_convert_q8_to_turbo4(
    const unsigned char* __restrict__ q8_cache,   // source Q8_0 cache
    unsigned char* __restrict__ turbo_cache,       // destination turbo4 cache
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads, int head_dim, int convert_len
) {
    const int h = blockIdx.x;
    const int pos = blockIdx.y;
    if (h >= n_kv_heads || pos >= convert_len) return;
    const int tid = threadIdx.x;

    const int blocks_per_head = head_dim / 32;
    const int q8_bytes_per_pos = n_kv_heads * blocks_per_head * 34;
    const int turbo_bytes_per_head = 4 + head_dim / 2;  // 68 for hd=128
    const int turbo_bytes_per_pos = n_kv_heads * turbo_bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    // Step 1: Dequant Q8_0 → FP32 in shared memory
    for (int blk = 0; blk < blocks_per_head; blk++) {
        const unsigned char* q8_blk = q8_cache + (size_t)pos * q8_bytes_per_pos
            + (h * blocks_per_head + blk) * 34;
        float scale = (float)(*(const _Float16*)q8_blk);
        signed char qval = (signed char)q8_blk[2 + tid];
        x[blk * 32 + tid] = scale * (float)qval;
    }
    __syncthreads();

    // Step 2: L2 norm
    float lsq = 0.0f;
    for (int d = tid; d < head_dim; d += 32) lsq += x[d] * x[d];
    scratch[tid] = lsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

    for (int d = tid; d < head_dim; d += 32) x[d] *= inv_norm;
    __syncthreads();

    // Step 3: FWHT via warp shuffle
    const int dpt = head_dim / 32;  // 4 elements per thread
    const int d0 = tid * dpt;
    float fa = x[d0], fb = x[d0+1], fc = x[d0+2], fd = x[d0+3];
    fwht_shfl_forward(fa, fb, fc, fd, signs1, signs2, tid);

    // Step 4: 4-bit quantize + norm correction
    float local_rsq = 0.0f;
    int idx0 = turbo_quantize_4bit(fa); local_rsq += TURBO_C4[idx0] * TURBO_C4[idx0];
    int idx1 = turbo_quantize_4bit(fb); local_rsq += TURBO_C4[idx1] * TURBO_C4[idx1];
    int idx2 = turbo_quantize_4bit(fc); local_rsq += TURBO_C4[idx2] * TURBO_C4[idx2];
    int idx3 = turbo_quantize_4bit(fd); local_rsq += TURBO_C4[idx3] * TURBO_C4[idx3];

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;

    // Step 5: Write turbo4 format
    unsigned char* out = turbo_cache + (size_t)pos * turbo_bytes_per_pos + h * turbo_bytes_per_head;
    if (tid == 0) *(float*)out = cnorm;
    out[4 + tid * 2]     = (unsigned char)((idx1 << 4) | (idx0 & 0xF));
    out[4 + tid * 2 + 1] = (unsigned char)((idx3 << 4) | (idx2 & 0xF));
}

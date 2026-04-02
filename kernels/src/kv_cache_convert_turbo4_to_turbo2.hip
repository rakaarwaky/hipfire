#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Convert turbo4 KV cache → turbo2 format for all positions up to convert_len.
// Grid: [n_kv_heads, convert_len]. Block: [32].
// Turbo4: [f32 norm][64 nibble-packed bytes] per head (68 bytes, hd=128).
// Turbo2: [f32 norm][32 byte-packed bytes] per head (36 bytes, hd=128).
// Re-quantizes from 4-bit to 2-bit centroids with new norm correction.
extern "C" __global__ void kv_cache_convert_turbo4_to_turbo2(
    const unsigned char* __restrict__ t4_cache,   // source turbo4
    unsigned char* __restrict__ t2_cache,          // destination turbo2
    int n_kv_heads, int head_dim, int convert_len
) {
    const int h = blockIdx.x;
    const int pos = blockIdx.y;
    if (h >= n_kv_heads || pos >= convert_len) return;
    const int tid = threadIdx.x;

    const int t4_bytes_per_head = 4 + head_dim / 2;  // 68
    const int t4_bytes_per_pos = n_kv_heads * t4_bytes_per_head;
    const int t2_bytes_per_head = 4 + head_dim / 4;  // 36
    const int t2_bytes_per_pos = n_kv_heads * t2_bytes_per_head;

    // Read turbo4: norm + 4 nibbles per thread (4 elements)
    const unsigned char* t4 = t4_cache + (size_t)pos * t4_bytes_per_pos + h * t4_bytes_per_head;
    float t4_norm = *(const float*)t4;
    unsigned char b0 = t4[4 + tid * 2];
    unsigned char b1 = t4[4 + tid * 2 + 1];
    int idx0 = b0 & 0xF;
    int idx1 = (b0 >> 4) & 0xF;
    int idx2 = b1 & 0xF;
    int idx3 = (b1 >> 4) & 0xF;

    // Dequant to centroid values, then re-quantize to 2-bit
    float v0 = TURBO_C4[idx0], v1 = TURBO_C4[idx1], v2 = TURBO_C4[idx2], v3 = TURBO_C4[idx3];
    int q0 = turbo_quantize_2bit(v0);
    int q1 = turbo_quantize_2bit(v1);
    int q2 = turbo_quantize_2bit(v2);
    int q3 = turbo_quantize_2bit(v3);

    // Norm correction: ratio of reconstructed norms
    extern __shared__ float scratch[];
    float t4_rsq = v0*v0 + v1*v1 + v2*v2 + v3*v3;
    float t2_rsq = TURBO_C2[q0]*TURBO_C2[q0] + TURBO_C2[q1]*TURBO_C2[q1]
                 + TURBO_C2[q2]*TURBO_C2[q2] + TURBO_C2[q3]*TURBO_C2[q3];

    scratch[tid] = t4_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float old_recon = sqrtf(scratch[0]);

    scratch[tid] = t2_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float new_recon = sqrtf(scratch[0]);

    float new_norm = (new_recon > 1e-10f) ? t4_norm * old_recon / new_recon : t4_norm;

    // Pack 4 × 2-bit indices into 1 byte
    unsigned char packed = (unsigned char)((q3 << 6) | (q2 << 4) | (q1 << 2) | q0);

    unsigned char* out = t2_cache + (size_t)pos * t2_bytes_per_pos + h * t2_bytes_per_head;
    if (tid == 0) *(float*)out = new_norm;
    out[4 + tid] = packed;
}

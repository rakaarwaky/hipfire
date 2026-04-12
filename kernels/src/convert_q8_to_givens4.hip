// Bulk convert Q8_0 KV cache → givens4 format.
// Reads Q8_0 blocks (34B per 32-element block, f16 scale + int8),
// dequants, applies Givens forward rotation, quantizes to 4-bit centroids,
// writes givens4 format [4B f32 scale][head_dim/2 packed nibbles].
//
// Called once at the Q8→givens4 switch point. Converts both K and V.
// Grid: [n_kv_heads, n_positions, 1]. Block: [32, 1, 1] (WAVE32).
// LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>
#include "turbo_common.h"
#include "givens_common.h"

extern "C" __global__ void convert_q8_to_givens4(
    const unsigned char* __restrict__ k_q8,
    const unsigned char* __restrict__ v_q8,
    unsigned char* __restrict__ k_g4,
    unsigned char* __restrict__ v_g4,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_kv_heads,
    int head_dim,
    int n_positions
) {
    const int h = blockIdx.x;
    const int pos = blockIdx.y;
    if (h >= n_kv_heads || pos >= n_positions) return;
    const int tid = threadIdx.x;

    // Q8_0 geometry
    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int q8_head_block_start = h * blocks_per_head;

    // givens4 geometry
    const int g4_bytes_per_head = 4 + head_dim / 2;
    const int g4_bytes_per_pos = n_kv_heads * g4_bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    for (int kv = 0; kv < 2; kv++) {
        const unsigned char* q8_src = (kv == 0) ? k_q8 : v_q8;
        unsigned char* g4_dst = (kv == 0) ? k_g4 : v_g4;

        // ── Dequant Q8_0 → float in LDS ──
        // Each thread reads from multiple blocks to fill head_dim elements
        for (int bi = 0; bi < blocks_per_head; bi++) {
            const unsigned char* blk = q8_src + (size_t)pos * total_blocks_per_pos * 34
                                     + (q8_head_block_start + bi) * 34;
            float scale = (float)(*(const _Float16*)blk);
            int base = bi * 32 + tid;
            x[base] = scale * (float)((signed char)blk[2 + tid]);
        }
        __syncthreads();

        // ── L2 normalize ──
        float lsq = 0.0f;
        for (int d = tid; d < head_dim; d += 32) lsq += x[d] * x[d];
        scratch[tid] = lsq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
        float orig_norm = sqrtf(scratch[0]);
        float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;
        for (int d = tid; d < head_dim; d += 32) x[d] *= inv_norm;
        __syncthreads();

        // ── Givens forward rotation (register-local) ──
        const int d0 = tid * 4;
        float fa = x[d0], fb = x[d0+1], fc = x[d0+2], fd = x[d0+3];
        givens_forward(fa, fb, fc, fd, cos_theta, sin_theta, tid);

        // ── 4-bit quantize ──
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

        // ── Write givens4 format ──
        unsigned char* out = g4_dst + (size_t)pos * g4_bytes_per_pos + h * g4_bytes_per_head;
        if (tid == 0) *(float*)out = cnorm;
        out[4 + tid * 2]     = (unsigned char)((idx1 << 4) | (idx0 & 0xF));
        out[4 + tid * 2 + 1] = (unsigned char)((idx3 << 4) | (idx2 & 0xF));
        __syncthreads();
    }
}

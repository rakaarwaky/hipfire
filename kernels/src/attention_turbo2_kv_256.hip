#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Turbo2 attention for head_dim=256. 32 threads × 8 dims each.
// 2-bit: 8 dims = 2 bytes per thread. bytes_per_head = 4 + 64 = 68.
extern "C" __global__ void attention_turbo2_kv_256(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads, int n_kv_heads, int head_dim, int max_seq,
    float scale_attn
) {
    int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;

    const int bytes_per_head = 4 + head_dim / 4;  // 68
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const int kv_head_off = kv_h * bytes_per_head;
    const int d0 = tid * 8;

    float* scores = sdata;

    // Register-only FWHT-256 on Q
    float mq0 = q[h * head_dim + d0];
    float mq1 = q[h * head_dim + d0 + 1];
    float mq2 = q[h * head_dim + d0 + 2];
    float mq3 = q[h * head_dim + d0 + 3];
    float mq4 = q[h * head_dim + d0 + 4];
    float mq5 = q[h * head_dim + d0 + 5];
    float mq6 = q[h * head_dim + d0 + 6];
    float mq7 = q[h * head_dim + d0 + 7];
    fwht_shfl_forward_256(mq0, mq1, mq2, mq3, mq4, mq5, mq6, mq7, signs1, signs2, tid);
    float mq[8] = {mq0, mq1, mq2, mq3, mq4, mq5, mq6, mq7};

    // K dot: 2-bit, each thread reads 2 bytes (8 dims, 4 values per byte)
    for (int t = 0; t < seq_len; t++) {
        const unsigned char* kb = k_cache + (size_t)t * bytes_per_pos + kv_head_off;
        float cnorm = *(const float*)kb;
        unsigned char b0 = kb[4 + tid * 2];
        unsigned char b1 = kb[4 + tid * 2 + 1];
        float partial = TURBO_C2_256[(b0)      & 0x3] * mq[0]
                       + TURBO_C2_256[(b0 >> 2) & 0x3] * mq[1]
                       + TURBO_C2_256[(b0 >> 4) & 0x3] * mq[2]
                       + TURBO_C2_256[(b0 >> 6) & 0x3] * mq[3]
                       + TURBO_C2_256[(b1)      & 0x3] * mq[4]
                       + TURBO_C2_256[(b1 >> 2) & 0x3] * mq[5]
                       + TURBO_C2_256[(b1 >> 4) & 0x3] * mq[6]
                       + TURBO_C2_256[(b1 >> 6) & 0x3] * mq[7];
        partial *= cnorm;
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor(partial, off);
        if (tid == 0) scores[t] = partial * scale_attn;
    }
    __syncthreads();

    // Parallel softmax
    float lmx = -1e30f;
    for (int t = tid; t < seq_len; t += 32) lmx = fmaxf(lmx, scores[t]);
    for (int off = 16; off > 0; off >>= 1) lmx = fmaxf(lmx, __shfl_xor(lmx, off));
    float mx = lmx;
    float lsm = 0.0f;
    for (int t = tid; t < seq_len; t += 32) {
        float e = expf(scores[t] - mx); scores[t] = e; lsm += e;
    }
    for (int off = 16; off > 0; off >>= 1) lsm += __shfl_xor(lsm, off);
    __syncthreads();
    float inv_sum = 1.0f / lsm;
    for (int t = tid; t < seq_len; t += 32) scores[t] *= inv_sum;
    __syncthreads();

    // V weighted sum: each thread owns 8 dims (2 bytes)
    float mv[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int t = 0; t < seq_len; t++) {
        float w = scores[t];
        if (w < 1e-8f) continue;
        const unsigned char* vb = v_cache + (size_t)t * bytes_per_pos + kv_head_off;
        float wn = w * *(const float*)vb;
        unsigned char b0 = vb[4 + tid * 2];
        unsigned char b1 = vb[4 + tid * 2 + 1];
        mv[0] += wn * TURBO_C2_256[(b0)      & 0x3];
        mv[1] += wn * TURBO_C2_256[(b0 >> 2) & 0x3];
        mv[2] += wn * TURBO_C2_256[(b0 >> 4) & 0x3];
        mv[3] += wn * TURBO_C2_256[(b0 >> 6) & 0x3];
        mv[4] += wn * TURBO_C2_256[(b1)      & 0x3];
        mv[5] += wn * TURBO_C2_256[(b1 >> 2) & 0x3];
        mv[6] += wn * TURBO_C2_256[(b1 >> 4) & 0x3];
        mv[7] += wn * TURBO_C2_256[(b1 >> 6) & 0x3];
    }

    // Register-only inverse FWHT-256
    fwht_shfl_inverse_256(mv[0], mv[1], mv[2], mv[3], mv[4], mv[5], mv[6], mv[7], signs1, signs2, tid);

    float* oh = out + h * head_dim;
    oh[d0] = mv[0]; oh[d0+1] = mv[1]; oh[d0+2] = mv[2]; oh[d0+3] = mv[3];
    oh[d0+4] = mv[4]; oh[d0+5] = mv[5]; oh[d0+6] = mv[6]; oh[d0+7] = mv[7];
}

//! Built-in HIP kernel sources for inference operations.

/// GEMV F32: y = alpha * A * x + beta * y
/// Uses shared memory reduction across wavefronts.
pub const GEMV_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_f32(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    float alpha, float beta
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += A[row * K + k] * x[k];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = alpha * sdata[0] + beta * y[row];
    }
}

extern "C" __global__ void gemv_f16(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    float alpha, float beta
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += (float)A[row * K + k] * (float)x[k];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = alpha * sdata[0] + beta * y[row];
    }
}
"#;

/// GEMV Q4_K: matrix-vector multiply with on-the-fly Q4_K dequantization.
/// A is stored as Q4_K blocks (144 bytes per 256 elements).
/// x is F32, y is F32. y = A_dequant * x.
///
/// Q4_K block layout (144 bytes for 256 elements):
///   [0:2]   f16 d (super-block scale)
///   [2:4]   f16 dmin (super-block min)
///   [4:16]  scales[12] (packed 6-bit scales/mins for 8 sub-blocks)
///   [16:144] qs[128] (4-bit quantized values, paired sub-blocks share 32 bytes)
///
/// Data layout: 4 groups of 64 elements. Each group has 2 sub-blocks sharing 32 bytes.
///   Group g (elements g*64..g*64+63):
///     sub-block 2g:   lower nibbles of qs[g*32..g*32+32] → elements g*64+0..g*64+31
///     sub-block 2g+1: upper nibbles of qs[g*32..g*32+32] → elements g*64+32..g*64+63
pub const GEMV_Q4K_SRC: &str = r#"
#include <hip/hip_runtime.h>

__device__ __forceinline__ float half_to_float(unsigned short h) {
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp  = (h >> 10) & 0x1F;
    unsigned int frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) return __int_as_float(sign << 31);
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
        exp = 127 - 15 + 1 + exp;
        return __int_as_float((sign << 31) | (exp << 23) | (frac << 13));
    }
    if (exp == 31) return __int_as_float((sign << 31) | (0xFF << 23) | (frac ? (frac << 13 | 1) : 0));
    return __int_as_float((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13));
}

// Q4_K GEMV v6: 32 threads (single warp), 8 elements per thread per Q4K block.
// No shared memory needed — warp shuffle only. Higher occupancy.
// launch_bounds: 32 threads, target 20 blocks/CU (10 waves/SIMD) for max occupancy.
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_q4k(
    const unsigned char* __restrict__ A_q4k,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A_q4k + (size_t)row * blocks_per_row * 144;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 144;
        float d = (float)*((const _Float16*)block);
        float dmin = (float)*((const _Float16*)(block + 2));
        const unsigned char* sc = block + 4;
        const float* xb = x + bi * 256;

        // Group 0: elements 0-63
        {
            unsigned char qbyte = block[16 + tid];
            float s0 = d * (float)(sc[0] & 63);
            float m0 = dmin * (float)(sc[4] & 63);
            sum += (s0 * (float)(qbyte & 0xF) - m0) * xb[tid];

            float s1 = d * (float)(sc[1] & 63);
            float m1 = dmin * (float)(sc[5] & 63);
            sum += (s1 * (float)(qbyte >> 4) - m1) * xb[32 + tid];
        }

        // Group 1: elements 64-127
        {
            unsigned char qbyte = block[48 + tid];
            float s2 = d * (float)(sc[2] & 63);
            float m2 = dmin * (float)(sc[6] & 63);
            sum += (s2 * (float)(qbyte & 0xF) - m2) * xb[64 + tid];

            float s3 = d * (float)(sc[3] & 63);
            float m3 = dmin * (float)(sc[7] & 63);
            sum += (s3 * (float)(qbyte >> 4) - m3) * xb[96 + tid];
        }

        // Group 2: elements 128-191 (different scale encoding)
        {
            unsigned char qbyte = block[80 + tid];
            float s4 = d * (float)((sc[8] & 0xF) | ((sc[0] >> 6) << 4));
            float m4 = dmin * (float)((sc[8] >> 4) | ((sc[4] >> 6) << 4));
            sum += (s4 * (float)(qbyte & 0xF) - m4) * xb[128 + tid];

            float s5 = d * (float)((sc[9] & 0xF) | ((sc[1] >> 6) << 4));
            float m5 = dmin * (float)((sc[9] >> 4) | ((sc[5] >> 6) << 4));
            sum += (s5 * (float)(qbyte >> 4) - m5) * xb[160 + tid];
        }

        // Group 3: elements 192-255
        {
            unsigned char qbyte = block[112 + tid];
            float s6 = d * (float)((sc[10] & 0xF) | ((sc[2] >> 6) << 4));
            float m6 = dmin * (float)((sc[10] >> 4) | ((sc[6] >> 6) << 4));
            sum += (s6 * (float)(qbyte & 0xF) - m6) * xb[192 + tid];

            float s7 = d * (float)((sc[11] & 0xF) | ((sc[3] >> 6) << 4));
            float m7 = dmin * (float)((sc[11] >> 4) | ((sc[7] >> 6) << 4));
            sum += (s7 * (float)(qbyte >> 4) - m7) * xb[224 + tid];
        }
    }

    // Warp shuffle reduction (5 steps, no shared memory)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}
"#;

/// GEMV Q8_0: matrix-vector multiply with on-the-fly Q8_0 dequantization.
/// Q8_0 block: 2 bytes f16 scale + 32 bytes int8 = 34 bytes per 32 elements.
pub const GEMV_Q8_0_SRC: &str = r#"
#include <hip/hip_runtime.h>

__device__ __forceinline__ float half_to_float_q8(unsigned short h) {
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp  = (h >> 10) & 0x1F;
    unsigned int frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) return __int_as_float(sign << 31);
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
        exp = 127 - 15 + 1 + exp;
        return __int_as_float((sign << 31) | (exp << 23) | (frac << 13));
    }
    if (exp == 31) return __int_as_float((sign << 31) | (0xFF << 23) | (frac ? (frac << 13 | 1) : 0));
    return __int_as_float((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13));
}

extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    if (row >= M) return;

    const int blocks_per_row = K / 32;
    const int bytes_per_block = 34;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * bytes_per_block;

    float sum = 0.0f;

    for (int elem = threadIdx.x; elem < K; elem += blockDim.x) {
        int block_idx = elem / 32;
        int within_block = elem % 32;

        const unsigned char* block = row_data + block_idx * bytes_per_block;

        unsigned short d_bits = block[0] | (block[1] << 8);
        float d = half_to_float_q8(d_bits);

        signed char qval = (signed char)block[2 + within_block];
        float w = d * (float)qval;
        sum += w * x[elem];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = sdata[0];
    }
}
"#;

/// GEMV Q6_K: matrix-vector multiply with on-the-fly Q6_K dequantization.
/// Q6_K block: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes per 256 elements.
pub const GEMV_Q6K_SRC: &str = r#"
#include <hip/hip_runtime.h>

__device__ __forceinline__ float half_to_float_q6(unsigned short h) {
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp  = (h >> 10) & 0x1F;
    unsigned int frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) return __int_as_float(sign << 31);
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
        exp = 127 - 15 + 1 + exp;
        return __int_as_float((sign << 31) | (exp << 23) | (frac << 13));
    }
    if (exp == 31) return __int_as_float((sign << 31) | (0xFF << 23) | (frac ? (frac << 13 | 1) : 0));
    return __int_as_float((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13));
}

extern "C" __global__ void gemv_q6k(
    const unsigned char* __restrict__ A_q6k,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    if (row >= M) return;

    const int blocks_per_row = K / 256;
    const int bytes_per_block = 210;
    const unsigned char* row_data = A_q6k + (size_t)row * blocks_per_row * bytes_per_block;

    float sum = 0.0f;

    for (int elem = threadIdx.x; elem < K; elem += blockDim.x) {
        int block_idx = elem / 256;
        int within_block = elem % 256;

        const unsigned char* block = row_data + block_idx * bytes_per_block;
        const unsigned char* ql = block;           // 128 bytes
        const unsigned char* qh = block + 128;     // 64 bytes
        const signed char* sc = (const signed char*)(block + 192); // 16 bytes
        unsigned short d_bits = block[208] | (block[209] << 8);
        float d = half_to_float_q6(d_bits);

        // Which group (0 or 1) of 128 elements
        int group = within_block / 128;
        int in_group = within_block % 128;

        // Advance pointers for group
        const unsigned char* g_ql = ql + group * 64;
        const unsigned char* g_qh = qh + group * 32;
        const signed char* g_sc = sc + group * 8;

        // Which of 4 sub-positions within group
        int sub = in_group / 32;  // 0,1,2,3
        int l = in_group % 32;
        int is = l / 16;  // scale sub-index

        int q;
        switch (sub) {
            case 0: q = ((g_ql[l] & 0xF) | (((g_qh[l] >> 0) & 3) << 4)) - 32; break;
            case 1: q = ((g_ql[l + 32] & 0xF) | (((g_qh[l] >> 2) & 3) << 4)) - 32; break;
            case 2: q = ((g_ql[l] >> 4) | (((g_qh[l] >> 4) & 3) << 4)) - 32; break;
            case 3: q = ((g_ql[l + 32] >> 4) | (((g_qh[l] >> 6) & 3) << 4)) - 32; break;
        }

        float scale = (float)g_sc[is + sub * 2];
        float w = d * scale * (float)q;
        sum += w * x[elem];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = sdata[0];
    }
}
"#;

/// RMSNorm: y[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
pub const RMSNORM_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n, float eps
) {
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[blockIdx.x * n + i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int idx = blockIdx.x * n + i;
        out[idx] = x[idx] * weight[i] * rms;
    }
}
"#;

/// Element-wise add
pub const ADD_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;

/// Element-wise multiply
pub const MUL_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void mul_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}
"#;

/// SiLU (Sigmoid Linear Unit): silu(x) = x * sigmoid(x)
pub const SILU_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void silu_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}
"#;

/// Fused SiLU(gate) * up: out[i] = silu(gate[i]) * up[i]
/// Saves one kernel launch + one intermediate buffer.
pub const SILU_MUL_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void silu_mul_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        out[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}
"#;

/// Softmax over last dimension (one block per row)
pub const SOFTMAX_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void softmax_f32(
    float* __restrict__ x,
    int n
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    float* row_data = x + row * n;

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        max_val = fmaxf(max_val, row_data[i]);
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_data[i] = expf(row_data[i] - max_val);
        sum += row_data[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_data[i] /= sum;
    }
}
"#;

/// RoPE (Rotary Positional Embedding)
pub const ROPE_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void rope_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    int pos,
    int n_heads_q,
    int n_heads_k,
    int head_dim,
    float freq_base
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    if (i >= half) return;

    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float val = (float)pos * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    for (int h = 0; h < n_heads_q; h++) {
        int base = h * head_dim;
        float q0 = q[base + i];
        float q1 = q[base + i + half];
        q[base + i]        = q0 * cos_val - q1 * sin_val;
        q[base + i + half] = q0 * sin_val + q1 * cos_val;
    }

    for (int h = 0; h < n_heads_k; h++) {
        int base = h * head_dim;
        float k0 = k[base + i];
        float k1 = k[base + i + half];
        k[base + i]        = k0 * cos_val - k1 * sin_val;
        k[base + i + half] = k0 * sin_val + k1 * cos_val;
    }
}
"#;

/// Single-head causal attention on GPU.
/// One thread block per query head. Handles GQA (kv_group heads share same KV).
/// q: [n_heads * head_dim], k_cache: [seq_len * n_kv_heads * head_dim],
/// v_cache: same layout, out: [n_heads * head_dim].
pub const ATTENTION_SRC: &str = r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void attention_f32(
    const float* __restrict__ q,          // [n_heads * head_dim]
    const float* __restrict__ k_cache,    // [max_seq * n_kv_heads * head_dim]
    const float* __restrict__ v_cache,    // [max_seq * n_kv_heads * head_dim]
    float* __restrict__ out,              // [n_heads * head_dim]
    int seq_len,        // number of positions cached (including current)
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,        // max sequence length (stride for cache indexing)
    float scale         // 1/sqrt(head_dim)
) {
    extern __shared__ float sdata[];

    const int h = blockIdx.x;  // query head index
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const float* q_head = q + h * head_dim;

    // Phase 1: compute attention scores (Q @ K^T) for all positions
    // Each thread handles a subset of positions
    float* scores = sdata;  // [seq_len] — reused for softmax

    // Find max for numerical stability
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        const float* k_t = k_cache + t * n_kv_heads * head_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_t[d];
        }
        float s = dot * scale;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    // Reduce max across threads
    sdata[nthreads + tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[nthreads + tid] = fmaxf(sdata[nthreads + tid], sdata[nthreads + tid + s]);
        __syncthreads();
    }
    float max_val = sdata[nthreads];
    __syncthreads();

    // Exp and sum
    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }

    sdata[nthreads + tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[nthreads + tid] += sdata[nthreads + tid + s];
        __syncthreads();
    }
    float sum_val = sdata[nthreads];
    __syncthreads();

    // Normalize
    for (int t = tid; t < seq_len; t += nthreads) {
        scores[t] /= sum_val;
    }
    __syncthreads();

    // Phase 2: weighted sum of values
    // Each thread computes a subset of output dimensions
    float* out_head = out + h * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            val += scores[t] * v_cache[t * n_kv_heads * head_dim + kv_h * head_dim + d];
        }
        out_head[d] = val;
    }
}
"#;

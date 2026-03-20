//! Built-in HIP kernel sources for inference operations.

/// GEMV (General Matrix-Vector Multiply): y = alpha * A * x + beta * y
/// A is M x K (row-major), x is K x 1, y is M x 1.
/// Each workgroup computes one row of the output.
pub const GEMV_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Shared memory reduction across wavefronts.
// RDNA wavefront = 32 threads, block may have multiple wavefronts.
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

    // Store partial sum in shared memory and reduce
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
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
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        y[row] = alpha * sdata[0] + beta * y[row];
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
    // One block per row
    extern __shared__ float sdata[];

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[blockIdx.x * n + i];
        sum_sq += v * v;
    }

    // Reduce within block
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
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

    // Find max
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

    // Exp and sum
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

    // Normalize
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
    int pos, int dim, int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    if (i >= half) return;

    float freq = 1.0f / powf(10000.0f, (float)(2 * i) / (float)head_dim);
    float val = (float)pos * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    // Apply to all heads in q and k
    int n_heads_q = dim / head_dim;
    for (int h = 0; h < n_heads_q; h++) {
        int base = h * head_dim;
        float q0 = q[base + i];
        float q1 = q[base + i + half];
        q[base + i]        = q0 * cos_val - q1 * sin_val;
        q[base + i + half] = q0 * sin_val + q1 * cos_val;
    }

    // k might have fewer heads (GQA)
    int n_heads_k = dim / head_dim;  // caller should pass k_dim instead
    for (int h = 0; h < n_heads_k; h++) {
        int base = h * head_dim;
        float k0 = k[base + i];
        float k1 = k[base + i + half];
        k[base + i]        = k0 * cos_val - k1 * sin_val;
        k[base + i + half] = k0 * sin_val + k1 * cos_val;
    }
}
"#;

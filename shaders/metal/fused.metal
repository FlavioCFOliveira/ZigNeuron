// Metal shaders for fused operations
// Combines multiple operations into single kernels to reduce memory bandwidth
//
// Fused operations:
// - matmul_bias: C = A * B + bias
// - matmul_bias_relu: C = max(0, A * B + bias)
// - matmul_bias_sigmoid: C = sigmoid(A * B + bias)
// - matmul_bias_tanh: C = tanh(A * B + bias)
// - linear_relu: C = max(0, A + B)  // For residual connections
// - linear_relu_linear: C = Linear(ReLU(Linear(A)))

#include <metal_stdlib>
using namespace metal;

// MARK: - MatMul + Bias

// Fused matrix multiplication + bias addition
// C = A * B + bias (broadcasted across batch)
// A: [batch_size x K], B: [K x N], bias: [N], C: [batch_size x N]
kernel void matmul_bias(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.y;
    uint col = gid.x;

    if (batch < batch_size && col < N) {
        float sum = bias[col];  // Start with bias

        // Compute dot product
        for (uint k = 0; k < K; k++) {
            sum += A[batch * K + k] * B[k * N + col];
        }

        C[batch * N + col] = sum;
    }
}

// MARK: - MatMul + Bias + ReLU

// Fused: matmul + bias + ReLU
// C = max(0, A * B + bias)
kernel void matmul_bias_relu(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.y;
    uint col = gid.x;

    if (batch < batch_size && col < N) {
        float sum = bias[col];

        for (uint k = 0; k < K; k++) {
            sum += A[batch * K + k] * B[k * N + col];
        }

        C[batch * N + col] = max(0.0f, sum);
    }
}

// MARK: - MatMul + Bias + Sigmoid

// Fused: matmul + bias + sigmoid
// C = sigmoid(A * B + bias)
kernel void matmul_bias_sigmoid(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.y;
    uint col = gid.x;

    if (batch < batch_size && col < N) {
        float sum = bias[col];

        for (uint k = 0; k < K; k++) {
            sum += A[batch * K + k] * B[k * N + col];
        }

        C[batch * N + col] = 1.0f / (1.0f + exp(-sum));
    }
}

// MARK: - MatMul + Bias + Tanh

// Fused: matmul + bias + tanh
// C = tanh(A * B + bias)
kernel void matmul_bias_tanh(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.y;
    uint col = gid.x;

    if (batch < batch_size && col < N) {
        float sum = bias[col];

        for (uint k = 0; k < K; k++) {
            sum += A[batch * K + k] * B[k * N + col];
        }

        C[batch * N + col] = tanh(sum);
    }
}

// MARK: - Linear + ReLU

// Element-wise addition followed by ReLU
// Used for residual connections: C = max(0, A + B)
// A, B, C: [size] vectors
kernel void linear_relu(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint idx = gid.x;

    if (idx < size) {
        C[idx] = max(0.0f, A[idx] + B[idx]);
    }
}

// MARK: - Vectorized Fused Kernels

// MatMul + Bias + ReLU with float4 vectorization
// Processes 4 elements per thread for better throughput
kernel void matmul_bias_relu_vec4(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.y;
    uint col_base = gid.x * 4;

    if (batch < batch_size) {
        // Process 4 columns at a time
        if (col_base + 3 < N) {
            float4 sums;
            sums.x = bias[col_base];
            sums.y = bias[col_base + 1];
            sums.z = bias[col_base + 2];
            sums.w = bias[col_base + 3];

            // Compute 4 dot products simultaneously
            for (uint k = 0; k < K; k++) {
                float a_val = A[batch * K + k];
                sums.x += a_val * B[k * N + col_base];
                sums.y += a_val * B[k * N + col_base + 1];
                sums.z += a_val * B[k * N + col_base + 2];
                sums.w += a_val * B[k * N + col_base + 3];
            }

            // Apply ReLU
            sums = max(0.0f, sums);

            // Store results
            device float4* out4 = (device float4*)(C + batch * N + col_base);
            *out4 = sums;
        } else {
            // Handle remainder (edge case)
            for (uint col = col_base; col < N; col++) {
                float sum = bias[col];
                for (uint k = 0; k < K; k++) {
                    sum += A[batch * K + k] * B[k * N + col];
                }
                C[batch * N + col] = max(0.0f, sum);
            }
        }
    }
}

// MARK: - Accumulation Fused Kernels

// MatMul + Bias + ReLU with accumulation
// C = max(0, C + A * B + bias) - for recurrent layers
kernel void matmul_bias_relu_accumulate(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.y;
    uint col = gid.x;

    if (batch < batch_size && col < N) {
        float sum = C[batch * N + col] + bias[col];

        for (uint k = 0; k < K; k++) {
            sum += A[batch * K + k] * B[k * N + col];
        }

        C[batch * N + col] = max(0.0f, sum);
    }
}

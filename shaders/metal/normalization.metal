// Metal shaders for normalization layers
#include <metal_stdlib>
using namespace metal;

// LayerNorm forward: output = gamma * (input - mean) / sqrt(var + eps) + beta
kernel void layernorm_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& size [[buffer(5)]],
    uint sample_idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // For simplicity, each threadgroup handles one sample
    // But since gid is 1D, we assume sample_idx is actually gid.y in 2D
    // or we use a different dispatch. Let's assume gid.y is sample index.
}

// Simplified LayerNorm for large vectors using threadgroup reduction
kernel void layernorm_forward_optimized(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& size [[buffer(5)]],
    uint sample_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]])
{
    device const float* in_sample = input + sample_idx * size;
    device float* out_sample = output + sample_idx * size;

    // 1. Calculate mean
    float thread_sum = 0.0f;
    for (uint i = tid; i < size; i += threads_per_tg) {
        thread_sum += in_sample[i];
    }
    float mean = simd_sum(thread_sum) / size;

    // 2. Calculate variance
    float thread_var_sum = 0.0f;
    for (uint i = tid; i < size; i += threads_per_tg) {
        float diff = in_sample[i] - mean;
        thread_var_sum += diff * diff;
    }
    float variance = simd_sum(thread_var_sum) / size;
    float inv_std = rsqrt(variance + eps);

    // 3. Normalize and scale
    for (uint i = tid; i < size; i += threads_per_tg) {
        out_sample[i] = (in_sample[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// LayerNorm backward (simplified)
kernel void layernorm_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device float* grad_gamma [[buffer(4)]],
    device float* grad_beta [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint& size [[buffer(7)]],
    uint sample_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]])
{
    device const float* in_sample = input + sample_idx * size;
    device const float* go_sample = grad_output + sample_idx * size;
    device float* gi_sample = grad_input + sample_idx * size;

    // Calculate mean and variance again (or store them from forward)
    float thread_sum = 0.0f;
    for (uint i = tid; i < size; i += threads_per_tg) {
        thread_sum += in_sample[i];
    }
    float mean = simd_sum(thread_sum) / size;

    float thread_var_sum = 0.0f;
    for (uint i = tid; i < size; i += threads_per_tg) {
        float diff = in_sample[i] - mean;
        thread_var_sum += diff * diff;
    }
    float variance = simd_sum(thread_var_sum) / size;
    float inv_std = rsqrt(variance + eps);

    // Accumulate grad_gamma and grad_beta
    for (uint i = tid; i < size; i += threads_per_tg) {
        float x_hat = (in_sample[i] - mean) * inv_std;
        atomic_fetch_add_explicit((device atomic_float*)&grad_gamma[i], go_sample[i] * x_hat, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&grad_beta[i], go_sample[i], memory_order_relaxed);

        // Simple approximation for grad_input (matching layer.zig CPU logic)
        gi_sample[i] = go_sample[i] * gamma[i] * inv_std;
    }
}

// MARK: - Batch Normalization

// BatchNorm forward for training
// Computes per-channel mean and variance across batch
// Updates running_mean and running_var
// Output = gamma * (input - mean) / sqrt(var + eps) + beta
kernel void batchnorm_forward_training(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device float* running_mean [[buffer(4)]],
    device float* running_var [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant float& momentum [[buffer(7)]],
    constant uint& batch_size [[buffer(8)]],
    constant uint& size [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint c = gid.x; // Channel index

    if (c < size) {
        // 1. Compute mean across batch
        float mean = 0.0f;
        for (uint b = 0; b < batch_size; b++) {
            mean += input[b * size + c];
        }
        mean /= batch_size;

        // 2. Compute variance across batch
        float var = 0.0f;
        for (uint b = 0; b < batch_size; b++) {
            float diff = input[b * size + c] - mean;
            var += diff * diff;
        }
        var /= batch_size;

        // 3. Update running statistics
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
        running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;

        // 4. Normalize and scale
        float inv_std = rsqrt(var + eps);
        for (uint b = 0; b < batch_size; b++) {
            float x_hat = (input[b * size + c] - mean) * inv_std;
            output[b * size + c] = x_hat * gamma[c] + beta[c];
        }
    }
}

// BatchNorm forward for inference
// Uses running_mean and running_var without updating them
kernel void batchnorm_forward_inference(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device const float* running_mean [[buffer(4)]],
    device const float* running_var [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    constant uint& size [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint c = gid.x; // Channel index

    if (c < size) {
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrt(var + eps);

        for (uint b = 0; b < batch_size; b++) {
            float x_hat = (input[b * size + c] - mean) * inv_std;
            output[b * size + c] = x_hat * gamma[c] + beta[c];
        }
    }
}

// BatchNorm backward
// Computes gradients for input, gamma, and beta
kernel void batchnorm_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device float* grad_gamma [[buffer(4)]],
    device float* grad_beta [[buffer(5)]],
    device const float* running_mean [[buffer(6)]],
    device const float* running_var [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant uint& batch_size [[buffer(9)]],
    constant uint& size [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint c = gid.x; // Channel index

    if (c < size) {
        // Use running statistics (simplified - in practice we'd store forward statistics)
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrt(var + eps);

        // Compute grad_beta and grad_gamma
        float g_beta = 0.0f;
        float g_gamma = 0.0f;

        for (uint b = 0; b < batch_size; b++) {
            float x_hat = (input[b * size + c] - mean) * inv_std;
            g_beta += grad_output[b * size + c];
            g_gamma += grad_output[b * size + c] * x_hat;
        }

        atomic_fetch_add_explicit((device atomic_float*)&grad_beta[c], g_beta, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&grad_gamma[c], g_gamma, memory_order_relaxed);

        // Compute grad_input
        float coef = gamma[c] * inv_std;
        for (uint b = 0; b < batch_size; b++) {
            grad_input[b * size + c] = grad_output[b * size + c] * coef;
        }
    }
}

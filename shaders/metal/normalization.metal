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

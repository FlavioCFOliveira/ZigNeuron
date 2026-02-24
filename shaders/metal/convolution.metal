// Metal shaders for convolution layers
#include <metal_stdlib>
using namespace metal;

// 1D Convolution forward
kernel void conv1d_forward(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& in_channels [[buffer(4)]],
    constant uint& out_channels [[buffer(5)]],
    constant uint& kernel_size [[buffer(6)]],
    constant uint& in_len [[buffer(7)]],
    constant uint& out_len [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint oc = gid.y;
    uint t = gid.x;
    uint batch = gid.z;

    if (oc < out_channels && t < out_len) {
        float sum = 0.0f;
        for (uint ic = 0; ic < in_channels; ic++) {
            for (uint k = 0; k < kernel_size; k++) {
                uint in_idx = (batch * in_channels + ic) * in_len + (t + k);
                uint w_idx = (oc * in_channels + ic) * kernel_size + k;
                sum += input[in_idx] * weights[w_idx];
            }
        }
        output[(batch * out_channels + oc) * out_len + t] = sum + bias[oc];
    }
}

// 1D Convolution backward
kernel void conv1d_backward(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* grad_after_act [[buffer(2)]],
    device float* grad_input [[buffer(3)]],
    device float* grad_weights [[buffer(4)]],
    device float* grad_bias [[buffer(5)]],
    constant uint& in_channels [[buffer(6)]],
    constant uint& out_channels [[buffer(7)]],
    constant uint& kernel_size [[buffer(8)]],
    constant uint& in_len [[buffer(9)]],
    constant uint& out_len [[buffer(10)]],
    constant uint& batch_size [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint oc = gid.y;
    uint t = gid.x;
    uint batch = gid.z;

    if (oc < out_channels && t < out_len) {
        float go = grad_after_act[(batch * out_channels + oc) * out_len + t];

        // grad_bias: each thread accumulates its own go, but we need atomic for global grad_bias
        atomic_fetch_add_explicit((device atomic_float*)&grad_bias[oc], go, memory_order_relaxed);

        for (uint ic = 0; ic < in_channels; ic++) {
            for (uint k = 0; k < kernel_size; k++) {
                uint in_idx = (batch * in_channels + ic) * in_len + (t + k);
                uint w_idx = (oc * in_channels + ic) * kernel_size + k;

                // grad_weights
                atomic_fetch_add_explicit((device atomic_float*)&grad_weights[w_idx], input[in_idx] * go, memory_order_relaxed);

                // grad_input
                atomic_fetch_add_explicit((device atomic_float*)&grad_input[in_idx], weights[w_idx] * go, memory_order_relaxed);
            }
        }
    }
}

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

// MARK: - 2D Convolution

// 2D Convolution forward
// Input: [batch_size, in_channels, input_h, input_w]
// Weights: [out_channels, in_channels, kernel_h, kernel_w]
// Output: [batch_size, out_channels, output_h, output_w]
kernel void conv2d_forward(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& in_channels [[buffer(4)]],
    constant uint& out_channels [[buffer(5)]],
    constant uint& kernel_h [[buffer(6)]],
    constant uint& kernel_w [[buffer(7)]],
    constant uint& input_h [[buffer(8)]],
    constant uint& input_w [[buffer(9)]],
    constant uint& output_h [[buffer(10)]],
    constant uint& output_w [[buffer(11)]],
    constant uint& stride_h [[buffer(12)]],
    constant uint& stride_w [[buffer(13)]],
    constant uint& padding_h [[buffer(14)]],
    constant uint& padding_w [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint oc = gid.z;
    uint oh = gid.y;
    uint ow = gid.x;

    if (oc < out_channels && oh < output_h && ow < output_w) {
        float sum = 0.0f;

        // Calculate input start position accounting for stride and padding
        int in_h_start = int(oh * stride_h) - int(padding_h);
        int in_w_start = int(ow * stride_w) - int(padding_w);

        for (uint ic = 0; ic < in_channels; ic++) {
            for (uint kh = 0; kh < kernel_h; kh++) {
                for (uint kw = 0; kw < kernel_w; kw++) {
                    int in_h = in_h_start + int(kh);
                    int in_w = in_w_start + int(kw);

                    // Check bounds
                    if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                        uint in_idx = (gid.z * in_channels + ic) * input_h * input_w + uint(in_h) * input_w + uint(in_w);
                        uint w_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }

        uint out_idx = (gid.z * out_channels + oc) * output_h * output_w + oh * output_w + ow;
        output[out_idx] = sum + bias[oc];
    }
}

// 2D Convolution backward
kernel void conv2d_backward(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* grad_output [[buffer(2)]],
    device float* grad_input [[buffer(3)]],
    device float* grad_weights [[buffer(4)]],
    device float* grad_bias [[buffer(5)]],
    constant uint& in_channels [[buffer(6)]],
    constant uint& out_channels [[buffer(7)]],
    constant uint& kernel_h [[buffer(8)]],
    constant uint& kernel_w [[buffer(9)]],
    constant uint& input_h [[buffer(10)]],
    constant uint& input_w [[buffer(11)]],
    constant uint& output_h [[buffer(12)]],
    constant uint& output_w [[buffer(13)]],
    constant uint& stride_h [[buffer(14)]],
    constant uint& stride_w [[buffer(15)]],
    constant uint& padding_h [[buffer(16)]],
    constant uint& padding_w [[buffer(17)]],
    constant uint& batch_size [[buffer(18)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint oc = gid.z;
    uint oh = gid.y;
    uint ow = gid.x;

    if (oc < out_channels && oh < output_h && ow < output_w) {
        uint batch = gid.z / out_channels;
        uint out_c = oc;

        float go = grad_output[(batch * out_channels + out_c) * output_h * output_w + oh * output_w + ow];

        // Accumulate grad_bias
        atomic_fetch_add_explicit((device atomic_float*)&grad_bias[out_c], go, memory_order_relaxed);

        int in_h_start = int(oh * stride_h) - int(padding_h);
        int in_w_start = int(ow * stride_w) - int(padding_w);

        for (uint ic = 0; ic < in_channels; ic++) {
            for (uint kh = 0; kh < kernel_h; kh++) {
                for (uint kw = 0; kw < kernel_w; kw++) {
                    int in_h = in_h_start + int(kh);
                    int in_w = in_w_start + int(kw);

                    if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                        uint in_idx = (batch * in_channels + ic) * input_h * input_w + uint(in_h) * input_w + uint(in_w);
                        uint w_idx = ((out_c * in_channels + ic) * kernel_h + kh) * kernel_w + kw;

                        // grad_weights
                        atomic_fetch_add_explicit((device atomic_float*)&grad_weights[w_idx], input[in_idx] * go, memory_order_relaxed);

                        // grad_input
                        atomic_fetch_add_explicit((device atomic_float*)&grad_input[in_idx], weights[w_idx] * go, memory_order_relaxed);
                    }
                }
            }
        }
    }
}

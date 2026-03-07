// CUDA Kernels for Convolution
// 1D and 2D convolution implementations

#include "common.h"

// 1D Convolution forward
// Input: [batch_size, in_channels, in_len]
// Weights: [out_channels, in_channels, kernel_size]
// Output: [batch_size, out_channels, out_len]
extern "C" __global__ void conv1d_forward(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int in_len, int out_len
) {
    int oc = blockIdx.y;      // output channel
    int t = blockIdx.x * blockDim.x + threadIdx.x;  // time position
    int batch = blockIdx.z;

    if (oc >= out_channels || t >= out_len || batch >= batch_size) return;

    float sum = bias[oc];

    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            int in_idx = ((batch * in_channels + ic) * in_len) + (t + k);
            int w_idx = ((oc * in_channels + ic) * kernel_size) + k;
            sum += input[in_idx] * weights[w_idx];
        }
    }

    int out_idx = ((batch * out_channels + oc) * out_len) + t;
    output[out_idx] = sum;
}

// 1D Convolution backward
extern "C" __global__ void conv1d_backward(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ grad_after_act,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weights,
    float* __restrict__ grad_bias,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int in_len, int out_len
) {
    int oc = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;

    if (oc >= out_channels || t >= out_len || batch >= batch_size) return;

    int out_idx = ((batch * out_channels + oc) * out_len) + t;
    float go = grad_after_act[out_idx];

    // Accumulate grad_bias atomically
    atomicAdd(&grad_bias[oc], go);

    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            int in_idx = ((batch * in_channels + ic) * in_len) + (t + k);
            int w_idx = ((oc * in_channels + ic) * kernel_size) + k;

            // grad_weights accumulation
            atomicAdd(&grad_weights[w_idx], input[in_idx] * go);

            // grad_input accumulation
            atomicAdd(&grad_input[in_idx], weights[w_idx] * go);
        }
    }
}

// 2D Convolution forward
// Input: [batch_size, in_channels, input_h, input_w]
// Weights: [out_channels, in_channels, kernel_h, kernel_w]
// Output: [batch_size, out_channels, output_h, output_w]
extern "C" __global__ void conv2d_forward(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int input_h, int input_w,
    int output_h, int output_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % out_channels;
    int batch = blockIdx.z / out_channels;

    if (oc >= out_channels || oh >= output_h || ow >= output_w || batch >= batch_size) return;

    float sum = bias[oc];

    int in_h_start = oh * stride_h - padding_h;
    int in_w_start = ow * stride_w - padding_w;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int in_h = in_h_start + kh;
                int in_w = in_w_start + kw;

                if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                    int in_idx = ((batch * in_channels + ic) * input_h + in_h) * input_w + in_w;
                    int w_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                    sum += input[in_idx] * weights[w_idx];
                }
            }
        }
    }

    int out_idx = ((batch * out_channels + oc) * output_h + oh) * output_w + ow;
    output[out_idx] = sum;
}

// 2D Convolution backward
extern "C" __global__ void conv2d_backward(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weights,
    float* __restrict__ grad_bias,
    int batch_size, int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int input_h, int input_w,
    int output_h, int output_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % out_channels;
    int batch = blockIdx.z / out_channels;

    if (oc >= out_channels || oh >= output_h || ow >= output_w || batch >= batch_size) return;

    int out_idx = ((batch * out_channels + oc) * output_h + oh) * output_w + ow;
    float go = grad_output[out_idx];

    // Accumulate grad_bias
    atomicAdd(&grad_bias[oc], go);

    int in_h_start = oh * stride_h - padding_h;
    int in_w_start = ow * stride_w - padding_w;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int in_h = in_h_start + kh;
                int in_w = in_w_start + kw;

                if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                    int in_idx = ((batch * in_channels + ic) * input_h + in_h) * input_w + in_w;
                    int w_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;

                    atomicAdd(&grad_weights[w_idx], input[in_idx] * go);
                    atomicAdd(&grad_input[in_idx], weights[w_idx] * go);
                }
            }
        }
    }
}

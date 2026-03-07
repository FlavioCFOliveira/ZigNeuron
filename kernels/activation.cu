// CUDA Kernels for Activation Functions
// Optimized for memory bandwidth with vectorized loads

#include "common.h"

// ReLU forward
extern "C" __global__ void relu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Vectorized loop
    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(input)[i];
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        reinterpret_cast<float4*>(output)[i] = val;
    }

    // Remainder
    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

// ReLU backward
extern "C" __global__ void relu_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 out = reinterpret_cast<const float4*>(output)[i];
        float4 go = reinterpret_cast<const float4*>(grad_output)[i];
        float4 gi;

        gi.x = (out.x > 0.0f) ? go.x : 0.0f;
        gi.y = (out.y > 0.0f) ? go.y : 0.0f;
        gi.z = (out.z > 0.0f) ? go.z : 0.0f;
        gi.w = (out.w > 0.0f) ? go.w : 0.0f;

        reinterpret_cast<float4*>(grad_input)[i] = gi;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        grad_input[i] = (output[i] > 0.0f) ? grad_output[i] : 0.0f;
    }
}

// Sigmoid forward
extern "C" __global__ void sigmoid_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(input)[i];
        float4 out;
        out.x = sigmoidf_stable(val.x);
        out.y = sigmoidf_stable(val.y);
        out.z = sigmoidf_stable(val.z);
        out.w = sigmoidf_stable(val.w);
        reinterpret_cast<float4*>(output)[i] = out;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        output[i] = sigmoidf_stable(input[i]);
    }
}

// Sigmoid backward
extern "C" __global__ void sigmoid_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 y = reinterpret_cast<const float4*>(output)[i];
        float4 go = reinterpret_cast<const float4*>(grad_output)[i];
        float4 gi;

        // dy/dx = y * (1 - y)
        gi.x = go.x * y.x * (1.0f - y.x);
        gi.y = go.y * y.y * (1.0f - y.y);
        gi.z = go.z * y.z * (1.0f - y.z);
        gi.w = go.w * y.w * (1.0f - y.w);

        reinterpret_cast<float4*>(grad_input)[i] = gi;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        float y = output[i];
        grad_input[i] = grad_output[i] * y * (1.0f - y);
    }
}

// Tanh forward
extern "C" __global__ void tanh_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(input)[i];
        float4 out;
        out.x = tanhf(val.x);
        out.y = tanhf(val.y);
        out.z = tanhf(val.z);
        out.w = tanhf(val.w);
        reinterpret_cast<float4*>(output)[i] = out;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

// Tanh backward
extern "C" __global__ void tanh_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 y = reinterpret_cast<const float4*>(output)[i];
        float4 go = reinterpret_cast<const float4*>(grad_output)[i];
        float4 gi;

        // dy/dx = 1 - y^2
        gi.x = go.x * (1.0f - y.x * y.x);
        gi.y = go.y * (1.0f - y.y * y.y);
        gi.z = go.z * (1.0f - y.z * y.z);
        gi.w = go.w * (1.0f - y.w * y.w);

        reinterpret_cast<float4*>(grad_input)[i] = gi;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        float y = output[i];
        grad_input[i] = grad_output[i] * (1.0f - y * y);
    }
}

// Linear (identity) forward
extern "C" __global__ void linear_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        reinterpret_cast<float4*>(output)[i] = reinterpret_cast<const float4*>(input)[i];
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        output[i] = input[i];
    }
}

// Linear (identity) backward
extern "C" __global__ void linear_backward(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        reinterpret_cast<float4*>(grad_input)[i] = reinterpret_cast<const float4*>(grad_output)[i];
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        grad_input[i] = grad_output[i];
    }
}

// GELU forward
extern "C" __global__ void gelu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        output[i] = geluf(input[i]);
    }
}

// GELU backward
extern "C" __global__ void gelu_backward(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        float x = input[i];
        float go = grad_output[i];

        // GELU derivative approximation
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;

        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_val = tanhf(tanh_arg);
        float sech_sq = 1.0f - tanh_val * tanh_val;

        float d_tanh = sech_sq * sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
        float d_gelu = 0.5f * (1.0f + tanh_val + x * d_tanh);

        grad_input[i] = go * d_gelu;
    }
}

// Softmax forward (per-sample)
extern "C" __global__ void softmax_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_samples, int num_classes
) {
    int sample = blockIdx.x;
    if (sample >= num_samples) return;

    const float* in_sample = input + sample * num_classes;
    float* out_sample = output + sample * num_classes;

    // Step 1: Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        max_val = fmaxf(max_val, in_sample[i]);
    }
    max_val = warpReduceMax(max_val);

    // Step 2: Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        float exp_val = expf(in_sample[i] - max_val);
        out_sample[i] = exp_val;
        sum_exp += exp_val;
    }
    sum_exp = warpReduceSum(sum_exp);

    // Step 3: Normalize
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        out_sample[i] /= sum_exp;
    }
}

// Softmax backward (using activated output)
extern "C" __global__ void softmax_backward(
    const float* __restrict__ activated_output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int num_samples, int num_classes
) {
    int sample = blockIdx.x;
    if (sample >= num_samples) return;

    const float* s_sample = activated_output + sample * num_classes;
    const float* go_sample = grad_output + sample * num_classes;
    float* gi_sample = grad_input + sample * num_classes;

    // Compute sum of go * s
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        sum += go_sample[i] * s_sample[i];
    }
    sum = warpReduceSum(sum);

    // gi_i = s_i * (go_i - sum(go_j * s_j))
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        gi_sample[i] = s_sample[i] * (go_sample[i] - sum);
    }
}

// Batched ReLU forward
extern "C" __global__ void relu_forward_batch(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int size
) {
    int batch = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batch_size || idx >= size) return;

    int offset = batch * size + idx;
    output[offset] = fmaxf(0.0f, input[offset]);
}

// Batched ReLU backward
extern "C" __global__ void relu_backward_batch(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int batch_size, int size
) {
    int batch = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batch_size || idx >= size) return;

    int offset = batch * size + idx;
    grad_input[offset] = (input[offset] > 0.0f) ? grad_output[offset] : 0.0f;
}

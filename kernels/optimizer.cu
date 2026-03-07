// CUDA Kernels for Optimizers
// SGD, Adam, and RMSprop implementations

#include "common.h"

// SGD weight update: w = w - lr * (g + wd * w)
extern "C" __global__ void sgd_update(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    int size,
    float learning_rate,
    float weight_decay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        float g = gradients[i];
        float w = weights[i];

        // Gradient clipping
        g = fminf(fmaxf(g, -5.0f), 5.0f);
        if (isnan(g)) g = 0.0f;

        float new_w = w - learning_rate * (g + weight_decay * w);

        // Weight clipping
        new_w = fminf(fmaxf(new_w, -100.0f), 100.0f);

        weights[i] = new_w;
    }
}

// SGD bias update (no weight decay)
extern "C" __global__ void sgd_update_bias(
    float* __restrict__ bias,
    const float* __restrict__ gradients,
    int size,
    float learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        float g = gradients[i];
        float b = bias[i];

        // Gradient clipping
        g = fminf(fmaxf(g, -5.0f), 5.0f);
        if (isnan(g)) g = 0.0f;

        float new_b = b - learning_rate * g;

        // Bias clipping
        new_b = fminf(fmaxf(new_b, -50.0f), 50.0f);

        bias[i] = new_b;
    }
}

// Accumulate bias gradients: gb = sum(grad_after_act) over batch
extern "C" __global__ void accumulate_bias(
    float* __restrict__ grad_bias,
    const float* __restrict__ grad_after_act,
    int batch_size, int bias_size
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= bias_size) return;

    float sum = 0.0f;
    for (int batch = 0; batch < batch_size; batch++) {
        sum += grad_after_act[batch * bias_size + col];
    }

    grad_bias[col] += sum;
}

// Adam optimizer update
extern "C" __global__ void adam_update(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float* __restrict__ m,
    float* __restrict__ v,
    int size,
    float lr, float beta1, float beta2, float eps,
    float bias_corr1, float bias_corr2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        float g = gradients[i];

        // Gradient clipping
        g = fminf(fmaxf(g, -5.0f), 5.0f);
        if (isnan(g)) g = 0.0f;

        // Update biased first moment estimate
        float m_val = beta1 * m[i] + (1.0f - beta1) * g;
        m[i] = m_val;

        // Update biased second raw moment estimate
        float v_val = beta2 * v[i] + (1.0f - beta2) * g * g;
        v[i] = v_val;

        // Compute bias-corrected estimates
        float m_hat = m_val / bias_corr1;
        float v_hat = v_val / bias_corr2;

        // Update weights
        float new_w = weights[i] - lr * m_hat / (sqrtf(v_hat) + eps);

        // Weight clipping
        new_w = fminf(fmaxf(new_w, -100.0f), 100.0f);

        weights[i] = new_w;
    }
}

// RMSprop optimizer update
extern "C" __global__ void rmsprop_update(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float* __restrict__ g_avg,
    int size,
    float lr, float rho, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        float g = gradients[i];

        // Gradient clipping
        g = fminf(fmaxf(g, -5.0f), 5.0f);
        if (isnan(g)) g = 0.0f;

        // Update moving average of squared gradients
        float g_avg_val = rho * g_avg[i] + (1.0f - rho) * g * g;
        g_avg[i] = g_avg_val;

        // Update weights
        float new_w = weights[i] - lr * g / (sqrtf(g_avg_val) + eps);

        // Weight clipping
        new_w = fminf(fmaxf(new_w, -100.0f), 100.0f);

        weights[i] = new_w;
    }
}

// Fill buffer with constant value
extern "C" __global__ void fill_constant(
    float* __restrict__ data,
    float value,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = {value, value, value, value};
        reinterpret_cast<float4*>(data)[i] = val;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        data[i] = value;
    }
}

// Scale buffer by constant
extern "C" __global__ void scale_buffer(
    float* __restrict__ data,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(data)[i];
        val.x *= scale;
        val.y *= scale;
        val.z *= scale;
        val.w *= scale;
        reinterpret_cast<float4*>(data)[i] = val;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        data[i] *= scale;
    }
}

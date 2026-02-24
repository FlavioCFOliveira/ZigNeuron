// Metal shaders for optimizers
#include <metal_stdlib>
using namespace metal;

// SGD weight update: w = w - lr * (g + wd * w)
kernel void sgd_update(
    device float* weights [[buffer(0)]],
    device const float* gradients [[buffer(1)]],
    constant float& learning_rate [[buffer(2)]],
    constant float& weight_decay [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float w = weights[gid];
        float g = gradients[gid];

        // Gradient clipping (to match network.zig implementation)
        const float max_grad = 5.0f;
        if (g > max_grad) g = max_grad;
        if (g < -max_grad) g = -max_grad;
        if (isnan(g)) g = 0.0f;

        float new_w = w - learning_rate * (g + weight_decay * w);

        // Weight clipping
        if (new_w > 100.0f) new_w = 100.0f;
        if (new_w < -100.0f) new_w = -100.0f;

        weights[gid] = new_w;
    }
}

// SGD bias update (usually no weight decay for bias)
kernel void sgd_update_bias(
    device float* bias [[buffer(0)]],
    device const float* gradients [[buffer(1)]],
    constant float& learning_rate [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float b = bias[gid];
        float g = gradients[gid];

        // Gradient clipping
        const float max_grad = 5.0f;
        if (g > max_grad) g = max_grad;
        if (g < -max_grad) g = -max_grad;
        if (isnan(g)) g = 0.0f;

        float new_b = b - learning_rate * g;

        // Bias clipping
        if (new_b > 50.0f) new_b = 50.0f;
        if (new_b < -50.0f) new_b = -50.0f;

        bias[gid] = new_b;
    }
}

// Accumulate bias gradients: gb = gb + sum(grad_after_act) over batch
kernel void accumulate_bias(
    device float* grad_bias [[buffer(0)]],
    device const float* grad_after_act [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& bias_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < bias_size) {
        float sum = 0.0f;
        for (uint i = 0; i < batch_size; i++) {
            sum += grad_after_act[i * bias_size + gid];
        }
        grad_bias[gid] += sum;
    }
}

// Adam optimizer update
kernel void adam_update(
    device float* weights [[buffer(0)]],
    device const float* gradients [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& bias_corr1 [[buffer(8)]],
    constant float& bias_corr2 [[buffer(9)]],
    constant uint& size [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float g = gradients[gid];
        // Clip gradients
        const float max_grad = 5.0f;
        if (g > max_grad) g = max_grad;
        if (g < -max_grad) g = -max_grad;
        if (isnan(g)) g = 0.0f;

        float m_val = beta1 * m[gid] + (1.0f - beta1) * g;
        float v_val = beta2 * v[gid] + (1.0f - beta2) * g * g;

        m[gid] = m_val;
        v[gid] = v_val;

        float m_hat = m_val / bias_corr1;
        float v_hat = v_val / bias_corr2;

        float new_w = weights[gid] - lr * m_hat / (sqrt(v_hat) + eps);

        // Weight clipping
        if (new_w > 100.0f) new_w = 100.0f;
        if (new_w < -100.0f) new_w = -100.0f;

        weights[gid] = new_w;
    }
}

// RMSprop optimizer update
kernel void rmsprop_update(
    device float* weights [[buffer(0)]],
    device const float* gradients [[buffer(1)]],
    device float* g_avg [[buffer(2)]],
    constant float& lr [[buffer(3)]],
    constant float& rho [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& size [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float g = gradients[gid];
        // Clip gradients
        const float max_grad = 5.0f;
        if (g > max_grad) g = max_grad;
        if (g < -max_grad) g = -max_grad;
        if (isnan(g)) g = 0.0f;

        float g_avg_val = rho * g_avg[gid] + (1.0f - rho) * g * g;
        g_avg[gid] = g_avg_val;

        float new_w = weights[gid] - lr * g / (sqrt(g_avg_val) + eps);

        // Weight clipping
        if (new_w > 100.0f) new_w = 100.0f;
        if (new_w < -100.0f) new_w = -100.0f;

        weights[gid] = new_w;
    }
}

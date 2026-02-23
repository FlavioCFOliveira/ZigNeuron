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

// Accumulate bias gradients: gb = gb + g
kernel void accumulate_bias(
    device float* grad_bias [[buffer(0)]],
    device const float* grad_after_act [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        grad_bias[gid] += grad_after_act[gid];
    }
}

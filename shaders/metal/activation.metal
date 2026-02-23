// Metal shaders for activation functions
// Includes forward and backward passes
#include <metal_stdlib>
using namespace metal;

// ReLU forward: max(0, x)
kernel void relu_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = input[gid];
        output[gid] = max(0.0f, x);
    }
}

// ReLU backward: gradient if y > 0, else 0 (using activated output y)
kernel void relu_backward(
    device const float* output [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float y = output[gid];
        grad_input[gid] = (y > 0.0f) ? grad_output[gid] : 0.0f;
    }
}

// Sigmoid forward: 1 / (1 + exp(-x))
kernel void sigmoid_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = input[gid];
        output[gid] = 1.0f / (1.0f + exp(-x));
    }
}

// Sigmoid backward: gradient * y * (1 - y) (using activated output y)
kernel void sigmoid_backward(
    device const float* output [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float y = output[gid];
        float grad = grad_output[gid];
        // Gradient: grad * y * (1 - y)
        grad_input[gid] = grad * y * (1.0f - y);
    }
}

// Tanh forward: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
kernel void tanh_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = input[gid];
        // Use tanh builtin for accuracy
        output[gid] = tanh(x);
    }
}

// Tanh backward: gradient * (1 - y^2) (using activated output y)
kernel void tanh_backward(
    device const float* output [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float y = output[gid];
        float grad = grad_output[gid];
        // Gradient: grad * (1 - y^2)
        grad_input[gid] = grad * (1.0f - y * y);
    }
}

// Linear forward: identity
kernel void linear_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        output[gid] = input[gid];
    }
}

// Linear backward: gradient * 1
kernel void linear_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        grad_input[gid] = grad_output[gid];
    }
}

// Batched ReLU forward
kernel void relu_forward_batch(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint idx = gid.x;

    if (batch < batch_size && idx < size) {
        uint offset = batch * size + idx;
        float x = input[offset];
        output[offset] = max(0.0f, x);
    }
}

// Batched ReLU backward
kernel void relu_backward_batch(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint idx = gid.x;

    if (batch < batch_size && idx < size) {
        uint offset = batch * size + idx;
        float x = input[offset];
        grad_input[offset] = (x > 0.0f) ? grad_output[offset] : 0.0f;
    }
}

// Softmax forward (requires special handling for numerical stability)
kernel void softmax_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    constant uint& num_classes [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint sample = gid.y;
    uint c_idx = gid.x;

    if (sample < size && c_idx < num_classes) {
        // Find max for numerical stability
        float max_val = input[sample * num_classes];
        for (uint c = 1; c < num_classes; c++) {
            max_val = max(max_val, input[sample * num_classes + c]);
        }

        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (uint c = 0; c < num_classes; c++) {
            sum_exp += exp(input[sample * num_classes + c] - max_val);
        }

        // Compute softmax
        output[sample * num_classes + c_idx] =
            exp(input[sample * num_classes + c_idx] - max_val) / sum_exp;
    }
}

// Softmax backward (Jacobian-vector product)
kernel void softmax_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint sample = gid.y;
    uint i = gid.x;

    if (sample < size && i < num_classes) {
        // First compute softmax for this sample
        // Find max for numerical stability
        float max_val = input[sample * num_classes];
        for (uint c = 1; c < num_classes; c++) {
            max_val = max(max_val, input[sample * num_classes + c]);
        }

        float sum_exp = 0.0f;
        for (uint c = 0; c < num_classes; c++) {
            sum_exp += exp(input[sample * num_classes + c] - max_val);
        }

        float s_i = exp(input[sample * num_classes + i] - max_val) / sum_exp;

        // Compute Jacobian-vector product: grad_input[i] = s_i * (grad_output[i] - sum(grad_output[j] * s_j))
        float sum_grad_s = 0.0f;
        for (uint j = 0; j < num_classes; j++) {
            float s_j = exp(input[sample * num_classes + j] - max_val) / sum_exp;
            sum_grad_s += grad_output[sample * num_classes + j] * s_j;
        }

        grad_input[sample * num_classes + i] = s_i * (grad_output[sample * num_classes + i] - sum_grad_s);
    }
}

// Metal shaders for loss functions
// Includes gradient computation for backpropagation
#include <metal_stdlib>
using namespace metal;

// MSE (Mean Squared Error) forward: (1/n) * sum((pred - target)^2)
kernel void mse_forward(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* loss [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < size; i++) {
            float diff = pred[i] - target[i];
            sum += diff * diff;
        }
        loss[0] = sum / n;
    }
}

// MSE backward: gradient = 2 * (pred - target) / n
kernel void mse_backward(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* grad_output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float diff = pred[gid] - target[gid];
        grad_output[gid] = 2.0f * diff / n;
    }
}

// Cross Entropy forward for logits: -sum(target * log(softmax(logits)))
kernel void cross_entropy_forward(
    device const float* logits [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* loss [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < size; i++) {
            // Find max for numerical stability
            float max_logit = logits[i * num_classes];
            for (uint c = 1; c < num_classes; c++) {
                max_logit = max(max_logit, logits[i * num_classes + c]);
            }

            // Compute sum of exponentials
            float sum_exp = 0.0f;
            for (uint c = 0; c < num_classes; c++) {
                sum_exp += exp(logits[i * num_classes + c] - max_logit);
            }

            // Compute cross entropy
            for (uint c = 0; c < num_classes; c++) {
                if (target[i * num_classes + c] > 0.0f) {
                    float prob = exp(logits[i * num_classes + c] - max_logit) / sum_exp;
                    sum -= target[i * num_classes + c] * log(prob);
                }
            }
        }
        loss[0] = sum / size;
    }
}

// Cross Entropy backward for logits: gradient = softmax(logits) - target
kernel void cross_entropy_backward(
    device const float* logits [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* grad_output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint sample = gid / num_classes;
    uint c_idx = gid % num_classes;

    if (sample < size) {
        // Find max for numerical stability
        float max_logit = logits[sample * num_classes];
        for (uint c = 1; c < num_classes; c++) {
            max_logit = max(max_logit, logits[sample * num_classes + c]);
        }

        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (uint c = 0; c < num_classes; c++) {
            sum_exp += exp(logits[sample * num_classes + c] - max_logit);
        }

        // Compute gradient: softmax(logits) - target
        float prob = exp(logits[sample * num_classes + c_idx] - max_logit) / sum_exp;
        grad_output[sample * num_classes + c_idx] = prob - target[sample * num_classes + c_idx];
    }
}

// Binary Cross Entropy forward: -[target * log(pred) + (1 - target) * log(1 - pred)]
kernel void binary_cross_entropy_forward(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* loss [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        float sum = 0.0f;
        const float eps = 1e-7f; // For numerical stability
        for (uint i = 0; i < size; i++) {
            float p = pred[i];
            float t = target[i];

            // Clip predictions to avoid log(0)
            if (p < eps) p = eps;
            if (p > 1.0f - eps) p = 1.0f - eps;

            sum -= t * log(p) + (1.0f - t) * log(1.0f - p);
        }
        loss[0] = sum / n;
    }
}

// Binary Cross Entropy backward: gradient = (pred - target) / n
kernel void binary_cross_entropy_backward(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* grad_output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        // Gradient simplifies to (pred - target) / n for sigmoid output
        float p = pred[gid];
        float t = target[gid];
        grad_output[gid] = (p - t) / n;
    }
}

// Batched MSE backward: gradient = 2 * (pred - target) / n
kernel void mse_backward_batch(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* grad_output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint idx = gid.x;

    if (batch < batch_size && idx < size) {
        uint offset = batch * size + idx;
        float diff = pred[offset] - target[offset];
        grad_output[offset] = 2.0f * diff / n;
    }
}

// Batched Cross Entropy backward
kernel void cross_entropy_backward_batch(
    device const float* logits [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* grad_output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    constant uint& num_classes [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint sample = gid.y;
    uint c_idx = gid.x;

    if (batch < batch_size && sample < size && c_idx < num_classes) {
        uint offset = (batch * size + sample) * num_classes + c_idx;

        // Find max for numerical stability
        float max_logit = logits[(batch * size + sample) * num_classes];
        for (uint c = 1; c < num_classes; c++) {
            max_logit = max(max_logit, logits[(batch * size + sample) * num_classes + c]);
        }

        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (uint c = 0; c < num_classes; c++) {
            sum_exp += exp(logits[(batch * size + sample) * num_classes + c] - max_logit);
        }

        // Compute gradient: softmax(logits) - target
        float prob = exp(logits[offset] - max_logit) / sum_exp;
        grad_output[offset] = prob - target[offset];
    }
}

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

// MSE backward - Vectorized
kernel void mse_backward(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* grad_output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = gid * 4;
    float n_f = (float)n;
    if (idx + 3 < size) {
        device const float4* p4 = (device const float4*)(pred + idx);
        device const float4* t4 = (device const float4*)(target + idx);
        device float4* go4 = (device float4*)(grad_output + idx);
        *go4 = 2.0f * (*p4 - *t4) / n_f;
    } else {
        for (uint i = idx; i < size; i++) {
            grad_output[i] = 2.0f * (pred[i] - target[i]) / n_f;
        }
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

// Cross Entropy backward - Optimized
kernel void cross_entropy_backward(
    device const float* logits [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* grad_output [[buffer(2)]],
    constant uint& num_samples [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]])
{
    uint sample = gid.y;
    if (sample >= num_samples) return;

    device const float* l_sample = logits + sample * num_classes;
    device const float* t_sample = target + sample * num_classes;
    device float* go_sample = grad_output + sample * num_classes;

    // 1. Find max for numerical stability
    float thread_max = -INFINITY;
    for (uint c = tid; c < num_classes; c += tg_size.x) {
        thread_max = max(thread_max, l_sample[c]);
    }
    float max_logit = simd_max(thread_max);

    // 2. Compute sum of exponentials
    float thread_sum = 0.0f;
    for (uint c = tid; c < num_classes; c += tg_size.x) {
        thread_sum += exp(l_sample[c] - max_logit);
    }
    float sum_exp = simd_sum(thread_sum);

    // 3. Compute gradient: softmax(logits) - target
    for (uint c = tid; c < num_classes; c += tg_size.x) {
        float prob = exp(l_sample[c] - max_logit) / sum_exp;
        go_sample[c] = prob - t_sample[c];
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

// Binary Cross Entropy backward - Vectorized
kernel void binary_cross_entropy_backward(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* grad_output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = gid * 4;
    float n_f = (float)n;
    if (idx + 3 < size) {
        device const float4* p4 = (device const float4*)(pred + idx);
        device const float4* t4 = (device const float4*)(target + idx);
        device float4* go4 = (device float4*)(grad_output + idx);
        *go4 = (*p4 - *t4) / n_f;
    } else {
        for (uint i = idx; i < size; i++) {
            grad_output[i] = (pred[i] - target[i]) / n_f;
        }
    }
}

// KL Divergence backward: grad_mu = mu, grad_log_var = 0.5 * (exp(log_var) - 1)
kernel void kl_divergence_backward(
    device const float* output [[buffer(0)]],
    device float* grad_output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        float mu = output[gid];
        float log_var = output[n + gid];

        grad_output[gid] = mu / n;
        grad_output[n + gid] = 0.5f * (exp(log_var) - 1.0f) / n;
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

// CUDA Kernels for Loss Functions
// MSE, Cross-Entropy, Binary Cross-Entropy

#include "common.h"

// MSE forward: (1/n) * sum((pred - target)^2)
extern "C" __global__ void mse_forward(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ loss,
    int size, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < size; i += stride) {
        float diff = pred[i] - target[i];
        sum += diff * diff;
    }

    // Warp reduction
    sum = warpReduceSum(sum);

    // Block reduction using shared memory
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) shared[wid] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) {
            total += shared[i];
        }
        atomicAdd(loss, total / n);
    }
}

// MSE backward: grad = 2 * (pred - target) / n
extern "C" __global__ void mse_backward(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ grad_output,
    int size, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float scale = 2.0f / n;

    for (int i = idx; i < size / 4; i += stride) {
        float4 p = reinterpret_cast<const float4*>(pred)[i];
        float4 t = reinterpret_cast<const float4*>(target)[i];
        float4 g;

        g.x = (p.x - t.x) * scale;
        g.y = (p.y - t.y) * scale;
        g.z = (p.z - t.z) * scale;
        g.w = (p.w - t.w) * scale;

        reinterpret_cast<float4*>(grad_output)[i] = g;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        grad_output[i] = (pred[i] - target[i]) * scale;
    }
}

// Batched MSE backward
extern "C" __global__ void mse_backward_batch(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ grad_output,
    int batch_size, int size, int n
) {
    int batch = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batch_size || idx >= size) return;

    int offset = batch * size + idx;
    float scale = 2.0f / n;
    float diff = pred[offset] - target[offset];
    grad_output[offset] = diff * scale;
}

// Binary Cross Entropy backward: grad = (pred - target) / n
extern "C" __global__ void binary_cross_entropy_backward(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ grad_output,
    int size, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float scale = 1.0f / n;

    for (int i = idx; i < size / 4; i += stride) {
        float4 p = reinterpret_cast<const float4*>(pred)[i];
        float4 t = reinterpret_cast<const float4*>(target)[i];
        float4 g;

        g.x = (p.x - t.x) * scale;
        g.y = (p.y - t.y) * scale;
        g.z = (p.z - t.z) * scale;
        g.w = (p.w - t.w) * scale;

        reinterpret_cast<float4*>(grad_output)[i] = g;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        grad_output[i] = (pred[i] - target[i]) * scale;
    }
}

// Cross Entropy backward (for one-hot targets)
extern "C" __global__ void cross_entropy_backward(
    const float* __restrict__ logits,
    const float* __restrict__ target,
    float* __restrict__ grad_output,
    int num_samples, int num_classes
) {
    int sample = blockIdx.y;
    if (sample >= num_samples) return;

    const float* l_sample = logits + sample * num_classes;
    const float* t_sample = target + sample * num_classes;
    float* go_sample = grad_output + sample * num_classes;

    // Step 1: Find max for numerical stability
    float thread_max = -INFINITY;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        thread_max = fmaxf(thread_max, l_sample[c]);
    }
    float max_logit = warpReduceMax(thread_max);

    // Step 2: Compute sum of exponentials
    float thread_sum = 0.0f;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        thread_sum += expf(l_sample[c] - max_logit);
    }
    float sum_exp = warpReduceSum(thread_sum);

    // Step 3: Compute gradient: softmax(logits) - target
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        float prob = expf(l_sample[c] - max_logit) / sum_exp;
        go_sample[c] = prob - t_sample[c];
    }
}

// KL Divergence backward: grad_mu = mu, grad_log_var = 0.5 * (exp(log_var) - 1)
extern "C" __global__ void kl_divergence_backward(
    const float* __restrict__ output,
    float* __restrict__ grad_output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float mu = output[idx];
        float log_var = output[n + idx];

        grad_output[idx] = mu / n;
        grad_output[n + idx] = 0.5f * (expf(log_var) - 1.0f) / n;
    }
}

// Cross Entropy forward (for reference/completeness)
extern "C" __global__ void cross_entropy_forward(
    const float* __restrict__ logits,
    const float* __restrict__ target,
    float* __restrict__ loss,
    int num_samples, int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx != 0) return;  // Only first thread computes

    float total_loss = 0.0f;

    for (int sample = 0; sample < num_samples; sample++) {
        const float* l_sample = logits + sample * num_classes;
        const float* t_sample = target + sample * num_classes;

        // Find max for stability
        float max_logit = l_sample[0];
        for (int c = 1; c < num_classes; c++) {
            max_logit = fmaxf(max_logit, l_sample[c]);
        }

        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(l_sample[c] - max_logit);
        }

        // Compute cross entropy
        for (int c = 0; c < num_classes; c++) {
            if (t_sample[c] > 0.0f) {
                float prob = expf(l_sample[c] - max_logit) / sum_exp;
                total_loss -= t_sample[c] * logf(fmaxf(prob, 1e-10f));
            }
        }
    }

    *loss = total_loss / num_samples;
}

// Binary Cross Entropy forward
extern "C" __global__ void binary_cross_entropy_forward(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ loss,
    int size, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx != 0) return;

    const float eps = 1e-7f;
    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        float p = pred[i];
        float t = target[i];

        // Clip predictions
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;

        sum -= t * logf(p) + (1.0f - t) * logf(1.0f - p);
    }

    *loss = sum / n;
}

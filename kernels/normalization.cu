// CUDA Kernels for Normalization Layers
// LayerNorm and BatchNorm implementations

#include "common.h"

// LayerNorm forward: output = gamma * (input - mean) / sqrt(var + eps) + beta
extern "C" __global__ void layernorm_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batch_size, int feature_size,
    float eps
) {
    int sample = blockIdx.x;
    if (sample >= batch_size) return;

    const float* in_sample = input + sample * feature_size;
    float* out_sample = output + sample * feature_size;

    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        sum += in_sample[i];
    }
    sum = warpReduceSum(sum);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) shared[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        sum = warpReduceSum(sum);
    }
    float mean = sum / feature_size;
    mean = __shfl_sync(0xFFFFFFFF, mean, 0);

    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        float diff = in_sample[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = warpReduceSum(var_sum);

    if (lane == 0) shared[wid] = var_sum;
    __syncthreads();

    if (wid == 0) {
        var_sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        var_sum = warpReduceSum(var_sum);
    }
    float variance = var_sum / feature_size;
    variance = __shfl_sync(0xFFFFFFFF, variance, 0);
    float inv_std = rsqrtf(variance + eps);

    // Step 3: Normalize and scale
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        float x_hat = (in_sample[i] - mean) * inv_std;
        out_sample[i] = x_hat * gamma[i] + beta[i];
    }
}

// LayerNorm backward
extern "C" __global__ void layernorm_backward(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const float* __restrict__ gamma,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int batch_size, int feature_size,
    float eps
) {
    int sample = blockIdx.x;
    if (sample >= batch_size) return;

    const float* in_sample = input + sample * feature_size;
    const float* go_sample = grad_output + sample * feature_size;
    float* gi_sample = grad_input + sample * feature_size;

    // Recompute mean and variance
    float sum = 0.0f;
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        sum += in_sample[i];
    }
    sum = warpReduceSum(sum);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) shared[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        sum = warpReduceSum(sum);
    }
    float mean = sum / feature_size;
    mean = __shfl_sync(0xFFFFFFFF, mean, 0);

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        float diff = in_sample[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = warpReduceSum(var_sum);

    if (lane == 0) shared[wid] = var_sum;
    __syncthreads();

    if (wid == 0) {
        var_sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        var_sum = warpReduceSum(var_sum);
    }
    float variance = var_sum / feature_size;
    variance = __shfl_sync(0xFFFFFFFF, variance, 0);
    float inv_std = rsqrtf(variance + eps);

    // Accumulate grad_gamma and grad_beta
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        float x_hat = (in_sample[i] - mean) * inv_std;
        atomicAdd(&grad_gamma[i], go_sample[i] * x_hat);
        atomicAdd(&grad_beta[i], go_sample[i]);
    }

    // Compute grad_input (simplified approximation)
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        gi_sample[i] = go_sample[i] * gamma[i] * inv_std;
    }
}

// BatchNorm forward for training
extern "C" __global__ void batchnorm_forward_training(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    int batch_size, int num_channels, int spatial_size,
    float eps, float momentum
) {
    int c = blockIdx.x;
    if (c >= num_channels) return;

    // Compute mean across batch and spatial dimensions
    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
            sum += input[((b * num_channels + c) * spatial_size) + s];
        }
    }
    sum = warpReduceSum(sum);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) shared[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        sum = warpReduceSum(sum);
    }
    float mean = sum / (batch_size * spatial_size);
    mean = __shfl_sync(0xFFFFFFFF, mean, 0);

    // Compute variance
    float var_sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
            float diff = input[((b * num_channels + c) * spatial_size) + s] - mean;
            var_sum += diff * diff;
        }
    }
    var_sum = warpReduceSum(var_sum);

    if (lane == 0) shared[wid] = var_sum;
    __syncthreads();

    if (wid == 0) {
        var_sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        var_sum = warpReduceSum(var_sum);
    }
    float variance = var_sum / (batch_size * spatial_size);
    variance = __shfl_sync(0xFFFFFFFF, variance, 0);

    // Update running statistics
    if (threadIdx.x == 0) {
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
        running_var[c] = (1.0f - momentum) * running_var[c] + momentum * variance;
    }

    // Normalize and scale
    float inv_std = rsqrtf(variance + eps);
    for (int b = 0; b < batch_size; b++) {
        for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
            int idx = ((b * num_channels + c) * spatial_size) + s;
            float x_hat = (input[idx] - mean) * inv_std;
            output[idx] = x_hat * gamma[c] + beta[c];
        }
    }
}

// BatchNorm forward for inference
extern "C" __global__ void batchnorm_forward_inference(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    int batch_size, int num_channels, int spatial_size,
    float eps
) {
    int c = blockIdx.x;
    if (c >= num_channels) return;

    float mean = running_mean[c];
    float variance = running_var[c];
    float inv_std = rsqrtf(variance + eps);

    for (int b = 0; b < batch_size; b++) {
        for (int s = threadIdx.x; s < spatial_size; s += blockDim.x) {
            int idx = ((b * num_channels + c) * spatial_size) + s;
            float x_hat = (input[idx] - mean) * inv_std;
            output[idx] = x_hat * gamma[c] + beta[c];
        }
    }
}

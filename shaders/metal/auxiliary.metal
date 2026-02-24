// Metal shaders for auxiliary layers (Dropout, Sampling)
#include <metal_stdlib>
using namespace metal;

// Philox counter-based RNG
uint32_t philox2x32_aux(uint32_t counter, uint32_t key) {
    uint32_t L = counter;
    uint32_t R = key;
    uint32_t M = 0xD2511F53;
    for (int i = 0; i < 10; i++) {
        uint64_t prod = (uint64_t)L * M;
        uint32_t hi = (uint32_t)(prod >> 32);
        uint32_t lo = (uint32_t)prod;
        L = R ^ hi;
        R = lo;
    }
    return L;
}

// Dropout forward: output = input * mask * scale
kernel void dropout_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* mask [[buffer(2)]],
    constant float& rate [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint64_t& seed [[buffer(5)]],
    constant uint& size [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        uint32_t rnd = philox2x32_aux(gid, (uint32_t)seed);
        float p = (float)rnd / 4294967296.0f;

        if (p > rate) {
            mask[gid] = 1.0f;
            output[gid] = input[gid] * scale;
        } else {
            mask[gid] = 0.0f;
            output[gid] = 0.0f;
        }
    }
}

// VAE Sampling forward: output = mu + epsilon * exp(0.5 * log_var)
kernel void vae_sampling_forward(
    device const float* input [[buffer(0)]], // [mu, log_var]
    device float* output [[buffer(1)]],     // [z]
    device float* epsilon [[buffer(2)]],
    constant uint64_t& seed [[buffer(3)]],
    constant uint& latent_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < latent_dim) {
        // Box-Muller for epsilon
        uint32_t u1_raw = philox2x32_aux(gid * 2, (uint32_t)seed);
        uint32_t u2_raw = philox2x32_aux(gid * 2 + 1, (uint32_t)seed);

        float u1 = max((float)u1_raw / 4294967296.0f, 1e-7f);
        float u2 = (float)u2_raw / 4294967296.0f;

        float mag = sqrt(-2.0f * log(u1));
        float eps = mag * cos(2.0f * M_PI_F * u2);
        epsilon[gid] = eps;

        float mu = input[gid];
        float log_var = input[latent_dim + gid];

        output[gid] = mu + eps * exp(0.5f * log_var);
    }
}

// VAE Sampling backward
kernel void vae_sampling_backward(
    device const float* input [[buffer(0)]], // [mu, log_var]
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    device const float* epsilon [[buffer(3)]],
    constant uint& latent_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < latent_dim) {
        float log_var = input[latent_dim + gid];
        float go = grad_output[gid];
        float eps = epsilon[gid];

        // grad_mu = grad_output
        grad_input[gid] = go;

        // grad_log_var = grad_output * epsilon * exp(0.5 * log_var) * 0.5
        grad_input[latent_dim + gid] = go * eps * exp(0.5f * log_var) * 0.5f;
    }
}

// Fill buffer with constant value
kernel void fill_constant(
    device float* data [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        data[gid] = value;
    }
}

// Scale buffer by constant
kernel void scale_buffer(
    device float* data [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        data[gid] *= scale;
    }
}

// Reverse sequence: output[t] = input[seq_len - 1 - t]
// Each element is a vector of size 'element_size'
kernel void reverse_sequence(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& element_size [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.y;
    uint i = gid.x;
    if (t < seq_len && i < element_size) {
        uint src_t = seq_len - 1 - t;
        output[t * element_size + i] = input[src_t * element_size + i];
    }
}

// Concatenate two buffers along the last dimension
kernel void concat_buffers(
    device const float* input1 [[buffer(0)]],
    device const float* input2 [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size1 [[buffer(3)]],
    constant uint& size2 [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.y;
    uint i = gid.x;
    uint total_size = size1 + size2;
    if (t < seq_len) {
        if (i < size1) {
            output[t * total_size + i] = input1[t * size1 + i];
        } else if (i < total_size) {
            output[t * total_size + i] = input2[t * size2 + (i - size1)];
        }
    }
}

// Split buffer into two along the last dimension
kernel void split_buffer(
    device const float* input [[buffer(0)]],
    device float* output1 [[buffer(1)]],
    device float* output2 [[buffer(2)]],
    constant uint& size1 [[buffer(3)]],
    constant uint& size2 [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.y;
    uint i = gid.x;
    uint total_size = size1 + size2;
    if (t < seq_len) {
        if (i < size1) {
            output1[t * size1 + i] = input[t * total_size + i];
        } else if (i < total_size) {
            output2[t * size2 + (i - size1)] = input[t * total_size + i];
        }
    }
}

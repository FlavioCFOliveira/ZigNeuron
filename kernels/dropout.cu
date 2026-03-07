// CUDA Kernels for Dropout
// Random number generation and masking

#include "common.h"

// Philox4x32_10 RNG state structure
struct Philox4x32_10 {
    uint4 state;
    uint2 key;

    __device__ Philox4x32_10(uint64_t seed, uint64_t subsequence) {
        state.x = (uint32_t)(subsequence);
        state.y = (uint32_t)(subsequence >> 32);
        state.z = 0;
        state.w = 0;
        key.x = (uint32_t)(seed);
        key.y = (uint32_t)(seed >> 32);
    }

    __device__ uint4 operator()() {
        uint4 s = state;
        uint2 k = key;

        #pragma unroll
        for (int i = 0; i < 10; i++) {
            uint32_t L0 = mul_lo(s.x, 0xD2511F53);
            uint32_t L1 = mul_lo(s.z, 0xD2511F53);
            uint32_t H0 = mul_hi(s.x, 0xD2511F53);
            uint32_t H1 = mul_hi(s.z, 0xD2511F53);

            s.x = H0 ^ s.y ^ k.x;
            s.z = H1 ^ s.w ^ k.y;
            s.y = L0;
            s.w = L1;

            uint32_t t = k.x;
            k.x = k.y;
            k.y = t;
        }

        state.x = s.z;
        state.y = s.w;
        state.z = s.x;
        state.w = s.y;

        return s;
    }

    __device__ uint32_t mul_lo(uint32_t a, uint32_t b) {
        return a * b;
    }

    __device__ uint32_t mul_hi(uint32_t a, uint32_t b) {
        return (uint32_t)(((uint64_t)a * b) >> 32);
    }
};

// Dropout forward: output = input * mask * scale
extern "C" __global__ void dropout_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ mask,
    int size,
    float rate,
    float scale,
    uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    Philox4x32_10 rng(seed, (uint64_t)idx * 4);

    for (int i = idx; i < size / 4; i += stride) {
        uint4 rand_vals = rng();

        float4 rand_floats;
        rand_floats.x = (float)rand_vals.x / 4294967296.0f;
        rand_floats.y = (float)rand_vals.y / 4294967296.0f;
        rand_floats.z = (float)rand_vals.z / 4294967296.0f;
        rand_floats.w = (float)rand_vals.w / 4294967296.0f;

        float4 in = reinterpret_cast<const float4*>(input)[i];
        float4 out, m;

        m.x = (rand_floats.x > rate) ? 1.0f : 0.0f;
        m.y = (rand_floats.y > rate) ? 1.0f : 0.0f;
        m.z = (rand_floats.z > rate) ? 1.0f : 0.0f;
        m.w = (rand_floats.w > rate) ? 1.0f : 0.0f;

        out.x = in.x * m.x * scale;
        out.y = in.y * m.y * scale;
        out.z = in.z * m.z * scale;
        out.w = in.w * m.w * scale;

        reinterpret_cast<float4*>(output)[i] = out;
        reinterpret_cast<float4*>(mask)[i] = m;
    }

    // Remainder
    int remainder_start = (size / 4) * 4;
    rng = Philox4x32_10(seed, (uint64_t)(remainder_start + threadIdx.x));
    for (int i = idx + remainder_start; i < size; i += stride) {
        uint4 rand_vals = rng();
        float p = (float)rand_vals.x / 4294967296.0f;
        mask[i] = (p > rate) ? 1.0f : 0.0f;
        output[i] = input[i] * mask[i] * scale;
    }
}

// Dropout backward (reuse mask from forward)
extern "C" __global__ void dropout_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ mask,
    float* __restrict__ grad_input,
    int size,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 go = reinterpret_cast<const float4*>(grad_output)[i];
        float4 m = reinterpret_cast<const float4*>(mask)[i];
        float4 gi;

        gi.x = go.x * m.x * scale;
        gi.y = go.y * m.y * scale;
        gi.z = go.z * m.z * scale;
        gi.w = go.w * m.w * scale;

        reinterpret_cast<float4*>(grad_input)[i] = gi;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        grad_input[i] = grad_output[i] * mask[i] * scale;
    }
}

// Dropout with inverted dropout (pre-scaled during forward)
extern "C" __global__ void dropout_inverted_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ mask,
    int size,
    float rate,
    float inv_scale,  // 1.0 / (1.0 - rate)
    uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    Philox4x32_10 rng(seed, (uint64_t)idx * 4);

    for (int i = idx; i < size / 4; i += stride) {
        uint4 rand_vals = rng();

        float4 rand_floats;
        rand_floats.x = (float)rand_vals.x / 4294967296.0f;
        rand_floats.y = (float)rand_vals.y / 4294967296.0f;
        rand_floats.z = (float)rand_vals.z / 4294967296.0f;
        rand_floats.w = (float)rand_vals.w / 4294967296.0f;

        float4 in = reinterpret_cast<const float4*>(input)[i];
        float4 out, m;

        // Inverted dropout: scale during forward
        m.x = (rand_floats.x > rate) ? inv_scale : 0.0f;
        m.y = (rand_floats.y > rate) ? inv_scale : 0.0f;
        m.z = (rand_floats.z > rate) ? inv_scale : 0.0f;
        m.w = (rand_floats.w > rate) ? inv_scale : 0.0f;

        out.x = in.x * m.x;
        out.y = in.y * m.y;
        out.z = in.z * m.z;
        out.w = in.w * m.w;

        reinterpret_cast<float4*>(output)[i] = out;
        reinterpret_cast<float4*>(mask)[i] = m;
    }

    // Remainder
    int remainder_start = (size / 4) * 4;
    rng = Philox4x32_10(seed, (uint64_t)(remainder_start + threadIdx.x));
    for (int i = idx + remainder_start; i < size; i += stride) {
        uint4 rand_vals = rng();
        float p = (float)rand_vals.x / 4294967296.0f;
        mask[i] = (p > rate) ? inv_scale : 0.0f;
        output[i] = input[i] * mask[i];
    }
}

// VAE Sampling forward: z = mu + epsilon * exp(0.5 * log_var)
extern "C" __global__ void vae_sampling_forward(
    const float* __restrict__ input,  // [mu, log_var]
    float* __restrict__ output,       // [z]
    float* __restrict__ epsilon,
    int latent_dim,
    uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= latent_dim) return;

    // Box-Muller transform for normal distribution
    uint32_t u1_raw = philox2x32(idx * 2, (uint32_t)seed);
    uint32_t u2_raw = philox2x32(idx * 2 + 1, (uint32_t)seed);

    float u1 = fmaxf((float)u1_raw / 4294967296.0f, 1e-7f);
    float u2 = (float)u2_raw / 4294967296.0f;

    float mag = sqrtf(-2.0f * logf(u1));
    float eps = mag * cosf(2.0f * M_PI * u2);
    epsilon[idx] = eps;

    float mu = input[idx];
    float log_var = input[latent_dim + idx];

    // z = mu + eps * exp(0.5 * log_var)
    output[idx] = mu + eps * expf(0.5f * log_var);
}

// VAE Sampling backward
extern "C" __global__ void vae_sampling_backward(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const float* __restrict__ epsilon,
    int latent_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= latent_dim) return;

    float log_var = input[latent_dim + idx];
    float go = grad_output[idx];
    float eps = epsilon[idx];

    // grad_mu = grad_output
    grad_input[idx] = go;

    // grad_log_var = grad_output * epsilon * exp(0.5 * log_var) * 0.5
    grad_input[latent_dim + idx] = go * eps * expf(0.5f * log_var) * 0.5f;
}

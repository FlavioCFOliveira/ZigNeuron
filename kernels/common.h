// CUDA Kernels Common Header
// Shared definitions and utilities for all kernels

#ifndef CUDA_KERNELS_COMMON_H
#define CUDA_KERNELS_COMMON_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

// Alignment macros
#define ALIGN(x) __align__(x)

// Force inline
#define INLINE __forceinline__

// Vector types
struct float4 {
    float x, y, z, w;
};

struct float2 {
    float x, y;
};

// Warp-level primitives
INLINE __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

INLINE __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

INLINE __device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        val = warpReduceSum(val);
    }
    return val;
}

INLINE __device__ float blockReduceMax(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;
        val = warpReduceMax(val);
    }
    return val;
}

// Math utilities
INLINE __device__ float sigmoidf_stable(float x) {
    // Clamp to avoid overflow
    x = fminf(fmaxf(x, -20.0f), 20.0f);
    return 1.0f / (1.0f + expf(-x));
}

INLINE __device__ float geluf(float x) {
    // GELU: x * Phi(x) where Phi is CDF of standard normal
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

// Philox2x32 RNG for deterministic random numbers
INLINE __device__ uint32_t philox2x32(uint32_t counter, uint32_t key) {
    uint32_t L = counter;
    uint32_t R = key;
    const uint32_t M = 0xD2511F53;

    #pragma unroll
    for (int i = 0; i < 10; i++) {
        uint64_t prod = (uint64_t)L * M;
        uint32_t hi = (uint32_t)(prod >> 32);
        uint32_t lo = (uint32_t)prod;
        L = R ^ hi;
        R = lo;
    }
    return L;
}

// Box-Muller transform for normal distribution
INLINE __device__ float2 box_muller(uint32_t u1_raw, uint32_t u2_raw) {
    float u1 = fmaxf((float)u1_raw / 4294967296.0f, 1e-7f);
    float u2 = (float)u2_raw / 4294967296.0f;

    float mag = sqrtf(-2.0f * logf(u1));
    float z0 = mag * cosf(2.0f * M_PI * u2);
    float z1 = mag * sinf(2.0f * M_PI * u2);

    return {z0, z1};
}

// Indexing macros
#define IDX2D(i, j, stride) ((i) * (stride) + (j))
#define IDX3D(i, j, k, stride1, stride2) (((i) * (stride1) + (j)) * (stride2) + (k))
#define IDX4D(i, j, k, l, s1, s2, s3) ((((i) * (s1) + (j)) * (s2) + (k)) * (s3) + (l))

#endif // CUDA_KERNELS_COMMON_H

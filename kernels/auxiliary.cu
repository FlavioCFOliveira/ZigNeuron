// CUDA Kernels for Auxiliary Operations
// Element-wise operations, filling, scaling, and sequence operations

#include "common.h"

// Map operations
extern "C" __global__ void map_exp(const float* __restrict__ in, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(in)[i];
        float4 result;
        result.x = expf(val.x);
        result.y = expf(val.y);
        result.z = expf(val.z);
        result.w = expf(val.w);
        reinterpret_cast<float4*>(out)[i] = result;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        out[i] = expf(in[i]);
    }
}

extern "C" __global__ void map_log(const float* __restrict__ in, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float eps = 1e-10f;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(in)[i];
        float4 result;
        result.x = logf(fmaxf(val.x, eps));
        result.y = logf(fmaxf(val.y, eps));
        result.z = logf(fmaxf(val.z, eps));
        result.w = logf(fmaxf(val.w, eps));
        reinterpret_cast<float4*>(out)[i] = result;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        out[i] = logf(fmaxf(in[i], eps));
    }
}

extern "C" __global__ void map_sqrt(const float* __restrict__ in, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(in)[i];
        float4 result;
        result.x = sqrtf(val.x);
        result.y = sqrtf(val.y);
        result.z = sqrtf(val.z);
        result.w = sqrtf(val.w);
        reinterpret_cast<float4*>(out)[i] = result;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        out[i] = sqrtf(in[i]);
    }
}

extern "C" __global__ void map_abs(const float* __restrict__ in, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(in)[i];
        float4 result;
        result.x = fabsf(val.x);
        result.y = fabsf(val.y);
        result.z = fabsf(val.z);
        result.w = fabsf(val.w);
        reinterpret_cast<float4*>(out)[i] = result;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        out[i] = fabsf(in[i]);
    }
}

extern "C" __global__ void map_square(const float* __restrict__ in, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(in)[i];
        float4 result;
        result.x = val.x * val.x;
        result.y = val.y * val.y;
        result.z = val.z * val.z;
        result.w = val.w * val.w;
        reinterpret_cast<float4*>(out)[i] = result;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        out[i] = in[i] * in[i];
    }
}

extern "C" __global__ void map_inv(const float* __restrict__ in, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(in)[i];
        float4 result;
        result.x = 1.0f / val.x;
        result.y = 1.0f / val.y;
        result.z = 1.0f / val.z;
        result.w = 1.0f / val.w;
        reinterpret_cast<float4*>(out)[i] = result;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        out[i] = 1.0f / in[i];
    }
}

// Element-wise operations
extern "C" __global__ void ew_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        reinterpret_cast<float4*>(c)[i] = vc;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void ew_sub(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 vc;
        vc.x = va.x - vb.x;
        vc.y = va.y - vb.y;
        vc.z = va.z - vb.z;
        vc.w = va.w - vb.w;
        reinterpret_cast<float4*>(c)[i] = vc;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        c[i] = a[i] - b[i];
    }
}

extern "C" __global__ void ew_mul(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 vc;
        vc.x = va.x * vb.x;
        vc.y = va.y * vb.y;
        vc.z = va.z * vb.z;
        vc.w = va.w * vb.w;
        reinterpret_cast<float4*>(c)[i] = vc;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        c[i] = a[i] * b[i];
    }
}

extern "C" __global__ void ew_div(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 vc;
        vc.x = va.x / vb.x;
        vc.y = va.y / vb.y;
        vc.z = va.z / vb.z;
        vc.w = va.w / vb.w;
        reinterpret_cast<float4*>(c)[i] = vc;
    }

    int remainder_start = (size / 4) * 4;
    for (int i = idx + remainder_start; i < size; i += stride) {
        c[i] = a[i] / b[i];
    }
}

// Reverse sequence: output[t] = input[seq_len - 1 - t]
extern "C" __global__ void reverse_sequence(
    const float* __restrict__ input,
    float* __restrict__ output,
    int seq_len, int element_size
) {
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (t >= seq_len || i >= element_size) return;

    int src_t = seq_len - 1 - t;
    output[t * element_size + i] = input[src_t * element_size + i];
}

// Concatenate two buffers along the last dimension
extern "C" __global__ void concat_buffers(
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    float* __restrict__ output,
    int size1, int size2, int seq_len
) {
    int t = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = size1 + size2;

    if (t >= seq_len) return;

    if (i < size1) {
        output[t * total_size + i] = input1[t * size1 + i];
    } else if (i < total_size) {
        output[t * total_size + i] = input2[t * size2 + (i - size1)];
    }
}

// Split buffer into two along the last dimension
extern "C" __global__ void split_buffer(
    const float* __restrict__ input,
    float* __restrict__ output1,
    float* __restrict__ output2,
    int size1, int size2, int seq_len
) {
    int t = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = size1 + size2;

    if (t >= seq_len) return;

    if (i < size1) {
        output1[t * size1 + i] = input[t * total_size + i];
    } else if (i < total_size) {
        output2[t * size2 + (i - size1)] = input[t * total_size + i];
    }
}

// Fill buffer with random normal values
extern "C" __global__ void fill_random_normal(
    float* __restrict__ data,
    int size,
    float mean, float std_dev,
    uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    // Box-Muller transform
    uint32_t u1_raw = philox2x32(idx * 2, (uint32_t)seed);
    uint32_t u2_raw = philox2x32(idx * 2 + 1, (uint32_t)seed);

    float u1 = fmaxf((float)u1_raw / 4294967296.0f, 1e-7f);
    float u2 = (float)u2_raw / 4294967296.0f;

    float mag = std_dev * sqrtf(-2.0f * logf(u1));
    float z0 = mag * cosf(2.0f * M_PI * u2) + mean;

    data[idx] = z0;
}

// Transpose: output[j, i] = input[i, j]
extern "C" __global__ void transpose_2d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    __shared__ float tile[32][33];  // Pad to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Coalesced read from input
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();

    // Coalesced write to output
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

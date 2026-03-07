// CUDA Kernels for Element-wise Operations
// Includes: basic math, activation functions, element-wise binary ops

#include "common.h"

// ============================================================================
// Element-wise Unary Operations
// ============================================================================

extern "C" __global__ void relu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Vectorized processing
    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(input)[i];
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        reinterpret_cast<float4*>(output)[i] = val;
    }

    // Remainder
    for (int i = idx + (size - size % 4); i < size; i += stride) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

extern "C" __global__ void relu_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 out = reinterpret_cast<const float4*>(output)[i];
        float4 go = reinterpret_cast<const float4*>(grad_output)[i];
        float4 gi;
        gi.x = (out.x > 0.0f) ? go.x : 0.0f;
        gi.y = (out.y > 0.0f) ? go.y : 0.0f;
        gi.z = (out.z > 0.0f) ? go.z : 0.0f;
        gi.w = (out.w > 0.0f) ? go.w : 0.0f;
        reinterpret_cast<float4*>(grad_input)[i] = gi;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        grad_input[i] = (output[i] > 0.0f) ? grad_output[i] : 0.0f;
    }
}

extern "C" __global__ void sigmoid_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(input)[i];
        val.x = sigmoidf_stable(val.x);
        val.y = sigmoidf_stable(val.y);
        val.z = sigmoidf_stable(val.z);
        val.w = sigmoidf_stable(val.w);
        reinterpret_cast<float4*>(output)[i] = val;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        output[i] = sigmoidf_stable(input[i]);
    }
}

extern "C" __global__ void sigmoid_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 y = reinterpret_cast<const float4*>(output)[i];
        float4 go = reinterpret_cast<const float4*>(grad_output)[i];
        float4 gi;
        gi.x = go.x * y.x * (1.0f - y.x);
        gi.y = go.y * y.y * (1.0f - y.y);
        gi.z = go.z * y.z * (1.0f - y.z);
        gi.w = go.w * y.w * (1.0f - y.w);
        reinterpret_cast<float4*>(grad_input)[i] = gi;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        float y = output[i];
        grad_input[i] = grad_output[i] * y * (1.0f - y);
    }
}

extern "C" __global__ void tanh_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(input)[i];
        val.x = tanhf(val.x);
        val.y = tanhf(val.y);
        val.z = tanhf(val.z);
        val.w = tanhf(val.w);
        reinterpret_cast<float4*>(output)[i] = val;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

extern "C" __global__ void tanh_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 y = reinterpret_cast<const float4*>(output)[i];
        float4 go = reinterpret_cast<const float4*>(grad_output)[i];
        float4 gi;
        float4 y_sq = {y.x * y.x, y.y * y.y, y.z * y.z, y.w * y.w};
        gi.x = go.x * (1.0f - y_sq.x);
        gi.y = go.y * (1.0f - y_sq.y);
        gi.z = go.z * (1.0f - y_sq.z);
        gi.w = go.w * (1.0f - y_sq.w);
        reinterpret_cast<float4*>(grad_input)[i] = gi;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        float y = output[i];
        grad_input[i] = grad_output[i] * (1.0f - y * y);
    }
}

extern "C" __global__ void gelu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        output[i] = geluf(input[i]);
    }
}

extern "C" __global__ void gelu_backward(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (int i = idx; i < size; i += stride) {
        float x = input[i];
        float x_sq = x * x;
        float x_cubed = x_sq * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_val = tanhf(tanh_arg);
        float sech_sq = 1.0f - tanh_val * tanh_val;
        float derivative = 0.5f * (1.0f + tanh_val + x * sech_sq * sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_sq));
        grad_input[i] = grad_output[i] * derivative;
    }
}

extern "C" __global__ void linear_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        reinterpret_cast<float4*>(output)[i] = reinterpret_cast<const float4*>(input)[i];
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        output[i] = input[i];
    }
}

extern "C" __global__ void linear_backward(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        reinterpret_cast<float4*>(grad_input)[i] = reinterpret_cast<const float4*>(grad_output)[i];
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        grad_input[i] = grad_output[i];
    }
}

// ============================================================================
// Softmax (requires warp-level reduction)
// ============================================================================

extern "C" __global__ void softmax_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_samples,
    int num_classes
) {
    int sample = blockIdx.x;
    if (sample >= num_samples) return;

    const float* in_sample = input + sample * num_classes;
    float* out_sample = output + sample * num_classes;

    // Step 1: Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        max_val = fmaxf(max_val, in_sample[i]);
    }
    max_val = warpReduceMax(max_val);

    // Step 2: Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        float exp_val = expf(in_sample[i] - max_val);
        out_sample[i] = exp_val;
        sum_exp += exp_val;
    }
    sum_exp = warpReduceSum(sum_exp);

    // Step 3: Normalize
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        out_sample[i] /= sum_exp;
    }
}

extern "C" __global__ void softmax_backward(
    const float* __restrict__ activated_output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int num_samples,
    int num_classes
) {
    int sample = blockIdx.x;
    if (sample >= num_samples) return;

    const float* s_sample = activated_output + sample * num_classes;
    const float* go_sample = grad_output + sample * num_classes;
    float* gi_sample = grad_input + sample * num_classes;

    // Compute sum(grad_output * softmax)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        sum += go_sample[i] * s_sample[i];
    }
    sum = warpReduceSum(sum);

    // Compute gradient: softmax * (grad_output - sum)
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        gi_sample[i] = s_sample[i] * (go_sample[i] - sum);
    }
}

// ============================================================================
// Element-wise Binary Operations
// ============================================================================

extern "C" __global__ void ew_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 vc = {va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w};
        reinterpret_cast<float4*>(c)[i] = vc;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void ew_sub(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 vc = {va.x - vb.x, va.y - vb.y, va.z - vb.z, va.w - vb.w};
        reinterpret_cast<float4*>(c)[i] = vc;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        c[i] = a[i] - b[i];
    }
}

extern "C" __global__ void ew_mul(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 vc = {va.x * vb.x, va.y * vb.y, va.z * vb.z, va.w * vb.w};
        reinterpret_cast<float4*>(c)[i] = vc;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        c[i] = a[i] * b[i];
    }
}

extern "C" __global__ void ew_div(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 vc = {va.x / vb.x, va.y / vb.y, va.z / vb.z, va.w / vb.w};
        reinterpret_cast<float4*>(c)[i] = vc;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        c[i] = a[i] / b[i];
    }
}

// ============================================================================
// Map Operations
// ============================================================================

extern "C" __global__ void map_exp(
    const float* __restrict__ in,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 v = reinterpret_cast<const float4*>(in)[i];
        v.x = expf(v.x);
        v.y = expf(v.y);
        v.z = expf(v.z);
        v.w = expf(v.w);
        reinterpret_cast<float4*>(out)[i] = v;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        out[i] = expf(in[i]);
    }
}

extern "C" __global__ void map_log(
    const float* __restrict__ in,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 v = reinterpret_cast<const float4*>(in)[i];
        v.x = logf(fmaxf(v.x, 1e-10f));
        v.y = logf(fmaxf(v.y, 1e-10f));
        v.z = logf(fmaxf(v.z, 1e-10f));
        v.w = logf(fmaxf(v.w, 1e-10f));
        reinterpret_cast<float4*>(out)[i] = v;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        out[i] = logf(fmaxf(in[i], 1e-10f));
    }
}

extern "C" __global__ void map_sqrt(
    const float* __restrict__ in,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 v = reinterpret_cast<const float4*>(in)[i];
        v.x = sqrtf(v.x);
        v.y = sqrtf(v.y);
        v.z = sqrtf(v.z);
        v.w = sqrtf(v.w);
        reinterpret_cast<float4*>(out)[i] = v;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        out[i] = sqrtf(in[i]);
    }
}

extern "C" __global__ void map_abs(
    const float* __restrict__ in,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 v = reinterpret_cast<const float4*>(in)[i];
        v.x = fabsf(v.x);
        v.y = fabsf(v.y);
        v.z = fabsf(v.z);
        v.w = fabsf(v.w);
        reinterpret_cast<float4*>(out)[i] = v;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        out[i] = fabsf(in[i]);
    }
}

extern "C" __global__ void map_square(
    const float* __restrict__ in,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 v = reinterpret_cast<const float4*>(in)[i];
        v.x = v.x * v.x;
        v.y = v.y * v.y;
        v.z = v.z * v.z;
        v.w = v.w * v.w;
        reinterpret_cast<float4*>(out)[i] = v;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        float v = in[i];
        out[i] = v * v;
    }
}

extern "C" __global__ void map_inv(
    const float* __restrict__ in,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size / 4; i += stride) {
        float4 v = reinterpret_cast<const float4*>(in)[i];
        v.x = 1.0f / v.x;
        v.y = 1.0f / v.y;
        v.z = 1.0f / v.z;
        v.w = 1.0f / v.w;
        reinterpret_cast<float4*>(out)[i] = v;
    }

    for (int i = idx + (size - size % 4); i < size; i += stride) {
        out[i] = 1.0f / in[i];
    }
}

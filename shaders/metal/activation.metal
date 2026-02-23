// Metal shaders for activation functions
// Includes forward and backward passes
#include <metal_stdlib>
using namespace metal;

// ReLU forward: max(0, x) - Vectorized
kernel void relu_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* in4 = (device const float4*)(input + idx);
        device float4* out4 = (device float4*)(output + idx);
        *out4 = max(0.0f, *in4);
    } else {
        for (uint i = idx; i < size; i++) {
            output[i] = max(0.0f, input[i]);
        }
    }
}

// ReLU backward - Vectorized
kernel void relu_backward(
    device const float* output [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* out4 = (device const float4*)(output + idx);
        device const float4* g_out4 = (device const float4*)(grad_output + idx);
        device float4* g_in4 = (device float4*)(grad_input + idx);

        float4 y = *out4;
        float4 go = *g_out4;
        *g_in4 = select(float4(0.0f), go, y > 0.0f);
    } else {
        for (uint i = idx; i < size; i++) {
            grad_input[i] = (output[i] > 0.0f) ? grad_output[i] : 0.0f;
        }
    }
}

// Sigmoid forward - Vectorized
kernel void sigmoid_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* in4 = (device const float4*)(input + idx);
        device float4* out4 = (device float4*)(output + idx);
        *out4 = 1.0f / (1.0f + exp(-(*in4)));
    } else {
        for (uint i = idx; i < size; i++) {
            output[i] = 1.0f / (1.0f + exp(-input[i]));
        }
    }
}

// Sigmoid backward - Vectorized
kernel void sigmoid_backward(
    device const float* output [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* out4 = (device const float4*)(output + idx);
        device const float4* g_out4 = (device const float4*)(grad_output + idx);
        device float4* g_in4 = (device float4*)(grad_input + idx);

        float4 y = *out4;
        float4 go = *g_out4;
        *g_in4 = go * y * (1.0f - y);
    } else {
        for (uint i = idx; i < size; i++) {
            float y = output[i];
            grad_input[i] = grad_output[i] * y * (1.0f - y);
        }
    }
}

// Tanh forward - Vectorized
kernel void tanh_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* in4 = (device const float4*)(input + idx);
        device float4* out4 = (device float4*)(output + idx);
        *out4 = tanh(*in4);
    } else {
        for (uint i = idx; i < size; i++) {
            output[i] = tanh(input[i]);
        }
    }
}

// Tanh backward - Vectorized
kernel void tanh_backward(
    device const float* output [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* out4 = (device const float4*)(output + idx);
        device const float4* g_out4 = (device const float4*)(grad_output + idx);
        device float4* g_in4 = (device float4*)(grad_input + idx);

        float4 y = *out4;
        float4 go = *g_out4;
        *g_in4 = go * (1.0f - y * y);
    } else {
        for (uint i = idx; i < size; i++) {
            float y = output[i];
            grad_input[i] = grad_output[i] * (1.0f - y * y);
        }
    }
}

// Linear forward - Vectorized
kernel void linear_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* in4 = (device const float4*)(input + idx);
        device float4* out4 = (device float4*)(output + idx);
        *out4 = *in4;
    } else {
        for (uint i = idx; i < size; i++) {
            output[i] = input[i];
        }
    }
}

// Linear backward - Vectorized
kernel void linear_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* g_out4 = (device const float4*)(grad_output + idx);
        device float4* g_in4 = (device float4*)(grad_input + idx);
        *g_in4 = *g_out4;
    } else {
        for (uint i = idx; i < size; i++) {
            grad_input[i] = grad_output[i];
        }
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

// Softmax forward - Optimized with threadgroup reduction and SIMD
kernel void softmax_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_samples [[buffer(2)]],
    constant uint& num_classes [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]])
{
    uint sample = gid.y;
    if (sample >= num_samples) return;

    device const float* in_sample = input + sample * num_classes;
    device float* out_sample = output + sample * num_classes;

    // 1. Find max value for stability
    float thread_max = -INFINITY;
    for (uint c = tid; c < num_classes; c += tg_size.x) {
        thread_max = max(thread_max, in_sample[c]);
    }
    float max_val = simd_max(thread_max);

    // 2. Compute sum of exponentials
    float thread_sum = 0.0f;
    for (uint c = tid; c < num_classes; c += tg_size.x) {
        thread_sum += exp(in_sample[c] - max_val);
    }
    float sum_exp = simd_sum(thread_sum);

    // 3. Compute softmax
    for (uint c = tid; c < num_classes; c += tg_size.x) {
        out_sample[c] = exp(in_sample[c] - max_val) / sum_exp;
    }
}

// Softmax backward - Optimized
kernel void softmax_backward(
    device const float* activated_output [[buffer(0)]], // Use activated output directly
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& num_samples [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]])
{
    uint sample = gid.y;
    if (sample >= num_samples) return;

    device const float* s_sample = activated_output + sample * num_classes;
    device const float* go_sample = grad_output + sample * num_classes;
    device float* gi_sample = grad_input + sample * num_classes;

    // Jacobian-vector product: gi_i = s_i * (go_i - sum(go_j * s_j))
    float thread_sum = 0.0f;
    for (uint j = tid; j < num_classes; j += tg_size.x) {
        thread_sum += go_sample[j] * s_sample[j];
    }
    float sum_grad_s = simd_sum(thread_sum);

    for (uint i = tid; i < num_classes; i += tg_size.x) {
        gi_sample[i] = s_sample[i] * (go_sample[i] - sum_grad_s);
    }
}

// Map functions - Vectorized
kernel void map_exp(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& size [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* in4 = (device const float4*)(in + idx);
        device float4* out4 = (device float4*)(out + idx);
        *out4 = exp(*in4);
    } else {
        for (uint i = idx; i < size; i++) out[i] = exp(in[i]);
    }
}
kernel void map_log(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& size [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* in4 = (device const float4*)(in + idx);
        device float4* out4 = (device float4*)(out + idx);
        *out4 = log(max(*in4, 1e-10f));
    } else {
        for (uint i = idx; i < size; i++) out[i] = log(max(in[i], 1e-10f));
    }
}
kernel void map_sqrt(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& size [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* in4 = (device const float4*)(in + idx);
        device float4* out4 = (device float4*)(out + idx);
        *out4 = sqrt(*in4);
    } else {
        for (uint i = idx; i < size; i++) out[i] = sqrt(in[i]);
    }
}
kernel void map_abs(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& size [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* in4 = (device const float4*)(in + idx);
        device float4* out4 = (device float4*)(out + idx);
        *out4 = abs(*in4);
    } else {
        for (uint i = idx; i < size; i++) out[i] = abs(in[i]);
    }
}
kernel void map_square(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& size [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* in4 = (device const float4*)(in + idx);
        device float4* out4 = (device float4*)(out + idx);
        *out4 = (*in4) * (*in4);
    } else {
        for (uint i = idx; i < size; i++) out[i] = in[i] * in[i];
    }
}
kernel void map_inv(device const float* in [[buffer(0)]], device float* out [[buffer(1)]], constant uint& size [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* in4 = (device const float4*)(in + idx);
        device float4* out4 = (device float4*)(out + idx);
        *out4 = 1.0f / (*in4);
    } else {
        for (uint i = idx; i < size; i++) out[i] = 1.0f / in[i];
    }
}

// Element-wise functions - Vectorized
kernel void ew_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]], constant uint& size [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* a4 = (device const float4*)(a + idx);
        device const float4* b4 = (device const float4*)(b + idx);
        device float4* c4 = (device float4*)(c + idx);
        *c4 = *a4 + *b4;
    } else {
        for (uint i = idx; i < size; i++) c[i] = a[i] + b[i];
    }
}
kernel void ew_sub(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]], constant uint& size [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* a4 = (device const float4*)(a + idx);
        device const float4* b4 = (device const float4*)(b + idx);
        device float4* c4 = (device float4*)(c + idx);
        *c4 = *a4 - *b4;
    } else {
        for (uint i = idx; i < size; i++) c[i] = a[i] - b[i];
    }
}
kernel void ew_mul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]], constant uint& size [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* a4 = (device const float4*)(a + idx);
        device const float4* b4 = (device const float4*)(b + idx);
        device float4* c4 = (device float4*)(c + idx);
        *c4 = *a4 * *b4;
    } else {
        for (uint i = idx; i < size; i++) c[i] = a[i] * b[i];
    }
}
kernel void ew_div(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]], constant uint& size [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    uint idx = gid * 4;
    if (idx + 3 < size) {
        device const float4* a4 = (device const float4*)(a + idx);
        device const float4* b4 = (device const float4*)(b + idx);
        device float4* c4 = (device float4*)(c + idx);
        *c4 = *a4 / *b4;
    } else {
        for (uint i = idx; i < size; i++) c[i] = a[i] / b[i];
    }
}

// Random normal (using Philox counter-based RNG)
uint32_t philox2x32(uint32_t counter, uint32_t key) {
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

kernel void fill_random_normal(
    device float* data [[buffer(0)]],
    constant float& mean [[buffer(1)]],
    constant float& std_dev [[buffer(2)]],
    constant uint64_t& seed [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        // Box-Muller transform
        uint32_t u1_raw = philox2x32(gid * 2, (uint32_t)seed);
        uint32_t u2_raw = philox2x32(gid * 2 + 1, (uint32_t)seed);

        float u1 = (float)u1_raw / 4294967296.0f;
        float u2 = (float)u2_raw / 4294967296.0f;

        // Ensure u1 is not zero for log
        u1 = max(u1, 1e-7f);

        float mag = std_dev * sqrt(-2.0f * log(u1));
        float z0 = mag * cos(2.0f * M_PI_F * u2) + mean;

        data[gid] = z0;
    }
}

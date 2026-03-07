// Metal shaders for attention layers
#include <metal_stdlib>
using namespace metal;

// Scaled Dot-Product Attention: Q*K^T / sqrt(d)
// Q: [seq_len, d_k], K: [seq_len, d_k], V: [seq_len, d_v]
// Output: softmax(Q*K^T / sqrt(d_k)) * V
//
// Uses device memory for attention scores to avoid stack overflow
// Supports arbitrary sequence lengths (not limited by stack size)
kernel void attention_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* attention_scores [[buffer(4)]],  // Temporary buffer for scores [seq_len * seq_len]
    constant uint& seq_len [[buffer(5)]],
    constant uint& d_k [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]])
{
    uint i = gid.y; // seq_idx (query position)
    uint j = gid.x; // feature_idx (value dimension)

    if (i >= seq_len || j >= d_k) return;

    // Each thread computes one row of attention (for query position i)
    // 1. Compute attention scores for query position i against all keys
    float max_score = -INFINITY;

    for (uint m = tid; m < seq_len; m += tg_size.x) {
        float score = 0.0f;
        for (uint k = 0; k < d_k; k++) {
            score += Q[i * d_k + k] * K[m * d_k + k];
        }
        score *= scale;
        attention_scores[i * seq_len + m] = score;
        max_score = max(max_score, score);
    }

    // Reduce max_score across threadgroup
    max_score = simd_max(max_score);

    // 2. Compute exp(score - max_score) and sum
    float sum_exp = 0.0f;
    for (uint m = tid; m < seq_len; m += tg_size.x) {
        float exp_val = exp(attention_scores[i * seq_len + m] - max_score);
        attention_scores[i * seq_len + m] = exp_val;
        sum_exp += exp_val;
    }

    // Reduce sum_exp across threadgroup
    sum_exp = simd_sum(sum_exp);

    // 3. Multiply by V (only for this thread's output position)
    float res = 0.0f;
    for (uint m = 0; m < seq_len; m++) {
        float attn_weight = attention_scores[i * seq_len + m] / sum_exp;
        res += attn_weight * V[m * d_k + j];
    }

    output[i * d_k + j] = res;
}

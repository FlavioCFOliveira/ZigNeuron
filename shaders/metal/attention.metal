// Metal shaders for attention layers
#include <metal_stdlib>
using namespace metal;

// Scaled Dot-Product Attention: Q*K^T / sqrt(d)
// Q: [seq_len, d_k], K: [seq_len, d_k], V: [seq_len, d_v]
// Output: softmax(Q*K^T / sqrt(d_k)) * V
kernel void attention_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& d_k [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]])
{
    // Simplified: each thread handles one element of the output
    // In a real implementation, we'd use shared memory for Q, K, V tiles
    uint i = gid.y; // seq_idx
    uint j = gid.x; // feature_idx

    if (i < seq_len && j < d_k) {
        // 1. Compute Q_i * K^T_m for all m (attention scores)
        float thread_scores[1024]; // Max seq_len for now
        float max_score = -INFINITY;

        for (uint m = 0; m < seq_len; m++) {
            float score = 0.0f;
            for (uint k = 0; k < d_k; k++) {
                score += Q[i * d_k + k] * K[m * d_k + k];
            }
            score *= scale;
            thread_scores[m] = score;
            max_score = max(max_score, score);
        }

        // 2. Softmax
        float sum_exp = 0.0f;
        for (uint m = 0; m < seq_len; m++) {
            thread_scores[m] = exp(thread_scores[m] - max_score);
            sum_exp += thread_scores[m];
        }

        // 3. Multiply by V
        float res = 0.0f;
        for (uint m = 0; m < seq_len; m++) {
            res += (thread_scores[m] / sum_exp) * V[m * d_k + j];
        }

        output[i * d_k + j] = res;
    }
}

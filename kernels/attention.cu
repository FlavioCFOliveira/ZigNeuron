// CUDA Kernels for Attention Mechanism
// Scaled dot-product attention implementation

#include "common.h"

// Scaled dot-product attention forward
// Q: [batch, num_heads, seq_len, head_dim]
// K: [batch, num_heads, seq_len, head_dim]
// V: [batch, num_heads, seq_len, head_dim]
// Output: [batch, num_heads, seq_len, head_dim]
extern "C" __global__ void attention_forward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    float* __restrict__ workspace,  // Temporary storage for attention scores
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale
) {
    int batch = blockIdx.z / num_heads;
    int head = blockIdx.z % num_heads;
    int q_pos = blockIdx.y;

    if (batch >= batch_size || q_pos >= seq_len) return;

    // Pointers to this head's data
    const float* Q_head = Q + ((batch * num_heads + head) * seq_len + q_pos) * head_dim;
    const float* K_head = K + (batch * num_heads + head) * seq_len * head_dim;
    const float* V_head = V + (batch * num_heads + head) * seq_len * head_dim;
    float* scores = workspace + ((batch * num_heads + head) * seq_len + q_pos) * seq_len;
    float* out_head = output + ((batch * num_heads + head) * seq_len + q_pos) * head_dim;

    // Step 1: Compute Q*K^T for this query position
    float max_score = -INFINITY;

    for (int k_pos = threadIdx.x; k_pos < seq_len; k_pos += blockDim.x) {
        float dot = 0.0f;
        const float* K_k = K_head + k_pos * head_dim;

        for (int d = 0; d < head_dim; d++) {
            dot += Q_head[d] * K_k[d];
        }

        float score = dot * scale;
        scores[k_pos] = score;
        max_score = fmaxf(max_score, score);
    }

    // Warp reduce max
    max_score = warpReduceMax(max_score);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) shared[wid] = max_score;
    __syncthreads();

    if (wid == 0) {
        max_score = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;
        max_score = warpReduceMax(max_score);
    }
    max_score = __shfl_sync(0xFFFFFFFF, max_score, 0);

    // Step 2: Compute exp and sum
    float sum_exp = 0.0f;
    for (int k_pos = threadIdx.x; k_pos < seq_len; k_pos += blockDim.x) {
        float exp_val = expf(scores[k_pos] - max_score);
        scores[k_pos] = exp_val;
        sum_exp += exp_val;
    }

    sum_exp = warpReduceSum(sum_exp);

    if (lane == 0) shared[wid] = sum_exp;
    __syncthreads();

    if (wid == 0) {
        sum_exp = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        sum_exp = warpReduceSum(sum_exp);
    }
    sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

    // Normalize scores
    for (int k_pos = threadIdx.x; k_pos < seq_len; k_pos += blockDim.x) {
        scores[k_pos] /= sum_exp;
    }
    __syncthreads();

    // Step 3: Compute weighted sum with V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int k_pos = 0; k_pos < seq_len; k_pos++) {
            sum += scores[k_pos] * V_head[k_pos * head_dim + d];
        }
        out_head[d] = sum;
    }
}

// Multi-head attention forward (fused QKV projection + attention)
// This is a simplified version - production code would use fused kernels
extern "C" __global__ void multihead_attention_forward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ W_q,
    const float* __restrict__ W_k,
    const float* __restrict__ W_v,
    const float* __restrict__ W_o,
    const float* __restrict__ b_q,
    const float* __restrict__ b_k,
    const float* __restrict__ b_v,
    const float* __restrict__ b_o,
    float* __restrict__ output,
    float* __restrict__ workspace,
    int batch_size, int seq_len, int num_heads, int head_dim, int model_dim
) {
    // Implementation would include:
    // 1. QKV linear projections
    // 2. Reshape to [batch, num_heads, seq_len, head_dim]
    // 3. Scaled dot-product attention
    // 4. Concatenate heads
    // 5. Output projection

    // This is a placeholder - full implementation requires careful memory management
}

// Attention with masking (for autoregressive models)
extern "C" __global__ void attention_masked_forward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ mask,  // Causal mask [seq_len, seq_len]
    float* __restrict__ output,
    float* __restrict__ workspace,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale
) {
    int batch = blockIdx.z / num_heads;
    int head = blockIdx.z % num_heads;
    int q_pos = blockIdx.y;

    if (batch >= batch_size || q_pos >= seq_len) return;

    const float* Q_head = Q + ((batch * num_heads + head) * seq_len + q_pos) * head_dim;
    const float* K_head = K + (batch * num_heads + head) * seq_len * head_dim;
    const float* V_head = V + (batch * num_heads + head) * seq_len * head_dim;
    float* scores = workspace + ((batch * num_heads + head) * seq_len + q_pos) * seq_len;
    float* out_head = output + ((batch * num_heads + head) * seq_len + q_pos) * head_dim;

    // Compute Q*K^T with causal masking
    float max_score = -INFINITY;

    for (int k_pos = threadIdx.x; k_pos < seq_len; k_pos += blockDim.x) {
        // Apply causal mask
        if (k_pos > q_pos) {
            scores[k_pos] = -INFINITY;
            continue;
        }

        float dot = 0.0f;
        const float* K_k = K_head + k_pos * head_dim;

        for (int d = 0; d < head_dim; d++) {
            dot += Q_head[d] * K_k[d];
        }

        float score = dot * scale;
        scores[k_pos] = score;
        max_score = fmaxf(max_score, score);
    }

    // Warp reduce max
    max_score = warpReduceMax(max_score);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) shared[wid] = max_score;
    __syncthreads();

    if (wid == 0) {
        max_score = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;
        max_score = warpReduceMax(max_score);
    }
    max_score = __shfl_sync(0xFFFFFFFF, max_score, 0);

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int k_pos = threadIdx.x; k_pos < seq_len; k_pos += blockDim.x) {
        float exp_val = expf(scores[k_pos] - max_score);
        // Causal mask: positions > q_pos should have been set to -inf
        if (scores[k_pos] == -INFINITY) exp_val = 0.0f;
        scores[k_pos] = exp_val;
        sum_exp += exp_val;
    }

    sum_exp = warpReduceSum(sum_exp);

    if (lane == 0) shared[wid] = sum_exp;
    __syncthreads();

    if (wid == 0) {
        sum_exp = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        sum_exp = warpReduceSum(sum_exp);
    }
    sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

    // Normalize
    for (int k_pos = threadIdx.x; k_pos < seq_len; k_pos += blockDim.x) {
        if (sum_exp > 0.0f) {
            scores[k_pos] /= sum_exp;
        }
    }
    __syncthreads();

    // Compute weighted sum with V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int k_pos = 0; k_pos <= q_pos; k_pos++) {  // Only sum valid positions
            sum += scores[k_pos] * V_head[k_pos * head_dim + d];
        }
        out_head[d] = sum;
    }
}

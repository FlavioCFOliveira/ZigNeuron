/// CUDA Kernel Collection for ZigNeuron
/// This file contains all CUDA kernels for neural network operations
/// Compile with: nvcc -ptx -arch=sm_60 -o kernels.ptx kernels.cu

// =============================================================================
// Helper Functions
// =============================================================================

__device__ inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ inline float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float sigmoid_derivative_from_output(float y) {
    return y * (1.0f - y);
}

__device__ inline float tanh_derivative_from_output(float y) {
    return 1.0f - y * y;
}

__device__ inline float gelu(float x) {
    // Approximation: x * sigmoid(1.702 * x)
    return x * sigmoid(1.702f * x);
}

// =============================================================================
// Matrix Operations
// =============================================================================

/// Matrix multiplication: C = A * B + (accumulate ? C : 0)
/// A: M x K, B: K x N, C: M x N
extern "C" __global__ void matmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int accumulate
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        const int idx = row * N + col;
        if (accumulate) {
            C[idx] += sum;
        } else {
            C[idx] = sum;
        }
    }
}

/// Matrix multiplication with transposed A: C = A^T * B
extern "C" __global__ void matmul_transpose_a(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int accumulate
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[k * M + row] * B[k * N + col];
        }
        const int idx = row * N + col;
        if (accumulate) {
            C[idx] += sum;
        } else {
            C[idx] = sum;
        }
    }
}

/// Matrix multiplication with transposed B: C = A * B^T
extern "C" __global__ void matmul_transpose_b(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int accumulate
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        const int idx = row * N + col;
        if (accumulate) {
            C[idx] += sum;
        } else {
            C[idx] = sum;
        }
    }
}

/// Batched matrix multiplication
extern "C" __global__ void matmul_batch(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size, int N, int K,
    int accumulate
) {
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch < batch_size && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[batch * K + k] * B[(batch * K + k) * N + col];
        }
        const int idx = batch * N + col;
        if (accumulate) {
            C[idx] += sum;
        } else {
            C[idx] = sum;
        }
    }
}

// =============================================================================
// Element-wise Operations
// =============================================================================

#define EW_KERNEL(name, op) \
    extern "C" __global__ void name(\
        const float* __restrict__ a,\
        const float* __restrict__ b,\
        float* __restrict__ c,\
        int n\
    ) {\
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;\
        if (idx < n) {\
            c[idx] = op;\
        }\
    }

EW_KERNEL(ew_add, a[idx] + b[idx])
EW_KERNEL(ew_sub, a[idx] - b[idx])
EW_KERNEL(ew_mul, a[idx] * b[idx])
EW_KERNEL(ew_div, a[idx] / b[idx])

/// Scale buffer
extern "C" __global__ void scale_buffer(
    const float* __restrict__ input,
    float scale,
    float* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale;
    }
}

/// Fill with constant
extern "C" __global__ void fill_constant(
    float* __restrict__ buffer,
    float value,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        buffer[idx] = value;
    }
}

// =============================================================================
// Map Operations
// =============================================================================

#define MAP_KERNEL(name, expr) \
    extern "C" __global__ void name(\
        const float* __restrict__ input,\
        float* __restrict__ output,\
        int n\
    ) {\
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;\
        if (idx < n) {\
            const float x = input[idx];\
            output[idx] = expr;\
        }\
    }

MAP_KERNEL(map_exp, expf(x))
MAP_KERNEL(map_log, logf(x))
MAP_KERNEL(map_sqrt, sqrtf(x))
MAP_KERNEL(map_abs, fabsf(x))
MAP_KERNEL(map_square, x * x)
MAP_KERNEL(map_inv, 1.0f / x)

// =============================================================================
// Activation Functions
// =============================================================================

extern "C" __global__ void relu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = relu(input[idx]);
    }
}

extern "C" __global__ void relu_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * relu_derivative(output[idx]);
    }
}

extern "C" __global__ void sigmoid_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sigmoid(input[idx]);
    }
}

extern "C" __global__ void sigmoid_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * sigmoid_derivative_from_output(output[idx]);
    }
}

extern "C" __global__ void tanh_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

extern "C" __global__ void tanh_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = grad_output[idx] * tanh_derivative_from_output(output[idx]);
    }
}

/// Softmax forward with numerical stability
extern "C" __global__ void softmax_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int features
) {
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size) {
        const int offset = batch * features;

        // Find max for numerical stability
        float max_val = input[offset];
        for (int i = 1; i < features; ++i) {
            max_val = fmaxf(max_val, input[offset + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < features; ++i) {
            const float exp_val = expf(input[offset + i] - max_val);
            output[offset + i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        const float inv_sum = 1.0f / sum;
        for (int i = 0; i < features; ++i) {
            output[offset + i] *= inv_sum;
        }
    }
}

extern "C" __global__ void softmax_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int batch_size, int features
) {
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size) {
        const int offset = batch * features;

        // Compute dot product: sum(grad_output * output)
        float dot = 0.0f;
        for (int i = 0; i < features; ++i) {
            dot += grad_output[offset + i] * output[offset + i];
        }

        // Gradient: output * (grad_output - dot)
        for (int i = 0; i < features; ++i) {
            grad_input[offset + i] = output[offset + i] * (grad_output[offset + i] - dot);
        }
    }
}

extern "C" __global__ void linear_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

extern "C" __global__ void linear_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = grad_output[idx];
    }
}

extern "C" __global__ void gelu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gelu(input[idx]);
    }
}

// =============================================================================
// Loss Functions
// =============================================================================

/// MSE loss gradient: grad = 2 * (output - target) / n
extern "C" __global__ void mse_backward(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ grad,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] = 2.0f * (output[idx] - target[idx]) / n;
    }
}

/// Cross-entropy loss gradient for softmax output
extern "C" __global__ void cross_entropy_backward(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ grad,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] = output[idx] - target[idx];
    }
}

/// Binary cross-entropy loss gradient
extern "C" __global__ void binary_cross_entropy_backward(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ grad,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Clip to avoid log(0)
        const float pred = fmaxf(0.0001f, fminf(0.9999f, output[idx]));
        grad[idx] = (pred - target[idx]) / (pred * (1.0f - pred));
    }
}

// =============================================================================
// Optimizers
// =============================================================================

/// SGD update: weights -= lr * (gradients + weight_decay * weights)
extern "C" __global__ void sgd_update(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float learning_rate,
    float weight_decay,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float grad = gradients[idx] + weight_decay * weights[idx];
        weights[idx] -= learning_rate * grad;
    }
}

/// SGD bias update (no weight decay): bias -= lr * gradients
extern "C" __global__ void sgd_update_bias(
    float* __restrict__ bias,
    const float* __restrict__ gradients,
    float learning_rate,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bias[idx] -= learning_rate * gradients[idx];
    }
}

/// Accumulate bias gradients
extern "C" __global__ void accumulate_bias(
    float* __restrict__ grad_bias,
    const float* __restrict__ grad_after_act,
    int batch_size, int features
) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature < features) {
        float sum = 0.0f;
        for (int batch = 0; batch < batch_size; ++batch) {
            sum += grad_after_act[batch * features + feature];
        }
        grad_bias[feature] = sum;
    }
}

/// Adam optimizer update
extern "C" __global__ void adam_update(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float* __restrict__ m,
    float* __restrict__ v,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    unsigned int t,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float grad = gradients[idx];

        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;

        // Update biased second moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

        // Compute bias-corrected estimates
        const float m_hat = m[idx] / (1.0f - powf(beta1, t));
        const float v_hat = v[idx] / (1.0f - powf(beta2, t));

        // Update weights
        weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

/// RMSprop optimizer update
extern "C" __global__ void rmsprop_update(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float* __restrict__ g_avg,
    float learning_rate,
    float rho,
    float epsilon,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float grad = gradients[idx];
        g_avg[idx] = rho * g_avg[idx] + (1.0f - rho) * grad * grad;
        weights[idx] -= learning_rate * grad / (sqrtf(g_avg[idx]) + epsilon);
    }
}

// =============================================================================
// Normalization
// =============================================================================

/// LayerNorm forward
extern "C" __global__ void layernorm_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float epsilon,
    int batch_size, int features
) {
    const int batch = blockIdx.x;

    if (batch < batch_size) {
        const int offset = batch * features;

        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < features; ++i) {
            mean += input[offset + i];
        }
        mean /= features;

        // Compute variance
        float var = 0.0f;
        for (int i = 0; i < features; ++i) {
            const float diff = input[offset + i] - mean;
            var += diff * diff;
        }
        var /= features;

        // Normalize and scale
        const float inv_std = 1.0f / sqrtf(var + epsilon);
        for (int i = 0; i < features; ++i) {
            const float normalized = (input[offset + i] - mean) * inv_std;
            output[offset + i] = gamma[i] * normalized + beta[i];
        }
    }
}

/// BatchNorm forward (training mode)
extern "C" __global__ void batchnorm_forward_training(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    float epsilon,
    float momentum,
    int batch_size, int features
) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature < features) {
        // Compute mean over batch
        float mean = 0.0f;
        for (int batch = 0; batch < batch_size; ++batch) {
            mean += input[batch * features + feature];
        }
        mean /= batch_size;

        // Compute variance
        float var = 0.0f;
        for (int batch = 0; batch < batch_size; ++batch) {
            const float diff = input[batch * features + feature] - mean;
            var += diff * diff;
        }
        var /= batch_size;

        // Update running statistics
        running_mean[feature] = momentum * running_mean[feature] + (1.0f - momentum) * mean;
        running_var[feature] = momentum * running_var[feature] + (1.0f - momentum) * var;

        // Normalize and scale
        const float inv_std = 1.0f / sqrtf(var + epsilon);
        for (int batch = 0; batch < batch_size; ++batch) {
            const int idx = batch * features + feature;
            const float normalized = (input[idx] - mean) * inv_std;
            output[idx] = gamma[feature] * normalized + beta[feature];
        }
    }
}

// =============================================================================
// Convolution
// =============================================================================

/// 1D Convolution forward
extern "C" __global__ void conv1d_forward(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int in_len, int out_len
) {
    const int batch = blockIdx.x;
    const int out_ch = blockIdx.y;
    const int out_pos = blockIdx.z * blockDim.x + threadIdx.x;

    if (batch >= batch_size || out_ch >= out_channels || out_pos >= out_len) return;

    float sum = bias[out_ch];

    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int k = 0; k < kernel_size; ++k) {
            const int in_pos = out_pos + k;
            if (in_pos < in_len) {
                const float in_val = input[((batch * in_channels) + in_ch) * in_len + in_pos];
                const float w_val = weights[((out_ch * in_channels) + in_ch) * kernel_size + k];
                sum += in_val * w_val;
            }
        }
    }

    output[((batch * out_channels) + out_ch) * out_len + out_pos] = sum;
}

// =============================================================================
// Dropout
// =============================================================================

/// Dropout forward with XOR-shift random number generation
extern "C" __global__ void dropout_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ mask,
    float rate,
    float scaling_factor,
    unsigned long long seed,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // XOR-shift random number generator
        unsigned long long x = seed + idx;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;

        const float rand_val = (float)(x & 0xFFFFFFFF) / 4294967296.0f;
        mask[idx] = (rand_val >= rate) ? 1.0f : 0.0f;
        output[idx] = input[idx] * mask[idx] * scaling_factor;
    }
}

// =============================================================================
// VAE
// =============================================================================

/// VAE reparameterization trick: z = mu + sigma * epsilon
extern "C" __global__ void vae_sampling_forward(
    const float* __restrict__ mu_sigma,  // Concatenated mu and log_var
    float* __restrict__ output,
    const float* __restrict__ epsilon,
    unsigned long long seed,
    int batch_size, int latent_dim
) {
    const int batch = blockIdx.x;
    const int dim = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch >= batch_size || dim >= latent_dim) return;

    const int mu_idx = batch * latent_dim + dim;
    const int log_var_idx = batch_size * latent_dim + mu_idx;

    const float mu = mu_sigma[mu_idx];
    const float log_var = mu_sigma[log_var_idx];
    const float sigma = expf(0.5f * log_var);

    output[mu_idx] = mu + sigma * epsilon[mu_idx];
}

// =============================================================================
// Attention
// =============================================================================

/// Scaled dot-product attention forward
/// Q, K, V: (seq_len, d_k), Output: (seq_len, d_k)
extern "C" __global__ void attention_forward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    float* __restrict__ attention_weights,  // Optional: for debugging/analysis
    int seq_len, int d_k,
    float scaling_factor
) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    if (row >= seq_len || col >= d_k) return;

    // Compute Q * K^T for this row
    // Store attention scores in shared memory
    extern __shared__ float scores[];

    // Each thread computes one attention score
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        float score = 0.0f;
        for (int k = 0; k < d_k; ++k) {
            score += Q[row * d_k + k] * K[j * d_k + k];
        }
        scores[j] = score * scaling_factor;
    }
    __syncthreads();

    // Softmax over attention scores
    float max_score = scores[0];
    for (int i = 1; i < seq_len; ++i) {
        max_score = fmaxf(max_score, scores[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }

    const float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < seq_len; ++i) {
        scores[i] *= inv_sum;
    }

    // Compute weighted sum of V
    float result = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        result += scores[i] * V[i * d_k + col];
    }

    output[row * d_k + col] = result;
}

// =============================================================================
// Recurrent Layers
// =============================================================================

/// LSTM forward step
/// Input: input_size, Hidden: hidden_size
/// Gates: input, forget, cell, output (concatenated in weights)
extern "C" __global__ void lstm_forward_step(
    const float* __restrict__ input,
    const float* __restrict__ hidden_prev,
    const float* __restrict__ cell_prev,
    const float* __restrict__ weights_input,
    const float* __restrict__ weights_hidden,
    const float* __restrict__ bias,
    float* __restrict__ hidden_out,
    float* __restrict__ cell_out,
    int batch_size, int input_size, int hidden_size
) {
    const int batch = blockIdx.x;
    const int hidden_idx = threadIdx.x;

    if (batch >= batch_size || hidden_idx >= hidden_size) return;

    // Compute gate inputs
    float i_gate = bias[hidden_idx];
    float f_gate = bias[hidden_size + hidden_idx];
    float c_gate = bias[2 * hidden_size + hidden_idx];
    float o_gate = bias[3 * hidden_size + hidden_idx];

    // Add input contribution
    for (int i = 0; i < input_size; ++i) {
        const float in_val = input[batch * input_size + i];
        i_gate += in_val * weights_input[i * 4 * hidden_size + hidden_idx];
        f_gate += in_val * weights_input[i * 4 * hidden_size + hidden_size + hidden_idx];
        c_gate += in_val * weights_input[i * 4 * hidden_size + 2 * hidden_size + hidden_idx];
        o_gate += in_val * weights_input[i * 4 * hidden_size + 3 * hidden_size + hidden_idx];
    }

    // Add hidden contribution
    for (int h = 0; h < hidden_size; ++h) {
        const float h_val = hidden_prev[batch * hidden_size + h];
        i_gate += h_val * weights_hidden[h * 4 * hidden_size + hidden_idx];
        f_gate += h_val * weights_hidden[h * 4 * hidden_size + hidden_size + hidden_idx];
        c_gate += h_val * weights_hidden[h * 4 * hidden_size + 2 * hidden_size + hidden_idx];
        o_gate += h_val * weights_hidden[h * 4 * hidden_size + 3 * hidden_size + hidden_idx];
    }

    // Apply activations
    i_gate = sigmoid(i_gate);
    f_gate = sigmoid(f_gate);
    c_gate = tanhf(c_gate);
    o_gate = sigmoid(o_gate);

    // Update cell state
    const int cell_idx = batch * hidden_size + hidden_idx;
    const float cell_new = f_gate * cell_prev[cell_idx] + i_gate * c_gate;
    cell_out[cell_idx] = cell_new;

    // Update hidden state
    hidden_out[cell_idx] = o_gate * tanhf(cell_new);
}

/// GRU forward step
extern "C" __global__ void gru_forward_step(
    const float* __restrict__ input,
    const float* __restrict__ hidden_prev,
    const float* __restrict__ weights_input,
    const float* __restrict__ weights_hidden,
    const float* __restrict__ bias,
    float* __restrict__ hidden_out,
    int batch_size, int input_size, int hidden_size
) {
    const int batch = blockIdx.x;
    const int hidden_idx = threadIdx.x;

    if (batch >= batch_size || hidden_idx >= hidden_size) return;

    // Compute gates
    float r_gate = bias[hidden_idx];
    float z_gate = bias[hidden_size + hidden_idx];
    float n_gate = bias[2 * hidden_size + hidden_idx];

    const float h_prev = hidden_prev[batch * hidden_size + hidden_idx];

    // Add input contribution
    for (int i = 0; i < input_size; ++i) {
        const float in_val = input[batch * input_size + i];
        r_gate += in_val * weights_input[i * 3 * hidden_size + hidden_idx];
        z_gate += in_val * weights_input[i * 3 * hidden_size + hidden_size + hidden_idx];
        n_gate += in_val * weights_input[i * 3 * hidden_size + 2 * hidden_size + hidden_idx];
    }

    // Add hidden contribution
    for (int h = 0; h < hidden_size; ++h) {
        const float h_val = hidden_prev[batch * hidden_size + h];
        r_gate += h_val * weights_hidden[h * 3 * hidden_size + hidden_idx];
        z_gate += h_val * weights_hidden[h * 3 * hidden_size + hidden_size + hidden_idx];
    }

    // Apply activations
    r_gate = sigmoid(r_gate);
    z_gate = sigmoid(z_gate);

    // Add r * h_prev contribution to n_gate
    for (int h = 0; h < hidden_size; ++h) {
        const float h_val = hidden_prev[batch * hidden_size + h];
        n_gate += (r_gate * h_val) * weights_hidden[h * 3 * hidden_size + 2 * hidden_size + hidden_idx];
    }
    n_gate = tanhf(n_gate);

    // Update hidden state
    const int idx = batch * hidden_size + hidden_idx;
    hidden_out[idx] = (1.0f - z_gate) * n_gate + z_gate * h_prev;
}

/// Simple RNN forward step
extern "C" __global__ void rnn_forward_step(
    const float* __restrict__ input,
    const float* __restrict__ hidden_prev,
    const float* __restrict__ weights_input,
    const float* __restrict__ weights_hidden,
    const float* __restrict__ bias,
    float* __restrict__ hidden_out,
    int batch_size, int input_size, int hidden_size
) {
    const int batch = blockIdx.x;
    const int hidden_idx = threadIdx.x;

    if (batch >= batch_size || hidden_idx >= hidden_size) return;

    float sum = bias[hidden_idx];

    // Add input contribution
    for (int i = 0; i < input_size; ++i) {
        sum += input[batch * input_size + i] * weights_input[i * hidden_size + hidden_idx];
    }

    // Add hidden contribution
    for (int h = 0; h < hidden_size; ++h) {
        sum += hidden_prev[batch * hidden_size + h] * weights_hidden[h * hidden_size + hidden_idx];
    }

    hidden_out[batch * hidden_size + hidden_idx] = tanhf(sum);
}

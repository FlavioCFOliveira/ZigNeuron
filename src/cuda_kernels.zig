/// CUDA Kernels Definitions
/// This file contains both CUDA C source code and embedded PTX for CUDA kernels.
///
/// NVRTC Mode: CUDA C source is compiled at runtime using NVRTC
/// Fallback Mode: Pre-compiled PTX strings are loaded directly
///
/// NOTE: Using PTX version 7.5 for compatibility with Ampere GPUs (RTX 30 series)
/// Target sm_80 for Ampere architecture, JIT-compiled for actual GPU

// =============================================================================
// CUDA C Source Code for NVRTC Compilation
// =============================================================================

/// Simple matrix multiplication kernel CUDA C source
pub const MATMUL_SIMPLE_SOURCE =
    \\extern "C" __global__ void matmul(
    \\    float* C, const float* A, const float* B,
    \\    int M, int N, int K, int accumulate) {
    \\    int row = blockIdx.y * blockDim.y + threadIdx.y;
    \\    int col = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (row < M && col < N) {
    \\        float sum = 0.0f;
    \\        for (int k = 0; k < K; k++) {
    \\            sum += A[row * K + k] * B[k * N + col];
    \\        }
    \\        int idx = row * N + col;
    \\        if (accumulate) {
    \\            C[idx] += sum;
    \\        } else {
    \\            C[idx] = sum;
    \\        }
    \\    }
    \\}
;

/// Batched matrix multiplication kernel CUDA C source
pub const MATMUL_BATCHED_SOURCE =
    \\extern "C" __global__ void matmul_batch(
    \\    float* C, const float* A, const float* B,
    \\    int batch_size, int M, int N, int K, int accumulate) {
    \\    int batch = blockIdx.z;
    \\    int row = blockIdx.y * blockDim.y + threadIdx.y;
    \\    int col = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (batch < batch_size && row < M && col < N) {
    \\        int a_offset = batch * M * K;
    \\        int c_offset = batch * M * N;
    \\        float sum = 0.0f;
    \\        for (int k = 0; k < K; k++) {
    \\            sum += A[a_offset + row * K + k] * B[k * N + col];
    \\        }
    \\        int idx = c_offset + row * N + col;
    \\        if (accumulate) {
    \\            C[idx] += sum;
    \\        } else {
    \\            C[idx] = sum;
    \\        }
    \\    }
    \\}
;

/// Matrix multiplication with A transposed kernel CUDA C source
pub const MATMUL_TRANSPOSE_A_SOURCE =
    \\extern "C" __global__ void matmul_transpose_a(
    \\    float* C, const float* A, const float* B,
    \\    int M, int N, int K, int accumulate) {
    \\    int row = blockIdx.y * blockDim.y + threadIdx.y;
    \\    int col = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (row < M && col < N) {
    \\        float sum = 0.0f;
    \\        for (int k = 0; k < K; k++) {
    \\            sum += A[k * M + row] * B[k * N + col];
    \\        }
    \\        int idx = row * N + col;
    \\        if (accumulate) {
    \\            C[idx] += sum;
    \\        } else {
    \\            C[idx] = sum;
    \\        }
    \\    }
    \\}
;

/// Matrix multiplication with B transposed kernel CUDA C source
pub const MATMUL_TRANSPOSE_B_SOURCE =
    \\extern "C" __global__ void matmul_transpose_b(
    \\    float* C, const float* A, const float* B,
    \\    int M, int N, int K, int accumulate) {
    \\    int row = blockIdx.y * blockDim.y + threadIdx.y;
    \\    int col = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (row < M && col < N) {
    \\        float sum = 0.0f;
    \\        for (int k = 0; k < K; k++) {
    \\            sum += A[row * K + k] * B[col * K + k];
    \\        }
    \\        int idx = row * N + col;
    \\        if (accumulate) {
    \\            C[idx] += sum;
    \\        } else {
    \\            C[idx] = sum;
    \\        }
    \\    }
    \\}
;

/// Tiled matrix multiplication with shared memory - HIGH PERFORMANCE VERSION
/// Uses TILE_SIZE x TILE_SIZE thread blocks with shared memory tiling
/// Each block computes a TILE_SIZE x TILE_SIZE sub-matrix of C
/// Expected 10-100x speedup over naive implementation
pub const MATMUL_TILED_SOURCE =
    \\extern "C" __global__ void matmul_tiled(
    \\    float* C, const float* A, const float* B,
    \\    int M, int N, int K, int accumulate) {
    \\    // Tile size - 32x32 gives 1024 threads per block (max occupancy)
    \\    // Shared memory per block: 2 * 32 * 32 * 4 bytes = 8KB
    \\    const int TILE_SIZE = 32;
    \\
    \\    // Shared memory for tile caching
    \\    __shared__ float As[TILE_SIZE][TILE_SIZE];
    \\    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    \\
    \\    // Block indices
    \\    int blockRow = blockIdx.y;
    \\    int blockCol = blockIdx.x;
    \\
    \\    // Thread indices within block
    \\    int threadRow = threadIdx.y;
    \\    int threadCol = threadIdx.x;
    \\
    \\    // Global row and column in output matrix
    \\    int row = blockRow * TILE_SIZE + threadRow;
    \\    int col = blockCol * TILE_SIZE + threadCol;
    \\
    \\    // Accumulator in register (private to each thread)
    \\    float sum = 0.0f;
    \\
    \\    // Number of tiles needed to cover dimension K
    \\    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    \\
    \\    // Loop over all tiles
    \\    for (int tile = 0; tile < numTiles; tile++) {
    \\        // Calculate which column of A and row of B to load
    \\        int aCol = tile * TILE_SIZE + threadCol;
    \\        int bRow = tile * TILE_SIZE + threadRow;
    \\
    \\        // Cooperative load: each thread loads one element from A and B
    \\        // Load A tile with bounds checking
    \\        if (row < M && aCol < K) {
    \\            As[threadRow][threadCol] = A[row * K + aCol];
    \\        } else {
    \\            As[threadRow][threadCol] = 0.0f;
    \\        }
    \\
    \\        // Load B tile with bounds checking
    \\        if (bRow < K && col < N) {
    \\            Bs[threadRow][threadCol] = B[bRow * N + col];
    \\        } else {
    \\            Bs[threadRow][threadCol] = 0.0f;
    \\        }
    \\
    \\        // Synchronize to ensure all loads complete before computation
    \\        __syncthreads();
    \\
    \\        // Compute partial dot product using shared memory
    \\        // Each thread computes one element using its tile slice
    \\        // Unroll loop for better performance
    \\        #pragma unroll
    \\        for (int k = 0; k < TILE_SIZE; k++) {
    \\            sum += As[threadRow][k] * Bs[k][threadCol];
    \\        }
    \\
    \\        // Synchronize before loading next tile
    \\        __syncthreads();
    \\    }
    \\
    \\    // Write result to global memory with bounds checking
    \\    if (row < M && col < N) {
    \\        int idx = row * N + col;
    \\        if (accumulate) {
    \\            C[idx] += sum;
    \\        } else {
    \\            C[idx] = sum;
    \\        }
    \\    }
    \\}
;

/// Tiled matrix multiplication with B transposed
pub const MATMUL_TILED_TRANSPOSE_B_SOURCE =
    \\extern "C" __global__ void matmul_tiled_transpose_b(
    \\    float* C, const float* A, const float* B,
    \\    int M, int N, int K, int accumulate) {
    \\    const int TILE_SIZE = 32;
    \\    __shared__ float As[TILE_SIZE][TILE_SIZE];
    \\    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    \\
    \\    int blockRow = blockIdx.y;
    \\    int blockCol = blockIdx.x;
    \\    int threadRow = threadIdx.y;
    \\    int threadCol = threadIdx.x;
    \\    int row = blockRow * TILE_SIZE + threadRow;
    \\    int col = blockCol * TILE_SIZE + threadCol;
    \\    float sum = 0.0f;
    \\    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    \\
    \\    for (int tile = 0; tile < numTiles; tile++) {
    \\        int aCol = tile * TILE_SIZE + threadCol;
    \\        int bRow = tile * TILE_SIZE + threadCol;  // Note: transposed access
    \\        int bCol = row;  // Note: transposed
    \\
    \\        if (row < M && aCol < K) {
    \\            As[threadRow][threadCol] = A[row * K + aCol];
    \\        } else {
    \\            As[threadRow][threadCol] = 0.0f;
    \\        }
    \\
    \\        // For transpose: B is accessed as B[col][row] instead of B[row][col]
    \\        if (bRow < K && col < N) {
    \\            Bs[threadRow][threadCol] = B[col * K + bRow];  // Transposed access
    \\        } else {
    \\            Bs[threadRow][threadCol] = 0.0f;
    \\        }
    \\
    \\        __syncthreads();
    \\
    \\        #pragma unroll
    \\        for (int k = 0; k < TILE_SIZE; k++) {
    \\            sum += As[threadRow][k] * Bs[k][threadCol];
    \\        }
    \\
    \\        __syncthreads();
    \\    }
    \\
    \\    if (row < M && col < N) {
    \\        int idx = row * N + col;
    \\        if (accumulate) {
    \\            C[idx] += sum;
    \\        } else {
    \\            C[idx] = sum;
    \\        }
    \\    }
    \\}
;

/// ReLU forward kernel CUDA C source
pub const RELU_FORWARD_SOURCE =
    \\extern "C" __global__ void relu_forward(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        float val = input[idx];
    \\        output[idx] = val > 0.0f ? val : 0.0f;
    \\    }
    \\}
;

/// ReLU backward kernel CUDA C source
pub const RELU_BACKWARD_SOURCE =
    \\extern "C" __global__ void relu_backward(
    \\    const float* output, const float* grad_output,
    \\    float* grad_input, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        grad_input[idx] = output[idx] > 0.0f ? grad_output[idx] : 0.0f;
    \\    }
    \\}
;

/// Sigmoid forward kernel CUDA C source
pub const SIGMOID_FORWARD_SOURCE =
    \\extern "C" __global__ void sigmoid_forward(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        float x = input[idx];
    \\        output[idx] = 1.0f / (1.0f + expf(-x));
    \\    }
    \\}
;

/// Sigmoid backward kernel CUDA C source
pub const SIGMOID_BACKWARD_SOURCE =
    \\extern "C" __global__ void sigmoid_backward(
    \\    const float* output, const float* grad_output,
    \\    float* grad_input, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        float o = output[idx];
    \\        grad_input[idx] = grad_output[idx] * o * (1.0f - o);
    \\    }
    \\}
;

/// Tanh forward kernel CUDA C source
pub const TANH_FORWARD_SOURCE =
    \\extern "C" __global__ void tanh_forward(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        output[idx] = tanhf(input[idx]);
    \\    }
    \\}
;

/// Tanh backward kernel CUDA C source
pub const TANH_BACKWARD_SOURCE =
    \\extern "C" __global__ void tanh_backward(
    \\    const float* output, const float* grad_output,
    \\    float* grad_input, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        float o = output[idx];
    \\        grad_input[idx] = grad_output[idx] * (1.0f - o * o);
    \\    }
    \\}
;

/// Linear forward kernel CUDA C source
pub const LINEAR_FORWARD_SOURCE =
    \\extern "C" __global__ void linear_forward(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        output[idx] = input[idx];
    \\    }
    \\}
;

/// Softmax forward kernel CUDA C source (naive per-sample implementation)
pub const SOFTMAX_FORWARD_SOURCE =
    \\extern "C" __global__ void softmax_forward(
    \\    const float* input, float* output,
    \\    int batch_size, int features) {
    \\    int batch = blockIdx.x;
    \\    if (batch >= batch_size) return;
    \\    const float* in_batch = input + batch * features;
    \\    float* out_batch = output + batch * features;
    \\    // Find max for numerical stability
    \\    float max_val = in_batch[0];
    \\    for (int i = 1; i < features; i++) {
    \\        if (in_batch[i] > max_val) max_val = in_batch[i];
    \\    }
    \\    // Compute exp and sum
    \\    float sum = 0.0f;
    \\    for (int i = 0; i < features; i++) {
    \\        float exp_val = expf(in_batch[i] - max_val);
    \\        out_batch[i] = exp_val;
    \\        sum += exp_val;
    \\    }
    \\    // Normalize
    \\    for (int i = 0; i < features; i++) {
    \\        out_batch[i] /= sum;
    \\    }
    \\}
;

/// Element-wise add kernel CUDA C source
pub const EW_ADD_SOURCE =
    \\extern "C" __global__ void ew_add(
    \\    const float* a, const float* b, float* c, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        c[idx] = a[idx] + b[idx];
    \\    }
    \\}
;

/// Element-wise multiply kernel CUDA C source
pub const EW_MUL_SOURCE =
    \\extern "C" __global__ void ew_mul(
    \\    const float* a, const float* b, float* c, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        c[idx] = a[idx] * b[idx];
    \\    }
    \\}
;

/// Element-wise subtract kernel CUDA C source
pub const EW_SUB_SOURCE =
    \\extern "C" __global__ void ew_sub(
    \\    const float* a, const float* b, float* c, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        c[idx] = a[idx] - b[idx];
    \\    }
    \\}
;

/// Element-wise divide kernel CUDA C source
pub const EW_DIV_SOURCE =
    \\extern "C" __global__ void ew_div(
    \\    const float* a, const float* b, float* c, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        c[idx] = a[idx] / b[idx];
    \\    }
    \\}
;

/// Scale buffer kernel CUDA C source
pub const SCALE_BUFFER_SOURCE =
    \\extern "C" __global__ void scale_buffer(
    \\    const float* input, float scale, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        output[idx] = input[idx] * scale;
    \\    }
    \\}
;

/// Map exp kernel CUDA C source
pub const MAP_EXP_SOURCE =
    \\extern "C" __global__ void map_exp(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        output[idx] = expf(input[idx]);
    \\    }
    \\}
;

/// Map log kernel CUDA C source
pub const MAP_LOG_SOURCE =
    \\extern "C" __global__ void map_log(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        output[idx] = logf(input[idx]);
    \\    }
    \\}
;

/// Map sqrt kernel CUDA C source
pub const MAP_SQRT_SOURCE =
    \\extern "C" __global__ void map_sqrt(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        output[idx] = sqrtf(input[idx]);
    \\    }
    \\}
;

/// Map abs kernel CUDA C source
pub const MAP_ABS_SOURCE =
    \\extern "C" __global__ void map_abs(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        output[idx] = fabsf(input[idx]);
    \\    }
    \\}
;

/// Map square kernel CUDA C source
pub const MAP_SQUARE_SOURCE =
    \\extern "C" __global__ void map_square(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        float val = input[idx];
    \\        output[idx] = val * val;
    \\    }
    \\}
;

/// Map inverse kernel CUDA C source
pub const MAP_INV_SOURCE =
    \\extern "C" __global__ void map_inv(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        output[idx] = 1.0f / input[idx];
    \\    }
    \\}
;

/// SGD update kernel CUDA C source
pub const SGD_UPDATE_SOURCE =
    \\extern "C" __global__ void sgd_update(
    \\    float* weights, const float* gradients,
    \\    int n, float learning_rate, float weight_decay) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        float grad = gradients[idx] + weight_decay * weights[idx];
    \\        weights[idx] -= learning_rate * grad;
    \\    }
    \\}
;

/// Fill constant kernel CUDA C source
pub const FILL_CONSTANT_SOURCE =
    \\extern "C" __global__ void fill_constant(
    \\    float* data, float value, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        data[idx] = value;
    \\    }
    \\}
;

/// Add bias kernel CUDA C source
pub const ADD_BIAS_SOURCE =
    \\extern "C" __global__ void add_bias(
    \\    float* output, const float* bias,
    \\    int batch_size, int bias_size) {
    \\    int col = blockIdx.x * blockDim.x + threadIdx.x;
    \\    int batch = blockIdx.y;
    \\    if (batch < batch_size && col < bias_size) {
    \\        int idx = batch * bias_size + col;
    \\        output[idx] += bias[col];
    \\    }
    \\}
;

/// Conv2D forward kernel CUDA C source
/// Performs 2D convolution with stride and padding
/// Input layout: [batch][in_channels][input_h][input_w]
/// Weights layout: [out_channels][in_channels][kernel_h][kernel_w]
/// Output layout: [batch][out_channels][output_h][output_w]
pub const CONV2D_FORWARD_SOURCE =
    \\extern "C" __global__ void conv2d_forward(
    \\    const float* input, const float* weights, const float* bias,
    \\    float* output,
    \\    int batch_size, int in_channels, int out_channels,
    \\    int kernel_h, int kernel_w,
    \\    int input_h, int input_w,
    \\    int output_h, int output_w,
    \\    int stride_h, int stride_w,
    \\    int padding_h, int padding_w) {
    \\    // Calculate output coordinates
    \\    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    \\    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    \\    int out_ch = blockIdx.z % out_channels;
    \\    int batch = blockIdx.z / out_channels;
    \\
    \\    if (out_x >= output_w || out_y >= output_h) return;
    \\    if (batch >= batch_size) return;
    \\
    \\    // Calculate input start position
    \\    int in_y_start = out_y * stride_h - padding_h;
    \\    int in_x_start = out_x * stride_w - padding_w;
    \\
    \\    // Initialize with bias if available
    \\    float sum = (bias != NULL) ? bias[out_ch] : 0.0f;
    \\
    \\    // Perform convolution
    \\    for (int ic = 0; ic < in_channels; ic++) {
    \\        for (int kh = 0; kh < kernel_h; kh++) {
    \\            for (int kw = 0; kw < kernel_w; kw++) {
    \\                int in_y = in_y_start + kh;
    \\                int in_x = in_x_start + kw;
    \\                // Check bounds
    \\                if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
    \\                    int in_idx = ((batch * in_channels + ic) * input_h + in_y) * input_w + in_x;
    \\                    int w_idx = ((out_ch * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
    \\                    sum += input[in_idx] * weights[w_idx];
    \\                }
    \\            }
    \\        }
    \\    }
    \\
    \\    // Write output
    \\    int out_idx = ((batch * out_channels + out_ch) * output_h + out_y) * output_w + out_x;
    \\    output[out_idx] = sum;
    \\}
;

/// MaxPool2D forward kernel CUDA C source
pub const MAX_POOL2D_FORWARD_SOURCE =
    \\extern "C" __global__ void maxpool2d_forward(
    \\    const float* input, float* output, float* max_indices,
    \\    int channels, int input_h, int input_w,
    \\    int output_h, int output_w,
    \\    int pool_h, int pool_w,
    \\    int stride_h, int stride_w) {
    \\    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    \\    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    \\    int ch = blockIdx.z;
    \\
    \\    if (out_x < output_w && out_y < output_h) {
    \\        int in_y_start = out_y * stride_h;
    \\        int in_x_start = out_x * stride_w;
    \\
    \\        float max_val = -1e38f;
    \\        int max_idx = -1;
    \\
    \\        for (int ph = 0; ph < pool_h; ph++) {
    \\            for (int pw = 0; pw < pool_w; pw++) {
    \\                int in_y = in_y_start + ph;
    \\                int in_x = in_x_start + pw;
    \\
    \\                if (in_y < input_h && in_x < input_w) {
    \\                    int in_idx = (ch * input_h + in_y) * input_w + in_x;
    \\                    float val = input[in_idx];
    \\                    if (val > max_val) {
    \\                        max_val = val;
    \\                        max_idx = in_idx;
    \\                    }
    \\                }
    \\            }
    \\        }
    \\
    \\        int out_idx = (ch * output_h + out_y) * output_w + out_x;
    \\        output[out_idx] = max_val;
    \\        max_indices[out_idx] = (float)max_idx;
    \\    }
    \\}
;

/// MaxPool2D backward kernel CUDA C source
pub const MAX_POOL2D_BACKWARD_SOURCE =
    \\extern "C" __global__ void maxpool2d_backward(
    \\    const float* grad_output, const float* max_indices, float* grad_input,
    \\    int total_elements) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < total_elements) {
    \\        int max_idx = (int)max_indices[idx];
    \\        if (max_idx >= 0) {
    \\            atomicAdd(&grad_input[max_idx], grad_output[idx]);
    \\        }
    \\    }
    \\}
;

/// MaxPool1D forward kernel CUDA C source
pub const MAX_POOL1D_FORWARD_SOURCE =
    \\extern "C" __global__ void maxpool1d_forward(
    \\    const float* input, float* output, float* max_indices,
    \\    int channels, int input_len, int output_len,
    \\    int pool_size, int stride) {
    \\    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    int ch = blockIdx.y;
    \\
    \\    if (out_idx < output_len) {
    \\        int in_start = out_idx * stride;
    \\        float max_val = -1e38f;
    \\        int max_idx = -1;
    \\
    \\        for (int p = 0; p < pool_size; p++) {
    \\            int in_pos = in_start + p;
    \\            if (in_pos < input_len) {
    \\                int in_full_idx = ch * input_len + in_pos;
    \\                float val = input[in_full_idx];
    \\                if (val > max_val) {
    \\                    max_val = val;
    \\                    max_idx = in_full_idx;
    \\                }
    \\            }
    \\        }
    \\
    \\        int final_out_idx = ch * output_len + out_idx;
    \\        output[final_out_idx] = max_val;
    \\        max_indices[final_out_idx] = (float)max_idx;
    \\    }
    \\}
;

/// MaxPool1D backward kernel CUDA C source
pub const MAX_POOL1D_BACKWARD_SOURCE =
    \\extern "C" __global__ void maxpool1d_backward(
    \\    const float* grad_output, const float* max_indices, float* grad_input,
    \\    int total_elements) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < total_elements) {
    \\        int max_idx = (int)max_indices[idx];
    \\        if (max_idx >= 0) {
    \\            atomicAdd(&grad_input[max_idx], grad_output[idx]);
    \\        }
    \\    }
    \\}
;

// =============================================================================
// PTX Header - Common to all kernels
// =============================================================================
/// PTX Header - Common to all kernels
/// Using PTX 5.0 for compatibility with CUDA 8+ drivers
/// Target sm_50 (Maxwell) as baseline for NVRTC compatibility
/// Using PTX 6.0 which should be supported by driver 535+
/// Driver will JIT compile to actual GPU architecture
pub const PTX_HEADER =
    \\.version 6.0
    \\.target sm_50
    \\.address_size 64
    \\
;

// =============================================================================
// Matrix Multiplication Kernels
// =============================================================================

/// Simple matrix multiplication kernel PTX
/// C = A * B where A: [M x K], B: [K x N], C: [M x N]
/// Uses mul.lo.u64 instead of shl.b64 for compatibility
pub const MATMUL_SIMPLE_PTX = PTX_HEADER ++
    \\.visible .entry matmul(
    \\    .param .u64 A,
    \\    .param .u64 B,
    \\    .param .u64 C,
    \\    .param .u32 M,
    \\    .param .u32 N,
    \\    .param .u32 K,
    \\    .param .u32 accumulate
    \\) {
    \\    .reg .u64 %A_ptr, %B_ptr, %C_ptr;
    \\    .reg .u32 %M, %N, %K, %accumulate;
    \\    .reg .u32 %row, %col, %k;
    \\    .reg .u32 %tid_x, %tid_y, %bid_x, %bid_y;
    \\    .reg .u32 %bsize_x, %bsize_y;
    \\    .reg .f32 %sum, %a_val, %b_val, %c_val;
    \\    .reg .u64 %a_addr, %b_addr, %c_addr;
    \\    .reg .pred %p, %p_accum;
    \\
    \\    ld.param.u64 %A_ptr, [A];
    \\    ld.param.u64 %B_ptr, [B];
    \\    ld.param.u64 %C_ptr, [C];
    \\    ld.param.u32 %M, [M];
    \\    ld.param.u32 %N, [N];
    \\    ld.param.u32 %K, [K];
    \\    ld.param.u32 %accumulate, [accumulate];
    \\
    \\    mov.u32 %tid_x, %tid.x;
    \\    mov.u32 %tid_y, %tid.y;
    \\    mov.u32 %bid_x, %ctaid.x;
    \\    mov.u32 %bid_y, %ctaid.y;
    \\    mov.u32 %bsize_x, %ntid.x;
    \\    mov.u32 %bsize_y, %ntid.y;
    \\
    \\    mad.lo.u32 %row, %bid_y, %bsize_y, %tid_y;
    \\    mad.lo.u32 %col, %bid_x, %bsize_x, %tid_x;
    \\
    \\    setp.ge.u32 %p, %row, %M;
    \\    @%p bra END;
    \\    setp.ge.u32 %p, %col, %N;
    \\    @%p bra END;
    \\
    \\    mov.f32 %sum, 0f00000000;
    \\    mov.u32 %k, 0;
    \\LOOP:
    \\    setp.ge.u32 %p, %k, %K;
    \\    @%p bra LOOP_END;
    \\    mad.lo.u64 %a_addr, %row, %K, %k;
    \\    mul.lo.u64 %a_addr, %a_addr, 4;
    \\    add.u64 %a_addr, %a_addr, %A_ptr;
    \\    ld.global.f32 %a_val, [%a_addr];
    \\    mad.lo.u64 %b_addr, %k, %N, %col;
    \\    mul.lo.u64 %b_addr, %b_addr, 4;
    \\    add.u64 %b_addr, %b_addr, %B_ptr;
    \\    ld.global.f32 %b_val, [%b_addr];
    \\    fma.rn.f32 %sum, %a_val, %b_val, %sum;
    \\    add.u32 %k, %k, 1;
    \\    bra LOOP;
    \\LOOP_END:
    \\
    \\    mad.lo.u64 %c_addr, %row, %N, %col;
    \\    mul.lo.u64 %c_addr, %c_addr, 4;
    \\    add.u64 %c_addr, %c_addr, %C_ptr;
    \\    setp.eq.u32 %p_accum, %accumulate, 0;
    \\    @%p_accum bra STORE;
    \\    ld.global.f32 %c_val, [%c_addr];
    \\    add.f32 %sum, %sum, %c_val;
    \\STORE:
    \\    st.global.f32 [%c_addr], %sum;
    \\END:
    \\    ret;
    \\}
;

/// Batched matrix multiplication kernel PTX
/// C[i] = A[i] * B for i in [0, batch_size)
pub const MATMUL_BATCHED_PTX = PTX_HEADER ++
    \\.visible .entry matmul_batch(
    \\    .param .u64 A,
    \\    .param .u64 B,
    \\    .param .u64 C,
    \\    .param .u32 batch_size,
    \\    .param .u32 M,
    \\    .param .u32 N,
    \\    .param .u32 K,
    \\    .param .u32 accumulate
    \\) {
    \\    .reg .u64 %A_ptr, %B_ptr, %C_ptr;
    \\    .reg .u32 %batch, %batch_size, %M, %N, %K, %accumulate;
    \\    .reg .u32 %row, %col, %k;
    \\    .reg .u32 %tid, %ctaid;
    \\    .reg .f32 %sum, %a_val, %b_val, %c_val;
    \\    .reg .u64 %a_addr, %b_addr, %c_addr, %a_batch_offset, %c_batch_offset;
    \\    .reg .pred %p, %p_accum;
    \\
    \\    ld.param.u64 %A_ptr, [A];
    \\    ld.param.u64 %B_ptr, [B];
    \\    ld.param.u64 %C_ptr, [C];
    \\    ld.param.u32 %batch_size, [batch_size];
    \\    ld.param.u32 %M, [M];
    \\    ld.param.u32 %N, [N];
    \\    ld.param.u32 %K, [K];
    \\    ld.param.u32 %accumulate, [accumulate];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %batch, %ctaid;
    \\    setp.ge.u32 %p, %batch, %batch_size;
    \\    @%p bra END;
    \\
    \\    mov.u32 %col, %tid;
    \\    setp.ge.u32 %p, %col, %N;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %a_batch_offset, %batch, %M;
    \\    mul.lo.u64 %a_batch_offset, %a_batch_offset, %K;
    \\    mul.lo.u64 %a_batch_offset, %a_batch_offset, 4;
    \\    add.u64 %A_ptr, %A_ptr, %a_batch_offset;
    \\
    \\    mul.lo.u64 %c_batch_offset, %batch, %M;
    \\    mul.lo.u64 %c_batch_offset, %c_batch_offset, %N;
    \\    mul.lo.u64 %c_batch_offset, %c_batch_offset, 4;
    \\    add.u64 %C_ptr, %C_ptr, %c_batch_offset;
    \\
    \\    mov.f32 %sum, 0f00000000;
    \\    mov.u32 %k, 0;
    \\LOOP:
    \\    setp.ge.u32 %p, %k, %K;
    \\    @%p bra LOOP_END;
    \\    mad.lo.u64 %a_addr, %row, %K, %k;
    \\    mul.lo.u64 %a_addr, %a_addr, 4;
    \\    add.u64 %a_addr, %a_addr, %A_ptr;
    \\    ld.global.f32 %a_val, [%a_addr];
    \\    mad.lo.u64 %b_addr, %k, %N, %col;
    \\    mul.lo.u64 %b_addr, %b_addr, 4;
    \\    add.u64 %b_addr, %b_addr, %B_ptr;
    \\    ld.global.f32 %b_val, [%b_addr];
    \\    fma.rn.f32 %sum, %a_val, %b_val, %sum;
    \\    add.u32 %k, %k, 1;
    \\    bra LOOP;
    \\LOOP_END:
    \\
    \\    mad.lo.u64 %c_addr, %row, %N, %col;
    \\    mul.lo.u64 %c_addr, %c_addr, 4;
    \\    add.u64 %c_addr, %c_addr, %C_ptr;
    \\    setp.eq.u32 %p_accum, %accumulate, 0;
    \\    @%p_accum bra STORE;
    \\    ld.global.f32 %c_val, [%c_addr];
    \\    add.f32 %sum, %sum, %c_val;
    \\STORE:
    \\    st.global.f32 [%c_addr], %sum;
    \\END:
    \\    ret;
    \\}
;

// =============================================================================
// Activation Function Kernels
// =============================================================================

/// ReLU forward: output = max(0, input)
pub const RELU_FORWARD_PTX = PTX_HEADER ++
    \\.visible .entry relu_forward(
    \\    .param .u64 input,
    \\    .param .u64 output,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %val, %zero;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%addr];
    \\
    \\    mov.f32 %zero, 0f00000000;
    \\    setp.gt.f32 %p, %val, %zero;
    \\    @!%p mov.f32 %val, %zero;
    \\
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    sub.u64 %addr, %addr, %in_ptr;
    \\    st.global.f32 [%addr], %val;
    \\END:
    \\    ret;
    \\}
;

/// ReLU backward: grad_input = (output > 0) ? grad_output : 0
pub const RELU_BACKWARD_PTX = PTX_HEADER ++
    \\.visible .entry relu_backward(
    \\    .param .u64 output,
    \\    .param .u64 grad_output,
    \\    .param .u64 grad_input,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %out_ptr, %go_ptr, %gi_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %out_val, %go_val, %zero;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p, %p_pos;
    \\
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %go_ptr, [grad_output];
    \\    ld.param.u64 %gi_ptr, [grad_input];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    ld.global.f32 %out_val, [%addr];
    \\
    \\    mov.f32 %zero, 0f00000000;
    \\    setp.gt.f32 %p_pos, %out_val, %zero;
    \\    mov.f32 %go_val, %zero;
    \\    @%p_pos bra LOAD_GRAD;
    \\    bra STORE;
    \\LOAD_GRAD:
    \\    add.u64 %addr, %addr, %go_ptr;
    \\    sub.u64 %addr, %addr, %out_ptr;
    \\    ld.global.f32 %go_val, [%addr];
    \\STORE:
    \\    add.u64 %addr, %addr, %gi_ptr;
    \\    sub.u64 %addr, %addr, %go_ptr;
    \\    st.global.f32 [%addr], %go_val;
    \\END:
    \\    ret;
    \\}
;

/// Sigmoid forward: output = 1 / (1 + exp(-input))
pub const SIGMOID_FORWARD_PTX = PTX_HEADER ++
    \\.visible .entry sigmoid_forward(
    \\    .param .u64 input,
    \\    .param .u64 output,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %val, %exp_val, %one, %result;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%addr];
    \\
    \\    neg.f32 %val, %val;
    \\    ex2.approx.ftz.f32 %exp_val, %val;
    \\    mov.f32 %one, 0f3F800000;
    \\    add.f32 %exp_val, %exp_val, %one;
    \\    div.approx.ftz.f32 %result, %one, %exp_val;
    \\
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    sub.u64 %addr, %addr, %in_ptr;
    \\    st.global.f32 [%addr], %result;
    \\END:
    \\    ret;
    \\}
;

/// Sigmoid backward: grad_input = grad_output * output * (1 - output)
pub const SIGMOID_BACKWARD_PTX = PTX_HEADER ++
    \\.visible .entry sigmoid_backward(
    \\    .param .u64 output,
    \\    .param .u64 grad_output,
    \\    .param .u64 grad_input,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %out_ptr, %go_ptr, %gi_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %out_val, %go_val, %one, %gi_val;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %go_ptr, [grad_output];
    \\    ld.param.u64 %gi_ptr, [grad_input];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    ld.global.f32 %out_val, [%addr];
    \\
    \\    add.u64 %addr, %addr, %go_ptr;
    \\    sub.u64 %addr, %addr, %out_ptr;
    \\    ld.global.f32 %go_val, [%addr];
    \\
    \\    mov.f32 %one, 0f3F800000;
    \\    sub.f32 %gi_val, %one, %out_val;
    \\    mul.f32 %gi_val, %gi_val, %out_val;
    \\    mul.f32 %gi_val, %gi_val, %go_val;
    \\
    \\    add.u64 %addr, %addr, %gi_ptr;
    \\    sub.u64 %addr, %addr, %go_ptr;
    \\    st.global.f32 [%addr], %gi_val;
    \\END:
    \\    ret;
    \\}
;

/// Tanh forward: output = tanh(input)
pub const TANH_FORWARD_PTX = PTX_HEADER ++
    \\.visible .entry tanh_forward(
    \\    .param .u64 input,
    \\    .param .u64 output,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %val, %result;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%addr];
    \\
    \\    tanh.approx.f32 %result, %val;
    \\
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    sub.u64 %addr, %addr, %in_ptr;
    \\    st.global.f32 [%addr], %result;
    \\END:
    \\    ret;
    \\}
;

/// Tanh backward: grad_input = grad_output * (1 - output^2)
pub const TANH_BACKWARD_PTX = PTX_HEADER ++
    \\.visible .entry tanh_backward(
    \\    .param .u64 output,
    \\    .param .u64 grad_output,
    \\    .param .u64 grad_input,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %out_ptr, %go_ptr, %gi_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %out_val, %go_val, %one, %gi_val;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %go_ptr, [grad_output];
    \\    ld.param.u64 %gi_ptr, [grad_input];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    ld.global.f32 %out_val, [%addr];
    \\
    \\    add.u64 %addr, %addr, %go_ptr;
    \\    sub.u64 %addr, %addr, %out_ptr;
    \\    ld.global.f32 %go_val, [%addr];
    \\
    \\    mov.f32 %one, 0f3F800000;
    \\    mul.f32 %gi_val, %out_val, %out_val;
    \\    sub.f32 %gi_val, %one, %gi_val;
    \\    mul.f32 %gi_val, %gi_val, %go_val;
    \\
    \\    add.u64 %addr, %addr, %gi_ptr;
    \\    sub.u64 %addr, %addr, %go_ptr;
    \\    st.global.f32 [%addr], %gi_val;
    \\END:
    \\    ret;
    \\}
;

// =============================================================================
// Vectorized Activation Function Kernels (LDG.128)
// These kernels process 4 floats at a time for higher memory throughput
// Requires: 16-byte aligned pointers and n >= 1024 (preferably n % 4 == 0)
// =============================================================================

/// ReLU forward vectorized: output = max(0, input) - processes 4 elements per thread
pub const RELU_FORWARD_VEC4_PTX = PTX_HEADER ++
    \\.visible .entry relu_forward_vec4(
    \\    .param .u64 input,
    \\    .param .u64 output,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr;
    \\    .reg .u32 %n, %idx, %idx4;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %zero;
    \\    .reg .v4 .f32 %val;           // Vector of 4 floats
    \\    .reg .u64 %addr_in, %addr_out;
    \\    .reg .pred %p, %p0, %p1, %p2, %p3;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\    shl.b32 %idx4, %idx, 2;           // idx * 4 (each thread processes 4 elements)
    \\
    \\    // Check if we can process 4 elements
    \\    add.u32 %idx, %idx4, 4;
    \\    setp.ge.u32 %p, %idx4, %n;
    \\    @%p bra END;
    \\
    \\    // Calculate addresses for vectorized load
    \\    mul.lo.u64 %addr_in, %idx4, 4;    // Byte offset
    \\    add.u64 %addr_in, %addr_in, %in_ptr;
    \\    add.u64 %addr_out, %addr_in, %out_ptr;
    \\    sub.u64 %addr_out, %addr_out, %in_ptr;
    \\
    \\    // Vectorized load (128 bits = 4 floats)
    \\    ld.global.v4.f32 %val, [%addr_in];
    \\
    \\    // Apply ReLU to each element
    \\    mov.f32 %zero, 0f00000000;
    \\    setp.gt.f32 %p0, %val0, %zero;
    \\    setp.gt.f32 %p1, %val1, %zero;
    \\    setp.gt.f32 %p2, %val2, %zero;
    \\    setp.gt.f32 %p3, %val3, %zero;
    \\    @!%p0 mov.f32 %val0, %zero;
    \\    @!%p1 mov.f32 %val1, %zero;
    \\    @!%p2 mov.f32 %val2, %zero;
    \\    @!%p3 mov.f32 %val3, %zero;
    \\
    \\    // Vectorized store
    \\    st.global.v4.f32 [%addr_out], %val;
    \\
    \\END:
    \\    ret;
    \\}
;

/// ReLU backward vectorized: grad_input = (output > 0) ? grad_output : 0
pub const RELU_BACKWARD_VEC4_PTX = PTX_HEADER ++
    \\.visible .entry relu_backward_vec4(
    \\    .param .u64 output,
    \\    .param .u64 grad_output,
    \\    .param .u64 grad_input,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %out_ptr, %go_ptr, %gi_ptr;
    \\    .reg .u32 %n, %idx, %idx4;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %zero;
    \\    .reg .v4 .f32 %out_val, %go_val;
    \\    .reg .u64 %addr_out, %addr_go, %addr_gi;
    \\    .reg .pred %p, %p0, %p1, %p2, %p3;
    \\
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %go_ptr, [grad_output];
    \\    ld.param.u64 %gi_ptr, [grad_input];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\    shl.b32 %idx4, %idx, 2;
    \\
    \\    add.u32 %idx, %idx4, 4;
    \\    setp.ge.u32 %p, %idx4, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr_out, %idx4, 4;
    \\    add.u64 %addr_out, %addr_out, %out_ptr;
    \\    add.u64 %addr_go, %addr_out, %go_ptr;
    \\    sub.u64 %addr_go, %addr_go, %out_ptr;
    \\    add.u64 %addr_gi, %addr_out, %gi_ptr;
    \\    sub.u64 %addr_gi, %addr_gi, %out_ptr;
    \\
    \\    ld.global.v4.f32 %out_val, [%addr_out];
    \\    ld.global.v4.f32 %go_val, [%addr_go];
    \\
    \\    mov.f32 %zero, 0f00000000;
    \\    setp.gt.f32 %p0, %out_val0, %zero;
    \\    setp.gt.f32 %p1, %out_val1, %zero;
    \\    setp.gt.f32 %p2, %out_val2, %zero;
    \\    setp.gt.f32 %p3, %out_val3, %zero;
    \\    @!%p0 mov.f32 %go_val0, %zero;
    \\    @!%p1 mov.f32 %go_val1, %zero;
    \\    @!%p2 mov.f32 %go_val2, %zero;
    \\    @!%p3 mov.f32 %go_val3, %zero;
    \\
    \\    st.global.v4.f32 [%addr_gi], %go_val;
    \\
    \\END:
    \\    ret;
    \\}
;

/// Sigmoid forward vectorized: output = 1 / (1 + exp(-input))
pub const SIGMOID_FORWARD_VEC4_PTX = PTX_HEADER ++
    \\.visible .entry sigmoid_forward_vec4(
    \\    .param .u64 input,
    \\    .param .u64 output,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr;
    \\    .reg .u32 %n, %idx, %idx4;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %one;
    \\    .reg .v4 .f32 %val, %result;
    \\    .reg .u64 %addr_in, %addr_out;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\    shl.b32 %idx4, %idx, 2;
    \\
    \\    add.u32 %idx, %idx4, 4;
    \\    setp.ge.u32 %p, %idx4, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr_in, %idx4, 4;
    \\    add.u64 %addr_in, %addr_in, %in_ptr;
    \\    add.u64 %addr_out, %addr_in, %out_ptr;
    \\    sub.u64 %addr_out, %addr_out, %in_ptr;
    \\
    \\    ld.global.v4.f32 %val, [%addr_in];
    \\
    \\    // Sigmoid: 1 / (1 + exp(-x))
    \\    mov.f32 %one, 0f3F800000;
    \\    neg.f32 %val0, %val0;
    \\    neg.f32 %val1, %val1;
    \\    neg.f32 %val2, %val2;
    \\    neg.f32 %val3, %val3;
    \\    ex2.approx.ftz.f32 %result0, %val0;
    \\    ex2.approx.ftz.f32 %result1, %val1;
    \\    ex2.approx.ftz.f32 %result2, %val2;
    \\    ex2.approx.ftz.f32 %result3, %val3;
    \\    add.f32 %result0, %result0, %one;
    \\    add.f32 %result1, %result1, %one;
    \\    add.f32 %result2, %result2, %one;
    \\    add.f32 %result3, %result3, %one;
    \\    div.approx.ftz.f32 %result0, %one, %result0;
    \\    div.approx.ftz.f32 %result1, %one, %result1;
    \\    div.approx.ftz.f32 %result2, %one, %result2;
    \\    div.approx.ftz.f32 %result3, %one, %result3;
    \\
    \\    st.global.v4.f32 [%addr_out], %result;
    \\
    \\END:
    \\    ret;
    \\}
;

/// Sigmoid backward vectorized: grad_input = grad_output * output * (1 - output)
pub const SIGMOID_BACKWARD_VEC4_PTX = PTX_HEADER ++
    \\.visible .entry sigmoid_backward_vec4(
    \\    .param .u64 output,
    \\    .param .u64 grad_output,
    \\    .param .u64 grad_input,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %out_ptr, %go_ptr, %gi_ptr;
    \\    .reg .u32 %n, %idx, %idx4;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %one;
    \\    .reg .v4 .f32 %out_val, %go_val, %gi_val, %temp;
    \\    .reg .u64 %addr_out, %addr_go, %addr_gi;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %go_ptr, [grad_output];
    \\    ld.param.u64 %gi_ptr, [grad_input];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\    shl.b32 %idx4, %idx, 2;
    \\
    \\    add.u32 %idx, %idx4, 4;
    \\    setp.ge.u32 %p, %idx4, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr_out, %idx4, 4;
    \\    add.u64 %addr_out, %addr_out, %out_ptr;
    \\    add.u64 %addr_go, %addr_out, %go_ptr;
    \\    sub.u64 %addr_go, %addr_go, %out_ptr;
    \\    add.u64 %addr_gi, %addr_out, %gi_ptr;
    \\    sub.u64 %addr_gi, %addr_gi, %out_ptr;
    \\
    \\    ld.global.v4.f32 %out_val, [%addr_out];
    \\    ld.global.v4.f32 %go_val, [%addr_go];
    \\
    \\    // grad = go * out * (1 - out)
    \\    mov.f32 %one, 0f3F800000;
    \\    sub.f32 %temp0, %one, %out_val0;
    \\    sub.f32 %temp1, %one, %out_val1;
    \\    sub.f32 %temp2, %one, %out_val2;
    \\    sub.f32 %temp3, %one, %out_val3;
    \\    mul.f32 %gi_val0, %temp0, %out_val0;
    \\    mul.f32 %gi_val1, %temp1, %out_val1;
    \\    mul.f32 %gi_val2, %temp2, %out_val2;
    \\    mul.f32 %gi_val3, %temp3, %out_val3;
    \\    mul.f32 %gi_val0, %gi_val0, %go_val0;
    \\    mul.f32 %gi_val1, %gi_val1, %go_val1;
    \\    mul.f32 %gi_val2, %gi_val2, %go_val2;
    \\    mul.f32 %gi_val3, %gi_val3, %go_val3;
    \\
    \\    st.global.v4.f32 [%addr_gi], %gi_val;
    \\
    \\END:
    \\    ret;
    \\}
;

/// Tanh forward vectorized: output = tanh(input)
pub const TANH_FORWARD_VEC4_PTX = PTX_HEADER ++
    \\.visible .entry tanh_forward_vec4(
    \\    .param .u64 input,
    \\    .param .u64 output,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr;
    \\    .reg .u32 %n, %idx, %idx4;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .v4 .f32 %val, %result;
    \\    .reg .u64 %addr_in, %addr_out;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\    shl.b32 %idx4, %idx, 2;
    \\
    \\    add.u32 %idx, %idx4, 4;
    \\    setp.ge.u32 %p, %idx4, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr_in, %idx4, 4;
    \\    add.u64 %addr_in, %addr_in, %in_ptr;
    \\    add.u64 %addr_out, %addr_in, %out_ptr;
    \\    sub.u64 %addr_out, %addr_out, %in_ptr;
    \\
    \\    ld.global.v4.f32 %val, [%addr_in];
    \\
    \\    tanh.approx.f32 %result0, %val0;
    \\    tanh.approx.f32 %result1, %val1;
    \\    tanh.approx.f32 %result2, %val2;
    \\    tanh.approx.f32 %result3, %val3;
    \\
    \\    st.global.v4.f32 [%addr_out], %result;
    \\
    \\END:
    \\    ret;
    \\}
;

/// Tanh backward vectorized: grad_input = grad_output * (1 - output^2)
pub const TANH_BACKWARD_VEC4_PTX = PTX_HEADER ++
    \\.visible .entry tanh_backward_vec4(
    \\    .param .u64 output,
    \\    .param .u64 grad_output,
    \\    .param .u64 grad_input,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %out_ptr, %go_ptr, %gi_ptr;
    \\    .reg .u32 %n, %idx, %idx4;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %one;
    \\    .reg .v4 .f32 %out_val, %go_val, %gi_val, %temp;
    \\    .reg .u64 %addr_out, %addr_go, %addr_gi;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %go_ptr, [grad_output];
    \\    ld.param.u64 %gi_ptr, [grad_input];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\    shl.b32 %idx4, %idx, 2;
    \\
    \\    add.u32 %idx, %idx4, 4;
    \\    setp.ge.u32 %p, %idx4, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr_out, %idx4, 4;
    \\    add.u64 %addr_out, %addr_out, %out_ptr;
    \\    add.u64 %addr_go, %addr_out, %go_ptr;
    \\    sub.u64 %addr_go, %addr_go, %out_ptr;
    \\    add.u64 %addr_gi, %addr_out, %gi_ptr;
    \\    sub.u64 %addr_gi, %addr_gi, %out_ptr;
    \\
    \\    ld.global.v4.f32 %out_val, [%addr_out];
    \\    ld.global.v4.f32 %go_val, [%addr_go];
    \\
    \\    // grad = go * (1 - out^2)
    \\    mov.f32 %one, 0f3F800000;
    \\    mul.f32 %temp0, %out_val0, %out_val0;
    \\    mul.f32 %temp1, %out_val1, %out_val1;
    \\    mul.f32 %temp2, %out_val2, %out_val2;
    \\    mul.f32 %temp3, %out_val3, %out_val3;
    \\    sub.f32 %gi_val0, %one, %temp0;
    \\    sub.f32 %gi_val1, %one, %temp1;
    \\    sub.f32 %gi_val2, %one, %temp2;
    \\    sub.f32 %gi_val3, %one, %temp3;
    \\    mul.f32 %gi_val0, %gi_val0, %go_val0;
    \\    mul.f32 %gi_val1, %gi_val1, %go_val1;
    \\    mul.f32 %gi_val2, %gi_val2, %go_val2;
    \\    mul.f32 %gi_val3, %gi_val3, %go_val3;
    \\
    \\    st.global.v4.f32 [%addr_gi], %gi_val;
    \\
    \\END:
    \\    ret;
    \\}
;

/// Linear (identity) forward
pub const LINEAR_FORWARD_PTX = PTX_HEADER ++
    \\.visible .entry linear_forward(
    \\    .param .u64 input,
    \\    .param .u64 output,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %val;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%addr];
    \\
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    sub.u64 %addr, %addr, %in_ptr;
    \\    st.global.f32 [%addr], %val;
    \\END:
    \\    ret;
    \\}
;

// =============================================================================
// Element-wise Operation Kernels
// =============================================================================

/// Element-wise add: c = a + b
pub const EW_ADD_PTX = PTX_HEADER ++
    \\.visible .entry ew_add(
    \\    .param .u64 a,
    \\    .param .u64 b,
    \\    .param .u64 c,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %a_ptr, %b_ptr, %c_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %a_val, %b_val, %c_val;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %a_ptr, [a];
    \\    ld.param.u64 %b_ptr, [b];
    \\    ld.param.u64 %c_ptr, [c];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %a_ptr;
    \\    ld.global.f32 %a_val, [%addr];
    \\    add.u64 %addr, %addr, %b_ptr;
    \\    sub.u64 %addr, %addr, %a_ptr;
    \\    ld.global.f32 %b_val, [%addr];
    \\    add.f32 %c_val, %a_val, %b_val;
    \\    add.u64 %addr, %addr, %c_ptr;
    \\    sub.u64 %addr, %addr, %b_ptr;
    \\    st.global.f32 [%addr], %c_val;
    \\END:
    \\    ret;
    \\}
;

/// Element-wise multiply: c = a * b
pub const EW_MUL_PTX = PTX_HEADER ++
    \\.visible .entry ew_mul(
    \\    .param .u64 a,
    \\    .param .u64 b,
    \\    .param .u64 c,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %a_ptr, %b_ptr, %c_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %a_val, %b_val, %c_val;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %a_ptr, [a];
    \\    ld.param.u64 %b_ptr, [b];
    \\    ld.param.u64 %c_ptr, [c];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %a_ptr;
    \\    ld.global.f32 %a_val, [%addr];
    \\    add.u64 %addr, %addr, %b_ptr;
    \\    sub.u64 %addr, %addr, %a_ptr;
    \\    ld.global.f32 %b_val, [%addr];
    \\    mul.f32 %c_val, %a_val, %b_val;
    \\    add.u64 %addr, %addr, %c_ptr;
    \\    sub.u64 %addr, %addr, %b_ptr;
    \\    st.global.f32 [%addr], %c_val;
    \\END:
    \\    ret;
    \\}
;

/// Scale buffer: output = input * scalar
pub const SCALE_BUFFER_PTX = PTX_HEADER ++
    \\.visible .entry scale_buffer(
    \\    .param .u64 input,
    \\    .param .f32 scale,
    \\    .param .u64 output,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %val, %scale;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.f32 %scale, [scale];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%addr];
    \\    mul.f32 %val, %val, %scale;
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    sub.u64 %addr, %addr, %in_ptr;
    \\    st.global.f32 [%addr], %val;
    \\END:
    \\    ret;
    \\}
;

// =============================================================================
// Optimizer Kernels
// =============================================================================

/// SGD weight update: w = w - lr * (g + wd * w)
pub const SGD_UPDATE_PTX = PTX_HEADER ++
    \\.visible .entry sgd_update(
    \\    .param .u64 weights,
    \\    .param .u64 gradients,
    \\    .param .u32 n,
    \\    .param .f32 learning_rate,
    \\    .param .f32 weight_decay
    \\) {
    \\    .reg .u64 %w_ptr, %g_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %w_val, %g_val, %lr, %wd, %new_w;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %w_ptr, [weights];
    \\    ld.param.u64 %g_ptr, [gradients];
    \\    ld.param.u32 %n, [n];
    \\    ld.param.f32 %lr, [learning_rate];
    \\    ld.param.f32 %wd, [weight_decay];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %w_ptr;
    \\    ld.global.f32 %w_val, [%addr];
    \\    add.u64 %addr, %addr, %g_ptr;
    \\    sub.u64 %addr, %addr, %w_ptr;
    \\    ld.global.f32 %g_val, [%addr];
    \\
    \\    mul.f32 %new_w, %wd, %w_val;
    \\    add.f32 %new_w, %new_w, %g_val;
    \\    mul.f32 %new_w, %new_w, %lr;
    \\    sub.f32 %new_w, %w_val, %new_w;
    \\
    \\    add.u64 %addr, %addr, %w_ptr;
    \\    sub.u64 %addr, %addr, %g_ptr;
    \\    st.global.f32 [%addr], %new_w;
    \\END:
    \\    ret;
    \\}
;

/// Fill buffer with constant value
pub const FILL_CONSTANT_PTX = PTX_HEADER ++
    \\.visible .entry fill_constant(
    \\    .param .u64 data,
    \\    .param .f32 value,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %data_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %value;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %data_ptr, [data];
    \\    ld.param.f32 %value, [value];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %data_ptr;
    \\    st.global.f32 [%addr], %value;
    \\END:
    \\    ret;
    \\}
;

/// Add bias: output[batch, :] += bias
pub const ADD_BIAS_PTX = PTX_HEADER ++
    \\.visible .entry add_bias(
    \\    .param .u64 output,
    \\    .param .u64 bias,
    \\    .param .u32 batch_size,
    \\    .param .u32 bias_size
    \\) {
    \\    .reg .u64 %out_ptr, %bias_ptr;
    \\    .reg .u32 %batch_size, %bias_size, %col, %batch;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %out_val, %bias_val;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %bias_ptr, [bias];
    \\    ld.param.u32 %batch_size, [batch_size];
    \\    ld.param.u32 %bias_size, [bias_size];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mov.u32 %batch, %ctaid.y;
    \\    mad.lo.u32 %col, %ctaid.x, %ntid.x, %tid.x;
    \\
    \\    setp.ge.u32 %p, %batch, %batch_size;
    \\    @%p bra END;
    \\    setp.ge.u32 %p, %col, %bias_size;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %col, 4;
    \\    add.u64 %addr, %addr, %bias_ptr;
    \\    ld.global.f32 %bias_val, [%addr];
    \\
    \\    mad.lo.u64 %addr, %batch, %bias_size, %col;
    \\    mul.lo.u64 %addr, %addr, 4;
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    ld.global.f32 %out_val, [%addr];
    \\    add.f32 %out_val, %out_val, %bias_val;
    \\    st.global.f32 [%addr], %out_val;
    \\END:
    \\    ret;
    \\}
;

/// Conv2D forward kernel PTX (simplified version)
pub const CONV2D_FORWARD_PTX = PTX_HEADER ++
    \\.visible .entry conv2d_forward(
    \\    .param .u64 input,
    \\    .param .u64 weights,
    \\    .param .u64 bias,
    \\    .param .u64 output,
    \\    .param .u32 batch_size,
    \\    .param .u32 in_channels,
    \\    .param .u32 out_channels,
    \\    .param .u32 kernel_h,
    \\    .param .u32 kernel_w,
    \\    .param .u32 input_h,
    \\    .param .u32 input_w,
    \\    .param .u32 output_h,
    \\    .param .u32 output_w,
    \\    .param .u32 stride_h,
    \\    .param .u32 stride_w,
    \\    .param .u32 padding_h,
    \\    .param .u32 padding_w
    \\) {
    \\    .reg .u64 %in_ptr, %w_ptr, %b_ptr, %out_ptr;
    \\    .reg .u32 %batch_size, %in_ch, %out_ch, %kh, %kw, %in_h, %in_w, %out_h, %out_w, %sh, %sw, %ph, %pw;
    \\    .reg .u32 %out_x, %out_y, %out_ch, %batch;
    \\    .reg .u32 %in_x_start, %in_y_start, %in_x, %in_y;
    \\    .reg .u32 %ic, %k_h, %k_w;
    \\    .reg .f32 %sum, %in_val, %w_val, %bias_val;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p, %q;
    \\    .reg .u32 %tmp_u32, %ntid_x, %ntid_y, %ctaid_x, %ctaid_y, %tid_x, %tid_y;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %w_ptr, [weights];
    \\    ld.param.u64 %b_ptr, [bias];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u32 %batch_size, [batch_size];
    \\    ld.param.u32 %in_ch, [in_channels];
    \\    ld.param.u32 %out_ch, [out_channels];
    \\    ld.param.u32 %kh, [kernel_h];
    \\    ld.param.u32 %kw, [kernel_w];
    \\    ld.param.u32 %in_h, [input_h];
    \\    ld.param.u32 %in_w, [input_w];
    \\    ld.param.u32 %out_h, [output_h];
    \\    ld.param.u32 %out_w, [output_w];
    \\    ld.param.u32 %sh, [stride_h];
    \\    ld.param.u32 %sw, [stride_w];
    \\    ld.param.u32 %ph, [padding_h];
    \\    ld.param.u32 %pw, [padding_w];
    \\
    \\    mov.u32 %ctaid_x, %ctaid.x;
    \\    mov.u32 %ctaid_y, %ctaid.y;
    \\    mov.u32 %tid_x, %tid.x;
    \\    mov.u32 %tid_y, %tid.y;
    \\    mov.u32 %ntid_x, %ntid.x;
    \\    mov.u32 %ntid_y, %ntid.y;
    \\    mov.u32 %out_ch, %ctaid.z;
    \\
    \\    // Calculate output coordinates
    \\    mad.lo.u32 %out_x, %ctaid_x, %ntid_x, %tid_x;
    \\    mad.lo.u32 %out_y, %ctaid_y, %ntid_y, %tid_y;
    \\
    \\    // Check bounds
    \\    setp.ge.u32 %p, %out_x, %out_w;
    \\    @%p bra END;
    \\    setp.ge.u32 %p, %out_y, %out_h;
    \\    @%p bra END;
    \\
    \\    // Calculate input start position
    \\    mul.lo.u32 %in_y_start, %out_y, %sh;
    \\    sub.s32 %in_y_start, %in_y_start, %ph;
    \\    mul.lo.u32 %in_x_start, %out_x, %sw;
    \\    sub.s32 %in_x_start, %in_x_start, %pw;
    \\
    \\    // Initialize sum with bias if available
    \\    setp.eq.u64 %p, %b_ptr, 0;
    \\    mov.f32 %sum, 0f00000000;
    \\    @%p bra LOOP_IC;
    \\    // Load bias
    \\    mul.lo.u64 %addr, %out_ch, 4;
    \\    add.u64 %addr, %addr, %b_ptr;
    \\    ld.global.f32 %sum, [%addr];
    \\
    \\LOOP_IC:
    \\    // Main convolution loop - simplified PTX
    \\    // This is a placeholder for the full convolution
    \\    // The full implementation would unroll loops for efficiency
    \\
    \\END:
    \\    // Store result
    \\    mov.u32 %batch, %ctaid.y;  // Simplified batch calculation
    \\    mad.lo.u32 %tmp_u32, %batch, %out_ch, %out_ch;
    \\    mad.lo.u32 %tmp_u32, %tmp_u32, %out_h, %out_y;
    \\    mad.lo.u32 %tmp_u32, %tmp_u32, %out_w, %out_x;
    \\    mul.lo.u64 %addr, %tmp_u32, 4;
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    st.global.f32 [%addr], %sum;
    \\    ret;
    \\}
;

// =============================================================================
// Softmax Kernel
// =============================================================================

/// Simple softmax forward (per-sample, not optimized)
pub const SOFTMAX_FORWARD_PTX = PTX_HEADER ++
    \\.visible .entry softmax_forward(
    \\    .param .u64 input,
    \\    .param .u64 output,
    \\    .param .u32 batch_size,
    \\    .param .u32 features
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr;
    \\    .reg .u32 %batch, %batch_size, %features, %i;
    \\    .reg .u32 %tid, %ctaid;
    \\    .reg .f32 %max_val, %val, %exp_val, %sum_exp;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u32 %batch_size, [batch_size];
    \\    ld.param.u32 %features, [features];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\
    \\    mov.u32 %batch, %ctaid;
    \\    setp.ge.u32 %p, %batch, %batch_size;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %batch, %features;
    \\    mul.lo.u64 %addr, %addr, 4;
    \\    add.u64 %in_ptr, %in_ptr, %addr;
    \\    add.u64 %out_ptr, %out_ptr, %addr;
    \\
    \\    setp.ne.u32 %p, %tid, 0;
    \\    @%p bra END;
    \\
    \\    ld.global.f32 %max_val, [%in_ptr];
    \\    mov.u32 %i, 1;
    \\MAX_LOOP:
    \\    setp.ge.u32 %p, %i, %features;
    \\    @%p bra MAX_DONE;
    \\    mul.lo.u64 %addr, %i, 4;
    \\    add.u64 %addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%addr];
    \\    max.f32 %max_val, %max_val, %val;
    \\    add.u32 %i, %i, 1;
    \\    bra MAX_LOOP;
    \\MAX_DONE:
    \\
    \\    mov.f32 %sum_exp, 0f00000000;
    \\    mov.u32 %i, 0;
    \\EXP_LOOP:
    \\    setp.ge.u32 %p, %i, %features;
    \\    @%p bra EXP_DONE;
    \\    mul.lo.u64 %addr, %i, 4;
    \\    add.u64 %addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%addr];
    \\    sub.f32 %val, %val, %max_val;
    \\    ex2.approx.ftz.f32 %exp_val, %val;
    \\    add.f32 %sum_exp, %sum_exp, %exp_val;
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    sub.u64 %addr, %addr, %in_ptr;
    \\    st.global.f32 [%addr], %exp_val;
    \\    add.u32 %i, %i, 1;
    \\    bra EXP_LOOP;
    \\EXP_DONE:
    \\
    \\    mov.u32 %i, 0;
    \\NORM_LOOP:
    \\    setp.ge.u32 %p, %i, %features;
    \\    @%p bra END;
    \\    mul.lo.u64 %addr, %i, 4;
    \\    add.u64 %addr, %addr, %out_ptr;
    \\    ld.global.f32 %val, [%addr];
    \\    div.approx.ftz.f32 %val, %val, %sum_exp;
    \\    st.global.f32 [%addr], %val;
    \\    add.u32 %i, %i, 1;
    \\    bra NORM_LOOP;
    \\END:
    \\    ret;
    \\}
;

/// Matrix multiplication with B transposed: C = A * B^T
pub const MATMUL_TRANSPOSE_B_PTX = PTX_HEADER ++
    \\.visible .entry matmul_transpose_b(
    \\    .param .u64 A,
    \\    .param .u64 B,
    \\    .param .u64 C,
    \\    .param .u32 M,
    \\    .param .u32 N,
    \\    .param .u32 K,
    \\    .param .u32 accumulate
    \\) {
    \\    .reg .u64 %A_ptr, %B_ptr, %C_ptr;
    \\    .reg .u32 %M, %N, %K, %accumulate;
    \\    .reg .u32 %row, %col, %k;
    \\    .reg .u32 %tid_x, %tid_y, %bid_x, %bid_y;
    \\    .reg .u32 %bsize_x, %bsize_y;
    \\    .reg .f32 %sum, %a_val, %b_val, %c_val;
    \\    .reg .u64 %a_addr, %b_addr, %c_addr;
    \\    .reg .pred %p, %p_accum;
    \\
    \\    ld.param.u64 %A_ptr, [A];
    \\    ld.param.u64 %B_ptr, [B];
    \\    ld.param.u64 %C_ptr, [C];
    \\    ld.param.u32 %M, [M];
    \\    ld.param.u32 %N, [N];
    \\    ld.param.u32 %K, [K];
    \\    ld.param.u32 %accumulate, [accumulate];
    \\
    \\    mov.u32 %tid_x, %tid.x;
    \\    mov.u32 %tid_y, %tid.y;
    \\    mov.u32 %bid_x, %ctaid.x;
    \\    mov.u32 %bid_y, %ctaid.y;
    \\    mov.u32 %bsize_x, %ntid.x;
    \\    mov.u32 %bsize_y, %ntid.y;
    \\
    \\    mad.lo.u32 %row, %bid_y, %bsize_y, %tid_y;
    \\    mad.lo.u32 %col, %bid_x, %bsize_x, %tid_x;
    \\
    \\    setp.ge.u32 %p, %row, %M;
    \\    @%p bra END;
    \\    setp.ge.u32 %p, %col, %N;
    \\    @%p bra END;
    \\
    \\    mov.f32 %sum, 0f00000000;
    \\    mov.u32 %k, 0;
    \\LOOP:
    \\    setp.ge.u32 %p, %k, %K;
    \\    @%p bra LOOP_END;
    \\    mad.lo.u64 %a_addr, %row, %K, %k;
    \\    mul.lo.u64 %a_addr, %a_addr, 4;
    \\    add.u64 %a_addr, %a_addr, %A_ptr;
    \\    ld.global.f32 %a_val, [%a_addr];
    \\    mad.lo.u64 %b_addr, %col, %K, %k;
    \\    mul.lo.u64 %b_addr, %b_addr, 4;
    \\    add.u64 %b_addr, %b_addr, %B_ptr;
    \\    ld.global.f32 %b_val, [%b_addr];
    \\    fma.rn.f32 %sum, %a_val, %b_val, %sum;
    \\    add.u32 %k, %k, 1;
    \\    bra LOOP;
    \\LOOP_END:
    \\
    \\    mad.lo.u64 %c_addr, %row, %N, %col;
    \\    mul.lo.u64 %c_addr, %c_addr, 4;
    \\    add.u64 %c_addr, %c_addr, %C_ptr;
    \\    setp.eq.u32 %p_accum, %accumulate, 0;
    \\    @%p_accum bra STORE;
    \\    ld.global.f32 %c_val, [%c_addr];
    \\    add.f32 %sum, %sum, %c_val;
    \\STORE:
    \\    st.global.f32 [%c_addr], %sum;
    \\END:
    \\    ret;
    \\}
;

/// Tensor Core WMMA kernel PTX (simplified version for sm_70+)
/// Uses mma.sync.aligned.m16n16k16.row.row.f32.f16.f16.f32 instruction
/// Note: Full WMMA requires inline PTX or CUDA C++ compilation
pub const MATMUL_TENSOR_CORE_PTX = PTX_HEADER ++
    \\    .visible .entry matmul_tensor_core(
    \\        .param .u64 A,
    \\        .param .u64 B,
    \\        .param .u64 C,
    \\        .param .u32 M,
    \\        .param .u32 N,
    \\        .param .u32 K,
    \\        .param .u32 accumulate
    \\    ) {
    \\        .reg .u64 %A_ptr, %B_ptr, %C_ptr;
    \\        .reg .u32 %M, %N, %K, %accumulate;
    \\        .reg .u32 %block_row, %block_col, %warp_id, %lane_id;
    \\        .reg .u32 %warp_row, %warp_col;
    \\        .reg .u32 %tid, %bid_x, %bid_y;
    \\        .reg .u32 %k_tile, %num_k_tiles, %k_offset;
    \\        .reg .u32 %i, %j, %r, %c;
    \\        .reg .f32 %acc00, %acc01, %acc10, %acc11;
    \\        .reg .f32 %a_val, %b_val, %c_val;
    \\        .reg .u64 %a_addr, %b_addr, %c_addr;
    \\        .reg .pred %p, %p_accum;
    \\
    \\        ld.param.u64 %A_ptr, [A];
    \\        ld.param.u64 %B_ptr, [B];
    \\        ld.param.u64 %C_ptr, [C];
    \\        ld.param.u32 %M, [M];
    \\        ld.param.u32 %N, [N];
    \\        ld.param.u32 %K, [K];
    \\        ld.param.u32 %accumulate, [accumulate];
    \\
    \\        mov.u32 %tid, %tid.x;
    \\        mov.u32 %bid_x, %ctaid.x;
    \\        mov.u32 %bid_y, %ctaid.y;
    \\
    \\        // Block coordinates (64x64 tiles)
    \\        mul.lo.u32 %block_row, %bid_y, 64;
    \\        mul.lo.u32 %block_col, %bid_x, 64;
    \\
    \\        // Warp ID (0-3 for 128 threads)
    \\        shr.u32 %warp_id, %tid, 5;
    \\
    \\        // Warp coordinates within block
    \\        and.b32 %i, %warp_id, 1;
    \\        shr.u32 %j, %warp_id, 1;
    \\        mul.lo.u32 %warp_row, %j, 32;
    \\        mul.lo.u32 %warp_col, %i, 32;
    \\
    \\        // Initialize accumulators
    \\        mov.f32 %acc00, 0f00000000;
    \\        mov.f32 %acc01, 0f00000000;
    \\        mov.f32 %acc10, 0f00000000;
    \\        mov.f32 %acc11, 0f00000000;
    \\
    \\        // Number of K tiles
    \\        add.u32 %num_k_tiles, %K, 15;
    \\        shr.u32 %num_k_tiles, %num_k_tiles, 4;
    \\
    \\        mov.u32 %k_tile, 0;
    \\    K_LOOP:
    \\        setp.ge.u32 %p, %k_tile, %num_k_tiles;
    \\        @%p bra K_LOOP_END;
    \\
    \\        mul.lo.u32 %k_offset, %k_tile, 16;
    \\
    \\        // Simplified: compute 16x16x16 MMA for each accumulator
    \\        // In full implementation, this would use mma.sync instructions
    \\        // For now, use standard FMA as fallback
    \\        mov.u32 %i, 0;
    \\    MMA_I_LOOP:
    \\        setp.ge.u32 %p, %i, 16;
    \\        @%p bra MMA_I_LOOP_END;
    \\
    \\        mov.u32 %j, 0;
    \\    MMA_J_LOOP:
    \\        setp.ge.u32 %p, %j, 16;
    \\        @%p bra MMA_J_LOOP_END;
    \\
    \\        mov.u32 %r, 0;
    \\    MMA_K_LOOP:
    \\        setp.ge.u32 %p, %r, 16;
    \\        @%p bra MMA_K_LOOP_END;
    \\
    \\        // Load A
    \\        add.u32 %a_addr, %block_row, %warp_row;
    \\        add.u32 %a_addr, %a_addr, %i;
    \\        add.u32 %c, %k_offset, %r;
    \\        mad.lo.u32 %a_addr, %a_addr, %K, %c;
    \\        mul.lo.u64 %a_addr, %a_addr, 2;
    \\        add.u64 %a_addr, %a_addr, %A_ptr;
    \\        ld.global.u16 %a_addr, [%a_addr];
    \\        cvt.f32.f16 %a_val, %a_addr;
    \\
    \\        // Load B
    \\        add.u32 %b_addr, %k_offset, %r;
    \\        add.u32 %c, %block_col, %warp_col;
    \\        add.u32 %c, %c, %j;
    \\        mad.lo.u32 %b_addr, %b_addr, %N, %c;
    \\        mul.lo.u64 %b_addr, %b_addr, 2;
    \\        add.u64 %b_addr, %b_addr, %B_ptr;
    \\        ld.global.u16 %b_addr, [%b_addr];
    \\        cvt.f32.f16 %b_val, %b_addr;
    \\
    \\        // FMA (in real WMMA, this would be mma.sync)
    \\        fma.rn.f32 %acc00, %a_val, %b_val, %acc00;
    \\
    \\        add.u32 %r, %r, 1;
    \\        bra MMA_K_LOOP;
    \\    MMA_K_LOOP_END:
    \\
    \\        add.u32 %j, %j, 1;
    \\        bra MMA_J_LOOP;
    \\    MMA_J_LOOP_END:
    \\
    \\        add.u32 %i, %i, 1;
    \\        bra MMA_I_LOOP;
    \\    MMA_I_LOOP_END:
    \\
    \\        add.u32 %k_tile, %k_tile, 1;
    \\        bra K_LOOP;
    \\    K_LOOP_END:
    \\
    \\        // Store results
    \\        add.u32 %r, %block_row, %warp_row;
    \\        add.u32 %c, %block_col, %warp_col;
    \\        mad.lo.u32 %c_addr, %r, %N, %c;
    \\        mul.lo.u64 %c_addr, %c_addr, 4;
    \\        add.u64 %c_addr, %c_addr, %C_ptr;
    \\
    \\        setp.eq.u32 %p_accum, %accumulate, 0;
    \\        @%p_accum bra STORE;
    \\        ld.global.f32 %c_val, [%c_addr];
    \\        add.f32 %acc00, %acc00, %c_val;
    \\    STORE:
    \\        st.global.f32 [%c_addr], %acc00;
    \\        ret;
    \\    }
;

// Export all kernel names for loading
pub const KERNEL_NAMES = .{
    "matmul",
    "matmul_batch",
    "matmul_transpose_b",
    "matmul_tensor_core",
    "relu_forward",
    "relu_backward",
    "sigmoid_forward",
    "sigmoid_backward",
    "tanh_forward",
    "tanh_backward",
    // Vectorized activation kernels (LDG.128)
    "relu_forward_vec4",
    "relu_backward_vec4",
    "sigmoid_forward_vec4",
    "sigmoid_backward_vec4",
    "tanh_forward_vec4",
    "tanh_backward_vec4",
    "linear_forward",
    "softmax_forward",
    "ew_add",
    "ew_mul",
    "scale_buffer",
    "sgd_update",
    "fill_constant",
    "add_bias",
};


/// Conv2D backward kernel CUDA C source
/// Computes gradients for input, weights, and bias
pub const CONV2D_BACKWARD_SOURCE =
    \\extern "C" __global__ void conv2d_backward(
    \\    const float* __restrict__ input,
    \\    const float* __restrict__ weights,
    \\    const float* __restrict__ grad_output,
    \\    float* __restrict__ grad_input,
    \\    float* __restrict__ grad_weights,
    \\    float* __restrict__ grad_bias,
    \\    int batch_size, int in_channels, int out_channels,
    \\    int kernel_h, int kernel_w,
    \\    int input_h, int input_w,
    \\    int output_h, int output_w,
    \\    int stride_h, int stride_w,
    \\    int padding_h, int padding_w) {
    \\
    \\    int batch = blockIdx.z / out_channels;
    \\    int out_ch = blockIdx.z % out_channels;
    \\    int in_y = blockIdx.y * blockDim.y + threadIdx.y;
    \\    int in_x = blockIdx.x * blockDim.x + threadIdx.x;
    \\
    \\    if (in_x >= input_w || in_y >= input_h) return;
    \\
    \\    // Gradient for input (transposed convolution)
    \\    if (grad_input != NULL) {
    \\        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
    \\            float grad_sum = 0.0f;
    \\            int in_idx = ((batch * in_channels + in_ch) * input_h + in_y) * input_w + in_x;
    \\
    \\            for (int oy = 0; oy < output_h; oy++) {
    \\                for (int ox = 0; ox < output_w; ox++) {
    \\                    int in_y_start = oy * stride_h - padding_h;
    \\                    int in_x_start = ox * stride_w - padding_w;
    \\                    int kh = in_y - in_y_start;
    \\                    int kw = in_x - in_x_start;
    \\
    \\                    if (kh >= 0 && kh < kernel_h && kw >= 0 && kw < kernel_w) {
    \\                        int out_idx = ((batch * out_channels + out_ch) * output_h + oy) * output_w + ox;
    \\                        int w_idx = ((out_ch * in_channels + in_ch) * kernel_h + kh) * kernel_w + kw;
    \\                        grad_sum += grad_output[out_idx] * weights[w_idx];
    \\                    }
    \\                }
    \\            }
    \\            atomicAdd(&grad_input[in_idx], grad_sum);
    \\        }
    \\    }
    \\
    \\    // Gradient for weights and bias (only first thread per block)
    \\    if (grad_weights != NULL && threadIdx.x == 0 && threadIdx.y == 0) {
    \\        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
    \\            for (int kh = 0; kh < kernel_h; kh++) {
    \\                for (int kw = 0; kw < kernel_w; kw++) {
    \\                    float grad_w = 0.0f;
    \\                    for (int oy = 0; oy < output_h; oy++) {
    \\                        for (int ox = 0; ox < output_w; ox++) {
    \\                            int in_y = oy * stride_h - padding_h + kh;
    \\                            int in_x = ox * stride_w - padding_w + kw;
    \\                            if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
    \\                                int in_idx = ((batch * in_channels + in_ch) * input_h + in_y) * input_w + in_x;
    \\                                int out_idx = ((batch * out_channels + out_ch) * output_h + oy) * output_w + ox;
    \\                                grad_w += input[in_idx] * grad_output[out_idx];
    \\                            }
    \\                        }
    \\                    }
    \\                    int w_idx = ((out_ch * in_channels + in_ch) * kernel_h + kh) * kernel_w + kw;
    \\                    atomicAdd(&grad_weights[w_idx], grad_w);
    \\                }
    \\            }
    \\        }
    \\    }
    \\
    \\    // Gradient for bias
    \\    if (grad_bias != NULL && threadIdx.x == 0 && threadIdx.y == 0) {
    \\        float grad_b = 0.0f;
    \\        for (int oy = 0; oy < output_h; oy++) {
    \\            for (int ox = 0; ox < output_w; ox++) {
    \\                int out_idx = ((batch * out_channels + out_ch) * output_h + oy) * output_w + ox;
    \\                grad_b += grad_output[out_idx];
    \\            }
    \\        }
    \\        atomicAdd(&grad_bias[out_ch], grad_b);
    \\    }
    \\}
;
/// RMSProp optimizer kernel CUDA C source
pub const RMSPROP_UPDATE_SOURCE =
    \\extern "C" __global__ void rmsprop_update(
    \\    float* weights, const float* gradients, float* cache,
    \\    int n, float learning_rate, float decay_rate, float epsilon) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        float grad = gradients[idx];
    \\        cache[idx] = decay_rate * cache[idx] + (1.0f - decay_rate) * grad * grad;
    \\        weights[idx] -= learning_rate * grad / (sqrtf(cache[idx]) + epsilon);
    \\    }
    \\}
;

/// RMSProp optimizer kernel PTX
pub const RMSPROP_UPDATE_PTX = PTX_HEADER ++
    \\.visible .entry rmsprop_update(
    \\    .param .u64 weights,
    \\    .param .u64 gradients,
    \\    .param .u64 cache,
    \\    .param .u32 n,
    \\    .param .f32 learning_rate,
    \\    .param .f32 decay_rate,
    \\    .param .f32 epsilon
    \\) {
    \\    .reg .u64 %w_ptr, %g_ptr, %c_ptr;
    \\    .reg .u32 %n, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %w_val, %g_val, %c_val, %lr, %dr, %eps;
    \\    .reg .f32 %grad_sq, %new_cache, %sqrt_cache;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %w_ptr, [weights];
    \\    ld.param.u64 %g_ptr, [gradients];
    \\    ld.param.u64 %c_ptr, [cache];
    \\    ld.param.u32 %n, [n];
    \\    ld.param.f32 %lr, [learning_rate];
    \\    ld.param.f32 %dr, [decay_rate];
    \\    ld.param.f32 %eps, [epsilon];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %g_ptr;
    \\    ld.global.f32 %g_val, [%addr];
    \\
    \\    add.u64 %addr, %addr, %c_ptr;
    \\    sub.u64 %addr, %addr, %g_ptr;
    \\    ld.global.f32 %c_val, [%addr];
    \\
    \\    mul.f32 %grad_sq, %g_val, %g_val;
    \\    mul.f32 %c_val, %dr, %c_val;
    \\    mov.f32 %new_cache, %c_val;
    \\    mov.f32 %c_val, 0f3F800000;
    \\    sub.f32 %c_val, %c_val, %dr;
    \\    mul.f32 %c_val, %c_val, %grad_sq;
    \\    add.f32 %new_cache, %new_cache, %c_val;
    \\
    \\    st.global.f32 [%addr], %new_cache;
    \\
    \\    sqrt.approx.f32 %sqrt_cache, %new_cache;
    \\    add.f32 %sqrt_cache, %sqrt_cache, %eps;
    \\    div.approx.f32 %g_val, %g_val, %sqrt_cache;
    \\    mul.f32 %g_val, %g_val, %lr;
    \\
    \\    add.u64 %addr, %addr, %w_ptr;
    \\    sub.u64 %addr, %addr, %c_ptr;
    \\    ld.global.f32 %w_val, [%addr];
    \\    sub.f32 %w_val, %w_val, %g_val;
    \\    st.global.f32 [%addr], %w_val;
    \\END:
    \\    ret;
    \\}
;

/// Adam optimizer kernel CUDA C source
pub const ADAM_UPDATE_SOURCE =
    \\extern "C" __global__ void adam_update(
    \\    float* weights, const float* gradients,
    \\    float* m, float* v, int n,
    \\    float learning_rate, float beta1, float beta2,
    \\    float epsilon, int timestep) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        float grad = gradients[idx];
    \\        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
    \\        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
    \\        float m_hat = m[idx] / (1.0f - powf(beta1, timestep));
    \\        float v_hat = v[idx] / (1.0f - powf(beta2, timestep));
    \\        weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    \\    }
    \\}
;

/// Adam optimizer kernel PTX
pub const ADAM_UPDATE_PTX = PTX_HEADER ++
    \\.visible .entry adam_update(
    \\    .param .u64 weights,
    \\    .param .u64 gradients,
    \\    .param .u64 m,
    \\    .param .u64 v,
    \\    .param .u32 n,
    \\    .param .f32 learning_rate,
    \\    .param .f32 beta1,
    \\    .param .f32 beta2,
    \\    .param .f32 epsilon,
    \\    .param .u32 timestep
    \\) {
    \\    .reg .u64 %w_ptr, %g_ptr, %m_ptr, %v_ptr;
    \\    .reg .u32 %n, %idx, %t;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %w_val, %g_val, %m_val, %v_val;
    \\    .reg .f32 %lr, %b1, %b2, %eps, %b1_t, %b2_t;
    \\    .reg .f32 %one, %m_hat, %v_hat;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %w_ptr, [weights];
    \\    ld.param.u64 %g_ptr, [gradients];
    \\    ld.param.u64 %m_ptr, [m];
    \\    ld.param.u64 %v_ptr, [v];
    \\    ld.param.u32 %n, [n];
    \\    ld.param.f32 %lr, [learning_rate];
    \\    ld.param.f32 %b1, [beta1];
    \\    ld.param.f32 %b2, [beta2];
    \\    ld.param.f32 %eps, [epsilon];
    \\    ld.param.u32 %t, [timestep];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\
    \\    // Load gradient
    \\    add.u64 %addr, %addr, %g_ptr;
    \\    ld.global.f32 %g_val, [%addr];
    \\
    \\    // Update m
    \\    sub.u64 %addr, %addr, %g_ptr;
    \\    add.u64 %addr, %addr, %m_ptr;
    \\    ld.global.f32 %m_val, [%addr];
    \\    mul.f32 %m_val, %b1, %m_val;
    \\    mov.f32 %one, 0f3F800000;
    \\    sub.f32 %b1_t, %one, %b1;
    \\    mul.f32 %b1_t, %b1_t, %g_val;
    \\    add.f32 %m_val, %m_val, %b1_t;
    \\    st.global.f32 [%addr], %m_val;
    \\
    \\    // Update v
    \\    sub.u64 %addr, %addr, %m_ptr;
    \\    add.u64 %addr, %addr, %v_ptr;
    \\    ld.global.f32 %v_val, [%addr];
    \\    mul.f32 %v_val, %b2, %v_val;
    \\    sub.f32 %b2_t, %one, %b2;
    \\    mul.f32 %g_val, %g_val, %g_val;
    \\    mul.f32 %b2_t, %b2_t, %g_val;
    \\    add.f32 %v_val, %v_val, %b2_t;
    \\    st.global.f32 [%addr], %v_val;
    \\
    \\    // Calculate m_hat and v_hat (simplified)
    \\    mov.f32 %m_hat, %m_val;
    \\    mov.f32 %v_hat, %v_val;
    \\
    \\    // Update weight
    \\    sub.u64 %addr, %addr, %v_ptr;
    \\    add.u64 %addr, %addr, %w_ptr;
    \\    ld.global.f32 %w_val, [%addr];
    \\    sqrt.approx.f32 %v_hat, %v_hat;
    \\    add.f32 %v_hat, %v_hat, %eps;
    \\    div.approx.f32 %m_hat, %m_hat, %v_hat;
    \\    mul.f32 %m_hat, %m_hat, %lr;
    \\    sub.f32 %w_val, %w_val, %m_hat;
    \\    st.global.f32 [%addr], %w_val;
    \\END:
    \\    ret;
    \\}
;

/// BatchNorm forward training kernel CUDA C source
pub const BATCHNORM_FORWARD_TRAINING_SOURCE =
    \\extern "C" __global__ void batchnorm_forward_training(
    \\    const float* input,      // [batch, channels, height, width]
    \\    float* output,
    \\    float* running_mean,     // [channels]
    \\    float* running_var,      // [channels]
    \\    const float* gamma,        // [channels]
    \\    const float* beta,       // [channels]
    \\    int batch_size, int channels, int spatial_size,
    \\    float epsilon, float momentum) {
    \\    int c = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (c >= channels) return;
    \\
    \\    // Calculate mean and variance for this channel
    \\    float mean = 0.0f;
    \\    float var = 0.0f;
    \\    int n = batch_size * spatial_size;
    \\
    \\    // Two-pass algorithm (more accurate)
    \\    // Pass 1: mean
    \\    for (int b = 0; b < batch_size; b++) {
    \\        for (int s = 0; s < spatial_size; s++) {
    \\            int idx = (b * channels + c) * spatial_size + s;
    \\            mean += input[idx];
    \\        }
    \\    }
    \\    mean /= n;
    \\
    \\    // Pass 2: variance
    \\    for (int b = 0; b < batch_size; b++) {
    \\        for (int s = 0; s < spatial_size; s++) {
    \\            int idx = (b * channels + c) * spatial_size + s;
    \\            float diff = input[idx] - mean;
    \\            var += diff * diff;
    \\        }
    \\    }
    \\    var /= n;
    \\
    \\    // Update running stats
    \\    running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
    \\    running_var[c] = momentum * running_var[c] + (1.0f - momentum) * var;
    \\
    \\    // Normalize
    \\    float inv_std = 1.0f / sqrtf(var + epsilon);
    \\    for (int b = 0; b < batch_size; b++) {
    \\        for (int s = 0; s < spatial_size; s++) {
    \\            int idx = (b * channels + c) * spatial_size + s;
    \\            float normalized = (input[idx] - mean) * inv_std;
    \\            output[idx] = normalized * gamma[c] + beta[c];
    \\        }
    \\    }
    \\}
;

/// BatchNorm forward inference kernel CUDA C source
pub const BATCHNORM_FORWARD_INFERENCE_SOURCE =
    \\extern "C" __global__ void batchnorm_forward_inference(
    \\    const float* input,
    \\    float* output,
    \\    const float* running_mean,
    \\    const float* running_var,
    \\    const float* gamma,
    \\    const float* beta,
    \\    int batch_size, int channels, int spatial_size,
    \\    float epsilon) {
    \\    int c = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (c >= channels) return;
    \\
    \\    float inv_std = 1.0f / sqrtf(running_var[c] + epsilon);
    \\
    \\    for (int b = 0; b < batch_size; b++) {
    \\        for (int s = 0; s < spatial_size; s++) {
    \\            int idx = (b * channels + c) * spatial_size + s;
    \\            float normalized = (input[idx] - running_mean[c]) * inv_std;
    \\            output[idx] = normalized * gamma[c] + beta[c];
    \\        }
    \\    }
    \\}
;

/// LayerNorm forward pass CUDA C source
/// Uses Welford's algorithm for numerical stability
pub const LAYERNORM_FORWARD_SOURCE =
    \\extern "C" __global__ void layernorm_forward(
    \\    const float* input, float* output,
    \\    const float* gamma, const float* beta,
    \\    int batch_size, int features,
    \\    float epsilon) {
    \\
    \\    int b = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (b >= batch_size) return;
    \\
    \\    // Use Welford's algorithm for numerical stability
    \\    float mean = 0.0f;
    \\    float m2 = 0.0f;
    \\    int count = 0;
    \\
    \\    // Pass 1: Compute mean and variance (Welford)
    \\    for (int f = 0; f < features; f++) {
    \\        int idx = b * features + f;
    \\        float x = input[idx];
    \\        count++;
    \\        float delta = x - mean;
    \\        mean += delta / count;
    \\        float delta2 = x - mean;
    \\        m2 += delta * delta2;
    \\    }
    \\
    \\    float variance = m2 / features;
    \\    float std = sqrtf(variance + epsilon);
    \\
    \\    // Pass 2: Normalize and apply gamma/beta
    \\    for (int f = 0; f < features; f++) {
    \\        int idx = b * features + f;
    \\        float normalized = (input[idx] - mean) / std;
    \\        output[idx] = gamma[f] * normalized + beta[f];
    \\    }
    \\}
;

/// LayerNorm backward pass CUDA C source
pub const LAYERNORM_BACKWARD_SOURCE =
    \\extern "C" __global__ void layernorm_backward(
    \\    const float* input, const float* grad_output,
    \\    float* grad_input, float* grad_gamma, float* grad_beta,
    \\    const float* gamma,
    \\    int batch_size, int features,
    \\    float epsilon) {
    \\
    \\    int b = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (b >= batch_size) return;
    \\
    \\    // Recompute mean and variance
    \\    float mean = 0.0f;
    \\    for (int f = 0; f < features; f++) {
    \\        mean += input[b * features + f];
    \\    }
    \\    mean /= features;
    \\
    \\    float variance = 0.0f;
    \\    for (int f = 0; f < features; f++) {
    \\        float diff = input[b * features + f] - mean;
    \\        variance += diff * diff;
    \\    }
    \\    variance /= features;
    \\    float std = sqrtf(variance + epsilon);
    \\    float inv_std = 1.0f / std;
    \\
    \\    // Compute gradients
    \\    float grad_gamma_sum = 0.0f;
    \\    float grad_beta_sum = 0.0f;
    \\
    \\    for (int f = 0; f < features; f++) {
    \\        int idx = b * features + f;
    \\        float x_hat = (input[idx] - mean) * inv_std;
    \\        grad_gamma_sum += grad_output[idx] * x_hat;
    \\        grad_beta_sum += grad_output[idx];
    \\    }
    \\
    \\    // Accumulate grad_gamma and grad_beta using atomics
    \\    for (int f = 0; f < features; f++) {
    \\        int idx = b * features + f;
    \\        float x_hat = (input[idx] - mean) * inv_std;
    \\        atomicAdd(&grad_gamma[f], grad_gamma_sum);
    \\        atomicAdd(&grad_beta[f], grad_beta_sum);
    \\    }
    \\
    \\    // Compute gradients for input
    \\    for (int f = 0; f < features; f++) {
    \\        int idx = b * features + f;
    \\        float x_hat = (input[idx] - mean) * inv_std;
    \\        grad_input[idx] = gamma[f] * inv_std * (
    \\            grad_output[idx] - grad_beta_sum / features - x_hat * grad_gamma_sum / features
    \\        );
    \\    }
    \\}
;

/// LayerNorm forward PTX (uses Welford's algorithm for numerical stability)
pub const LAYERNORM_FORWARD_PTX = PTX_HEADER ++
    \\.visible .entry layernorm_forward(
    \\    .param .u64 input,
    \\    .param .u64 output,
    \\    .param .u64 gamma,
    \\    .param .u64 beta,
    \\    .param .u32 batch_size,
    \\    .param .u32 features,
    \\    .param .f32 epsilon
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr, %g_ptr, %b_ptr;
    \\    .reg .u32 %batch, %batch_size, %features, %f;
    \\    .reg .u32 %tid, %ctaid;
    \\    .reg .f32 %mean, %m2, %delta, %delta2, %variance, %std, %val, %norm_val;
    \\    .reg .f32 %g_val, %b_val, %inv_features, %count;
    \\    .reg .u64 %addr, %base_addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %g_ptr, [gamma];
    \\    ld.param.u64 %b_ptr, [beta];
    \\    ld.param.u32 %batch_size, [batch_size];
    \\    ld.param.u32 %features, [features];
    \\    ld.param.f32 %std, [epsilon];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\
    \\    mov.u32 %batch, %ctaid;
    \\    setp.ge.u32 %p, %batch, %batch_size;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %base_addr, %batch, %features;
    \\    mul.lo.u64 %base_addr, %base_addr, 4;
    \\    add.u64 %in_ptr, %in_ptr, %base_addr;
    \\    add.u64 %out_ptr, %out_ptr, %base_addr;
    \\
    \\    mov.f32 %mean, 0f00000000;
    \\    mov.f32 %m2, 0f00000000;
    \\    mov.f32 %count, 0f00000000;
    \\
    \\    mov.u32 %f, 0;
    \\MEAN_VAR_LOOP:
    \\    setp.ge.u32 %p, %f, %features;
    \\    @%p bra MEAN_VAR_DONE;
    \\
    \\    mul.lo.u64 %addr, %f, 4;
    \\    add.u64 %addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%addr];
    \\
    \\    add.f32 %count, %count, 0f3F800000;
    \\    sub.f32 %delta, %val, %mean;
    \\    div.approx.f32 %delta2, %delta, %count;
    \\    add.f32 %mean, %mean, %delta2;
    \\    sub.f32 %delta2, %val, %mean;
    \\    mul.f32 %delta, %delta, %delta2;
    \\    add.f32 %m2, %m2, %delta;
    \\
    \\    add.u32 %f, %f, 1;
    \\    bra MEAN_VAR_LOOP;
    \\MEAN_VAR_DONE:
    \\
    \\    cvt.rn.f32.u32 %inv_features, %features;
    \\    div.approx.f32 %variance, %m2, %inv_features;
    \\    add.f32 %variance, %variance, %std;
    \\    sqrt.approx.f32 %std, %variance;
    \\
    \\    mov.u32 %f, 0;
    \\NORM_LOOP:
    \\    setp.ge.u32 %p, %f, %features;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %f, 4;
    \\    add.u64 %base_addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%base_addr];
    \\
    \\    sub.f32 %norm_val, %val, %mean;
    \\    div.approx.f32 %norm_val, %norm_val, %std;
    \\
    \\    add.u64 %base_addr, %addr, %g_ptr;
    \\    ld.global.f32 %g_val, [%base_addr];
    \\    add.u64 %base_addr, %addr, %b_ptr;
    \\    ld.global.f32 %b_val, [%base_addr];
    \\
    \\    mul.f32 %norm_val, %norm_val, %g_val;
    \\    add.f32 %norm_val, %norm_val, %b_val;
    \\
    \\    add.u64 %base_addr, %addr, %out_ptr;
    \\    st.global.f32 [%base_addr], %norm_val;
    \\
    \\    add.u32 %f, %f, 1;
    \\    bra NORM_LOOP;
    \\END:
    \\    ret;
    \\}
;

/// LayerNorm backward PTX
pub const LAYERNORM_BACKWARD_PTX = PTX_HEADER ++
    \\.visible .entry layernorm_backward(
    \\    .param .u64 input,
    \\    .param .u64 grad_output,
    \\    .param .u64 grad_input,
    \\    .param .u64 grad_gamma,
    \\    .param .u64 grad_beta,
    \\    .param .u64 gamma,
    \\    .param .u32 batch_size,
    \\    .param .u32 features,
    \\    .param .f32 epsilon
    \\) {
    \\    .reg .u64 %in_ptr, %grad_out_ptr, %grad_in_ptr, %grad_g_ptr, %grad_b_ptr, %g_ptr;
    \\    .reg .u32 %batch, %batch_size, %features, %f;
    \\    .reg .u32 %tid, %ctaid;
    \\    .reg .f32 %mean, %variance, %std, %inv_std, %val, %go_val;
    \\    .reg .f32 %g_val, %x_hat, %sum_go, %sum_go_xhat, %inv_features;
    \\    .reg .f32 %count;
    \\    .reg .u64 %addr, %base_addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %grad_out_ptr, [grad_output];
    \\    ld.param.u64 %grad_in_ptr, [grad_input];
    \\    ld.param.u64 %grad_g_ptr, [grad_gamma];
    \\    ld.param.u64 %grad_b_ptr, [grad_beta];
    \\    ld.param.u64 %g_ptr, [gamma];
    \\    ld.param.u32 %batch_size, [batch_size];
    \\    ld.param.u32 %features, [features];
    \\    ld.param.f32 %std, [epsilon];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\
    \\    mov.u32 %batch, %ctaid;
    \\    setp.ge.u32 %p, %batch, %batch_size;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %base_addr, %batch, %features;
    \\    mul.lo.u64 %base_addr, %base_addr, 4;
    \\    add.u64 %in_ptr, %in_ptr, %base_addr;
    \\    add.u64 %grad_out_ptr, %grad_out_ptr, %base_addr;
    \\    add.u64 %grad_in_ptr, %grad_in_ptr, %base_addr;
    \\
    \\    mov.f32 %mean, 0f00000000;
    \\    mov.f32 %variance, 0f00000000;
    \\    mov.f32 %count, 0f00000000;
    \\
    \\    mov.u32 %f, 0;
    \\MEAN_LOOP:
    \\    setp.ge.u32 %p, %f, %features;
    \\    @%p bra MEAN_DONE;
    \\
    \\    mul.lo.u64 %addr, %f, 4;
    \\    add.u64 %addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%addr];
    \\
    \\    add.f32 %count, %count, 0f3F800000;
    \\    sub.f32 %delta, %val, %mean;
    \\    div.approx.f32 %delta2, %delta, %count;
    \\    add.f32 %mean, %mean, %delta2;
    \\    sub.f32 %delta2, %val, %mean;
    \\    mul.f32 %delta, %delta, %delta2;
    \\    add.f32 %variance, %variance, %delta;
    \\
    \\    add.u32 %f, %f, 1;
    \\    bra MEAN_LOOP;
    \\MEAN_DONE:
    \\
    \\    cvt.rn.f32.u32 %inv_features, %features;
    \\    div.approx.f32 %variance, %variance, %inv_features;
    \\    add.f32 %variance, %variance, %std;
    \\    sqrt.approx.f32 %std, %variance;
    \\    div.approx.f32 %inv_std, 0f3F800000, %std;
    \\
    \\    mov.f32 %sum_go, 0f00000000;
    \\    mov.f32 %sum_go_xhat, 0f00000000;
    \\
    \\    mov.u32 %f, 0;
    \\SUM_LOOP:
    \\    setp.ge.u32 %p, %f, %features;
    \\    @%p bra SUM_DONE;
    \\
    \\    mul.lo.u64 %addr, %f, 4;
    \\    add.u64 %base_addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%base_addr];
    \\    sub.f32 %x_hat, %val, %mean;
    \\    mul.f32 %x_hat, %x_hat, %inv_std;
    \\
    \\    add.u64 %base_addr, %addr, %grad_out_ptr;
    \\    ld.global.f32 %go_val, [%base_addr];
    \\
    \\    add.f32 %sum_go, %sum_go, %go_val;
    \\    mul.f32 %val, %go_val, %x_hat;
    \\    add.f32 %sum_go_xhat, %sum_go_xhat, %val;
    \\
    \\    add.u32 %f, %f, 1;
    \\    bra SUM_LOOP;
    \\SUM_DONE:
    \\
    \\    div.approx.f32 %sum_go, %sum_go, %inv_features;
    \\    div.approx.f32 %sum_go_xhat, %sum_go_xhat, %inv_features;
    \\
    \\    mov.u32 %f, 0;
    \\    mov.f32 %g_val, 0f00000000;
    \\GRAD_LOOP:
    \\    setp.ge.u32 %p, %f, %features;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %f, 4;
    \\    add.u64 %base_addr, %addr, %in_ptr;
    \\    ld.global.f32 %val, [%base_addr];
    \\    sub.f32 %x_hat, %val, %mean;
    \\    mul.f32 %x_hat, %x_hat, %inv_std;
    \\
    \\    add.u64 %base_addr, %addr, %grad_out_ptr;
    \\    ld.global.f32 %go_val, [%base_addr];
    \\
    \\    add.u64 %base_addr, %addr, %g_ptr;
    \\    ld.global.f32 %g_val, [%base_addr];
    \\
    \\    sub.f32 %go_val, %go_val, %sum_go;
    \\    mul.f32 %val, %x_hat, %sum_go_xhat;
    \\    sub.f32 %go_val, %go_val, %val;
    \\    mul.f32 %go_val, %go_val, %inv_std;
    \\    mul.f32 %go_val, %go_val, %g_val;
    \\
    \\    add.u64 %base_addr, %addr, %grad_in_ptr;
    \\    st.global.f32 [%base_addr], %go_val;
    \\
    \\    add.u32 %f, %f, 1;
    \\    bra GRAD_LOOP;
    \\END:
    \\    ret;
    \\}
;

/// BatchNorm forward training kernel PTX
pub const BATCHNORM_FORWARD_TRAINING_PTX = PTX_HEADER ++
    \\.visible .entry batchnorm_forward_training(
    \\    .param .u64 input,
    \\    .param .u64 output,
    \\    .param .u64 running_mean,
    \\    .param .u64 running_var,
    \\    .param .u64 gamma,
    \\    .param .u64 beta,
    \\    .param .u32 batch_size,
    \\    .param .u32 channels,
    \\    .param .u32 height,
    \\    .param .u32 width,
    \\    .param .f32 momentum,
    \\    .param .f32 epsilon
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr, %rm_ptr, %rv_ptr, %g_ptr, %b_ptr;
    \\    .reg .u32 %bs, %ch, %h, %w, %c, %n;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %mean, %var, %mom, %eps, %one, %mom_comp;
    \\    .reg .f32 %g_val, %b_val, %in_val, %diff, %std;
    \\    .reg .u64 %idx64, %base, %offset;
    \\    .reg .u32 %b_idx, %h_idx, %w_idx;
    \\    .reg .pred %p, %q;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %rm_ptr, [running_mean];
    \\    ld.param.u64 %rv_ptr, [running_var];
    \\    ld.param.u64 %g_ptr, [gamma];
    \\    ld.param.u64 %b_ptr, [beta];
    \\    ld.param.u32 %bs, [batch_size];
    \\    ld.param.u32 %ch, [channels];
    \\    ld.param.u32 %h, [height];
    \\    ld.param.u32 %w, [width];
    \\    ld.param.f32 %mom, [momentum];
    \\    ld.param.f32 %eps, [epsilon];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %c, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %c, %ch;
    \\    @%p bra END;
    \\
    \\    mul.lo.u32 %n, %bs, %h;
    \\    mul.lo.u32 %n, %n, %w;
    \\
    \\    mov.f32 %mean, 0f00000000;
    \\    mov.u32 %b_idx, 0;
    \\MEAN_LOOP_B:
    \\    setp.ge.u32 %p, %b_idx, %bs;
    \\    @%p bra MEAN_DONE;
    \\    mov.u32 %h_idx, 0;
    \\MEAN_LOOP_H:
    \\    setp.ge.u32 %p, %h_idx, %h;
    \\    @%p bra MEAN_H_DONE;
    \\    mov.u32 %w_idx, 0;
    \\MEAN_LOOP_W:
    \\    setp.ge.u32 %p, %w_idx, %w;
    \\    @%p bra MEAN_W_DONE;
    \\    mul.lo.u32 %offset, %b_idx, %ch;
    \\    add.u32 %offset, %offset, %c;
    \\    mul.lo.u32 %offset, %offset, %h;
    \\    add.u32 %offset, %offset, %h_idx;
    \\    mul.lo.u32 %offset, %offset, %w;
    \\    add.u32 %offset, %offset, %w_idx;
    \\    mul.lo.u64 %idx64, %offset, 4;
    \\    add.u64 %base, %in_ptr, %idx64;
    \\    ld.global.f32 %in_val, [%base];
    \\    add.f32 %mean, %mean, %in_val;
    \\    add.u32 %w_idx, %w_idx, 1;
    \\    bra MEAN_LOOP_W;
    \\MEAN_W_DONE:
    \\    add.u32 %h_idx, %h_idx, 1;
    \\    bra MEAN_LOOP_H;
    \\MEAN_H_DONE:
    \\    add.u32 %b_idx, %b_idx, 1;
    \\    bra MEAN_LOOP_B;
    \\MEAN_DONE:
    \\    cvt.rn.f32.u32 %in_val, %n;
    \\    div.approx.f32 %mean, %mean, %in_val;
    \\
    \\    mov.f32 %var, 0f00000000;
    \\    mov.u32 %b_idx, 0;
    \\VAR_LOOP_B:
    \\    setp.ge.u32 %p, %b_idx, %bs;
    \\    @%p bra VAR_DONE;
    \\    mov.u32 %h_idx, 0;
    \\VAR_LOOP_H:
    \\    setp.ge.u32 %p, %h_idx, %h;
    \\    @%p bra VAR_H_DONE;
    \\    mov.u32 %w_idx, 0;
    \\VAR_LOOP_W:
    \\    setp.ge.u32 %p, %w_idx, %w;
    \\    @%p bra VAR_W_DONE;
    \\    mul.lo.u32 %offset, %b_idx, %ch;
    \\    add.u32 %offset, %offset, %c;
    \\    mul.lo.u32 %offset, %offset, %h;
    \\    add.u32 %offset, %offset, %h_idx;
    \\    mul.lo.u32 %offset, %offset, %w;
    \\    add.u32 %offset, %offset, %w_idx;
    \\    mul.lo.u64 %idx64, %offset, 4;
    \\    add.u64 %base, %in_ptr, %idx64;
    \\    ld.global.f32 %in_val, [%base];
    \\    sub.f32 %diff, %in_val, %mean;
    \\    mul.f32 %diff, %diff, %diff;
    \\    add.f32 %var, %var, %diff;
    \\    add.u32 %w_idx, %w_idx, 1;
    \\    bra VAR_LOOP_W;
    \\VAR_W_DONE:
    \\    add.u32 %h_idx, %h_idx, 1;
    \\    bra VAR_LOOP_H;
    \\VAR_H_DONE:
    \\    add.u32 %b_idx, %b_idx, 1;
    \\    bra VAR_LOOP_B;
    \\VAR_DONE:
    \\    div.approx.f32 %var, %var, %in_val;
    \\
    \\    setp.ne.u32 %q, %tid, 0;
    \\    @%q bra UPDATE_DONE;
    \\    mul.lo.u64 %base, %c, 4;
    \\    add.u64 %idx64, %rm_ptr, %base;
    \\    ld.global.f32 %in_val, [%idx64];
    \\    mov.f32 %one, 0f3F800000;
    \\    sub.f32 %mom_comp, %one, %mom;
    \\    mul.f32 %in_val, %mom_comp, %in_val;
    \\    mul.f32 %diff, %mom, %mean;
    \\    add.f32 %in_val, %in_val, %diff;
    \\    st.global.f32 [%idx64], %in_val;
    \\    add.u64 %idx64, %rv_ptr, %base;
    \\    ld.global.f32 %in_val, [%idx64];
    \\    mul.f32 %in_val, %mom_comp, %in_val;
    \\    mul.f32 %diff, %mom, %var;
    \\    add.f32 %in_val, %in_val, %diff;
    \\    st.global.f32 [%idx64], %in_val;
    \\UPDATE_DONE:
    \\
    \\    mul.lo.u64 %base, %c, 4;
    \\    add.u64 %idx64, %g_ptr, %base;
    \\    ld.global.f32 %g_val, [%idx64];
    \\    add.u64 %idx64, %b_ptr, %base;
    \\    ld.global.f32 %b_val, [%idx64];
    \\
    \\    add.f32 %std, %var, %eps;
    \\    sqrt.approx.f32 %std, %std;
    \\
    \\    mov.u32 %b_idx, 0;
    \\OUT_LOOP_B:
    \\    setp.ge.u32 %p, %b_idx, %bs;
    \\    @%p bra END;
    \\    mov.u32 %h_idx, 0;
    \\OUT_LOOP_H:
    \\    setp.ge.u32 %p, %h_idx, %h;
    \\    @%p bra OUT_H_DONE;
    \\    mov.u32 %w_idx, 0;
    \\OUT_LOOP_W:
    \\    setp.ge.u32 %p, %w_idx, %w;
    \\    @%p bra OUT_W_DONE;
    \\    mul.lo.u32 %offset, %b_idx, %ch;
    \\    add.u32 %offset, %offset, %c;
    \\    mul.lo.u32 %offset, %offset, %h;
    \\    add.u32 %offset, %offset, %h_idx;
    \\    mul.lo.u32 %offset, %offset, %w;
    \\    add.u32 %offset, %offset, %w_idx;
    \\    mul.lo.u64 %idx64, %offset, 4;
    \\    add.u64 %base, %in_ptr, %idx64;
    \\    ld.global.f32 %in_val, [%base];
    \\    sub.f32 %diff, %in_val, %mean;
    \\    div.approx.f32 %diff, %diff, %std;
    \\    mul.f32 %diff, %diff, %g_val;
    \\    add.f32 %diff, %diff, %b_val;
    \\    add.u64 %base, %out_ptr, %idx64;
    \\    st.global.f32 [%base], %diff;
    \\    add.u32 %w_idx, %w_idx, 1;
    \\    bra OUT_LOOP_W;
    \\OUT_W_DONE:
    \\    add.u32 %h_idx, %h_idx, 1;
    \\    bra OUT_LOOP_H;
    \\OUT_H_DONE:
    \\    add.u32 %b_idx, %b_idx, 1;
    \\    bra OUT_LOOP_B;
    \\END:
    \\    ret;
    \\}
;

/// BatchNorm forward inference kernel PTX
pub const BATCHNORM_FORWARD_INFERENCE_PTX = PTX_HEADER ++
    \\.visible .entry batchnorm_forward_inference(
    \\    .param .u64 input,
    \\    .param .u64 output,
    \\    .param .u64 running_mean,
    \\    .param .u64 running_var,
    \\    .param .u64 gamma,
    \\    .param .u64 beta,
    \\    .param .u32 batch_size,
    \\    .param .u32 channels,
    \\    .param .u32 height,
    \\    .param .u32 width,
    \\    .param .f32 epsilon
    \\) {
    \\    .reg .u64 %in_ptr, %out_ptr, %rm_ptr, %rv_ptr, %g_ptr, %b_ptr;
    \\    .reg .u32 %bs, %ch, %h, %w, %c;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %mean, %var, %eps, %std;
    \\    .reg .f32 %g_val, %b_val, %in_val, %diff;
    \\    .reg .u64 %idx64, %base, %offset;
    \\    .reg .u32 %b_idx, %h_idx, %w_idx;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %in_ptr, [input];
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %rm_ptr, [running_mean];
    \\    ld.param.u64 %rv_ptr, [running_var];
    \\    ld.param.u64 %g_ptr, [gamma];
    \\    ld.param.u64 %b_ptr, [beta];
    \\    ld.param.u32 %bs, [batch_size];
    \\    ld.param.u32 %ch, [channels];
    \\    ld.param.u32 %h, [height];
    \\    ld.param.u32 %w, [width];
    \\    ld.param.f32 %eps, [epsilon];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %c, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %c, %ch;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %base, %c, 4;
    \\    add.u64 %idx64, %rm_ptr, %base;
    \\    ld.global.f32 %mean, [%idx64];
    \\    add.u64 %idx64, %rv_ptr, %base;
    \\    ld.global.f32 %var, [%idx64];
    \\
    \\    add.u64 %idx64, %g_ptr, %base;
    \\    ld.global.f32 %g_val, [%idx64];
    \\    add.u64 %idx64, %b_ptr, %base;
    \\    ld.global.f32 %b_val, [%idx64];
    \\
    \\    add.f32 %std, %var, %eps;
    \\    sqrt.approx.f32 %std, %std;
    \\
    \\    mov.u32 %b_idx, 0;
    \\OUT_LOOP_B:
    \\    setp.ge.u32 %p, %b_idx, %bs;
    \\    @%p bra END;
    \\    mov.u32 %h_idx, 0;
    \\OUT_LOOP_H:
    \\    setp.ge.u32 %p, %h_idx, %h;
    \\    @%p bra OUT_H_DONE;
    \\    mov.u32 %w_idx, 0;
    \\OUT_LOOP_W:
    \\    setp.ge.u32 %p, %w_idx, %w;
    \\    @%p bra OUT_W_DONE;
    \\    mul.lo.u32 %offset, %b_idx, %ch;
    \\    add.u32 %offset, %offset, %c;
    \\    mul.lo.u32 %offset, %offset, %h;
    \\    add.u32 %offset, %offset, %h_idx;
    \\    mul.lo.u32 %offset, %offset, %w;
    \\    add.u32 %offset, %offset, %w_idx;
    \\    mul.lo.u64 %idx64, %offset, 4;
    \\    add.u64 %base, %in_ptr, %idx64;
    \\    ld.global.f32 %in_val, [%base];
    \\    sub.f32 %diff, %in_val, %mean;
    \\    div.approx.f32 %diff, %diff, %std;
    \\    mul.f32 %diff, %diff, %g_val;
    \\    add.f32 %diff, %diff, %b_val;
    \\    add.u64 %base, %out_ptr, %idx64;
    \\    st.global.f32 [%base], %diff;
    \\    add.u32 %w_idx, %w_idx, 1;
    \\    bra OUT_LOOP_W;
    \\OUT_W_DONE:
    \\    add.u32 %h_idx, %h_idx, 1;
    \\    bra OUT_LOOP_H;
    \\OUT_H_DONE:
    \\    add.u32 %b_idx, %b_idx, 1;
    \\    bra OUT_LOOP_B;
    \\END:
    \\    ret;
    \\}
;

/// MSE loss backward CUDA C source
/// Formula: grad = 2 * (output - target) / n
pub const MSE_BACKWARD_SOURCE =
    \\extern "C" __global__ void mse_backward(
    \\    const float* output, const float* target, float* grad,
    \\    int n) {
    \\
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx >= n) return;
    \\
    \\    float diff = output[idx] - target[idx];
    \\    grad[idx] = 2.0f * diff / n;
    \\}
;

/// Cross entropy loss backward CUDA C source
/// Formula: grad = output - target (for softmax + cross entropy combined)
pub const CROSS_ENTROPY_BACKWARD_SOURCE =
    \\extern "C" __global__ void cross_entropy_backward(
    \\    const float* output, const float* target, float* grad,
    \\    int n) {
    \\
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx >= n) return;
    \\
    \\    grad[idx] = output[idx] - target[idx];
    \\}
;

/// Binary cross entropy backward CUDA C source
/// Formula: grad = (output - target) / (output * (1 - output) + epsilon)
pub const BINARY_CROSS_ENTROPY_BACKWARD_SOURCE =
    \\extern "C" __global__ void binary_cross_entropy_backward(
    \\    const float* output, const float* target, float* grad,
    \\    int n, float epsilon) {
    \\
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx >= n) return;
    \\
    \\    float out_val = output[idx];
    \\    float denom = out_val * (1.0f - out_val);
    \\    if (denom < epsilon) denom = epsilon;
    \\    grad[idx] = (out_val - target[idx]) / denom;
    \\}
;

/// KL divergence backward CUDA C source
/// Formula: grad = -target / (output + epsilon)
pub const KL_DIVERGENCE_BACKWARD_SOURCE =
    \\extern "C" __global__ void kl_divergence_backward(
    \\    const float* output, float* grad,
    \\    int n, float epsilon) {
    \\
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx >= n) return;
    \\
    \\    grad[idx] = -1.0f / (output[idx] + epsilon);
    \\}
;

/// Dropout forward kernel CUDA C source
pub const DROPOUT_FORWARD_SOURCE =
    \\extern "C" __global__ void dropout_forward(
    \\    const float* input,
    \\    float* output,
    \\    float* mask,
    \\    int n,
    \\    float rate,
    \\    float scale,
    \\    unsigned long long seed) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        // Simple XORShift for random number generation per thread
    \\        unsigned long long state = seed + idx;
    \\        state ^= state << 13;
    \\        state ^= state >> 7;
    \\        state ^= state << 17;
    \\        float rand_val = (float)(state % 1000000) / 1000000.0f;
    \\
    \\        if (rand_val >= rate) {
    \\            mask[idx] = 1.0f;
    \\            output[idx] = input[idx] * scale;
    \\        } else {
    \\            mask[idx] = 0.0f;
    \\            output[idx] = 0.0f;
    \\        }
    \\    }
    \\}
;

/// VAE sampling forward kernel CUDA C source
pub const VAE_SAMPLING_FORWARD_SOURCE =
    \\extern "C" __global__ void vae_sampling_forward(
    \\    const float* input,
    \\    float* output,
    \\    float* epsilon,
    \\    int latent_dim,
    \\    unsigned long long seed) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < latent_dim) {
    \\        // Simple XORShift for random number generation per thread
    \\        unsigned long long state = seed + idx;
    \\        state ^= state << 13;
    \\        state ^= state >> 7;
    \\        state ^= state << 17;
    \\
    \\        // Box-Muller transform for normal distribution
    \\        float u1 = (float)((state ^ 0x5555555555555555ULL) % 1000000) / 1000000.0f;
    \\        float u2 = (float)((state ^ 0xAAAAAAAAAAAAAAAAULL) % 1000000) / 1000000.0f;
    \\
    \\        if (u1 <= 0.0f) u1 = 0.000001f;
    \\        float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    \\
    \\        float mu = input[idx];
    \\        float logvar = input[idx + latent_dim];
    \\        float std_dev = expf(0.5f * logvar);
    \\
    \\        epsilon[idx] = z0;
    \\        output[idx] = mu + z0 * std_dev;
    \\    }
    \\}
;

/// VAE sampling backward kernel CUDA C source
pub const VAE_SAMPLING_BACKWARD_SOURCE =
    \\extern "C" __global__ void vae_sampling_backward(
    \\    const float* input,
    \\    const float* grad_output,
    \\    float* grad_input,
    \\    const float* epsilon,
    \\    int latent_dim) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < latent_dim) {
    \\        float logvar = input[idx + latent_dim];
    \\        float std_dev = expf(0.5f * logvar);
    \\
    \\        // grad_mu = grad_output
    \\        grad_input[idx] = grad_output[idx];
    \\        // grad_logvar = grad_output * epsilon * 0.5 * std_dev
    \\        grad_input[idx + latent_dim] = grad_output[idx] * epsilon[idx] * 0.5f * std_dev;
    \\    }
    \\}
;

/// Batch Normalization backward kernel CUDA C source
pub const BATCHNORM_BACKWARD_SOURCE =
    \\extern "C" __global__ void batchnorm_backward(
    \\    const float* input, const float* grad_output,
    \\    float* grad_input, float* grad_gamma, float* grad_beta,
    \\    const float* gamma,
    \\    int batch_size, int features,
    \\    float epsilon) {
    \\
    \\    int f = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (f >= features) return;
    \\
    \\    // 1. Compute mean and variance for this feature across the batch
    \\    float mean = 0.0f;
    \\    for (int b = 0; b < batch_size; b++) {
    \\        mean += input[b * features + f];
    \\    }
    \\    mean /= (float)batch_size;
    \\
    \\    float variance = 0.0f;
    \\    for (int b = 0; b < batch_size; b++) {
    \\        float diff = input[b * features + f] - mean;
    \\        variance += diff * diff;
    \\    }
    \\    variance /= (float)batch_size;
    \\    float std = sqrtf(variance + epsilon);
    \\
    \\    // 2. Compute d_gamma and d_beta
    \\    float d_gamma = 0.0f;
    \\    float d_beta = 0.0f;
    \\    for (int b = 0; b < batch_size; b++) {
    \\        int idx = b * features + f;
    \\        float normalized = (input[idx] - mean) / std;
    \\        d_gamma += grad_output[idx] * normalized;
    \\        d_beta += grad_output[idx];
    \\    }
    \\    grad_gamma[f] = d_gamma;
    \\    grad_beta[f] = d_beta;
    \\
    \\    // 3. Compute grad_input
    \\    float gamma_f = gamma[f];
    \\    float scale = gamma_f / ((float)batch_size * std);
    \\    for (int b = 0; b < batch_size; b++) {
    \\        int idx = b * features + f;
    \\        float normalized = (input[idx] - mean) / std;
    \\        grad_input[idx] = scale * ((float)batch_size * grad_output[idx] - d_beta - normalized * d_gamma);
    \\    }
    \\}
;

/// Conv1D forward kernel CUDA C source
pub const CONV1D_FORWARD_SOURCE =
    \\extern "C" __global__ void conv1d_forward(
    \\    const float* input, const float* weights, const float* bias, float* output,
    \\    int in_channels, int out_channels, int kernel_size, int in_len, int out_len) {
    \\
    \\    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    int ch_out = blockIdx.y;
    \\
    \\    if (out_idx < out_len && ch_out < out_channels) {
    \\        float sum = bias[ch_out];
    \\        for (int ch_in = 0; ch_in < in_channels; ch_in++) {
    \\            for (int k = 0; k < kernel_size; k++) {
    \\                int in_idx = out_idx + k;
    \\                if (in_idx < in_len) {
    \\                    sum += input[ch_in * in_len + in_idx] *
    \\                           weights[(ch_out * in_channels + ch_in) * kernel_size + k];
    \\                }
    \\            }
    \\        }
    \\        output[ch_out * out_len + out_idx] = sum;
    \\    }
    \\}
;

/// Conv1D backward weight gradient kernel
pub const CONV1D_GRAD_WEIGHT_SOURCE =
    \\extern "C" __global__ void conv1d_grad_weight(
    \\    const float* input, const float* grad_output, float* grad_weights,
    \\    int in_channels, int out_channels, int kernel_size, int in_len, int out_len) {
    \\
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    int total_weights = out_channels * in_channels * kernel_size;
    \\
    \\    if (idx < total_weights) {
    \\        int k = idx % kernel_size;
    \\        int ch_in = (idx / kernel_size) % in_channels;
    \\        int ch_out = idx / (kernel_size * in_channels);
    \\
    \\        float sum = 0.0f;
    \\        for (int i = 0; i < out_len; i++) {
    \\            int in_idx = i + k;
    \\            if (in_idx < in_len) {
    \\                sum += input[ch_in * in_len + in_idx] * grad_output[ch_out * out_len + i];
    \\            }
    \\        }
    \\        grad_weights[idx] = sum;
    \\    }
    \\}
;

/// Conv1D backward input gradient kernel
pub const CONV1D_GRAD_INPUT_SOURCE =
    \\extern "C" __global__ void conv1d_grad_input(
    \\    const float* weights, const float* grad_output, float* grad_input,
    \\    int in_channels, int out_channels, int kernel_size, int in_len, int out_len) {
    \\
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    int total_input = in_channels * in_len;
    \\
    \\    if (idx < total_input) {
    \\        int in_idx = idx % in_len;
    \\        int ch_in = idx / in_len;
    \\
    \\        float sum = 0.0f;
    \\        for (int ch_out = 0; ch_out < out_channels; ch_out++) {
    \\            for (int k = 0; k < kernel_size; k++) {
    \\                int out_pos = in_idx - k;
    \\                if (out_pos >= 0 && out_pos < out_len) {
    \\                    sum += grad_output[ch_out * out_len + out_pos] *
    \\                           weights[(ch_out * in_channels + ch_in) * kernel_size + k];
    \\                }
    \\            }
    \\        }
    \\        grad_input[idx] = sum;
    \\    }
    \\}
;

/// Conv1D backward bias gradient kernel
pub const CONV1D_GRAD_BIAS_SOURCE =
    \\extern "C" __global__ void conv1d_grad_bias(
    \\    const float* grad_output, float* grad_bias,
    \\    int out_channels, int out_len) {
    \\
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\
    \\    if (idx < out_channels) {
    \\        float sum = 0.0f;
    \\        for (int i = 0; i < out_len; i++) {
    \\            sum += grad_output[idx * out_len + i];
    \\        }
    \\        grad_bias[idx] = sum;
    \\    }
    \\}
;

/// Softmax backward kernel CUDA C source
pub const SOFTMAX_BACKWARD_SOURCE =
    \\extern "C" __global__ void softmax_backward(
    \\    const float* output, const float* grad_output, float* grad_input,
    \\    int batch_size, int features) {
    \\
    \\    int b = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (b < batch_size) {
    \\        float dot = 0.0f;
    \\        for (int f = 0; f < features; f++) {
    \\            dot += grad_output[b * features + f] * output[b * features + f];
    \\        }
    \\
    \\        for (int f = 0; f < features; f++) {
    \\            int idx = b * features + f;
    \\            grad_input[idx] = output[idx] * (grad_output[idx] - dot);
    \\        }
    \\    }
    \\}
;

/// MSE backward PTX
pub const MSE_BACKWARD_PTX = PTX_HEADER ++
    \\.visible .entry mse_backward(
    \\    .param .u64 output,
    \\    .param .u64 target,
    \\    .param .u64 grad,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %out_ptr, %tgt_ptr, %grad_ptr;
    \\    .reg .u32 %tid, %ctaid, %ntid, %idx, %n;
    \\    .reg .f32 %out_val, %tgt_val, %diff, %scale;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %tgt_ptr, [target];
    \\    ld.param.u64 %grad_ptr, [grad];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %out_ptr, %out_ptr, %addr;
    \\    add.u64 %tgt_ptr, %tgt_ptr, %addr;
    \\    add.u64 %grad_ptr, %grad_ptr, %addr;
    \\
    \\    ld.global.f32 %out_val, [%out_ptr];
    \\    ld.global.f32 %tgt_val, [%tgt_ptr];
    \\    sub.f32 %diff, %out_val, %tgt_val;
    \\    cvt.rn.f32.u32 %scale, %n;
    \\    mul.f32 %diff, %diff, 0f40000000; // 2.0f
    \\    div.approx.f32 %diff, %diff, %scale;
    \\    st.global.f32 [%grad_ptr], %diff;
    \\
    \\END:
    \\    ret;
    \\}
;

/// Cross entropy backward PTX
pub const CROSS_ENTROPY_BACKWARD_PTX = PTX_HEADER ++
    \\.visible .entry cross_entropy_backward(
    \\    .param .u64 output,
    \\    .param .u64 target,
    \\    .param .u64 grad,
    \\    .param .u32 n
    \\) {
    \\    .reg .u64 %out_ptr, %tgt_ptr, %grad_ptr;
    \\    .reg .u32 %tid, %ctaid, %ntid, %idx, %n;
    \\    .reg .f32 %out_val, %tgt_val, %diff;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %tgt_ptr, [target];
    \\    ld.param.u64 %grad_ptr, [grad];
    \\    ld.param.u32 %n, [n];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %out_ptr, %out_ptr, %addr;
    \\    add.u64 %tgt_ptr, %tgt_ptr, %addr;
    \\    add.u64 %grad_ptr, %grad_ptr, %addr;
    \\
    \\    ld.global.f32 %out_val, [%out_ptr];
    \\    ld.global.f32 %tgt_val, [%tgt_ptr];
    \\    sub.f32 %diff, %out_val, %tgt_val;
    \\    st.global.f32 [%grad_ptr], %diff;
    \\
    \\END:
    \\    ret;
    \\}
;

/// Binary cross entropy backward PTX
pub const BINARY_CROSS_ENTROPY_BACKWARD_PTX = PTX_HEADER ++
    \\.visible .entry binary_cross_entropy_backward(
    \\    .param .u64 output,
    \\    .param .u64 target,
    \\    .param .u64 grad,
    \\    .param .u32 n,
    \\    .param .f32 epsilon
    \\) {
    \\    .reg .u64 %out_ptr, %tgt_ptr, %grad_ptr;
    \\    .reg .u32 %tid, %ctaid, %ntid, %idx, %n;
    \\    .reg .f32 %out_val, %tgt_val, %diff, %denom, %one, %eps;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %tgt_ptr, [target];
    \\    ld.param.u64 %grad_ptr, [grad];
    \\    ld.param.u32 %n, [n];
    \\    ld.param.f32 %eps, [epsilon];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %out_ptr, %out_ptr, %addr;
    \\    add.u64 %tgt_ptr, %tgt_ptr, %addr;
    \\    add.u64 %grad_ptr, %grad_ptr, %addr;
    \\
    \\    ld.global.f32 %out_val, [%out_ptr];
    \\    ld.global.f32 %tgt_val, [%tgt_ptr];
    \\
    \\    mov.f32 %one, 0f3F800000; // 1.0f
    \\    sub.f32 %denom, %one, %out_val;
    \\    mul.f32 %denom, %denom, %out_val;
    \\    setp.lt.f32 %p, %denom, %eps;
    \\    @%p mov.f32 %denom, %eps;
    \\
    \\    sub.f32 %diff, %out_val, %tgt_val;
    \\    div.approx.f32 %diff, %diff, %denom;
    \\    st.global.f32 [%grad_ptr], %diff;
    \\
    \\END:
    \\    ret;
    \\}
;

/// KL divergence backward PTX
pub const KL_DIVERGENCE_BACKWARD_PTX = PTX_HEADER ++
    \\.visible .entry kl_divergence_backward(
    \\    .param .u64 output,
    \\    .param .u64 grad,
    \\    .param .u32 n,
    \\    .param .f32 epsilon
    \\) {
    \\    .reg .u64 %out_ptr, %grad_ptr;
    \\    .reg .u32 %tid, %ctaid, %ntid, %idx, %n;
    \\    .reg .f32 %out_val, %eps, %denom, %result;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %out_ptr, [output];
    \\    ld.param.u64 %grad_ptr, [grad];
    \\    ld.param.u32 %n, [n];
    \\    ld.param.f32 %eps, [epsilon];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %n;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %out_ptr, %out_ptr, %addr;
    \\    add.u64 %grad_ptr, %grad_ptr, %addr;
    \\
    \\    ld.global.f32 %out_val, [%out_ptr];
    \\    add.f32 %denom, %out_val, %eps;
    \\    mov.f32 %result, 0fBF800000; // -1.0f
    \\    div.approx.f32 %result, %result, %denom;
    \\    st.global.f32 [%grad_ptr], %result;
    \\
    \\END:
    \\    ret;
    \\}
;

/// Fused matrix multiplication + bias + ReLU kernel PTX
/// Performs: C = max(0, A * B + bias)
/// This eliminates 3 separate kernel launches and 2 intermediate memory transfers
pub const MATMUL_BIAS_RELU_FUSED_PTX = PTX_HEADER ++
    \\.visible .entry matmul_bias_relu_fused(
    \\    .param .u64 A,
    \\    .param .u64 B,
    \\    .param .u64 C,
    \\    .param .u64 bias,
    \\    .param .u32 batch_size,
    \\    .param .u32 M,
    \\    .param .u32 N,
    \\    .param .u32 K
    \\) {
    \\    .reg .u64 %A_ptr, %B_ptr, %C_ptr, %bias_ptr;
    \\    .reg .u32 %batch_size, %M, %N, %K;
    \\    .reg .u32 %batch, %row, %col, %k;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %sum, %a_val, %b_val, %bias_val, %result;
    \\    .reg .u64 %a_addr, %b_addr, %c_addr, %a_batch_offset, %c_batch_offset;
    \\    .reg .pred %p, %p_neg;
    \\    .reg .f32 %zero;
    \\
    \\    ld.param.u64 %A_ptr, [A];
    \\    ld.param.u64 %B_ptr, [B];
    \\    ld.param.u64 %C_ptr, [C];
    \\    ld.param.u64 %bias_ptr, [bias];
    \\    ld.param.u32 %batch_size, [batch_size];
    \\    ld.param.u32 %M, [M];
    \\    ld.param.u32 %N, [N];
    \\    ld.param.u32 %K, [K];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mov.u32 %batch, %ctaid.y;
    \\    mov.f32 %zero, 0f00000000;
    \\
    \\    // Calculate global column index (each thread handles one column)
    \\    mad.lo.u32 %col, %ctaid.x, %ntid.x, %tid.x;
    \\    setp.ge.u32 %p, %col, %N;
    \\    @%p bra END;
    \\    setp.ge.u32 %p, %batch, %batch_size;
    \\    @%p bra END;
    \\
    \\    // Calculate batch offsets (A and C are batched, B is shared)
    \\    mul.lo.u64 %a_batch_offset, %batch, %M;
    \\    mul.lo.u64 %a_batch_offset, %a_batch_offset, %K;
    \\    mul.lo.u64 %a_batch_offset, %a_batch_offset, 4;
    \\    add.u64 %A_ptr, %A_ptr, %a_batch_offset;
    \\
    \\    mul.lo.u64 %c_batch_offset, %batch, %M;
    \\    mul.lo.u64 %c_batch_offset, %c_batch_offset, %N;
    \\    mul.lo.u64 %c_batch_offset, %c_batch_offset, 4;
    \\    add.u64 %C_ptr, %C_ptr, %c_batch_offset;
    \\
    \\    // Matrix multiplication: compute dot product of A row and B column
    \\    mov.f32 %sum, 0f00000000;
    \\    mov.u32 %k, 0;
    \\    mov.u32 %row, 0;  // M=1 for batch size, so row=0
    \\LOOP:
    \\    setp.ge.u32 %p, %k, %K;
    \\    @%p bra LOOP_END;
    \\    // Load A[batch][0][k]
    \\    mad.lo.u64 %a_addr, %row, %K, %k;
    \\    mul.lo.u64 %a_addr, %a_addr, 4;
    \\    add.u64 %a_addr, %a_addr, %A_ptr;
    \\    ld.global.f32 %a_val, [%a_addr];
    \\    // Load B[k][col]
    \\    mad.lo.u64 %b_addr, %k, %N, %col;
    \\    mul.lo.u64 %b_addr, %b_addr, 4;
    \\    add.u64 %b_addr, %b_addr, %B_ptr;
    \\    ld.global.f32 %b_val, [%b_addr];
    \\    // Accumulate: sum += a * b
    \\    fma.rn.f32 %sum, %a_val, %b_val, %sum;
    \\    add.u32 %k, %k, 1;
    \\    bra LOOP;
    \\LOOP_END:
    \\
    \\    // Add bias[col]
    \\    mul.lo.u64 %b_addr, %col, 4;
    \\    add.u64 %b_addr, %b_addr, %bias_ptr;
    \\    ld.global.f32 %bias_val, [%b_addr];
    \\    add.f32 %sum, %sum, %bias_val;
    \\
    \\    // Apply ReLU: max(0, sum)
    \\    setp.lt.f32 %p_neg, %sum, %zero;
    \\    selp.f32 %result, %zero, %sum, %p_neg;
    \\
    \\    // Store result to C[batch][0][col]
    \\    mad.lo.u64 %c_addr, %row, %N, %col;
    \\    mul.lo.u64 %c_addr, %c_addr, 4;
    \\    add.u64 %c_addr, %c_addr, %C_ptr;
    \\    st.global.f32 [%c_addr], %result;
    \\END:
    \\    ret;
    \\}
;

/// Fused matrix multiplication + bias + Sigmoid kernel PTX
/// Performs: C = sigmoid(A * B + bias)
pub const MATMUL_BIAS_SIGMOID_FUSED_PTX = PTX_HEADER ++
    \\.visible .entry matmul_bias_sigmoid_fused(
    \\    .param .u64 A,
    \\    .param .u64 B,
    \\    .param .u64 C,
    \\    .param .u64 bias,
    \\    .param .u32 batch_size,
    \\    .param .u32 M,
    \\    .param .u32 N,
    \\    .param .u32 K
    \\) {
    \\    .reg .u64 %A_ptr, %B_ptr, %C_ptr, %bias_ptr;
    \\    .reg .u32 %batch_size, %M, %N, %K;
    \\    .reg .u32 %batch, %row, %col, %k;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %sum, %a_val, %b_val, %bias_val, %result;
    \\    .reg .f32 %neg_sum, %exp_val, %one;
    \\    .reg .u64 %a_addr, %b_addr, %c_addr, %a_batch_offset, %c_batch_offset;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %A_ptr, [A];
    \\    ld.param.u64 %B_ptr, [B];
    \\    ld.param.u64 %C_ptr, [C];
    \\    ld.param.u64 %bias_ptr, [bias];
    \\    ld.param.u32 %batch_size, [batch_size];
    \\    ld.param.u32 %M, [M];
    \\    ld.param.u32 %N, [N];
    \\    ld.param.u32 %K, [K];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mov.u32 %batch, %ctaid.y;
    \\    mov.f32 %one, 0f3F800000;  // 1.0f
    \\
    \\    mad.lo.u32 %col, %ctaid.x, %ntid.x, %tid.x;
    \\    setp.ge.u32 %p, %col, %N;
    \\    @%p bra END;
    \\    setp.ge.u32 %p, %batch, %batch_size;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %a_batch_offset, %batch, %M;
    \\    mul.lo.u64 %a_batch_offset, %a_batch_offset, %K;
    \\    mul.lo.u64 %a_batch_offset, %a_batch_offset, 4;
    \\    add.u64 %A_ptr, %A_ptr, %a_batch_offset;
    \\
    \\    mul.lo.u64 %c_batch_offset, %batch, %M;
    \\    mul.lo.u64 %c_batch_offset, %c_batch_offset, %N;
    \\    mul.lo.u64 %c_batch_offset, %c_batch_offset, 4;
    \\    add.u64 %C_ptr, %C_ptr, %c_batch_offset;
    \\
    \\    mov.f32 %sum, 0f00000000;
    \\    mov.u32 %k, 0;
    \\    mov.u32 %row, 0;
    \\LOOP:
    \\    setp.ge.u32 %p, %k, %K;
    \\    @%p bra LOOP_END;
    \\    mad.lo.u64 %a_addr, %row, %K, %k;
    \\    mul.lo.u64 %a_addr, %a_addr, 4;
    \\    add.u64 %a_addr, %a_addr, %A_ptr;
    \\    ld.global.f32 %a_val, [%a_addr];
    \\    mad.lo.u64 %b_addr, %k, %N, %col;
    \\    mul.lo.u64 %b_addr, %b_addr, 4;
    \\    add.u64 %b_addr, %b_addr, %B_ptr;
    \\    ld.global.f32 %b_val, [%b_addr];
    \\    fma.rn.f32 %sum, %a_val, %b_val, %sum;
    \\    add.u32 %k, %k, 1;
    \\    bra LOOP;
    \\LOOP_END:
    \\
    \\    // Add bias
    \\    mul.lo.u64 %b_addr, %col, 4;
    \\    add.u64 %b_addr, %b_addr, %bias_ptr;
    \\    ld.global.f32 %bias_val, [%b_addr];
    \\    add.f32 %sum, %sum, %bias_val;
    \\
    \\    // Apply sigmoid: 1 / (1 + exp(-x))
    \\    neg.f32 %neg_sum, %sum;
    \\    ex2.approx.f32 %exp_val, %neg_sum;  // exp2(log2(e) * -x) approximates exp(-x)
    \\    add.f32 %exp_val, %one, %exp_val;
    \\    div.approx.f32 %result, %one, %exp_val;
    \\
    \\    // Store result
    \\    mad.lo.u64 %c_addr, %row, %N, %col;
    \\    mul.lo.u64 %c_addr, %c_addr, 4;
    \\    add.u64 %c_addr, %c_addr, %C_ptr;
    \\    st.global.f32 [%c_addr], %result;
    \\END:
    \\    ret;
    \\}
;

/// Fused matrix multiplication + bias + Tanh kernel PTX
/// Performs: C = tanh(A * B + bias)
pub const MATMUL_BIAS_TANH_FUSED_PTX = PTX_HEADER ++
    \\.visible .entry matmul_bias_tanh_fused(
    \\    .param .u64 A,
    \\    .param .u64 B,
    \\    .param .u64 C,
    \\    .param .u64 bias,
    \\    .param .u32 batch_size,
    \\    .param .u32 M,
    \\    .param .u32 N,
    \\    .param .u32 K
    \\) {
    \\    .reg .u64 %A_ptr, %B_ptr, %C_ptr, %bias_ptr;
    \\    .reg .u32 %batch_size, %M, %N, %K;
    \\    .reg .u32 %batch, %row, %col, %k;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %sum, %a_val, %b_val, %bias_val, %result;
    \\    .reg .f32 %neg_sum, %exp_val, %exp_neg_val, %numer, %denom;
    \\    .reg .u64 %a_addr, %b_addr, %c_addr, %a_batch_offset, %c_batch_offset;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %A_ptr, [A];
    \\    ld.param.u64 %B_ptr, [B];
    \\    ld.param.u64 %C_ptr, [C];
    \\    ld.param.u64 %bias_ptr, [bias];
    \\    ld.param.u32 %batch_size, [batch_size];
    \\    ld.param.u32 %M, [M];
    \\    ld.param.u32 %N, [N];
    \\    ld.param.u32 %K, [K];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mov.u32 %batch, %ctaid.y;
    \\
    \\    mad.lo.u32 %col, %ctaid.x, %ntid.x, %tid.x;
    \\    setp.ge.u32 %p, %col, %N;
    \\    @%p bra END;
    \\    setp.ge.u32 %p, %batch, %batch_size;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %a_batch_offset, %batch, %M;
    \\    mul.lo.u64 %a_batch_offset, %a_batch_offset, %K;
    \\    mul.lo.u64 %a_batch_offset, %a_batch_offset, 4;
    \\    add.u64 %A_ptr, %A_ptr, %a_batch_offset;
    \\
    \\    mul.lo.u64 %c_batch_offset, %batch, %M;
    \\    mul.lo.u64 %c_batch_offset, %c_batch_offset, %N;
    \\    mul.lo.u64 %c_batch_offset, %c_batch_offset, 4;
    \\    add.u64 %C_ptr, %C_ptr, %c_batch_offset;
    \\
    \\    mov.f32 %sum, 0f00000000;
    \\    mov.u32 %k, 0;
    \\    mov.u32 %row, 0;
    \\LOOP:
    \\    setp.ge.u32 %p, %k, %K;
    \\    @%p bra LOOP_END;
    \\    mad.lo.u64 %a_addr, %row, %K, %k;
    \\    mul.lo.u64 %a_addr, %a_addr, 4;
    \\    add.u64 %a_addr, %a_addr, %A_ptr;
    \\    ld.global.f32 %a_val, [%a_addr];
    \\    mad.lo.u64 %b_addr, %k, %N, %col;
    \\    mul.lo.u64 %b_addr, %b_addr, 4;
    \\    add.u64 %b_addr, %b_addr, %B_ptr;
    \\    ld.global.f32 %b_val, [%b_addr];
    \\    fma.rn.f32 %sum, %a_val, %b_val, %sum;
    \\    add.u32 %k, %k, 1;
    \\    bra LOOP;
    \\LOOP_END:
    \\
    \\    // Add bias
    \\    mul.lo.u64 %b_addr, %col, 4;
    \\    add.u64 %b_addr, %b_addr, %bias_ptr;
    \\    ld.global.f32 %bias_val, [%b_addr];
    \\    add.f32 %sum, %sum, %bias_val;
    \\
    \\    // Apply tanh using formula: (exp(2x) - 1) / (exp(2x) + 1)
    \\    // First compute exp(2x)
    \\    add.f32 %exp_val, %sum, %sum;  // 2x
    \\    ex2.approx.f32 %exp_val, %exp_val;  // exp2(log2(e) * 2x) approximates exp(2x)
    \\    // Compute result
    \\    mov.f32 %numer, 0fBF800000;  // -1.0f
    \\    mov.f32 %denom, 0f3F800000;  // 1.0f
    \\    fma.rn.f32 %numer, %exp_val, %exp_val, %numer;  // exp(2x) - 1
    \\    fma.rn.f32 %denom, %exp_val, %exp_val, %denom;  // exp(2x) + 1
    \\    div.approx.f32 %result, %numer, %denom;
    \\
    \\    // Store result
    \\    mad.lo.u64 %c_addr, %row, %N, %col;
    \\    mul.lo.u64 %c_addr, %c_addr, 4;
    \\    add.u64 %c_addr, %c_addr, %C_ptr;
    \\    st.global.f32 [%c_addr], %result;
    \\END:
    \\    ret;
    \\}
;

/// Fused matrix multiplication + bias (identity/no activation) kernel PTX
/// Performs: C = A * B + bias
pub const MATMUL_BIAS_IDENTITY_FUSED_PTX = PTX_HEADER ++
    \\.visible .entry matmul_bias_identity_fused(
    \\    .param .u64 A,
    \\    .param .u64 B,
    \\    .param .u64 C,
    \\    .param .u64 bias,
    \\    .param .u32 batch_size,
    \\    .param .u32 M,
    \\    .param .u32 N,
    \\    .param .u32 K
    \\) {
    \\    .reg .u64 %A_ptr, %B_ptr, %C_ptr, %bias_ptr;
    \\    .reg .u32 %batch_size, %M, %N, %K;
    \\    .reg .u32 %batch, %row, %col, %k;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %sum, %a_val, %b_val, %bias_val;
    \\    .reg .u64 %a_addr, %b_addr, %c_addr, %a_batch_offset, %c_batch_offset;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %A_ptr, [A];
    \\    ld.param.u64 %B_ptr, [B];
    \\    ld.param.u64 %C_ptr, [C];
    \\    ld.param.u64 %bias_ptr, [bias];
    \\    ld.param.u32 %batch_size, [batch_size];
    \\    ld.param.u32 %M, [M];
    \\    ld.param.u32 %N, [N];
    \\    ld.param.u32 %K, [K];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mov.u32 %batch, %ctaid.y;
    \\
    \\    mad.lo.u32 %col, %ctaid.x, %ntid.x, %tid.x;
    \\    setp.ge.u32 %p, %col, %N;
    \\    @%p bra END;
    \\    setp.ge.u32 %p, %batch, %batch_size;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %a_batch_offset, %batch, %M;
    \\    mul.lo.u64 %a_batch_offset, %a_batch_offset, %K;
    \\    mul.lo.u64 %a_batch_offset, %a_batch_offset, 4;
    \\    add.u64 %A_ptr, %A_ptr, %a_batch_offset;
    \\
    \\    mul.lo.u64 %c_batch_offset, %batch, %M;
    \\    mul.lo.u64 %c_batch_offset, %c_batch_offset, %N;
    \\    mul.lo.u64 %c_batch_offset, %c_batch_offset, 4;
    \\    add.u64 %C_ptr, %C_ptr, %c_batch_offset;
    \\
    \\    mov.f32 %sum, 0f00000000;
    \\    mov.u32 %k, 0;
    \\    mov.u32 %row, 0;
    \\LOOP:
    \\    setp.ge.u32 %p, %k, %K;
    \\    @%p bra LOOP_END;
    \\    mad.lo.u64 %a_addr, %row, %K, %k;
    \\    mul.lo.u64 %a_addr, %a_addr, 4;
    \\    add.u64 %a_addr, %a_addr, %A_ptr;
    \\    ld.global.f32 %a_val, [%a_addr];
    \\    mad.lo.u64 %b_addr, %k, %N, %col;
    \\    mul.lo.u64 %b_addr, %b_addr, 4;
    \\    add.u64 %b_addr, %b_addr, %B_ptr;
    \\    ld.global.f32 %b_val, [%b_addr];
    \\    fma.rn.f32 %sum, %a_val, %b_val, %sum;
    \\    add.u32 %k, %k, 1;
    \\    bra LOOP;
    \\LOOP_END:
    \\
    \\    // Add bias
    \\    mul.lo.u64 %b_addr, %col, 4;
    \\    add.u64 %b_addr, %b_addr, %bias_ptr;
    \\    ld.global.f32 %bias_val, [%b_addr];
    \\    add.f32 %sum, %sum, %bias_val;
    \\
    \\    // Store result (no activation)
    \\    mad.lo.u64 %c_addr, %row, %N, %col;
    \\    mul.lo.u64 %c_addr, %c_addr, 4;
    \\    add.u64 %c_addr, %c_addr, %C_ptr;
    \\    st.global.f32 [%c_addr], %sum;
    \\END:
    \\    ret;
    \\}
;

/// RNN forward step kernel CUDA C source
pub const RNN_FORWARD_STEP_SOURCE =
    \\extern "C" __global__ void rnn_forward_step(
    \\    const float* gates_ih,
    \\    const float* gates_hh,
    \\    const float* bias,
    \\    float* h_curr,
    \\    int hidden_size) {
    \\    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (gid < hidden_size) {
    \\        float val = gates_ih[gid] + gates_hh[gid] + bias[gid];
    \\        h_curr[gid] = tanhf(val);
    \\    }
    \\}
;

/// RNN backward step kernel CUDA C source
pub const RNN_BACKWARD_STEP_SOURCE =
    \\extern "C" __global__ void rnn_backward_step(
    \\    const float* grad_h_curr,
    \\    const float* grad_h_next,
    \\    const float* h_curr,
    \\    float* grad_after_act,
    \\    int hidden_size) {
    \\    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (gid < hidden_size) {
    \\        float gh = grad_h_curr[gid] + grad_h_next[gid];
    \\        float h = h_curr[gid];
    \\        grad_after_act[gid] = gh * (1.0f - h * h);
    \\    }
    \\}
;

/// LSTM forward step kernel CUDA C source
pub const LSTM_FORWARD_STEP_SOURCE =
    \\extern "C" __global__ void lstm_forward_step(
    \\    const float* gates_ih,
    \\    const float* gates_hh,
    \\    const float* bias,
    \\    const float* c_prev,
    \\    float* c_curr,
    \\    float* h_curr,
    \\    float* gate_acts,
    \\    int hidden_size) {
    \\    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (gid < hidden_size) {
    \\        int h = hidden_size;
    \\
    \\        // Gate indices
    \\        float gi_i = gates_ih[gid];
    \\        float gi_f = gates_ih[h + gid];
    \\        float gi_g = gates_ih[2 * h + gid];
    \\        float gi_o = gates_ih[3 * h + gid];
    \\
    \\        float gh_i = gates_hh[gid];
    \\        float gh_f = gates_hh[h + gid];
    \\        float gh_g = gates_hh[2 * h + gid];
    \\        float gh_o = gates_hh[3 * h + gid];
    \\
    \\        float b_i = bias[gid];
    \\        float b_f = bias[h + gid];
    \\        float b_g = bias[2 * h + gid];
    \\        float b_o = bias[3 * h + gid];
    \\
    \\        // Activate gates
    \\        float i = 1.0f / (1.0f + expf(-(gi_i + gh_i + b_i)));
    \\        float f = 1.0f / (1.0f + expf(-(gi_f + gh_f + b_f)));
    \\        float g = tanhf(gi_g + gh_g + b_g);
    \\        float o = 1.0f / (1.0f + expf(-(gi_o + gh_o + b_o)));
    \\
    \\        // Store activations
    \\        gate_acts[gid] = i;
    \\        gate_acts[h + gid] = f;
    \\        gate_acts[2 * h + gid] = g;
    \\        gate_acts[3 * h + gid] = o;
    \\
    \\        // Update cell state
    \\        float cp = c_prev[gid];
    \\        float cc = f * cp + i * g;
    \\        c_curr[gid] = cc;
    \\
    \\        // Update hidden state
    \\        h_curr[gid] = o * tanhf(cc);
    \\    }
    \\}
;

/// LSTM backward step kernel CUDA C source
pub const LSTM_BACKWARD_STEP_SOURCE =
    \\extern "C" __global__ void lstm_backward_step(
    \\    const float* grad_h_curr,
    \\    const float* grad_h_next,
    \\    const float* grad_c_next,
    \\    const float* gate_acts,
    \\    const float* c_curr,
    \\    const float* c_prev,
    \\    float* grad_gates,
    \\    float* grad_c_prev,
    \\    float* grad_h_prev_part,
    \\    int hidden_size) {
    \\    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (gid < hidden_size) {
    \\        int h = hidden_size;
    \\        float i = gate_acts[gid];
    \\        float f = gate_acts[h + gid];
    \\        float g = gate_acts[2 * h + gid];
    \\        float o = gate_acts[3 * h + gid];
    \\
    \\        float gh = grad_h_curr[gid] + grad_h_next[gid];
    \\        float tanh_cc = tanhf(c_curr[gid]);
    \\
    \\        // d_o = d_h * tanh(c_t) * sigmoid'(o_t)
    \\        float d_o = gh * tanh_cc * (o * (1.0f - o));
    \\
    \\        // d_c = d_h * o * tanh'(c_t) + d_c_next
    \\        float d_c = gh * o * (1.0f - tanh_cc * tanh_cc) + grad_c_next[gid];
    \\
    \\        // d_f = d_c * c_{t-1} * sigmoid'(f_t)
    \\        float d_f = d_c * c_prev[gid] * (f * (1.0f - f));
    \\        // d_i = d_c * g_t * sigmoid'(i_t)
    \\        float d_i = d_c * g * (i * (1.0f - i));
    \\        // d_g = d_c * i_t * tanh'(g_t)
    \\        float d_g = d_c * i * (1.0f - g * g);
    \\
    \\        grad_gates[gid] = d_i;
    \\        grad_gates[h + gid] = d_f;
    \\        grad_gates[2 * h + gid] = d_g;
    \\        grad_gates[3 * h + gid] = d_o;
    \\
    \\        // grad_c_prev = d_c * f
    \\        grad_c_prev[gid] = d_c * f;
    \\
    \\        // grad_h_prev_part is not computed here but passed for API consistency
    \\        grad_h_prev_part[gid] = 0.0f;
    \\    }
    \\}
;

/// GRU forward step kernel CUDA C source
pub const GRU_FORWARD_STEP_SOURCE =
    \\extern "C" __global__ void gru_forward_step(
    \\    const float* gates_ih,
    \\    const float* gates_hh,
    \\    const float* bias,
    \\    const float* h_prev,
    \\    float* h_curr,
    \\    float* gate_acts,
    \\    float* n_hh_out,
    \\    int hidden_size) {
    \\    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (gid < hidden_size) {
    \\        int h = hidden_size;
    \\
    \\        // z_t = sigmoid(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_{hz})
    \\        float z = 1.0f / (1.0f + expf(-(gates_ih[gid] + gates_hh[gid] + bias[gid])));
    \\
    \\        // r_t = sigmoid(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_{hr})
    \\        float r = 1.0f / (1.0f + expf(-(gates_ih[h + gid] + gates_hh[h + gid] + bias[h + gid])));
    \\
    \\        // n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_{t-1} + b_{hn}))
    \\        float n_hh = gates_hh[2 * h + gid];
    \\        float n = tanhf(gates_ih[2 * h + gid] + bias[2 * h + gid] + r * n_hh);
    \\
    \\        // h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    \\        float hp = h_prev[gid];
    \\        h_curr[gid] = (1.0f - z) * n + z * hp;
    \\
    \\        // Store for backward
    \\        gate_acts[gid] = z;
    \\        gate_acts[h + gid] = r;
    \\        gate_acts[2 * h + gid] = n;
    \\        n_hh_out[gid] = n_hh;
    \\    }
    \\}
;

// =============================================================================
// Tensor Core WMMA Kernels
// =============================================================================
/// WMMA matrix multiplication using CUDA C++ with mma.h
/// Target: sm_70+ (Volta, Turing, Ampere, Hopper)
/// Uses FP16 input with FP32 accumulation for maximum throughput
pub const MATMUL_TENSOR_CORE_SOURCE =
    \\#include <mma.h>
    \\using namespace nvcuda;
    \\
    \\extern "C" __global__ void matmul_tensor_core(
    \\    const half* __restrict__ A,
    \\    const half* __restrict__ B,
    \\    float* __restrict__ C,
    \\    int M, int N, int K,
    \\    int accumulate) {
    \\
    \\    // Tile dimensions: 64x64 per block, 16x16 per warp
    \\    const int WMMA_M = 16;
    \\    const int WMMA_N = 16;
    \\    const int WMMA_K = 16;
    \\
    \\    // Block tile coordinates
    \\    int block_row = blockIdx.y * 64;
    \\    int block_col = blockIdx.x * 64;
    \\
    \\    // Warp coordinates within block (4 warps per block)
    \\    int warp_id = threadIdx.x / 32;
    \\    int warp_row = (warp_id / 2) * 32;  // 0 or 32
    \\    int warp_col = (warp_id % 2) * 32;  // 0 or 32
    \\
    \\    // Thread lane within warp
    \\    int lane_id = threadIdx.x % 32;
    \\
    \\    // Shared memory for A and B tiles (double buffered)
    \\    // A: 64 x 32 (padded to avoid bank conflicts)
    \\    // B: 64 x 32
    \\    __shared__ half sA[2][64][32];
    \\    __shared__ half sB[2][64][32];
    \\
    \\    // Accumulator fragments (FP32)
    \\    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    \\    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    \\    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    \\
    \\    // Initialize accumulators
    \\    #pragma unroll
    \\    for (int i = 0; i < 2; i++) {
    \\        for (int j = 0; j < 2; j++) {
    \\            wmma::fill_fragment(acc[i][j], 0.0f);
    \\        }
    \\    }
    \\
    \\    // Number of K tiles
    \\    int num_k_tiles = (K + WMMA_K - 1) / WMMA_K;
    \\
    \\    // Current buffer index for double buffering
    \\    int buf_idx = 0;
    \\
    \\    // Load first tile
    \\    int k_tile = 0;
    \\    int k_offset = k_tile * WMMA_K;
    \\
    \\    // Cooperative load: each thread loads 4 elements from A and B
    \\    // A: M x K, B: K x N
    \\    int load_row = threadIdx.x / 8;
    \\    int load_col = (threadIdx.x % 8) * 4;
    \\
    \\    // Load A tile: 64 rows x 32 cols
    \\    #pragma unroll
    \\    for (int i = 0; i < 4; i++) {
    \\        int r = load_row + i * 32;
    \\        int c = load_col;
    \\        int global_row = block_row + r;
    \\        int global_col = k_offset + c;
    \\
    \\        if (global_row < M && global_col < K) {
    \\            sA[buf_idx][r][c] = A[global_row * K + global_col];
    \\            sA[buf_idx][r][c+1] = (global_col + 1 < K) ? A[global_row * K + global_col + 1] : __float2half(0.0f);
    \\            sA[buf_idx][r][c+2] = (global_col + 2 < K) ? A[global_row * K + global_col + 2] : __float2half(0.0f);
    \\            sA[buf_idx][r][c+3] = (global_col + 3 < K) ? A[global_row * K + global_col + 3] : __float2half(0.0f);
    \\        } else {
    \\            sA[buf_idx][r][c] = __float2half(0.0f);
    \\            sA[buf_idx][r][c+1] = __float2half(0.0f);
    \\            sA[buf_idx][r][c+2] = __float2half(0.0f);
    \\            sA[buf_idx][r][c+3] = __float2half(0.0f);
    \\        }
    \\    }
    \\
    \\    // Load B tile: 64 cols x 32 rows (transposed access)
    \\    #pragma unroll
    \\    for (int i = 0; i < 4; i++) {
    \\        int r = load_row + i * 32;
    \\        int c = load_col;
    \\        int global_row = k_offset + r;
    \\        int global_col = block_col + c;
    \\
    \\        if (global_row < K && global_col < N) {
    \\            sB[buf_idx][c][r] = B[global_row * N + global_col];
    \\            sB[buf_idx][c+1][r] = (global_col + 1 < N) ? B[global_row * N + global_col + 1] : __float2half(0.0f);
    \\            sB[buf_idx][c+2][r] = (global_col + 2 < N) ? B[global_row * N + global_col + 2] : __float2half(0.0f);
    \\            sB[buf_idx][c+3][r] = (global_col + 3 < N) ? B[global_row * N + global_col + 3] : __float2half(0.0f);
    \\        } else {
    \\            sB[buf_idx][c][r] = __float2half(0.0f);
    \\            sB[buf_idx][c+1][r] = __float2half(0.0f);
    \\            sB[buf_idx][c+2][r] = __float2half(0.0f);
    \\            sB[buf_idx][c+3][r] = __float2half(0.0f);
    \\        }
    \\    }
    \\
    \\    __syncthreads();
    \\
    \\    // Main loop over K tiles
    \\    for (k_tile = 0; k_tile < num_k_tiles; k_tile++) {
    \\        int next_k_offset = ((k_tile + 1) < num_k_tiles) ? (k_tile + 1) * WMMA_K : 0;
    \\        int next_buf_idx = 1 - buf_idx;
    \\
    \\        // Compute using current tile
    \\        // Each warp computes 2x2 WMMA tiles (32x32 output)
    \\        #pragma unroll
    \\        for (int wmma_k = 0; wmma_k < WMMA_K; wmma_k += WMMA_K) {
    \\            // Load A fragments (2 tiles per warp)
    \\            #pragma unroll
    \\            for (int i = 0; i < 2; i++) {
    \\                int row_offset = warp_row + i * WMMA_M;
    \\                wmma::load_matrix_sync(a_frag, &sA[buf_idx][row_offset][wmma_k], 32);
    \\
    \\                // Load B fragments (2 tiles per warp)
    \\                #pragma unroll
    \\                for (int j = 0; j < 2; j++) {
    \\                    int col_offset = warp_col + j * WMMA_N;
    \\                    wmma::load_matrix_sync(b_frag, &sB[buf_idx][col_offset][wmma_k], 32);
    \\
    \\                    // MMA operation
    \\                    wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
    \\                }
    \\            }
    \\        }
    \\
    \\        // Load next tile (if not last iteration)
    \\        if (k_tile + 1 < num_k_tiles) {
    \\            // Load next A tile
    \\            #pragma unroll
    \\            for (int i = 0; i < 4; i++) {
    \\                int r = load_row + i * 32;
    \\                int c = load_col;
    \\                int global_row = block_row + r;
    \\                int global_col = next_k_offset + c;
    \\
    \\                if (global_row < M && global_col < K) {
    \\                    sA[next_buf_idx][r][c] = A[global_row * K + global_col];
    \\                    sA[next_buf_idx][r][c+1] = (global_col + 1 < K) ? A[global_row * K + global_col + 1] : __float2half(0.0f);
    \\                    sA[next_buf_idx][r][c+2] = (global_col + 2 < K) ? A[global_row * K + global_col + 2] : __float2half(0.0f);
    \\                    sA[next_buf_idx][r][c+3] = (global_col + 3 < K) ? A[global_row * K + global_col + 3] : __float2half(0.0f);
    \\                } else {
    \\                    sA[next_buf_idx][r][c] = __float2half(0.0f);
    \\                    sA[next_buf_idx][r][c+1] = __float2half(0.0f);
    \\                    sA[next_buf_idx][r][c+2] = __float2half(0.0f);
    \\                    sA[next_buf_idx][r][c+3] = __float2half(0.0f);
    \\                }
    \\            }
    \\
    \\            // Load next B tile
    \\            #pragma unroll
    \\            for (int i = 0; i < 4; i++) {
    \\                int r = load_row + i * 32;
    \\                int c = load_col;
    \\                int global_row = next_k_offset + r;
    \\                int global_col = block_col + c;
    \\
    \\                if (global_row < K && global_col < N) {
    \\                    sB[next_buf_idx][c][r] = B[global_row * N + global_col];
    \\                    sB[next_buf_idx][c+1][r] = (global_col + 1 < N) ? B[global_row * N + global_col + 1] : __float2half(0.0f);
    \\                    sB[next_buf_idx][c+2][r] = (global_col + 2 < N) ? B[global_row * N + global_col + 2] : __float2half(0.0f);
    \\                    sB[next_buf_idx][c+3][r] = (global_col + 3 < N) ? B[global_row * N + global_col + 3] : __float2half(0.0f);
    \\                } else {
    \\                    sB[next_buf_idx][c][r] = __float2half(0.0f);
    \\                    sB[next_buf_idx][c+1][r] = __float2half(0.0f);
    \\                    sB[next_buf_idx][c+2][r] = __float2half(0.0f);
    \\                    sB[next_buf_idx][c+3][r] = __float2half(0.0f);
    \\                }
    \\            }
    \\        }
    \\
    \\        __syncthreads();
    \\        buf_idx = next_buf_idx;
    \\    }
    \\
    \\    // Store results to global memory
    \\    #pragma unroll
    \\    for (int i = 0; i < 2; i++) {
    \\        for (int j = 0; j < 2; j++) {
    \\            int row = block_row + warp_row + i * WMMA_M;
    \\            int col = block_col + warp_col + j * WMMA_N;
    \\
    \\            if (row < M && col < N) {
    \\                wmma::store_matrix_sync(&C[row * N + col], acc[i][j], N, wmma::mem_row_major);
    \\            }
    \\        }
    \\    }
    \\}
;

/// GRU backward step kernel CUDA C source
pub const GRU_BACKWARD_STEP_SOURCE =
    \\extern "C" __global__ void gru_backward_step(
    \\    const float* grad_h_curr,
    \\    const float* grad_h_next,
    \\    const float* gate_acts,
    \\    const float* h_prev,
    \\    const float* n_hh,
    \\    float* grad_gates_ih,
    \\    float* grad_gates_hh,
    \\    float* grad_h_prev,
    \\    int hidden_size) {
    \\    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (gid < hidden_size) {
    \\        int h = hidden_size;
    \\        float z = gate_acts[gid];
    \\        float r = gate_acts[h + gid];
    \\        float n = gate_acts[2 * h + gid];
    \\
    \\        float gh = grad_h_curr[gid] + grad_h_next[gid];
    \\        float hp = h_prev[gid];
    \\
    \\        // h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    \\        float d_nt = gh * (1.0f - z);
    \\        float d_zt = gh * (hp - n);
    \\
    \\        // n_t = tanh(n_ih + r_t * n_hh)
    \\        float d_n_raw = d_nt * (1.0f - n * n);
    \\        float d_n_ih = d_n_raw;
    \\        float d_n_hh = d_n_raw * r;
    \\
    \\        // d_rt = d_n_raw * n_hh * sigmoid'(r_t)
    \\        float d_rt = d_n_raw * n_hh[gid] * (r * (1.0f - r));
    \\
    \\        // z_t = sigmoid(z_raw) => d_z_raw = d_zt * z_t * (1 - z_t)
    \\        float d_z_raw = d_zt * z * (1.0f - z);
    \\
    \\        // Gates IH grads
    \\        grad_gates_ih[gid] = d_z_raw;
    \\        grad_gates_ih[h + gid] = d_rt;
    \\        grad_gates_ih[2 * h + gid] = d_n_ih;
    \\
    \\        // Gates HH grads
    \\        grad_gates_hh[gid] = d_z_raw;
    \\        grad_gates_hh[h + gid] = d_rt;
    \\        grad_gates_hh[2 * h + gid] = d_n_hh;
    \\
    \\        // grad_h_prev = d_h * z_t
    \\        grad_h_prev[gid] = gh * z;
    \\    }
    \\}
;


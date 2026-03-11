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

// =============================================================================
// PTX Header - Common to all kernels
// =============================================================================
pub const PTX_HEADER =
    \\.version 7.5
    \\.target sm_80
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
    \\    .visible .entry conv2d_forward(
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

// Export all kernel names for loading
pub const KERNEL_NAMES = .{
    "matmul",
    "matmul_batch",
    "matmul_transpose_b",
    "relu_forward",
    "relu_backward",
    "sigmoid_forward",
    "sigmoid_backward",
    "tanh_forward",
    "tanh_backward",
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

/// MSE backward PTX
pub const MSE_BACKWARD_PTX = PTX_HEADER ++
    \\ .visible .entry mse_backward(
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
    \\ .visible .entry cross_entropy_backward(
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
    \\ .visible .entry binary_cross_entropy_backward(
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
    \\ .visible .entry kl_divergence_backward(
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

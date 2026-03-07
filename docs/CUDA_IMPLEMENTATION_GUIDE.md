# CUDA Implementation Guide for ZigNeuron

## Executive Summary

This document provides comprehensive guidance for implementing high-performance CUDA kernels for the ZigNeuron neural network library. The implementation targets modern NVIDIA architectures (Turing/Ampere/Ada/Hopper/Blackwell) with compute capability 7.5+.

## 1. cuBLAS vs Custom Kernels Decision Matrix

| Operation | Recommendation | Rationale |
|-----------|---------------|-----------|
| **Matrix Multiplication (M,N,K > 512)** | **cuBLAS** | cuBLAS achieves 90%+ of peak TFLOPS with expert tuning |
| **Matrix Multiplication (small sizes)** | Custom kernel | cuBLAS has high launch overhead for small problems |
| **Batched MatMul** | cuBLAS `cublasSgemmBatched` | Optimized for batch dimension parallelism |
| **Transpose variants** | cuBLAS with params | `CUBLAS_OP_T` flags handle transpose efficiently |
| **Element-wise ops** | Custom kernel | Memory-bound, fusion opportunities |
| **Activations** | Custom kernel | Element-wise, fusion critical |
| **Softmax** | Custom kernel | Requires warp-level reduction primitives |
| **LayerNorm** | Custom kernel | Requires parallel reduction across features |
| **Conv1D/Conv2D** | cuDNN for training | cuDNN has optimized winograd/algo selection |
| **Attention** | Flash Attention custom | Memory-bound, custom kernel beats cuDNN |
| **Optimizers** | Custom kernel | Memory-bound, element-wise updates |

### cuBLAS Integration Pattern

```cuda
// For large matmul: Use cuBLAS with tensor cores
// For small matmul (< 256x256): Use custom shared-memory tiled kernel
// Threshold determined empirically: ~256x256x256

#define CUBLAS_SMALL_THRESHOLD 256

void matmulDispatch(const float* A, const float* B, float* C,
                    int M, int N, int K, bool transposeA, bool transposeB) {
    if (M * N * K > CUBLAS_SMALL_THRESHOLD * CUBLAS_SMALL_THRESHOLD * CUBLAS_SMALL_THRESHOLD) {
        cublasSgemm(handle,
                    transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
                    transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                    M, N, K,
                    &alpha, A, lda, B, ldb, &beta, C, ldc);
    } else {
        // Launch custom small-kernel
        matmul_small_kernel<<<grid, block>>>(A, B, C, M, N, K);
    }
}
```

## 2. Kernel Configuration Recommendations

### 2.1 Matrix Multiplication

```cuda
// Configuration for shared memory tiled matmul
#define TILE_M 128  // Rows per block
#define TILE_N 128  // Cols per block
#define TILE_K 8    // K dimension chunk (for shared mem)

// Block dimensions: 128 threads = 4 warps
// Each warp computes 32x32 output tile (1 thread = 4x4 elements)
dim3 block(16, 8, 1);  // 128 threads

// Grid dimensions
dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, batch_size);

// Shared memory per block: 2 * TILE_K * (TILE_M + TILE_N) floats
// ~ 2 * 8 * 256 * 4 = 16KB per block
// Occupancy: 48KB SMEM / 16KB = 3 blocks per SM (good)
```

### 2.2 Element-wise Operations

```cuda
// Configuration for element-wise ops
// Process 4 elements per thread (vectorized loads/stores)
#define ELEMWISE_VEC_SIZE 4

// 256 threads per block (8 warps)
// Each block processes 1024 elements
dim3 block(256, 1, 1);
dim3 grid((size + 1023) / 1024, 1, 1);

// For batched operations, use 3D grid:
dim3 grid((inner_dim + 255) / 256, outer_dim, batch_size);
```

### 2.3 Softmax

```cuda
// Configuration for softmax (per-row operation)
// One warp per row for efficient reduction
#define SOFTMAX_WARP_SIZE 32

// Block: 128 threads = 4 warps
// Each block processes 4 rows
dim3 block(128, 1, 1);
dim3 grid((num_samples + 3) / 4, 1, 1);

// Use warp shuffle for intra-warp reduction
// Final reduction across warps via shared memory
```

### 2.4 LayerNorm

```cuda
// Configuration for LayerNorm (per-sample operation)
// One threadblock per sample
#define LAYERNORM_THREADS 256

dim3 block(LAYERNORM_THREADS, 1, 1);
dim3 grid(batch_size, 1, 1);

// Use warp shuffle for reduction within warps
// Use shared memory for inter-warp reduction
// Each thread processes feature_dim / 256 elements
```

### 2.5 Convolution

```cuda
// Configuration for Conv1D
// Each thread computes one output element
// Block: 256 threads (1D)
// Grid: (out_len, out_channels, batch_size)

dim3 block(256, 1, 1);
dim3 grid((out_len + 255) / 256, out_channels, batch_size);

// For Conv2D:
// Block: 16x16 threads (2D tile)
// Grid: ((out_w + 15) / 16, (out_h + 15) / 16, out_channels * batch_size)

dim3 block(16, 16, 1);
dim3 grid((out_w + 15) / 16, (out_h + 15) / 16, out_channels * batch_size);
```

### 2.6 Attention

```cuda
// Flash Attention inspired configuration
// One threadblock per query sequence position
// Shared memory for KV cache tiling

#define FLASH_ATTN_BLOCK_SIZE 128
#define FLASH_ATTN_HEAD_DIM 64

dim3 block(128, 1, 1);  // One warp per 32 heads
dim3 grid(seq_len, num_heads, batch_size);

// SMEM: Q tile (128 * 64 * 2 bytes) + K tile + V tile
// ~ 16KB + 16KB + 16KB = 48KB (fits in 64KB L1/SMEM)
```

## 3. Memory Coalescing Strategies

### 3.1 Global Memory Access Patterns

```cuda
// GOOD: Coalesced access - contiguous threads access contiguous memory
__global__ void coalesced_copy(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes 4 elements (float4 vectorized)
    for (int i = idx; i < n / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(in)[i];
        reinterpret_cast<float4*>(out)[i] = val;
    }

    // Handle remainder
    for (int i = idx * 4 + n - (n % 4); i < n; i++) {
        out[i] = in[i];
    }
}

// BAD: Strided access causes uncoalesced memory transactions
__global__ void strided_access(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread accesses non-contiguous elements
    out[idx] = in[idx * 32];  // Stride of 32 - very bad!
}

// GOOD: Transpose with shared memory to enable coalescing
__global__ void transpose_coalesced(float* out, const float* in, int rows, int cols) {
    __shared__ float tile[32][33];  // Pad to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Coalesced read from input
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }
    __syncthreads();

    // Coalesced write to output
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < rows && y < cols) {
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### 3.2 Shared Memory Usage Patterns

```cuda
// Tiled matrix multiplication with shared memory
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    const int TILE_M = 128;
    const int TILE_N = 128;
    const int TILE_K = 8;

    __shared__ float As[TILE_M][TILE_K + 1];  // +1 for padding
    __shared__ float Bs[TILE_K][TILE_N + 1];

    // Each thread computes 4x4 output elements
    float sum[4][4] = {{0}};

    for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        // Load A tile (coalesced)
        // Each thread loads 4 elements from A
        // ... load code ...

        // Load B tile (coalesced)
        // Each thread loads 4 elements from B
        // ... load code ...

        __syncthreads();

        // Compute on tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // Load from shared memory (fast)
            float a[4] = {As[threadIdx.y * 4 + 0][k],
                          As[threadIdx.y * 4 + 1][k],
                          As[threadIdx.y * 4 + 2][k],
                          As[threadIdx.y * 4 + 3][k]};
            float b[4] = {Bs[k][threadIdx.x * 4 + 0],
                          Bs[k][threadIdx.x * 4 + 1],
                          Bs[k][threadIdx.x * 4 + 2],
                          Bs[k][threadIdx.x * 4 + 3]};

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    sum[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    // Write results (coalesced)
    // ... store code ...
}
```

## 4. Tensor Core Usage (WMMA)

### 4.1 When to Use Tensor Cores

```cuda
// Use WMMA for: M, N, K >= 16 and multiples of 8
// Best performance: M, N, K multiples of 16 (TF32) or 8 (FP16)

// FP16 Tensor Core matmul (SM 7.0+)
__global__ void matmul_wmma_fp16(const half* A, const half* B, float* C,
                                  int M, int N, int K) {
    using namespace nvcuda::wmma;

    // Declare fragments
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;

    // Initialize accumulator
    fill_fragment(acc_frag, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += 16) {
        // Load A and B fragments
        load_matrix_sync(a_frag, A + blockIdx.y * 16 * K + k, K);
        load_matrix_sync(b_frag, B + k * N + blockIdx.x * 16, N);

        // MMA operation
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store result
    store_matrix_sync(C + blockIdx.y * 16 * N + blockIdx.x * 16, acc_frag, N, mem_row_major);
}
```

### 4.2 Tensor Core Guidelines

| Data Type | MMA Shape | Accumulator | Min Arch | Notes |
|-----------|-----------|-------------|----------|-------|
| FP16 | 16x16x16 | FP32 | SM 7.0 | 2x throughput vs FP32 |
| BF16 | 16x16x16 | FP32 | SM 8.0 | Better range than FP16 |
| TF32 | 16x16x8 | FP32 | SM 8.0 | ~10x throughput, same range as FP32 |
| FP64 | 8x8x4 | FP64 | SM 8.0 | Limited support |
| INT8 | 16x16x16 | INT32 | SM 7.5 | 4x throughput |

## 5. Warp-Level Primitives

### 5.1 Warp Shuffle for Reductions

```cuda
// Warp-level sum reduction using __shfl_down_sync
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level sum reduction using warp shuffle + shared memory
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp reduction
    val = warpReduceSum(val);

    // First thread in warp writes to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Final reduction across warps
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
        val = warpReduceSum(val);
    }
    return val;
}

// Usage in softmax
__global__ void softmax_warp(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Find max (warp shuffle reduction)
    float max_val = input[idx];
    max_val = warpReduceMax(max_val);  // Similar to sum

    // Step 2: Compute exp and sum
    float exp_val = exp(input[idx] - max_val);
    float sum_exp = warpReduceSum(exp_val);

    // Step 3: Normalize
    output[idx] = exp_val / sum_exp;
}
```

### 5.2 Warp-Level Matrix Operations

```cuda
// For operations that map well to warp execution
// Example: Each warp processes one attention head

__global__ void attention_warp_per_head(const float* Q, const float* K, const float* V,
                                        float* output, int seq_len, int head_dim) {
    int head_idx = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp handles different query positions
    for (int q_pos = warp_id; q_pos < seq_len; q_pos += warps_per_block) {
        // Load Q for this position (broadcast to all lanes)
        float q_val = Q[head_idx * seq_len * head_dim + q_pos * head_dim + lane_id];

        // Compute attention scores with all K positions
        float max_score = -INFINITY;
        for (int k_pos = 0; k_pos < seq_len; k_pos++) {
            float k_val = K[head_idx * seq_len * head_dim + k_pos * head_dim + lane_id];
            float score = q_val * k_val;  // Dot product

            // Warp reduction for dot product
            score = warpReduceSum(score);

            if (lane_id == 0) {
                max_score = fmaxf(max_score, score);
            }
        }
        max_score = __shfl_sync(0xFFFFFFFF, max_score, 0);

        // Softmax computation...
    }
}
```

## 6. Stream and Event Management

### 6.1 Async Execution Pattern

```cuda
// Create multiple streams for overlapping
#define NUM_STREAMS 4

class CudaStreamPool {
    cudaStream_t streams[NUM_STREAMS];
    int current;

public:
    void init() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
        }
        current = 0;
    }

    cudaStream_t get() {
        return streams[current++ % NUM_STREAMS];
    }

    void synchronizeAll() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamSynchronize(streams[i]);
        }
    }
};

// Usage for model parallelism
void forwardPassAsync(Layer* layers, int num_layers, float* input, float* output) {
    CudaStreamPool streams;
    streams.init();

    float* current = input;
    for (int i = 0; i < num_layers; i++) {
        cudaStream_t stream = streams.get();

        // Launch layer computation asynchronously
        layers[i].forwardAsync(current, output, stream);

        // Optionally overlap with next layer data transfer
        if (i < num_layers - 1) {
            cudaMemcpyAsync(layers[i+1].input_buffer, current, size, cudaMemcpyDeviceToDevice, stream);
        }

        current = output;
    }

    streams.synchronizeAll();
}
```

### 6.2 CUDA Graphs for Reduced Launch Overhead

```cuda
// For repeated inference with same graph structure
class CudaGraphExecutor {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    bool captured;

public:
    void captureBegin(cudaStream_t stream) {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        captured = false;
    }

    void captureEnd(cudaStream_t stream) {
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
        captured = true;
    }

    void launch(cudaStream_t stream) {
        if (captured) {
            cudaGraphLaunch(graphExec, stream);
        }
    }
};

// Usage for inference loops
void inferenceWithGraph(Model* model, float* input, float* output, int iterations) {
    CudaGraphExecutor graph;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Capture graph
    graph.captureBegin(stream);
    model->forward(input, output, stream);
    graph.captureEnd(stream);

    // Execute repeatedly
    for (int i = 0; i < iterations; i++) {
        graph.launch(stream);
    }

    cudaStreamSynchronize(stream);
}
```

### 6.3 Event-Based Timing and Synchronization

```cuda
// Precise kernel timing
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} KernelTimer;

void timeKernel(void (*kernel)(), dim3 grid, dim3 block, void** args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    kernel<<<grid, block>>>();
    cudaDeviceSynchronize();

    // Timed execution
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<grid, block>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Average kernel time: %.3f ms\n", ms / 100);
}
```

## 7. PTX vs CUDA C Compilation

### 7.1 Recommendation: CUDA C + Separate Compilation

For Zig integration, I recommend the following approach:

```
Project Structure:
├── kernels/
│   ├── matmul.cu          # CUDA C source
│   ├── activation.cu
│   ├── loss.cu
│   └── ...
├── ptx/                   # Pre-compiled PTX (optional)
│   ├── matmul.ptx
│   └── ...
└── src/
    └── cuda.zig           # Zig bindings
```

### 7.2 Build Process

```zig
// build.zig additions
const std = @import("std");

pub fn build(b: *std.Build) void {
    // ... existing build config ...

    // CUDA compilation step
    const cuda_compile = b.addSystemCommand(&[_][]const u8{
        "nvcc",
        "-O3",
        "-arch=sm_75",          // Minimum architecture
        "-gencode=arch=compute_75,code=sm_75",   // Turing
        "-gencode=arch=compute_80,code=sm_80",   // Ampere
        "-gencode=arch=compute_86,code=sm_86",   // Ampere RTX
        "-gencode=arch=compute_89,code=sm_89",   // Ada
        "-gencode=arch=compute_90,code=sm_90",   // Hopper
        "-ptx",                  // Generate PTX for JIT compilation
        "-dc",                   // Separate compilation
        "-o", "kernels.ptx",
        "kernels/matmul.cu",
        "kernels/activation.cu",
        // ... more files ...
    });

    // Link PTX at runtime or embed as string
    // Option 1: Embed PTX as string literal
    const ptx_string = @embedFile("kernels.ptx");
}
```

### 7.3 Runtime PTX Loading

```cuda
// In cuda.zig - Runtime kernel loading
const CUmodule = opaque{};
const CUfunction = opaque{};

extern "C" {
    fn cuModuleLoadData(module: **CUmodule, ptx: [*]const u8) c_int;
    fn cuModuleGetFunction(func: **CUfunction, module: *CUmodule, name: [*]const u8) c_int;
    fn cuLaunchKernel(func: *CUfunction,
                      gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint,
                      blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint,
                      sharedMemBytes: c_uint,
                      hStream: ?*anyopaque,
                      kernelParams: [*]?*anyopaque,
                      extra: ?*anyopaque) c_int;
}

pub const CudaKernel = struct {
    function: *CUfunction,

    pub fn launch(self: *CudaKernel, grid: dim3, block: dim3, args: []?*anyopaque) !void {
        const result = cuLaunchKernel(
            self.function,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            0, null,
            args.ptr, null
        );
        if (result != 0) return error.KernelLaunchFailed;
    }
};
```

## 8. Kernel Implementation Templates

### 8.1 Matrix Multiplication (Tiled with SMEM)

```cuda
// File: kernels/matmul.cu

#define TILE_M 128
#define TILE_N 128
#define TILE_K 8
#define WARP_M 32
#define WARP_N 32

extern "C" __global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    bool accumulate
) {
    // 4 warps per block, each warp computes 32x32 tile
    // Block computes 128x128 output tile

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Compute warp position within block
    int warp_row = (warp_id / 2) * WARP_M;
    int warp_col = (warp_id % 2) * WARP_N;

    // Thread position within warp (each thread computes 4x4 elements)
    int thread_row = (lane_id / 8) * 4;
    int thread_col = (lane_id % 8) * 4;

    // Global position
    int global_row = blockIdx.y * TILE_M + warp_row + thread_row;
    int global_col = blockIdx.x * TILE_N + warp_col + thread_col;

    // Accumulator registers
    float sum[4][4] = {{0}};

    // Loop over K tiles
    for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        // Load A tile into SMEM (coalesced)
        // Each thread loads multiple elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int load_row = warp_row + thread_row + i;
            int load_col = threadIdx.x % TILE_K;

            if (global_row + i < M && tile_k + load_col < K) {
                As[load_row][load_col] = A[(global_row + i) * K + tile_k + load_col];
            } else {
                As[load_row][load_col] = 0.0f;
            }
        }

        // Load B tile into SMEM (coalesced)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int load_row = threadIdx.x / TILE_N;
            int load_col = warp_col + thread_col + j;

            if (tile_k + load_row < K && global_col + j < N) {
                Bs[load_row][load_col] = B[(tile_k + load_row) * N + global_col + j];
            } else {
                Bs[load_row][load_col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute on tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a[4], b[4];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a[i] = As[warp_row + thread_row + i][k];
            }

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                b[j] = Bs[k][warp_col + thread_col + j];
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    sum[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

    // Store results (coalesced)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (global_row + i < M && global_col + j < N) {
                int idx = (global_row + i) * N + global_col + j;
                if (accumulate) {
                    C[idx] += sum[i][j];
                } else {
                    C[idx] = sum[i][j];
                }
            }
        }
    }
}

// Batched version
extern "C" __global__ void matmul_batched(
    const float* __restrict__ A,  // [batch, M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [batch, M, N]
    int batch_size, int M, int N, int K,
    bool accumulate
) {
    int batch = blockIdx.z;
    if (batch >= batch_size) return;

    const float* A_batch = A + batch * M * K;
    float* C_batch = C + batch * M * N;

    // Reuse the tiled matmul logic
    // ... same as above using A_batch and C_batch ...
}
```

### 8.2 Activation Functions

```cuda
// File: kernels/activation.cu

// Vectorized loads for element-wise ops
struct float4 {
    float x, y, z, w;
};

extern "C" __global__ void relu_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Vectorized loop
    for (int i = idx; i < size / 4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(input)[i];
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        reinterpret_cast<float4*>(output)[i] = val;
    }

    // Remainder
    for (int i = idx * 4 + (size - size % 4); i < size; i++) {
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

    if (idx < size) {
        grad_input[idx] = (output[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

extern "C" __global__ void sigmoid_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = input[idx];
        // Stable sigmoid: 1 / (1 + exp(-x))
        output[idx] = 1.0f / (1.0f + expf(-fminf(fmaxf(x, -20.0f), 20.0f)));
    }
}

extern "C" __global__ void sigmoid_backward(
    const float* __restrict__ output,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float y = output[idx];
        grad_input[idx] = grad_output[idx] * y * (1.0f - y);
    }
}

extern "C" __global__ void softmax_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_samples, int num_classes
) {
    int sample = blockIdx.x;
    if (sample >= num_samples) return;

    const float* in_sample = input + sample * num_classes;
    float* out_sample = output + sample * num_classes;

    // Step 1: Find max (warp reduction)
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        max_val = fmaxf(max_val, in_sample[i]);
    }

    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

    // Step 2: Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        float exp_val = expf(in_sample[i] - max_val);
        out_sample[i] = exp_val;
        sum_exp += exp_val;
    }

    // Warp reduce sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

    // Step 3: Normalize
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        out_sample[i] /= sum_exp;
    }
}
```

### 8.3 Optimizers

```cuda
// File: kernels/optimizer.cu

extern "C" __global__ void sgd_update(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    int size,
    float learning_rate,
    float weight_decay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float g = gradients[idx];
        float w = weights[idx];

        // Gradient clipping
        g = fminf(fmaxf(g, -5.0f), 5.0f);
        if (isnan(g)) g = 0.0f;

        float new_w = w - learning_rate * (g + weight_decay * w);
        new_w = fminf(fmaxf(new_w, -100.0f), 100.0f);

        weights[idx] = new_w;
    }
}

extern "C" __global__ void adam_update(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float* __restrict__ m,
    float* __restrict__ v,
    int size,
    float lr, float beta1, float beta2, float eps,
    float bias_corr1, float bias_corr2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float g = gradients[idx];
        g = fminf(fmaxf(g, -5.0f), 5.0f);
        if (isnan(g)) g = 0.0f;

        float m_val = beta1 * m[idx] + (1.0f - beta1) * g;
        float v_val = beta2 * v[idx] + (1.0f - beta2) * g * g;

        m[idx] = m_val;
        v[idx] = v_val;

        float m_hat = m_val / bias_corr1;
        float v_hat = v_val / bias_corr2;

        float new_w = weights[idx] - lr * m_hat / (sqrtf(v_hat) + eps);
        new_w = fminf(fmaxf(new_w, -100.0f), 100.0f);

        weights[idx] = new_w;
    }
}

extern "C" __global__ void rmsprop_update(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float* __restrict__ g_avg,
    int size,
    float lr, float rho, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float g = gradients[idx];
        g = fminf(fmaxf(g, -5.0f), 5.0f);
        if (isnan(g)) g = 0.0f;

        float g_avg_val = rho * g_avg[idx] + (1.0f - rho) * g * g;
        g_avg[idx] = g_avg_val;

        float new_w = weights[idx] - lr * g / (sqrtf(g_avg_val) + eps);
        new_w = fminf(fmaxf(new_w, -100.0f), 100.0f);

        weights[idx] = new_w;
    }
}
```

### 8.4 Layer Normalization

```cuda
// File: kernels/normalization.cu

extern "C" __global__ void layernorm_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int batch_size, int feature_size,
    float eps
) {
    int sample = blockIdx.x;
    if (sample >= batch_size) return;

    const float* in_sample = input + sample * feature_size;
    float* out_sample = output + sample * feature_size;

    // Compute mean (warp shuffle reduction)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        sum += in_sample[i];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);
    float mean = sum / feature_size;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        float diff = in_sample[i] - mean;
        var_sum += diff * diff;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    }
    var_sum = __shfl_sync(0xFFFFFFFF, var_sum, 0);
    float variance = var_sum / feature_size;
    float inv_std = rsqrtf(variance + eps);

    // Normalize and scale
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x) {
        float x_hat = (in_sample[i] - mean) * inv_std;
        out_sample[i] = x_hat * gamma[i] + beta[i];
    }
}
```

### 8.5 Convolution

```cuda
// File: kernels/convolution.cu

extern "C" __global__ void conv1d_forward(
    const float* __restrict__ input,    // [batch, in_channels, in_len]
    const float* __restrict__ weights,  // [out_channels, in_channels, kernel_size]
    const float* __restrict__ bias,     // [out_channels]
    float* __restrict__ output,         // [batch, out_channels, out_len]
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int in_len, int out_len
) {
    int oc = blockIdx.y;      // output channel
    int t = blockIdx.x * blockDim.x + threadIdx.x;  // time position
    int batch = blockIdx.z;

    if (oc >= out_channels || t >= out_len || batch >= batch_size) return;

    float sum = bias[oc];

    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            int in_idx = ((batch * in_channels + ic) * in_len) + (t + k);
            int w_idx = ((oc * in_channels + ic) * kernel_size) + k;
            sum += input[in_idx] * weights[w_idx];
        }
    }

    int out_idx = ((batch * out_channels + oc) * out_len) + t;
    output[out_idx] = sum;
}

// Conv1D backward with atomic adds for grad_weights
extern "C" __global__ void conv1d_backward(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ grad_after_act,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weights,
    float* __restrict__ grad_bias,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int in_len, int out_len
) {
    // Each thread handles one output gradient
    int oc = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;

    if (oc >= out_channels || t >= out_len || batch >= batch_size) return;

    int out_idx = ((batch * out_channels + oc) * out_len) + t;
    float go = grad_after_act[out_idx];

    // Accumulate grad_bias
    atomicAdd(&grad_bias[oc], go);

    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            int in_idx = ((batch * in_channels + ic) * in_len) + (t + k);
            int w_idx = ((oc * in_channels + ic) * kernel_size) + k;

            // grad_weights
            atomicAdd(&grad_weights[w_idx], input[in_idx] * go);

            // grad_input
            atomicAdd(&grad_input[in_idx], weights[w_idx] * go);
        }
    }
}
```

### 8.6 Attention

```cuda
// File: kernels/attention.cu

// Simplified scaled dot-product attention
extern "C" __global__ void attention_forward(
    const float* __restrict__ Q,  // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    float* __restrict__ workspace,  // For attention scores
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale
) {
    // One threadblock per (batch, head, query_position)
    int batch = blockIdx.z / num_heads;
    int head = blockIdx.z % num_heads;
    int q_pos = blockIdx.y;

    if (batch >= batch_size || q_pos >= seq_len) return;

    // Pointers to this query's data
    const float* Q_q = Q + ((batch * num_heads + head) * seq_len + q_pos) * head_dim;
    const float* K_head = K + (batch * num_heads + head) * seq_len * head_dim;
    const float* V_head = V + (batch * num_heads + head) * seq_len * head_dim;
    float* scores = workspace + ((batch * num_heads + head) * seq_len + q_pos) * seq_len;

    // Compute Q*K^T for this query position
    // Each thread computes scores for multiple key positions
    float max_score = -INFINITY;

    for (int k_pos = threadIdx.x; k_pos < seq_len; k_pos += blockDim.x) {
        float dot = 0.0f;
        const float* K_k = K_head + k_pos * head_dim;

        for (int d = 0; d < head_dim; d++) {
            dot += Q_q[d] * K_k[d];
        }

        float score = dot * scale;
        scores[k_pos] = score;
        max_score = fmaxf(max_score, score);
    }

    // Warp reduce max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, offset));
    }
    max_score = __shfl_sync(0xFFFFFFFF, max_score, 0);

    // Softmax
    float sum_exp = 0.0f;
    for (int k_pos = threadIdx.x; k_pos < seq_len; k_pos += blockDim.x) {
        float exp_val = expf(scores[k_pos] - max_score);
        scores[k_pos] = exp_val;
        sum_exp += exp_val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }
    sum_exp = __shfl_sync(0xFFFFFFFF, sum_exp, 0);

    // Normalize
    for (int k_pos = threadIdx.x; k_pos < seq_len; k_pos += blockDim.x) {
        scores[k_pos] /= sum_exp;
    }
    __syncthreads();

    // Compute weighted sum with V
    float* out_q = output + ((batch * num_heads + head) * seq_len + q_pos) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int k_pos = 0; k_pos < seq_len; k_pos++) {
            sum += scores[k_pos] * V_head[k_pos * head_dim + d];
        }
        out_q[d] = sum;
    }
}
```

### 8.7 Dropout

```cuda
// File: kernels/dropout.cu

// Philox4x32_10 RNG
struct Philox4x32_10 {
    uint4 state;
    uint2 key;

    __device__ Philox4x32_10(uint64_t seed, uint64_t subsequence) {
        state.x = (uint32_t)(subsequence);
        state.y = (uint32_t)(subsequence >> 32);
        state.z = 0;
        state.w = 0;
        key.x = (uint32_t)(seed);
        key.y = (uint32_t)(seed >> 32);
    }

    __device__ uint4 operator()() {
        uint4 s = state;
        uint2 k = key;

        #pragma unroll
        for (int i = 0; i < 10; i++) {
            uint32_t L0 = mul_lo(s.x, 0xD2511F53);
            uint32_t L1 = mul_lo(s.z, 0xD2511F53);
            uint32_t H0 = mul_hi(s.x, 0xD2511F53);
            uint32_t H1 = mul_hi(s.z, 0xD2511F53);

            s.x = H0 ^ s.y ^ k.x;
            s.z = H1 ^ s.w ^ k.y;
            s.y = L0;
            s.w = L1;

            uint32_t t = k.x;
            k.x = k.y;
            k.y = t;
        }

        state.x = s.z;
        state.y = s.w;
        state.z = s.x;
        state.w = s.y;

        return s;
    }

    __device__ uint32_t mul_lo(uint32_t a, uint32_t b) {
        return a * b;
    }

    __device__ uint32_t mul_hi(uint32_t a, uint32_t b) {
        return (uint32_t)(((uint64_t)a * b) >> 32);
    }
};

extern "C" __global__ void dropout_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ mask,
    int size,
    float rate,
    float scale,
    uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    Philox4x32_10 rng(seed, (uint64_t)idx * 4);

    for (int i = idx; i < size / 4; i += stride) {
        uint4 rand_vals = rng();

        float4 rand_floats;
        rand_floats.x = (float)rand_vals.x / 4294967296.0f;
        rand_floats.y = (float)rand_vals.y / 4294967296.0f;
        rand_floats.z = (float)rand_vals.z / 4294967296.0f;
        rand_floats.w = (float)rand_vals.w / 4294967296.0f;

        float4 in = reinterpret_cast<const float4*>(input)[i];
        float4 out, m;

        m.x = (rand_floats.x > rate) ? 1.0f : 0.0f;
        m.y = (rand_floats.y > rate) ? 1.0f : 0.0f;
        m.z = (rand_floats.z > rate) ? 1.0f : 0.0f;
        m.w = (rand_floats.w > rate) ? 1.0f : 0.0f;

        out.x = in.x * m.x * scale;
        out.y = in.y * m.y * scale;
        out.z = in.z * m.z * scale;
        out.w = in.w * m.w * scale;

        reinterpret_cast<float4*>(output)[i] = out;
        reinterpret_cast<float4*>(mask)[i] = m;
    }

    // Remainder
    rng = Philox4x32_10(seed, (uint64_t)(size - size % 4 + threadIdx.x));
    for (int i = idx + (size - size % 4); i < size; i += stride) {
        uint4 rand_vals = rng();
        float p = (float)rand_vals.x / 4294967296.0f;
        mask[i] = (p > rate) ? 1.0f : 0.0f;
        output[i] = input[i] * mask[i] * scale;
    }
}
```

## 9. Performance Tuning Guidelines

### 9.1 Occupancy Optimization

```cuda
// Target 50%+ occupancy (128 threads per SM on modern GPUs)
// For 128 threads/block: 32 warps/SM = 100% occupancy on Ampere

// Use launch bounds to hint occupancy
__launch_bounds__(128, 4)  // 128 threads, min 4 blocks per SM
__global__ void my_kernel(...) {
    // ...
}
```

### 9.2 Memory Bandwidth Optimization

```cuda
// For memory-bound kernels (activations, optimizers):
// 1. Vectorize loads (float4)
// 2. Use LDG (read-only cache) for inputs
__ldg(&input[idx])  // Load through texture cache

// 3. Avoid bank conflicts in shared memory
// Pad arrays: float tile[32][32] -> float tile[32][33]
```

### 9.3 Instruction-Level Optimization

```cuda
// 1. Use intrinsics for faster math
// Instead of: 1.0f / sqrtf(x)
// Use: rsqrtf(x)

// 2. Use __fmaf_rn for fused multiply-add
__fmaf_rn(a, b, c);  // a * b + c in single instruction

// 3. Prefer float over double
// Use: sqrtf, sinf, cosf, expf
// Avoid: sqrt, sin, cos, exp
```

## 10. Zig Integration Summary

### 10.1 Type Mappings

| CUDA Type | Zig Type |
|-----------|----------|
| `float` | `f32` |
| `int` | `c_int` |
| `unsigned int` | `c_uint` |
| `bool` | `c_int` (0/1) |
| `size_t` | `usize` |
| `dim3` | `extern struct { x: u32, y: u32, z: u32 }` |
| `cudaStream_t` | `*opaque{}` |
| `CUdeviceptr` | `u64` |

### 10.2 Error Handling Pattern

```zig
pub const CudaError = error{
    OutOfMemory,
    InvalidValue,
    NotInitialized,
    DeviceUnavailable,
    KernelLaunchFailed,
};

fn checkCuda(result: c_int) CudaError!void {
    if (result != 0) {
        return switch (result) {
            1 => error.InvalidValue,
            2 => error.OutOfMemory,
            3 => error.NotInitialized,
            100 => error.DeviceUnavailable,
            else => error.KernelLaunchFailed,
        };
    }
}
```

### 10.3 Recommended Implementation Order

1. **Phase 1: Core Infrastructure**
   - CUDA context initialization
   - Memory allocation/deallocation
   - Basic kernel launching

2. **Phase 2: Core Operations**
   - Element-wise operations (vectorized)
   - Matrix multiplication (cuBLAS + custom)
   - Activation functions

3. **Phase 3: Training Support**
   - Loss functions
   - Backward passes
   - Optimizers

4. **Phase 4: Advanced Features**
   - Convolution (cuDNN)
   - LayerNorm
   - Attention
   - Dropout

5. **Phase 5: Optimization**
   - Tensor cores
   - Multi-stream execution
   - CUDA graphs
   - Profiling integration

## References

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - Reference GEMM implementations
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Optimized attention

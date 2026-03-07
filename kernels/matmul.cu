// CUDA Kernels for Matrix Multiplication
// Optimized for NVIDIA GPUs with shared memory tiling

#include "common.h"

// Tiled matrix multiplication with shared memory
// C = A * B where A: [M x K], B: [K x N], C: [M x N]
extern "C" __global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int accumulate
) {
    const int TILE_M = 128;
    const int TILE_N = 128;
    const int TILE_K = 8;

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int warp_row = (warp_id / 2) * 64;
    int warp_col = (warp_id % 2) * 64;

    int thread_row = (lane_id / 8) * 4;
    int thread_col = (lane_id % 8) * 4;

    int global_row = blockIdx.y * TILE_M + warp_row + thread_row;
    int global_col = blockIdx.x * TILE_N + warp_col + thread_col;

    float sum[4][4] = {{0.0f}};

    for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        // Load A tile
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

        // Load B tile
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

    // Store results
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

// Simple matrix multiplication (for small sizes)
extern "C" __global__ void matmul_simple(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int accumulate
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        if (accumulate) {
            C[row * N + col] += sum;
        } else {
            C[row * N + col] = sum;
        }
    }
}

// Batched matrix multiplication
extern "C" __global__ void matmul_batched(
    const float* __restrict__ A,  // [batch_size, M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [batch_size, M, N]
    int batch_size, int M, int N, int K,
    int accumulate
) {
    int batch = blockIdx.z;
    if (batch >= batch_size) return;

    const float* A_batch = A + batch * M * K;
    float* C_batch = C + batch * M * N;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A_batch[row * K + k] * B[k * N + col];
        }
        if (accumulate) {
            C_batch[row * N + col] += sum;
        } else {
            C_batch[row * N + col] = sum;
        }
    }
}

// Matrix multiplication with B transposed: C = A * B^T
extern "C" __global__ void matmul_transpose_b(
    const float* __restrict__ A,
    const float* __restrict__ B,  // [N x K]
    float* __restrict__ C,
    int M, int N, int K,
    int accumulate
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // B is [N x K], so B^T is [K x N]
            // B^T[k, col] = B[col, k]
            sum += A[row * K + k] * B[col * K + k];
        }
        if (accumulate) {
            C[row * N + col] += sum;
        } else {
            C[row * N + col] = sum;
        }
    }
}

// Matrix multiplication with A transposed: C = A^T * B
extern "C" __global__ void matmul_transpose_a(
    const float* __restrict__ A,  // [K x M]
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int accumulate
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A is [K x M], so A^T is [M x K]
            // A^T[row, k] = A[k * M + row]
            sum += A[k * M + row] * B[k * N + col];
        }
        if (accumulate) {
            C[row * N + col] += sum;
        } else {
            C[row * N + col] = sum;
        }
    }
}

// Batched matrix multiplication with B transposed
extern "C" __global__ void matmul_batched_transpose_b(
    const float* __restrict__ A,  // [batch_size, M, K]
    const float* __restrict__ B,  // [N, K]
    float* __restrict__ C,        // [batch_size, M, N]
    int batch_size, int M, int N, int K,
    int accumulate
) {
    int batch = blockIdx.z;
    if (batch >= batch_size) return;

    const float* A_batch = A + batch * M * K;
    float* C_batch = C + batch * M * N;

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        // Compute dot product for this column across all rows
        for (int row = 0; row < M; row++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A_batch[row * K + k] * B[col * K + k];
            }
            if (accumulate) {
                C_batch[row * N + col] += sum;
            } else {
                C_batch[row * N + col] = sum;
            }
        }
    }
}

// Add bias to output: output += bias (broadcasted over batch)
extern "C" __global__ void add_bias(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch_size, int bias_size
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;

    if (col < bias_size && batch < batch_size) {
        output[batch * bias_size + col] += bias[col];
    }
}

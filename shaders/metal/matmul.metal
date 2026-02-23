// Metal shader for matrix multiplication
// Optimized for Apple Silicon GPU
#include <metal_stdlib>
using namespace metal;

// Basic matrix multiplication: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
kernel void test_write(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid == 0) out[0] = 1.23f;
}

kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Batched matrix multiplication
// Each batch: A[batch] (1xK) * B (KxN) = C[batch] (1xN)
kernel void matmul_batch(
    device const float* A [[buffer(0)]],  // [batch_size, K]
    device const float* B [[buffer(1)]],  // [K, N]
    device float* C [[buffer(2)]],        // [batch_size, N]
    constant uint& batch_size [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint col = gid.x;

    if (batch < batch_size && col < N) {
        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            sum += A[batch * K + k] * B[k * N + col];
        }
        C[batch * N + col] = sum;
    }
}

// Optimized matrix multiplication with threadgroup memory
// Uses tiling for better cache locality
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]])
{
    const uint TILE_SIZE = 16;

    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    uint row = tg_pos.y * TILE_SIZE + tid.y;
    uint col = tg_pos.x * TILE_SIZE + tid.x;

    float sum = 0.0;

    // Loop over tiles
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        uint rowA = row;
        uint colA = t * TILE_SIZE + tid.x;
        if (rowA < M && colA < K) {
            tileA[tid.y][tid.x] = A[rowA * K + colA];
        } else {
            tileA[tid.y][tid.x] = 0.0;
        }

        // Load tile from B
        uint rowB = t * TILE_SIZE + tid.y;
        uint colB = col;
        if (rowB < K && colB < N) {
            tileB[tid.y][tid.x] = B[rowB * N + colB];
        } else {
            tileB[tid.y][tid.x] = 0.0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply tiles
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y][k] * tileB[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Matrix multiplication: C = A * B^T
// A: [M x K], B: [N x K], C: [M x N]
// Required for backpropagation: dX = dY * W^T
kernel void matmul_transpose_b(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            // B is [N x K], so B^T is [K x N]
            // Standard matmul: C[row, col] = sum(A[row, k] * B_T[k, col])
            // Since B_T[k, col] = B[col, k]
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

// Matrix multiplication: C = A^T * B
// A: [K x M], B: [K x N], C: [M x N]
// Required for backpropagation: dW = X^T * dY
kernel void matmul_transpose_a(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            // A is [K x M], so A^T is [M x K]
            // Standard matmul: C[row, col] = sum(A_T[row, k] * B[k, col])
            // Since A_T[row, k] = A[k * M + row]
            sum += A[k * M + row] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Add bias to output: output = output + bias (broadcasted over batch)
kernel void add_bias(
    device float* output [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& bias_size [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < bias_size && gid.y < batch_size) {
        output[gid.y * bias_size + gid.x] += bias[gid.x];
    }
}

// Batched matrix multiplication: C = A * B^T
kernel void matmul_batch_transpose_b(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint col = gid.x;

    if (batch < batch_size && col < N) {
        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            sum += A[batch * K + k] * B[col * K + k];
        }
        C[batch * N + col] = sum;
    }
}

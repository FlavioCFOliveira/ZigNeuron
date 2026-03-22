# CUDA Backend Best Practices

Recommended patterns and practices for optimal use of the ZigNeuron CUDA backend.

## Table of Contents

- [General Principles](#general-principles)
- [Memory Management](#memory-management)
- [Performance Optimization](#performance-optimization)
- [Kernel Design](#kernel-design)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Debugging](#debugging)

## General Principles

### 1. GPU-First Design

Design your code with GPU execution as the primary path:

```zig
// Good: GPU first, CPU fallback
pub fn forwardPass(backend: Backend, input: []const f32, output: []f32) !void {
    switch (backend) {
        .cuda => |cuda_backend| {
            try cuda_backend.reluForward(input, output);
        },
        .cpu => {
            try cpu_relu_forward(input, output);
        },
    }
}

// Avoid: CPU-first thinking
pub fn forwardPass(input: []const f32, output: []f32) !void {
    // CPU implementation first, GPU as afterthought
}
```

### 2. Minimize CPU-GPU Synchronization

```zig
// Bad: Sync after every operation
for (layers) |layer| {
    try backend.matmul(...);
    try backend.synchronize();  // Unnecessary
    try backend.addBias(...);
    try backend.synchronize();  // Unnecessary
}

// Good: Sync only when needed
try backend.matmul(...);
try backend.addBias(...);
try backend.reluForward(...);
try backend.synchronize();  // Only at end
```

### 3. Batch Operations

```zig
// Bad: Process samples individually
for (samples) |sample| {
    try backend.process(sample);
}

// Good: Process as batch
try backend.processBatch(samples);
```

## Memory Management

### Buffer Lifecycle

```zig
// Pattern: RAII for device buffers
const DeviceBufferGuard = struct {
    buffer: cuda.DeviceBuffer,
    backend: *cuda.CudaBackend,

    pub fn init(backend: *cuda.CudaBackend, size: usize) !DeviceBufferGuard {
        const buffer = try backend.allocBuffer(size);
        return DeviceBufferGuard{
            .buffer = buffer,
            .backend = backend,
        };
    }

    pub fn deinit(self: *DeviceBufferGuard) void {
        self.backend.freeBuffer(self.buffer);
    }
};

// Usage
var d_input = try DeviceBufferGuard.init(backend, size);
defer d_input.deinit();
```

### Memory Pool Benefits

The memory pool automatically handles reuse:

```zig
// These allocations reuse pooled memory
var buf1 = try backend.allocBuffer(1024 * @sizeOf(f32));
var buf2 = try backend.allocBuffer(1024 * @sizeOf(f32));

backend.freeBuffer(buf1);
backend.freeBuffer(buf2);

// Next allocation likely reuses pooled memory
var buf3 = try backend.allocBuffer(1024 * @sizeOf(f32));
```

### Avoiding Memory Leaks

```zig
// Good: Explicit cleanup
try {
    var d_buffer = try backend.allocBuffer(size);
    defer backend.freeBuffer(d_buffer);

    // Use buffer...
} catch |err| {
    std.log.err("Allocation failed: {}", .{err});
    return err;
}

// Good: Error set handling
const CudaOperationError = error{
    OutOfMemory,
    InvalidBuffer,
    KernelFailed,
};
```

### Alignment for Vectorized Operations

```zig
// Good: 16-byte aligned for float4 operations
var input = try allocator.alignedAlloc(f32, 16, size);
var output = try allocator.alignedAlloc(f32, 16, size);
defer allocator.free(input);
defer allocator.free(output);

// This enables vectorized kernels automatically
try backend.reluForward(input, output);
```

## Performance Optimization

### Matrix Multiplication

```zig
// Use Tensor Cores when available
if (backend.shouldUseTensorCores(M, N, K)) {
    try backend.matMulTensorCore(A, B, C, M, N, K, false);
} else {
    try backend.matMul(A, B, C, M, N, K, false, false, false);
}

// Ensure dimensions are aligned for Tensor Cores
const M_aligned = (M + 15) / 16 * 16;
const N_aligned = (N + 15) / 16 * 16;
const K_aligned = (K + 15) / 16 * 16;
```

### Element-wise Operations

```zig
// Vectorized kernels are automatic for aligned data > 1024 elements
// Ensure 16-byte alignment
const size = 10000;
var input = try allocator.alignedAlloc(f32, 16, size);
```

### Layer Fusion

```zig
// Bad: Separate operations
try backend.matMulBatch(A, B, temp, batch_size, N, K, false);
try backend.addBias(temp, bias, batch_size, N);
try backend.reluForward(temp, output);

// Good: Fused operation (if available)
try backend.matMulBiasActivation(
    A, B, output, bias, batch_size, N, K,
    "matmul_bias_relu_fused",
);
```

### Optimizer Updates

```zig
// Use in-place operations where possible
// SGD automatically handles weight decay

// Adam: Reuse momentum buffers
var m: []f32 = try allocator.alloc(f32, weights.len);
var v: []f32 = try allocator.alloc(f32, weights.len);
defer allocator.free(m);
defer allocator.free(v);

// Initialize
@memset(m, 0);
@memset(v, 0);

// Update each iteration
try backend.adamUpdate(
    weights, gradients, m, v,
    learning_rate, beta1, beta2, epsilon, timestep,
);
```

### Batch Normalization

```zig
// Use running statistics for inference
if (is_training) {
    try backend.batchNormForwardTraining(
        input, output, gamma, beta,
        running_mean, running_var,
        batch_size, num_features,
        momentum, epsilon,
    );
} else {
    try backend.batchNormForwardInference(
        input, output, gamma, beta,
        running_mean, running_var,
        batch_size, num_features, epsilon,
    );
}
```

## Kernel Design

### Grid Configuration

```zig
// Element-wise: One thread per element
const block_size: u32 = 256;
const grid_size = (n + block_size - 1) / block_size;

// Matrix: 2D grid for 2D output
const tile_size: u32 = 32;
const grid_x = (n + tile_size - 1) / tile_size;
const grid_y = (m + tile_size - 1) / tile_size;

// Softmax: One block per sample
const block = 128;  // One warp per block
const grid = batch_size;
```

### Memory Coalescing

```zig
// CUDA kernel: Coalesced access pattern
// Each thread accesses consecutive elements
__global__ void coalesced_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        out[i] = in[i] * 2.0f;  // Coalesced
    }
}
```

### Shared Memory Usage

```zig
// Tiled matrix multiplication
const TILE_SIZE: u32 = 32;
const shared_mem_bytes = 2 * TILE_SIZE * TILE_SIZE * @sizeOf(f32);  // 8KB

// Check against device limit
if (shared_mem_bytes > backend.device_props.max_shared_memory_per_block) {
    // Reduce tile size
}
```

## Error Handling

### Graceful Degradation

```zig
pub fn tryGpuOperation() !void {
    // Check CUDA availability
    if (!cuda.CudaBackend.isAvailable()) {
        std.log.info("CUDA not available, using CPU");
        return try cpuOperation();
    }

    // Try GPU operation
    var backend = cuda.CudaBackend.init(allocator) catch |err| {
        std.log.warn("Failed to initialize CUDA: {}, falling back to CPU", .{err});
        return try cpuOperation();
    };
    defer backend.deinit();

    // Try operation
    backend.gpuOperation() catch |err| {
        std.log.warn("GPU operation failed: {}, using CPU", .{err});
        return try cpuOperation();
    };
}
```

### Validation Before Operations

```zig
fn validateInputs(input: []const f32, output: []f32) !void {
    // Check sizes match
    if (input.len != output.len) {
        return error.SizeMismatch;
    }

    // Check for NaN/Inf
    for (input) |val| {
        if (std.math.isNan(val)) return error.InputContainsNaN;
        if (std.math.isInf(val)) return error.InputContainsInf;
    }

    // Check buffer alignment for vectorized ops
    const input_aligned = @intFromPtr(input.ptr) % 16 == 0;
    const output_aligned = @intFromPtr(output.ptr) % 16 == 0;
    if (!input_aligned or !output_aligned) {
        std.log.debug("Buffers not 16-byte aligned, using scalar kernels");
    }
}
```

### Error Context

```zig
// Provide context in error messages
fn matMulWithContext(
    backend: *cuda.CudaBackend,
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
) !void {
    backend.matMul(a, b, c, m, n, k, false, false, false) catch |err| {
        std.log.err("matMul failed: m={d}, n={d}, k={d}, error={s}", .{
            m, n, k, @errorName(err),
        });
        return err;
    };
}
```

## Testing

### Numerical Accuracy Tests

```zig
test "GPU matches CPU numerical accuracy" {
    if (!cuda.CudaBackend.isAvailable()) return error.SkipZigTest;

    var backend = try cuda.CudaBackend.init(std.testing.allocator);
    defer backend.deinit();

    // Generate random input
    var input: [1000]f32 = undefined;
    var rng = std.rand.DefaultPrng.init(42);
    for (&input) |*val| val.* = rng.random().float(f32);

    // CPU reference
    var cpu_output: [1000]f32 = undefined;
    for (input, 0..) |val, i| {
        cpu_output[i] = @max(0.0, val);  // ReLU
    }

    // GPU result
    var gpu_output: [1000]f32 = undefined;
    try backend.reluForward(&input, &gpu_output);

    // Compare with tolerance
    const TOLERANCE: f32 = 1e-5;
    for (cpu_output, gpu_output) |cpu, gpu| {
        const diff = @abs(cpu - gpu);
        try std.testing.expect(diff < TOLERANCE);
    }
}
```

### Memory Leak Tests

```zig
test "No memory leaks" {
    if (!cuda.CudaBackend.isAvailable()) return error.SkipZigTest;

    var gpa = std.heap.GeneralPurposeAllocator(.{ .enable_memory_limit = true }){};
    defer {
        const status = gpa.deinit();
        try std.testing.expect(status == .ok);
    }

    var backend = try cuda.CudaBackend.init(gpa.allocator());
    defer backend.deinit();

    // Perform many operations
    for (0..1000) |_| {
        var buffer = try backend.allocBuffer(1024 * @sizeOf(f32));
        backend.freeBuffer(buffer);
    }

    // Check for leaks
    try backend.synchronize();
}
```

### Stress Tests

```zig
test "Large tensor handling" {
    if (!cuda.CudaBackend.isAvailable()) return error.SkipZigTest;

    var backend = try cuda.CudaBackend.init(std.testing.allocator);
    defer backend.deinit();

    const large_size: usize = 10_000_000;  // 10M elements
    var input = try allocator.alloc(f32, large_size);
    defer allocator.free(input);
    var output = try allocator.alloc(f32, large_size);
    defer allocator.free(output);

    // Should not crash or OOM
    try backend.reluForward(input, output);
    try backend.synchronize();
}
```

## Debugging

### Debug Builds

```zig
// In debug mode, add validation
const is_debug = builtin.mode == .Debug;

if (is_debug) {
    // Validate inputs
    for (input) |val| {
        if (std.math.isNan(val)) @panic("NaN in input");
    }

    // Check outputs
    for (output) |val| {
        if (std.math.isNan(val)) {
            std.log.err("NaN detected in output at index {d}", .{i});
        }
    }
}
```

### Logging Levels

```zig
const std = @import("std");

// Use appropriate log levels
std.log.err("Critical error: {}", .{err});    // Always shown
std.log.warn("Warning: {}", .{warning});       // Important issues
std.log.info("Info: {}", .{info});              // General info
std.log.debug("Debug: {}", .{debug_info});     // Detailed debugging

// Set log level at runtime or compile time
// zig build -Dlog-level=debug
```

### Profiling

```zig
// Simple timing wrapper
fn timeOperation(
    comptime name: []const u8,
    backend: *cuda.CudaBackend,
    operation: anytype,
    args: anytype,
) !void {
    var timer = try std.time.Timer.start();

    try @call(.auto, operation, args);
    try backend.synchronize();

    const elapsed = timer.read();
    std.log.info("{s}: {d:.3} ms", .{ name, @as(f64, elapsed) / 1e6 });
}

// Usage
try timeOperation("matMul", backend, cuda.CudaBackend.matMul, .{
    backend, A, B, C, M, N, K, false, false, false,
});
```

## See Also

- [User Guide](./user-guide.md) - Usage tutorials
- [API Reference](./api.md) - Complete API documentation
- [Developer Guide](./developer-guide.md) - Implementation details
- [Troubleshooting](./troubleshooting.md) - Common issues

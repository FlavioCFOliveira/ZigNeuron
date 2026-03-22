# CUDA Backend User Guide

A comprehensive guide for using the ZigNeuron CUDA backend.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Basic Operations](#basic-operations)
- [Neural Network Layers](#neural-network-layers)
- [Training Workflows](#training-workflows)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Introduction

The CUDA backend provides GPU acceleration for neural network operations on NVIDIA hardware. It supports:

- **Dynamic driver loading** - Works without static CUDA dependencies
- **Automatic kernel loading** - PTX and NVRTC runtime compilation
- **Memory pooling** - Efficient buffer reuse
- **Async execution** - Non-blocking operations with streams
- **Tensor Cores** - Accelerated FP16 operations (sm_70+)

## Prerequisites

### Hardware Requirements

| GPU Generation | Compute Capability | Status |
|----------------|---------------------|--------|
| Maxwell | 5.x | Limited support |
| Pascal | 6.x | Supported |
| Volta | 7.0 | Supported |
| Turing | 7.5 | Supported |
| Ampere | 8.x | **Recommended** |
| Ada Lovelace | 8.9 | **Recommended** |
| Hopper | 9.0 | Experimental |

**Minimum Requirements:**
- NVIDIA GPU with Compute Capability 6.0+
- CUDA Driver 11.0+
- 4GB+ VRAM (8GB+ recommended for training)

### Software Requirements

- Linux or Windows (macOS not supported)
- NVIDIA drivers installed
- Zig compiler

### Verify CUDA Installation

```bash
# Check if NVIDIA driver is installed
nvidia-smi

# Expected output shows GPU info, driver version, CUDA version
```

## Quick Start

### 1. Check CUDA Availability

```zig
const std = @import("std");
const cuda = @import("ZigNeuron").cuda;

pub fn main() !void {
    // Check if CUDA is available
    if (!cuda.CudaBackend.isAvailable()) {
        std.log.info("CUDA not available, using CPU fallback");
        return;
    }

    // Get device count
    const device_count = cuda.CudaBackend.getDeviceCount();
    std.log.info("Found {d} CUDA device(s)", .{device_count});
}
```

### 2. Initialize Backend

```zig
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
const allocator = gpa.allocator();

// Initialize CUDA backend
var backend = try cuda.CudaBackend.init(allocator);
defer backend.deinit();

std.log.info("CUDA backend initialized successfully");
```

### 3. Perform Operations

```zig
// Allocate host memory
const size: usize = 1024;
var input = try allocator.alloc(f32, size);
defer allocator.free(input);
var output = try allocator.alloc(f32, size);
defer allocator.free(output);

// Initialize input data
for (input, 0..) |*val, i| {
    val.* = @floatFromInt(i);
}

// Allocate device buffer
var d_buffer = try backend.allocBuffer(size * @sizeOf(f32));
defer backend.freeBuffer(d_buffer);

// Upload data
try backend.upload(d_buffer.ptr, input);

// Perform ReLU activation
try backend.reluForward(input, output);

std.log.info("Operation completed successfully");
```

## Basic Operations

### Memory Management

```zig
// Allocate device buffer
var d_buffer = try backend.allocBuffer(1024 * @sizeOf(f32));
defer backend.freeBuffer(d_buffer);

// Upload from host to device
try backend.upload(d_buffer.ptr, host_data);

// Download from device to host
try backend.download(host_result, d_buffer.ptr);

// Async operations (non-blocking)
try backend.uploadAsync(d_buffer.ptr, host_data);
try backend.downloadAsync(host_result, d_buffer.ptr);

// Synchronize when needed
try backend.synchronize();
```

### Matrix Multiplication

```zig
// Standard matrix multiplication
// C = A * B where A: MxK, B: KxN, C: MxN
const M: usize = 128;
const N: usize = 64;
const K: usize = 256;

var A = try allocator.alloc(f32, M * K);
var B = try allocator.alloc(f32, K * N);
var C = try allocator.alloc(f32, M * N);
defer allocator.free(A);
defer allocator.free(B);
defer allocator.free(C);

// Initialize A and B...

// Perform matrix multiplication
try backend.matMul(
    A, B, C,
    M, N, K,
    false,  // transpose_a
    false,  // transpose_b
    false,  // accumulate (if true: C += A*B)
);
```

### Element-wise Operations

```zig
var a: [100]f32 = undefined;
var b: [100]f32 = undefined;
var c: [100]f32 = undefined;

// Element-wise addition
try backend.elementWiseOp(.add, &a, &b, &c);

// Element-wise subtraction
try backend.elementWiseOp(.sub, &a, &b, &c);

// Element-wise multiplication
try backend.elementWiseOp(.mul, &a, &b, &c);

// Element-wise division
try backend.elementWiseOp(.div, &a, &b, &c);

// Scalar multiplication
try backend.scale(&a, 2.5, &c);
```

### Activation Functions

```zig
var input: [1000]f32 = undefined;
var output: [1000]f32 = undefined;

// ReLU
try backend.reluForward(&input, &output);

// Sigmoid
try backend.sigmoidForward(&input, &output);

// Tanh
try backend.tanhForward(&input, &output);

// Softmax (per-row)
const batch_size: usize = 10;
const features: usize = 100;
try backend.softmaxForward(&input, &output, batch_size, features);
```

## Neural Network Layers

### Dense Layer Forward Pass

```zig
/// Dense layer: output = activation(input * weights + bias)
fn denseLayerForward(
    backend: *cuda.CudaBackend,
    input: []const f32,
    weights: []const f32,
    bias: []const f32,
    output: []f32,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
) !void {
    // Allocate temporary buffer for linear output
    var linear_output = try allocator.alloc(f32, batch_size * output_size);
    defer allocator.free(linear_output);

    // Matrix multiplication: input * weights
    // input: [batch_size, input_size]
    // weights: [input_size, output_size]
    // output: [batch_size, output_size]
    try backend.matMulBatch(
        input,
        weights,
        linear_output,
        batch_size,
        output_size,
        input_size,
        false,  // accumulate
    );

    // Add bias
    try backend.addBias(linear_output, bias, batch_size, output_size);

    // Apply activation (e.g., ReLU)
    try backend.reluForward(linear_output, output);
}
```

### Convolutional Layer

```zig
/// Conv2D forward pass
fn conv2dForward(
    backend: *cuda.CudaBackend,
    input: []const f32,     // [batch, in_channels, height, width]
    weights: []const f32,   // [out_channels, in_channels, kernel_h, kernel_w]
    bias: []const f32,      // [out_channels]
    output: []f32,          // [batch, out_channels, out_h, out_w]
    params: Conv2DParams,
) !void {
    // Convolution is performed by specialized CUDA kernels
    // See developer guide for kernel implementation details
    try backend.conv2dForward(
        input,
        weights,
        bias,
        output,
        params.batch_size,
        params.in_channels,
        params.out_channels,
        params.kernel_h,
        params.kernel_w,
        params.input_h,
        params.input_w,
        params.output_h,
        params.output_w,
        params.stride_h,
        params.stride_w,
        params.padding_h,
        params.padding_w,
    );
}
```

### Recurrent Layers (LSTM)

```zig
/// LSTM cell forward pass
fn lstmCellForward(
    backend: *cuda.CudaBackend,
    input: []const f32,     // [batch, input_size]
    hidden: []f32,          // [batch, hidden_size]
    cell: []f32,            // [batch, hidden_size]
    weights: LSTMWeights,    // Weight matrices
    params: LSTMParams,
) !void {
    // LSTM gates: forget, input, cell, output
    // Uses specialized LSTM kernels
    try backend.lstmForward(
        input,
        hidden,
        cell,
        weights.input_weights,
        weights.hidden_weights,
        weights.bias,
        params.batch_size,
        params.input_size,
        params.hidden_size,
    );
}
```

## Training Workflows

### Forward Pass

```zig
fn forwardPass(
    backend: *cuda.CudaBackend,
    input: []const f32,
    output: []f32,
    layers: []const Layer,
) !void {
    var current = input;
    for (layers) |layer| {
        switch (layer.type) {
            .dense => {
                try backend.matMulBatch(
                    current,
                    layer.weights,
                    layer.output_buffer,
                    layer.batch_size,
                    layer.output_size,
                    layer.input_size,
                    false,
                );
                try backend.addBias(
                    layer.output_buffer,
                    layer.bias,
                    layer.batch_size,
                    layer.output_size,
                );
                try applyActivation(backend, layer);
                current = layer.output_buffer;
            },
            // ... other layer types
        }
    }
    @memcpy(output, current);
}
```

### Backward Pass

```zig
fn backwardPass(
    backend: *cuda.CudaBackend,
    output: []const f32,
    target: []const f32,
    layers: []Layer,
) !void {
    // Compute loss gradient
    var grad_output = try allocator.alloc(f32, output.len);
    defer allocator.free(grad_output);
    try backend.mseBackward(output, target, grad_output);

    // Backpropagate through layers (in reverse)
    var i: usize = layers.len;
    while (i > 0) {
        i -= 1;
        const layer = &layers[i];

        // Compute gradients for this layer
        try computeLayerGradients(backend, layer, grad_output);

        // Update weights
        try backend.sgdUpdate(
            layer.weights,
            layer.weight_gradients,
            layer.learning_rate,
            layer.weight_decay,
        );

        // Propagate gradient to previous layer
        if (i > 0) {
            try propagateGradient(backend, layer, grad_output);
        }
    }
}
```

### Optimizer Updates

```zig
/// SGD optimizer
fn sgdUpdate(
    backend: *cuda.CudaBackend,
    weights: []f32,
    gradients: []const f32,
    learning_rate: f32,
    weight_decay: f32,
) !void {
    try backend.sgdUpdate(
        weights,
        gradients,
        learning_rate,
        weight_decay,
    );
}

/// Adam optimizer
fn adamUpdate(
    backend: *cuda.CudaBackend,
    weights: []f32,
    gradients: []const f32,
    m: []f32,
    v: []f32,
    t: u32,
    config: AdamConfig,
) !void {
    try backend.adamUpdate(
        weights,
        gradients,
        m,
        v,
        config.learning_rate,
        config.beta1,
        config.beta2,
        config.epsilon,
        t,
    );
}
```

### Complete Training Loop

```zig
fn train(
    backend: *cuda.CudaBackend,
    dataset: Dataset,
    model: *Model,
    epochs: usize,
    batch_size: usize,
) !void {
    var rng = std.rand.DefaultPrng.init(42);
    var adam_step: u32 = 0;

    for (0..epochs) |epoch| {
        var total_loss: f32 = 0;
        var batch_count: usize = 0;

        // Shuffle data
        var indices = try allocator.alloc(usize, dataset.size);
        defer allocator.free(indices);
        for (indices, 0..) |*idx, i| idx.* = i;
        std.rand.shuffle(usize, indices, rng.random());

        // Process batches
        var i: usize = 0;
        while (i + batch_size <= dataset.size) : (i += batch_size) {
            // Get batch
            const batch_input = getBatchInput(dataset, indices, i, batch_size);
            const batch_target = getBatchTarget(dataset, indices, i, batch_size);

            // Forward pass
            try forwardPass(backend, batch_input, model.output_buffer, model.layers);

            // Compute loss
            const loss = try computeLoss(backend, model.output_buffer, batch_target);
            total_loss += loss;

            // Backward pass
            try backwardPass(backend, model.output_buffer, batch_target, model.layers);

            // Update weights (Adam)
            adam_step += 1;
            for (model.layers) |*layer| {
                if (layer.hasWeights()) {
                    try backend.adamUpdate(
                        layer.weights,
                        layer.weight_gradients,
                        layer.m_buffer,
                        layer.v_buffer,
                        0.001,  // learning_rate
                        0.9,    // beta1
                        0.999,  // beta2
                        1e-8,   // epsilon
                        adam_step,
                    );
                }
            }

            batch_count += 1;
        }

        const avg_loss = total_loss / @as(f32, @floatFromInt(batch_count));
        std.log.info("Epoch {d}: avg_loss = {d:.6}", .{ epoch + 1, avg_loss });
    }
}
```

## Performance Optimization

### Memory Transfer Optimization

```zig
// BAD: Frequent small transfers
for (0..1000) |_| {
    try backend.upload(d_small.ptr, small_data);
    try backend.download(result, d_small.ptr);
}

// GOOD: Batch transfers
// Allocate larger buffer
var d_large = try backend.allocBuffer(1000 * @sizeOf(f32));
defer backend.freeBuffer(d_large);

// Single upload
try backend.upload(d_large.ptr, large_data);

// Process on GPU
for (0..1000) |i| {
    // Use pointer arithmetic to access elements
    const offset = i * @sizeOf(f32);
    // ... process ...
}

// Single download
try backend.download(result, d_large.ptr);
```

### Tensor Core Usage

```zig
// Tensor Cores are automatically used for compatible operations
// when available (sm_70+) and dimensions are multiples of 16

const M: usize = 256;  // Must be multiple of 16
const N: usize = 256;  // Must be multiple of 16
const K: usize = 256;  // Must be multiple of 16

// This will use Tensor Cores if available
try backend.matMul(A, B, C, M, N, K, false, false, false);

// Check if Tensor Cores were used
if (backend.context.device_props.hasTensorCores()) {
    std.log.info("Tensor Cores available and used");
}
```

### Async Execution

```zig
// Upload data asynchronously
var h_input = try allocator.alloc(f32, size);
var h_output = try allocator.alloc(f32, size);
var d_buffer = try backend.allocBuffer(size * @sizeOf(f32));

try backend.uploadAsync(d_buffer.ptr, h_input);

// Launch kernel while data is still uploading
// (if using separate streams, currently uses default stream)
try backend.reluForward(h_input, h_output);

// Synchronize when results needed
try backend.synchronize();
```

### Vectorized Operations

```zig
// The backend automatically uses vectorized kernels for
// large aligned tensors (>= 1024 elements, 16-byte aligned)

// Ensure alignment for best performance
var input = try allocator.alignedAlloc(f32, 16, size);
var output = try allocator.alignedAlloc(f32, 16, size);
defer allocator.free(input);
defer allocator.free(output);

// This will use vec4 kernel automatically
try backend.reluForward(input, output);
```

## Troubleshooting

### "CUDA driver not found"

**Cause:** NVIDIA drivers not installed or not in library path.

**Solution:**
```bash
# Ubuntu
sudo apt update
sudo apt install nvidia-driver-535

# Verify installation
nvidia-smi
```

### "Out of memory"

**Cause:** GPU memory exhausted.

**Solutions:**
1. Reduce batch size
2. Enable memory pooling (automatic)
3. Use gradient checkpointing
4. Process in chunks

```zig
// Process large arrays in chunks
const chunk_size: usize = 10000;
var offset: usize = 0;
while (offset < total_size) : (offset += chunk_size) {
    const end = @min(offset + chunk_size, total_size);
    const chunk = data[offset..end];
    try backend.processChunk(chunk);
}
```

### "Invalid PTX"

**Cause:** PTX code incompatible with GPU architecture.

**Solutions:**
1. Ensure PTX version matches GPU capability
2. Use NVRTC for runtime compilation
3. Update NVIDIA drivers

```zig
// Fallback to NVRTC if embedded PTX fails
try context.compileAndLoadKernel("my_kernel", cuda_source);
```

### Performance Issues

**Symptoms:** GPU utilization low (< 50%).

**Diagnostic:**
```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Profile with Nsight
nsys profile -o report ./your_app
```

**Solutions:**
1. Increase batch size
2. Minimize CPU-GPU transfers
3. Use async operations
4. Check kernel launch configuration

## Best Practices

### 1. Always Check Availability

```zig
if (!cuda.CudaBackend.isAvailable()) {
    // Use CPU fallback
    return;
}
```

### 2. Use Proper Error Handling

```zig
backend.reluForward(input, output) catch |err| {
    std.log.err("ReLU failed: {s}", .{@errorName(err)});
    return err;
};
```

### 3. Manage Memory Carefully

```zig
// Use defer to ensure cleanup
var buffer = try backend.allocBuffer(size);
defer backend.freeBuffer(buffer);

// Or use RAII pattern
const DeviceBuffer = struct {
    buffer: cuda.DeviceBuffer,
    backend: *cuda.CudaBackend,

    pub fn deinit(self: *DeviceBuffer) void {
        self.backend.freeBuffer(self.buffer);
    }
};
```

### 4. Batch Operations

```zig
// Prefer batched operations over loops
try backend.matMulBatch(A, B, C, batch_size, N, K, false);

// Instead of:
for (0..batch_size) |i| {
    try backend.matMul(A[i], B, C[i], M, N, K, false, false, false);
}
```

### 5. Profile Before Optimizing

```zig
// Use CUDA events for timing
var start: cuda.CUevent = undefined;
var stop: cuda.CUevent = undefined;
backend.driver.eventCreate(&start, .DEFAULT);
backend.driver.eventCreate(&stop, .DEFAULT);

backend.driver.eventRecord(start, null);
try backend.matMul(A, B, C, M, N, K, false, false, false);
backend.driver.eventRecord(stop, null);
backend.driver.eventSynchronize(stop);

var ms: f32 = undefined;
backend.driver.eventElapsedTime(&ms, start, stop);
std.log.info("MatMul took {d:.3} ms", .{ms});
```

### 6. Use Appropriate Data Types

```zig
// FP32 is default and most compatible
// FP16 can be used with Tensor Cores for 2x throughput
// (requires explicit conversion in current implementation)
```

### 7. Document Device Requirements

```zig
/// This function requires:
/// - CUDA-capable GPU (sm_60+)
/// - 4GB+ VRAM
/// - CUDA Driver 11.0+
pub fn trainLargeModel(...) !void {
    // ...
}
```

## See Also

- [API Reference](./api.md) - Complete API documentation
- [Developer Guide](./developer-guide.md) - Implementation details
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions

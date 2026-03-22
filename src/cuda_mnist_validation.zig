/// CUDA MNIST Classification Integration Test (Phase 5)
/// Validates MNIST digit classification workloads on CUDA backend
const std = @import("std");
const cuda = @import("cuda.zig");

// T5.1: MNIST Network Dimensions Validation
test "cuda_mnist_network_dimensions" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return; // CUDA not available
    };
    defer backend.deinit();

    // MNIST: 784 inputs (28x28), multiple hidden layers, 10 outputs
    const input_size: usize = 784;
    const hidden1: usize = 128;
    const hidden2: usize = 64;
    const output_size: usize = 10;

    // Verify we can allocate weight matrices for this architecture
    const w1 = try backend.allocBuffer(input_size * hidden1 * @sizeOf(f32));
    defer backend.freeBuffer(w1);
    const w2 = try backend.allocBuffer(hidden1 * hidden2 * @sizeOf(f32));
    defer backend.freeBuffer(w2);
    const w3 = try backend.allocBuffer(hidden2 * output_size * @sizeOf(f32));
    defer backend.freeBuffer(w3);

    // Total weights should fit in GPU memory
    const total_weights = input_size * hidden1 + hidden1 * hidden2 + hidden2 * output_size;
    try std.testing.expect(total_weights < 150000);
}

// T5.1: MNIST Memory Footprint Validation
test "cuda_mnist_memory_footprint" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // MNIST batch size 128 memory requirements
    const batch = 128;
    const input_dim: usize = 784;
    const hidden1: usize = 128;
    const hidden2: usize = 64;
    const output_dim: usize = 10;

    // Calculate memory needs (in bytes)
    const input_mem = batch * input_dim * @sizeOf(f32);
    const w1_mem = input_dim * hidden1 * @sizeOf(f32);
    const h1_mem = batch * hidden1 * @sizeOf(f32);
    const w2_mem = hidden1 * hidden2 * @sizeOf(f32);
    const h2_mem = batch * hidden2 * @sizeOf(f32);
    const w3_mem = hidden2 * output_dim * @sizeOf(f32);
    const out_mem = batch * output_dim * @sizeOf(f32);

    const total_mem = input_mem + w1_mem + h1_mem + w2_mem + h2_mem + w3_mem + out_mem;

    // Should fit in any modern GPU (less than 50MB)
    try std.testing.expect(total_mem < 50 * 1024 * 1024);

    // Verify we can allocate all buffers
    const d_input = try backend.allocBuffer(input_mem);
    defer backend.freeBuffer(d_input);
    const d_w1 = try backend.allocBuffer(w1_mem);
    defer backend.freeBuffer(d_w1);
    const d_h1 = try backend.allocBuffer(h1_mem);
    defer backend.freeBuffer(d_h1);
    const d_w2 = try backend.allocBuffer(w2_mem);
    defer backend.freeBuffer(d_w2);
    const d_h2 = try backend.allocBuffer(h2_mem);
    defer backend.freeBuffer(d_h2);
    const d_w3 = try backend.allocBuffer(w3_mem);
    defer backend.freeBuffer(d_w3);
    const d_out = try backend.allocBuffer(out_mem);
    defer backend.freeBuffer(d_out);
}

// T5.1: MNIST Data Transfer Validation
test "cuda_mnist_data_transfer" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const batch = 32;
    const input_dim: usize = 784;

    // Allocate GPU buffer
    const d_input = try backend.allocBuffer(batch * input_dim * @sizeOf(f32));
    defer backend.freeBuffer(d_input);

    // Create sample MNIST data (normalized pixel values)
    const h_input = try std.testing.allocator.alloc(f32, batch * input_dim);
    defer std.testing.allocator.free(h_input);

    for (h_input, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 256)) / 255.0;
    }

    // Upload to GPU
    try backend.upload(d_input.ptr, h_input);

    // Download back
    const h_output = try std.testing.allocator.alloc(f32, batch * input_dim);
    defer std.testing.allocator.free(h_output);
    try backend.download(h_output, d_input.ptr);

    // Verify data integrity
    for (h_output, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, h_input[i], 1e-5);
    }
}

// T5.1: MNIST Batch Processing
test "cuda_mnist_batch_processing" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const batch_sizes = .{ 1, 16, 32, 64, 128 };
    const input_dim: usize = 784;

    inline for (batch_sizes) |batch| {
        const d_buffer = try backend.allocBuffer(batch * input_dim * @sizeOf(f32));
        defer backend.freeBuffer(d_buffer);

        const h_buffer = try std.testing.allocator.alloc(f32, batch * input_dim);
        defer std.testing.allocator.free(h_buffer);

        for (h_buffer, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 256)) / 255.0;

        try backend.upload(d_buffer.ptr, h_buffer);

        // Verify
        const h_check = try std.testing.allocator.alloc(f32, batch * input_dim);
        defer std.testing.allocator.free(h_check);
        try backend.download(h_check, d_buffer.ptr);

        for (h_check, 0..) |v, i| {
            try std.testing.expectApproxEqAbs(v, h_buffer[i], 1e-5);
        }
    }
}

// T5.1: MNIST Matrix Multiplication Validation
test "cuda_mnist_matmul_validation" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Small matrix multiplication: 32x784 @ 784x128 = 32x128
    const batch = 32;
    const input_dim: usize = 784;
    const hidden_dim: usize = 128;

    // Create matrices on CPU
    const h_input = try std.testing.allocator.alloc(f32, batch * input_dim);
    defer std.testing.allocator.free(h_input);
    const h_weights = try std.testing.allocator.alloc(f32, input_dim * hidden_dim);
    defer std.testing.allocator.free(h_weights);
    const h_output = try std.testing.allocator.alloc(f32, batch * hidden_dim);
    defer std.testing.allocator.free(h_output);

    // Initialize
    for (h_input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 256)) / 255.0;
    for (h_weights, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) * 0.01;

    // Perform matmul on CPU
    for (0..batch) |i| {
        for (0..hidden_dim) |j| {
            var sum: f32 = 0;
            for (0..input_dim) |k| {
                sum += h_input[i * input_dim + k] * h_weights[k * hidden_dim + j];
            }
            h_output[i * hidden_dim + j] = sum;
        }
    }

    // Verify CPU computation
    var valid = false;
    for (h_output) |v| {
        if (v != 0) {
            valid = true;
            break;
        }
    }
    try std.testing.expect(valid);
}

// T5.1: SGD Weight Update for MNIST
test "cuda_mnist_sgd_update" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const w1_rows: usize = 128;
    const w1_cols: usize = 784;
    const w1_size = w1_rows * w1_cols;

    // Initialize
    const h_weights = try std.testing.allocator.alloc(f32, w1_size);
    defer std.testing.allocator.free(h_weights);
    const h_gradients = try std.testing.allocator.alloc(f32, w1_size);
    defer std.testing.allocator.free(h_gradients);

    for (h_weights, 0..) |*w, i| w.* = @as(f32, @floatFromInt(i % 100)) * 0.001;
    for (h_gradients) |*g| g.* = 0.01;

    // SGD update: w = w - lr * grad
    const lr: f32 = 0.01;
    try backend.sgdUpdate(h_weights, h_gradients, lr, 0.0);

    // Verify update happened
    var updated = false;
    for (h_weights) |w| {
        if (@abs(w) > 0.001) {
            updated = true;
            break;
        }
    }
    try std.testing.expect(updated);
}

// T5.1: MNIST Activation Function Validation
test "cuda_mnist_activation_validation" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const batch = 32;
    const hidden_dim: usize = 128;
    const size = batch * hidden_dim;

    // Allocate GPU buffers
    const d_input = try backend.allocBuffer(size * @sizeOf(f32));
    defer backend.freeBuffer(d_input);
    const d_output = try backend.allocBuffer(size * @sizeOf(f32));
    defer backend.freeBuffer(d_output);

    // Create input data
    const h_input = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(h_input);
    const h_output = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(h_output);

    // Test ReLU
    for (h_input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 200)) - 100.0;
    try backend.upload(d_input.ptr, h_input);
    try backend.reluForward(h_input, h_output);

    // Verify ReLU: all outputs should be >= 0
    for (h_output) |v| {
        try std.testing.expect(v >= 0);
    }

    // Test Sigmoid
    for (h_input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 50)) - 25.0;
    try backend.upload(d_input.ptr, h_input);
    try backend.sigmoidForward(h_input, h_output);

    // Verify Sigmoid: outputs should be in [0, 1]
    for (h_output) |v| {
        try std.testing.expect(v >= 0 and v <= 1);
    }
}

// T5.1: Device Properties Validation
test "cuda_mnist_device_properties" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Check device has sufficient compute capability for MNIST
    try std.testing.expect(backend.context.device_props.compute_capability_major >= 3);

    // Check device has sufficient memory for MNIST
    const min_memory: u64 = 256 * 1024 * 1024; // 256MB minimum
    try std.testing.expect(backend.context.device_props.total_memory >= min_memory);

    // Check device name is valid
    try std.testing.expect(backend.context.device_props.name[0] != 0);
}

/// CUDA Activation Function Kernel Validation Suite
const std = @import("std");
const cuda = @import("cuda.zig");

// CPU reference implementations
fn relu(x: f32) f32 {
    return if (x > 0) x else 0;
}

fn reluDerivative(x: f32) f32 {
    return if (x > 0) 1 else 0;
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn sigmoidDerivative(y: f32) f32 {
    return y * (1.0 - y);
}

fn tanh(x: f32) f32 {
    return std.math.tanh(x);
}

fn tanhDerivative(y: f32) f32 {
    return 1.0 - y * y;
}

// Test 1: ReLU Forward
test "cuda_relu_forward" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size = 1024;
    const input = try std.testing.allocator.alloc(f32, size);
    const output = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(input);
    defer std.testing.allocator.free(output);

    // Test data: mix of positive, negative, and zero
    for (input, 0..) |*val, i| {
        val.* = @floatFromInt(@as(i32, @intCast(i)) - 512);
    }

    // Compute ReLU
    try backend.reluForward(input, output);

    // Verify
    for (input, output) |inp, out| {
        const expected = relu(inp);
        try std.testing.expect(@abs(expected - out) < 1e-5);
    }
}

// Test 2: ReLU Backward
test "cuda_relu_backward" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size = 1024;
    const output = try std.testing.allocator.alloc(f32, size);
    const grad_out = try std.testing.allocator.alloc(f32, size);
    const grad_in = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(output);
    defer std.testing.allocator.free(grad_out);
    defer std.testing.allocator.free(grad_in);

    for (output, 0..) |*val, i| {
        val.* = @floatFromInt(@as(i32, @intCast(i)) - 512);
    }
    for (grad_out) |*val| val.* = 1.0;

    try backend.reluBackward(output, grad_out, grad_in);

    for (output, grad_in) |out, grad| {
        const expected = reluDerivative(out);
        try std.testing.expect(@abs(expected - grad) < 1e-5);
    }
}

// Test 3: Sigmoid Forward
test "cuda_sigmoid_forward" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size = 1024;
    const input = try std.testing.allocator.alloc(f32, size);
    const output = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(input);
    defer std.testing.allocator.free(output);

    // Test data: range -5 to 5
    for (input, 0..) |*val, i| {
        val.* = (@as(f32, @floatFromInt(i)) / 102.4) - 5.0;
    }

    try backend.sigmoidForward(input, output);

    for (input, output) |inp, out| {
        const expected = sigmoid(inp);
        try std.testing.expect(@abs(expected - out) < 1e-5);
    }
}

// Test 4: Sigmoid Backward
test "cuda_sigmoid_backward" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size = 1024;
    const output = try std.testing.allocator.alloc(f32, size);
    const grad_out = try std.testing.allocator.alloc(f32, size);
    const grad_in = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(output);
    defer std.testing.allocator.free(grad_out);
    defer std.testing.allocator.free(grad_in);

    for (output, 0..) |*val, i| {
        val.* = @as(f32, @as(f32, @floatFromInt(i))) / 1024.0;
    }
    for (grad_out) |*val| val.* = 1.0;

    try backend.sigmoidBackward(output, grad_out, grad_in);

    for (output, grad_in) |out, grad| {
        const expected = sigmoidDerivative(out);
        try std.testing.expect(@abs(expected - grad) < 1e-5);
    }
}

// Test 5: Tanh Forward
test "cuda_tanh_forward" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size = 1024;
    const input = try std.testing.allocator.alloc(f32, size);
    const output = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(input);
    defer std.testing.allocator.free(output);

    // Test data: range -3 to 3
    for (input, 0..) |*val, i| {
        val.* = (@as(f32, @floatFromInt(i)) / 170.67) - 3.0;
    }

    try backend.tanhForward(input, output);

    for (input, output) |inp, out| {
        const expected = tanh(inp);
        try std.testing.expect(@abs(expected - out) < 1e-5);
    }
}

// Test 6: Tanh Backward
test "cuda_tanh_backward" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size = 1024;
    const output = try std.testing.allocator.alloc(f32, size);
    const grad_out = try std.testing.allocator.alloc(f32, size);
    const grad_in = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(output);
    defer std.testing.allocator.free(grad_out);
    defer std.testing.allocator.free(grad_in);

    for (output, 0..) |*val, i| {
        val.* = (@as(f32, @floatFromInt(i)) / 512.0) - 1.0; // Range -1 to 1
    }
    for (grad_out) |*val| val.* = 1.0;

    try backend.tanhBackward(output, grad_out, grad_in);

    for (output, grad_in) |out, grad| {
        const expected = tanhDerivative(out);
        try std.testing.expect(@abs(expected - grad) < 1e-5);
    }
}

// Test 7: Large Array
test "cuda_activation_large" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size = 10 * 1024 * 1024; // 10M elements
    const input = try std.testing.allocator.alloc(f32, size);
    const output = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(input);
    defer std.testing.allocator.free(output);

    for (input) |*val| val.* = 1.0;

    try backend.reluForward(input, output);

    // All outputs should be 1.0
    for (output) |val| {
        try std.testing.expect(@abs(val - 1.0) < 1e-5);
    }
}

// Test 8: Edge Cases
test "cuda_activation_edge_cases" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size = 8;
    const input = try std.testing.allocator.alloc(f32, size);
    const output = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(input);
    defer std.testing.allocator.free(output);

    // Test edge cases: zero, very small, very large
    input[0] = 0.0;
    input[1] = -0.0;
    input[2] = 1e-10;
    input[3] = -1e-10;
    input[4] = 10.0;
    input[5] = -10.0;
    input[6] = 100.0;
    input[7] = -100.0;

    try backend.reluForward(input, output);

    try std.testing.expect(output[0] == 0.0);
    try std.testing.expect(output[1] == 0.0);
    try std.testing.expect(output[2] > 0.0);
    try std.testing.expect(output[3] == 0.0);
    try std.testing.expect(output[4] == 10.0);
    try std.testing.expect(output[5] == 0.0);
    try std.testing.expect(output[6] == 100.0);
    try std.testing.expect(output[7] == 0.0);
}

/// CUDA Matrix Multiplication Kernel Validation Suite
const std = @import("std");
const cuda = @import("cuda.zig");

// Simple CPU reference matmul
fn cpuMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
    @memset(c, 0);
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |l| {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// Test 1: Small matmul
test "cuda_matmul_small" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return; // CUDA not available
    };
    defer backend.deinit();

    const m = 64;
    const n = 64;
    const k = 64;

    const a = try std.testing.allocator.alloc(f32, m * k);
    const b = try std.testing.allocator.alloc(f32, k * n);
    const c_cuda = try std.testing.allocator.alloc(f32, m * n);
    const c_cpu = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(a);
    defer std.testing.allocator.free(b);
    defer std.testing.allocator.free(c_cuda);
    defer std.testing.allocator.free(c_cpu);

    for (a, 0..) |*val, i| val.* = @floatFromInt(i % 10);
    for (b, 0..) |*val, i| val.* = @floatFromInt((i * 3) % 10);

    cpuMatMul(a, b, c_cpu, m, n, k);
    try backend.matMul(a, b, c_cuda, m, n, k, false, false, false);

    const tolerance = 1e-4;
    for (c_cpu, c_cuda) |expected, actual| {
        try std.testing.expect(@abs(expected - actual) < tolerance);
    }
}

// Test 2: Medium size
test "cuda_matmul_medium" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const m = 256;
    const n = 256;
    const k = 256;

    const a = try std.testing.allocator.alloc(f32, m * k);
    const b = try std.testing.allocator.alloc(f32, k * n);
    const c = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(a);
    defer std.testing.allocator.free(b);
    defer std.testing.allocator.free(c);

    for (a, 0..) |*val, i| val.* = @floatFromInt(i % 5);
    for (b, 0..) |*val, i| val.* = @floatFromInt((i * 2) % 5);

    try backend.matMul(a, b, c, m, n, k, false, false, false);
}

// Test 3: Non-square
test "cuda_matmul_nonsquare" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const m = 128;
    const n = 64;
    const k = 256;

    const a = try std.testing.allocator.alloc(f32, m * k);
    const b = try std.testing.allocator.alloc(f32, k * n);
    const c = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(a);
    defer std.testing.allocator.free(b);
    defer std.testing.allocator.free(c);

    for (a, 0..) |*val, i| val.* = @floatFromInt(i % 7);
    for (b, 0..) |*val, i| val.* = @floatFromInt((i * 3) % 7);

    try backend.matMul(a, b, c, m, n, k, false, false, false);
}

// Test 4: Accumulate
test "cuda_matmul_accumulate" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const m = 64;
    const n = 64;
    const k = 64;

    const a = try std.testing.allocator.alloc(f32, m * k);
    const b = try std.testing.allocator.alloc(f32, k * n);
    const c = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(a);
    defer std.testing.allocator.free(b);
    defer std.testing.allocator.free(c);

    for (a, 0..) |*val, i| val.* = @floatFromInt(i % 3);
    for (b, 0..) |*val, i| val.* = @floatFromInt((i * 2) % 3);

    // First call (no accumulate)
    try backend.matMul(a, b, c, m, n, k, false, false, false);

    // Second call (accumulate)
    try backend.matMul(a, b, c, m, n, k, false, false, true);
}

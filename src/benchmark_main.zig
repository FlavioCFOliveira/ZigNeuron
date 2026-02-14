/// Benchmark suite for ZigNeuron with GPU/CPU comparison
const std = @import("std");
const benchmark = @import("benchmark.zig");
const backend_module = @import("backend.zig");
const activation = @import("activation.zig");
const loss = @import("loss.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== ZigNeuron Performance Benchmarks ===\n\n", .{});

    // Get available backends
    const cpu_backend = backend_module.Backend{ .cpu = {} };

    var results: std.ArrayList(benchmark.BenchmarkResult) = .{
        .items = &[_]benchmark.BenchmarkResult{},
        .capacity = 0,
    };
    defer results.deinit(allocator);

    // Benchmark 1: Matrix multiplication (128x128) - CPU
    std.debug.print("Benchmark 1: Matrix Multiplication (128x128)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkMatrixMul(allocator, cpu_backend, 128, 128, 128, 100);
        try results.append(allocator, cpu_result);
        benchmark.printResult(cpu_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 2: Matrix multiplication (256x256) - CPU
    std.debug.print("Benchmark 2: Matrix Multiplication (256x256)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkMatrixMul(allocator, cpu_backend, 256, 256, 256, 100);
        try results.append(allocator, cpu_result);
        benchmark.printResult(cpu_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 3: Forward pass (small network) - CPU
    std.debug.print("Benchmark 3: Forward Pass (Small Network)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkForwardPass(allocator, cpu_backend, 8, 1, 3, 1000);
        try results.append(allocator, cpu_result);
        benchmark.printResult(cpu_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 4: Forward pass (larger network) - CPU
    std.debug.print("Benchmark 4: Forward Pass (Larger Network)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkForwardPass(allocator, cpu_backend, 16, 1, 4, 500);
        try results.append(allocator, cpu_result);
        benchmark.printResult(cpu_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 5: Training step - CPU
    std.debug.print("Benchmark 5: Training Step (small)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkTraining(allocator, cpu_backend, 4, 1, 3, 10, 50);
        try results.append(allocator, cpu_result);
        benchmark.printResult(cpu_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 6: Activation functions - CPU
    std.debug.print("Benchmark 6: Activation Forward (1024 elements)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkActivationForward(allocator, cpu_backend, .relu, 1024, 1000);
        try results.append(allocator, cpu_result);
        benchmark.printResult(cpu_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 7: Activation functions - larger size - CPU
    std.debug.print("Benchmark 7: Activation Forward (4096 elements)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkActivationForward(allocator, cpu_backend, .tanh, 4096, 500);
        try results.append(allocator, cpu_result);
        benchmark.printResult(cpu_result);
    }
    std.debug.print("\n", .{});

    // Print comparison
    benchmark.compareBackends(results.items);

    std.debug.print("\n=== Benchmark Complete ===\n", .{});
}

/// Benchmark suite for ZigNeuron
const std = @import("std");
const benchmark = @import("benchmark.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== ZigNeuron Performance Benchmarks ===\n\n", .{});

    // Benchmark 1: Forward pass with small network
    std.debug.print("Benchmark 1: Forward Pass (Small Network)\n", .{});
    const result1 = try benchmark.benchmarkForwardPass(allocator, 4, 1, 3, 1000);
    benchmark.printResult(result1);
    std.debug.print("\n", .{});

    // Benchmark 2: Forward pass with larger network
    std.debug.print("Benchmark 2: Forward Pass (Larger Network)\n", .{});
    const result2 = try benchmark.benchmarkForwardPass(allocator, 8, 1, 4, 1000);
    benchmark.printResult(result2);
    std.debug.print("\n", .{});

    // Benchmark 3: Training step
    std.debug.print("Benchmark 3: Training Step\n", .{});
    const result3 = try benchmark.benchmarkTraining(allocator, 4, 1, 3, 10, 100);
    benchmark.printResult(result3);
    std.debug.print("\n", .{});

    // Benchmark 4: Activation function
    std.debug.print("Benchmark 4: Softmax Forward (1024 elements)\n", .{});
    const result4 = try benchmark.benchmarkActivationForward(allocator, .softmax, 1024, 1000);
    benchmark.printResult(result4);
    std.debug.print("\n", .{});

    std.debug.print("=== Benchmark Complete ===\n", .{});
}

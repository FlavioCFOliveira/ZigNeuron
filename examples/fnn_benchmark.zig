/// Comprehensive FNN Benchmark for CPU vs Metal GPU comparison
const std = @import("std");
const zn = @import("ZigNeuron");

const BenchmarkResult = struct {
    backend_name: []const u8,
    total_time_ms: u64,
    avg_forward_time_ms: f32,
    avg_backward_time_ms: f32,
    memory_used_bytes: u64,
    iterations: usize,
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n╔═══════════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║  ZigNeuron FNN Benchmark - CPU vs Metal GPU Performance      ║\n", .{});
    std.debug.print("╚═══════════════════════════════════════════════════════════════╝\n", .{});

    // Benchmark Part 1: Sinewave Regression
    try benchmarkSinewaveRegression(allocator);

    // Benchmark Part 2: Binary Classification
    try benchmarkBinaryClassification(allocator);

    // Benchmark Part 3: Multi-class Classification
    try benchmarkMulticlassClassification(allocator);

    std.debug.print("\n✓ All benchmarks completed successfully!\n", .{});
}

fn benchmarkSinewaveRegression(allocator: std.mem.Allocator) !void {
    const separator = "═══════════════════════════════════════════════════════════════";
    std.debug.print("\n{s}\n", .{separator});
    std.debug.print("  PART 1: Sinewave Regression (1->16->16->1)\n", .{});
    std.debug.print("{s}\n\n", .{separator});

    // Test with CPU backend
    std.debug.print("Testing with CPU backend...\n", .{});
    const cpu_result = try benchmarkSinewave(allocator, zn.backend.Backend{ .type = .cpu, .metal_ctx = null }, "CPU");
    printBenchmarkResult(cpu_result);

    // Test with Metal backend (if available)
    std.debug.print("\nTesting with Metal GPU backend...\n", .{});
    const metal_result = try benchmarkSinewave(allocator, zn.backend.Backend{ .gpu = .metal }, "Metal");
    printBenchmarkResult(metal_result);

    // Print comparison
    printComparison(cpu_result, metal_result);
}

fn benchmarkSinewave(allocator: std.mem.Allocator, backend: zn.backend.Backend, backend_name: []const u8) !BenchmarkResult {
    const start_time = std.time.milliTimestamp();
    var memory_used: u64 = 0;

    // Create network
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();

    _ = try network.addDense(1, 16, .relu);
    _ = try network.addDense(16, 16, .relu);
    _ = try network.addDense(16, 1, .linear);

    // Generate training data
    const training_data = try generateSinewaveData(allocator, 1000);
    defer allocator.free(training_data);

    const targets = try generateSinewaveTargets(allocator, training_data);
    defer allocator.free(targets);

    // Training parameters
    const epochs: usize = 100;
    const learning_rate: f32 = 0.01;
    const loss_fn = zn.loss.Loss{ .mse = {} };

    // Benchmark training
    var total_forward_time: u64 = 0;
    var total_backward_time: u64 = 0;
    var forward_count: usize = 0;
    var backward_count: usize = 0;

    for (0..epochs) |epoch| {
        var epoch_loss: f32 = 0;

        for (training_data, targets) |data, target| {
            // Forward pass
            const forward_start = std.time.microTimestamp();
            var output: [1]f32 = undefined;
            _ = try network.forward(data, &output);
            const forward_end = std.time.microTimestamp();
            total_forward_time += @as(u64, @intCast(forward_end - forward_start));
            forward_count += 1;

            // Backward pass (training step)
            const backward_start = std.time.microTimestamp();
            const sample_loss = try network.trainStep(data, target, learning_rate, loss_fn);
            const backward_end = std.time.microTimestamp();
            total_backward_time += @as(u64, @intCast(backward_end - backward_start));
            backward_count += 1;

            epoch_loss += sample_loss;
        }

        if (epoch % 10 == 0) {
            std.debug.print("  Epoch {}: Loss = {d:.4}\n", .{ epoch, epoch_loss / @as(f32, @floatFromInt(training_data.len)) });
        }
    }

    const end_time = std.time.milliTimestamp();
    memory_used = @as(u64, @intCast(@sizeOf(zn.network.Network) + training_data.len * @sizeOf(f32) * 2));

    return BenchmarkResult{
        .backend_name = backend_name,
        .total_time_ms = @as(u64, @intCast(end_time - start_time)),
        .avg_forward_time_ms = @as(f32, @floatFromInt(total_forward_time)) / @as(f32, @floatFromInt(forward_count)) / 1000.0,
        .avg_backward_time_ms = @as(f32, @floatFromInt(total_backward_time)) / @as(f32, @floatFromInt(backward_count)) / 1000.0,
        .memory_used_bytes = memory_used,
        .iterations = epochs * training_data.len,
    };
}

fn benchmarkBinaryClassification(allocator: std.mem.Allocator) !void {
    const separator = "═══════════════════════════════════════════════════════════════";
    std.debug.print("\n{s}\n", .{separator});
    std.debug.print("  PART 2: Binary Classification (2->8->8->1)\n", .{});
    std.debug.print("{s}\n\n", .{separator});

    // Test with CPU backend
    std.debug.print("Testing with CPU backend...\n", .{});
    const cpu_result = try benchmarkBinaryClassificationImpl(allocator, zn.backend.Backend{ .type = .cpu, .metal_ctx = null }, "CPU");
    printBenchmarkResult(cpu_result);

    // Test with Metal backend (if available)
    std.debug.print("\nTesting with Metal GPU backend...\n", .{});
    const metal_result = try benchmarkBinaryClassificationImpl(allocator, zn.backend.Backend{ .gpu = .metal }, "Metal");
    printBenchmarkResult(metal_result);

    // Print comparison
    printComparison(cpu_result, metal_result);
}

fn benchmarkBinaryClassificationImpl(allocator: std.mem.Allocator, backend: zn.backend.Backend, backend_name: []const u8) !BenchmarkResult {
    const start_time = std.time.milliTimestamp();

    // Create network
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();

    _ = try network.addDense(2, 8, .relu);
    _ = try network.addDense(8, 8, .relu);
    _ = try network.addDense(8, 1, .sigmoid);

    // Generate spiral-like data
    const training_data = try generateSpiralData(allocator, 2000);
    defer allocator.free(training_data);

    const targets = try generateSpiralTargets(allocator, training_data);
    defer allocator.free(targets);

    // Training parameters
    const epochs: usize = 100;
    const learning_rate: f32 = 0.1;
    const loss_fn = zn.loss.Loss{ .mse = {} };

    // Benchmark training
    var total_forward_time: u64 = 0;
    var total_backward_time: u64 = 0;
    var forward_count: usize = 0;
    var backward_count: usize = 0;

    for (0..epochs) |epoch| {
        var epoch_loss: f32 = 0;

        for (training_data, targets) |data, target| {
            // Forward pass
            const forward_start = std.time.microTimestamp();
            var output: [1]f32 = undefined;
            _ = try network.forward(data, &output);
            const forward_end = std.time.microTimestamp();
            total_forward_time += @as(u64, @intCast(forward_end - forward_start));
            forward_count += 1;

            // Backward pass (training step)
            const backward_start = std.time.microTimestamp();
            const sample_loss = try network.trainStep(data, target, learning_rate, loss_fn);
            const backward_end = std.time.microTimestamp();
            total_backward_time += @as(u64, @intCast(backward_end - backward_start));
            backward_count += 1;

            epoch_loss += sample_loss;
        }

        if (epoch % 10 == 0) {
            std.debug.print("  Epoch {}: Loss = {d:.4}\n", .{ epoch, epoch_loss / @as(f32, @floatFromInt(training_data.len)) });
        }
    }

    const end_time = std.time.milliTimestamp();

    return BenchmarkResult{
        .backend_name = backend_name,
        .total_time_ms = @as(u64, @intCast(end_time - start_time)),
        .avg_forward_time_ms = @as(f32, @floatFromInt(total_forward_time)) / @as(f32, @floatFromInt(forward_count)) / 1000.0,
        .avg_backward_time_ms = @as(f32, @floatFromInt(total_backward_time)) / @as(f32, @floatFromInt(backward_count)) / 1000.0,
        .memory_used_bytes = @as(u64, @intCast(training_data.len * @sizeOf(f32) * 4)), // Estimate
        .iterations = epochs * training_data.len,
    };
}

fn benchmarkMulticlassClassification(allocator: std.mem.Allocator) !void {
    const separator = "═══════════════════════════════════════════════════════════════";
    std.debug.print("\n{s}\n", .{separator});
    std.debug.print("  PART 3: Multi-class Classification (4->12->12->3)\n", .{});
    std.debug.print("{s}\n\n", .{separator});

    // Test with CPU backend
    std.debug.print("Testing with CPU backend...\n", .{});
    const cpu_result = try benchmarkMulticlassClassificationImpl(allocator, zn.backend.Backend{ .type = .cpu, .metal_ctx = null }, "CPU");
    printBenchmarkResult(cpu_result);

    // Test with Metal backend (if available)
    std.debug.print("\nTesting with Metal GPU backend...\n", .{});
    const metal_result = try benchmarkMulticlassClassificationImpl(allocator, zn.backend.Backend{ .gpu = .metal }, "Metal");
    printBenchmarkResult(metal_result);

    // Print comparison
    printComparison(cpu_result, metal_result);
}

fn benchmarkMulticlassClassificationImpl(allocator: std.mem.Allocator, backend: zn.backend.Backend, backend_name: []const u8) !BenchmarkResult {
    const start_time = std.time.milliTimestamp();

    // Create network
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();

    _ = try network.addDense(4, 12, .relu);
    _ = try network.addDense(12, 12, .relu);
    _ = try network.addDense(12, 3, .softmax);

    // Generate Iris-like data
    const training_data = try generateIrisData(allocator, 1500);
    defer allocator.free(training_data);

    const targets = try generateIrisTargets(allocator, training_data);
    defer allocator.free(targets);

    // Training parameters
    const epochs: usize = 100;
    const learning_rate: f32 = 0.05;
    const loss_fn = zn.loss.Loss{ .cross_entropy = {} };

    // Benchmark training
    var total_forward_time: u64 = 0;
    var total_backward_time: u64 = 0;
    var forward_count: usize = 0;
    var backward_count: usize = 0;

    for (0..epochs) |epoch| {
        var epoch_loss: f32 = 0;

        for (training_data, targets) |data, target| {
            // Forward pass
            const forward_start = std.time.microTimestamp();
            var output: [3]f32 = undefined;
            _ = try network.forward(data, &output);
            const forward_end = std.time.microTimestamp();
            total_forward_time += @as(u64, @intCast(forward_end - forward_start));
            forward_count += 1;

            // Backward pass (training step)
            const backward_start = std.time.microTimestamp();
            const sample_loss = try network.trainStep(data, target, learning_rate, loss_fn);
            const backward_end = std.time.microTimestamp();
            total_backward_time += @as(u64, @intCast(backward_end - backward_start));
            backward_count += 1;

            epoch_loss += sample_loss;
        }

        if (epoch % 10 == 0) {
            std.debug.print("  Epoch {}: Loss = {d:.4}\n", .{ epoch, epoch_loss / @as(f32, @floatFromInt(training_data.len)) });
        }
    }

    const end_time = std.time.milliTimestamp();

    return BenchmarkResult{
        .backend_name = backend_name,
        .total_time_ms = @as(u64, @intCast(end_time - start_time)),
        .avg_forward_time_ms = @as(f32, @floatFromInt(total_forward_time)) / @as(f32, @floatFromInt(forward_count)) / 1000.0,
        .avg_backward_time_ms = @as(f32, @floatFromInt(total_backward_time)) / @as(f32, @floatFromInt(backward_count)) / 1000.0,
        .memory_used_bytes = @as(u64, @intCast(training_data.len * @sizeOf(f32) * 8)), // Estimate
        .iterations = epochs * training_data.len,
    };
}

fn printBenchmarkResult(result: BenchmarkResult) void {
    std.debug.print("\n  Backend: {s}\n", .{result.backend_name});
    std.debug.print("  ───────────────────────────────────────────────────────────────\n", .{});
    std.debug.print("  Total Training Time: {d} ms\n", .{result.total_time_ms});
    std.debug.print("  Average Forward Pass: {d:.4} ms\n", .{result.avg_forward_time_ms});
    std.debug.print("  Average Backward Pass: {d:.4} ms\n", .{result.avg_backward_time_ms});
    std.debug.print("  Memory Used: {d} KB\n", .{result.memory_used_bytes / 1024});
    std.debug.print("  Iterations: {d}\n", .{result.iterations});
    std.debug.print("  ───────────────────────────────────────────────────────────────\n", .{});
}

fn printComparison(cpu: BenchmarkResult, metal: BenchmarkResult) void {
    std.debug.print("\n╔═══════════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║                    PERFORMANCE COMPARISON                     ║\n", .{});
    std.debug.print("╚═══════════════════════════════════════════════════════════════╝\n", .{});

    const speedup_total = @as(f32, @floatFromInt(cpu.total_time_ms)) / @as(f32, @floatFromInt(metal.total_time_ms));
    const speedup_forward = cpu.avg_forward_time_ms / metal.avg_forward_time_ms;
    const speedup_backward = cpu.avg_backward_time_ms / metal.avg_backward_time_ms;

    std.debug.print("\n  Speedup (Metal vs CPU):\n", .{});
    std.debug.print("  ───────────────────────────────────────────────────────────────\n", .{});
    std.debug.print("  Total Training Time: {d:.2}x faster\n", .{speedup_total});
    std.debug.print("  Forward Pass: {d:.2}x faster\n", .{speedup_forward});
    std.debug.print("  Backward Pass: {d:.2}x faster\n", .{speedup_backward});
    std.debug.print("  ───────────────────────────────────────────────────────────────\n", .{});

    // Classify speedup
    if (speedup_total > 10.0) {
        std.debug.print("  🚀 EXCELLENT: Metal GPU provides massive speedup!\n", .{});
    } else if (speedup_total > 5.0) {
        std.debug.print("  ⚡ GREAT: Metal GPU provides significant speedup!\n", .{});
    } else if (speedup_total > 2.0) {
        std.debug.print("  ✨ GOOD: Metal GPU provides moderate speedup!\n", .{});
    } else if (speedup_total > 1.0) {
        std.debug.print("  👍 FAIR: Metal GPU provides slight speedup!\n", .{});
    } else {
        std.debug.print("  ⚠️  LIMITED: Metal GPU performance similar to CPU\n", .{});
    }
}

// Helper functions for data generation
fn generateSinewaveData(allocator: std.mem.Allocator, count: usize) ![][]const f32 {
    const data = try allocator.alloc([]const f32, count);
    for (0..count) |i| {
        const x = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(count));
        const input = try allocator.alloc(f32, 1);
        input[0] = x;
        data[i] = input;
    }
    return data;
}

fn generateSinewaveTargets(allocator: std.mem.Allocator, data: []const []const f32) ![][]const f32 {
    const targets = try allocator.alloc([]const f32, data.len);
    for (data, 0..) |input, i| {
        const target = try allocator.alloc(f32, 1);
        target[0] = @sin(2 * std.math.pi * input[0]) + 0.1 * @sin(10 * std.math.pi * input[0]);
        targets[i] = target;
    }
    return targets;
}

fn generateSpiralData(allocator: std.mem.Allocator, count: usize) ![][]const f32 {
    const data = try allocator.alloc([]const f32, count);
    for (0..count) |i| {
        const input = try allocator.alloc(f32, 2);
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(count)) * 4 * std.math.pi;
        const label = i % 2;
        const sign: f32 = if (label == 0) 1.0 else -1.0;
        input[0] = sign * t * @cos(t) / (4 * std.math.pi);
        input[1] = sign * t * @sin(t) / (4 * std.math.pi);
        data[i] = input;
    }
    return data;
}

fn generateSpiralTargets(allocator: std.mem.Allocator, data: []const []const f32) ![][]const f32 {
    const targets = try allocator.alloc([]const f32, data.len);
    for (targets, 0..) |*target, i| {
        target.* = try allocator.alloc(f32, 1);
        target.*[0] = @as(f32, @floatFromInt(i % 2));
    }
    return targets;
}

fn generateIrisData(allocator: std.mem.Allocator, count: usize) ![][]const f32 {
    const data = try allocator.alloc([]const f32, count);
    for (0..count) |i| {
        const input = try allocator.alloc(f32, 4);
        const class = i % 3;
        if (class == 0) {
            // Class 0: Sepal-like
            input[0] = 5.0 + @as(f32, @floatFromInt(i % 10)) / 10.0;
            input[1] = 3.5 + @as(f32, @floatFromInt(i % 8)) / 10.0;
            input[2] = 1.5 + @as(f32, @floatFromInt(i % 5)) / 10.0;
            input[3] = 0.2 + @as(f32, @floatFromInt(i % 3)) / 10.0;
        } else if (class == 1) {
            // Class 1: Versicolor-like
            input[0] = 6.0 + @as(f32, @floatFromInt(i % 10)) / 10.0;
            input[1] = 2.8 + @as(f32, @floatFromInt(i % 8)) / 10.0;
            input[2] = 4.5 + @as(f32, @floatFromInt(i % 5)) / 10.0;
            input[3] = 1.5 + @as(f32, @floatFromInt(i % 3)) / 10.0;
        } else {
            // Class 2: Virginica-like
            input[0] = 6.5 + @as(f32, @floatFromInt(i % 10)) / 10.0;
            input[1] = 3.0 + @as(f32, @floatFromInt(i % 8)) / 10.0;
            input[2] = 5.5 + @as(f32, @floatFromInt(i % 5)) / 10.0;
            input[3] = 2.0 + @as(f32, @floatFromInt(i % 3)) / 10.0;
        }
        data[i] = input;
    }
    return data;
}

fn generateIrisTargets(allocator: std.mem.Allocator, data: []const []const f32) ![][]const f32 {
    const targets = try allocator.alloc([]const f32, data.len);
    for (targets, 0..) |*target, i| {
        target.* = try allocator.alloc(f32, 3);
        const class = i % 3;
        target.*[0] = if (class == 0) 1.0 else 0.0;
        target.*[1] = if (class == 1) 1.0 else 0.0;
        target.*[2] = if (class == 2) 1.0 else 0.0;
    }
    return targets;
}

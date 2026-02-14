/// Performance benchmarks for ZigNeuron
const std = @import("std");
const activation = @import("activation.zig");
const loss = @import("loss.zig");
const layer = @import("layer.zig");
const network = @import("network.zig");

pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    total_time_ns: u64,
    avg_time_per_iter_ns: u64,
    operations_per_second: f64,
};

pub fn benchmarkForwardPass(allocator: std.mem.Allocator, input_size: usize, output_size: usize, layers: usize, iterations: usize) !BenchmarkResult {
    const net = try network.Network.init(allocator);
    defer net.deinit();

    var current_size = input_size;
    for (0..layers) |i| {
        const next_size = if (i == layers - 1) output_size else current_size * 2;
        _ = try net.addDense(current_size, next_size, .relu);
        current_size = next_size;
    }

    const input = try allocator.alloc(f32, input_size);
    defer allocator.free(input);
    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 10)) / 10.0;
    }

    const output = try allocator.alloc(f32, current_size);
    defer allocator.free(output);

    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        _ = try net.forward(input, output);
    }
    const end = std.time.nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = total_time_ns / @as(u64, @intCast(iterations));
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0);

    return BenchmarkResult{
        .name = "forward_pass",
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
    };
}

pub fn benchmarkTraining(allocator: std.mem.Allocator, input_size: usize, output_size: usize, layers: usize, samples: usize, epochs: usize) !BenchmarkResult {
    const net = try network.Network.init(allocator);
    defer net.deinit();

    var current_size = input_size;
    for (0..layers) |i| {
        const next_size = if (i == layers - 1) output_size else current_size * 2;
        _ = try net.addDense(current_size, next_size, .relu);
        current_size = next_size;
    }

    const training_data = try allocator.alloc([]const f32, samples);
    defer allocator.free(training_data);
    const training_targets = try allocator.alloc([]const f32, samples);
    defer allocator.free(training_targets);

    for (0..samples) |i| {
        const sample = try allocator.alloc(f32, input_size);
        for (sample, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt((i + j) % 10)) / 10.0;
        }
        training_data[i] = sample;

        const target = try allocator.alloc(f32, current_size);
        for (target, 0..) |*v, j| {
            v.* = @as(f32, @floatFromInt((i * j) % 2)) / 1.0;
        }
        training_targets[i] = target;
    }

    const loss_fn = loss.Loss{ .mse = {} };
    const learning_rate: f32 = 0.1;

    const start = std.time.nanoTimestamp();
    for (0..epochs) |_| {
        for (training_data, training_targets) |sample, target| {
            _ = try net.trainStep(sample, target, learning_rate, loss_fn);
        }
    }
    const end = std.time.nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = total_time_ns / @as(u64, @intCast(epochs * samples));
    const ops_per_sec = @as(f64, @floatFromInt(epochs * samples)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0);

    return BenchmarkResult{
        .name = "training_step",
        .iterations = epochs * samples,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
    };
}

pub fn benchmarkActivationForward(allocator: std.mem.Allocator, act: activation.Activation, size: usize, iterations: usize) !BenchmarkResult {
    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);

    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 100)) / 10.0;
    }

    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        try act.softmaxForward(input, output);
    }
    const end = std.time.nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = total_time_ns / @as(u64, @intCast(iterations));
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0);

    return BenchmarkResult{
        .name = "activation_forward",
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
    };
}

pub fn printResult(result: BenchmarkResult) void {
    std.debug.print("{s}:\n", .{result.name});
    std.debug.print("  Iterations: {}\n", .{result.iterations});
    std.debug.print("  Total time: {} ns\n", .{result.total_time_ns});
    std.debug.print("  Avg per iter: {} ns\n", .{result.avg_time_per_iter_ns});
    std.debug.print("  Ops/sec: {d:.2}\n", .{result.operations_per_second});
}

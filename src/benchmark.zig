/// Performance benchmarks for ZigNeuron
const std = @import("std");
const activation = @import("activation.zig");
const loss = @import("loss.zig");
const layer = @import("layer.zig");
const network = @import("network.zig");
const backend_module = @import("backend.zig");

pub const BenchmarkResult = struct {
    name: []const u8,
    backend: []const u8,
    iterations: usize,
    total_time_ns: u64,
    avg_time_per_iter_ns: u64,
    operations_per_second: f64,
};

// Get current time in nanoseconds (Zig 0.16+ compatible)
// Uses std.os.linux.clock_gettime for high-resolution timing
fn nanoTimestamp() u64 {
    const os = std.os;

    // timespec struct for clock_gettime
    var tv: os.linux.timespec = undefined;

    // CLOCK_MONOTONIC = 1 (from linux/time.h)
    const CLOCK_MONOTONIC: os.linux.clockid_t = .MONOTONIC;

    // Call clock_gettime
    _ = os.linux.clock_gettime(CLOCK_MONOTONIC, &tv);

    // Return time in nanoseconds
    return @as(u64, @intCast(tv.sec)) * 1_000_000_000 + @as(u64, @intCast(tv.nsec));
}

pub fn benchmarkForwardPass(allocator: std.mem.Allocator, backend: backend_module.Backend, input_size: usize, output_size: usize, layers: usize, iterations: usize) !BenchmarkResult {
    const net = try network.Network.init(allocator, backend);
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

    const start = nanoTimestamp();
    for (0..iterations) |_| {
        _ = try net.forward(input, output);
    }
    const end = nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = total_time_ns / @as(u64, @intCast(iterations));
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0);

    const backend_name = switch (backend) {
        .cpu => "CPU",
        .gpu => |gpu| switch (gpu) {
            .metal => "Metal",
            .vulkan => "Vulkan",
        },
    };

    return BenchmarkResult{
        .name = "forward_pass",
        .backend = backend_name,
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
    };
}

pub fn benchmarkTraining(allocator: std.mem.Allocator, backend: backend_module.Backend, input_size: usize, output_size: usize, layers: usize, samples: usize, epochs: usize) !BenchmarkResult {
    const net = try network.Network.init(allocator, backend);
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

    const start = nanoTimestamp();
    for (0..epochs) |_| {
        for (training_data, training_targets) |sample, target| {
            _ = try net.trainStep(sample, target, learning_rate, loss_fn);
        }
    }
    const end = nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = total_time_ns / @as(u64, @intCast(epochs * samples));
    const ops_per_sec = @as(f64, @floatFromInt(epochs * samples)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0);

    const backend_name = switch (backend) {
        .cpu => "CPU",
        .gpu => |gpu| switch (gpu) {
            .metal => "Metal",
            .vulkan => "Vulkan",
        },
    };

    return BenchmarkResult{
        .name = "training_step",
        .backend = backend_name,
        .iterations = epochs * samples,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
    };
}

pub fn benchmarkActivationForward(allocator: std.mem.Allocator, backend: backend_module.Backend, act: activation.Activation, size: usize, iterations: usize) !BenchmarkResult {
    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);

    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 100)) / 10.0;
    }

    const start = nanoTimestamp();
    for (0..iterations) |_| {
        try backend.activationForward(act, input, output);
    }
    const end = nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = total_time_ns / @as(u64, @intCast(iterations));
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0);

    const backend_name = switch (backend) {
        .cpu => "CPU",
        .gpu => |gpu| switch (gpu) {
            .metal => "Metal",
            .vulkan => "Vulkan",
        },
    };

    return BenchmarkResult{
        .name = "activation_forward",
        .backend = backend_name,
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
    };
}

pub fn benchmarkMatrixMul(allocator: std.mem.Allocator, backend: backend_module.Backend, m: usize, n: usize, k: usize, iterations: usize) !BenchmarkResult {
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    for (a, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 10)) / 10.0;
    }
    for (b, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt((i * 3) % 10)) / 10.0;
    }

    const start = nanoTimestamp();
    for (0..iterations) |_| {
        backend.matMul(a, null, b, null, c, null, m, n, k, false) catch unreachable;
    }
    const end = nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = total_time_ns / @as(u64, @intCast(iterations));
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0);

    const backend_name = switch (backend) {
        .cpu => "CPU",
        .gpu => |gpu| switch (gpu) {
            .metal => "Metal",
            .vulkan => "Vulkan",
        },
    };

    return BenchmarkResult{
        .name = "matrix_mul",
        .backend = backend_name,
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
    };
}

pub fn printResult(result: BenchmarkResult) void {
    std.debug.print("{s} ({s}):\n", .{ result.name, result.backend });
    std.debug.print("  Iterations: {}\n", .{result.iterations});
    std.debug.print("  Total time: {} ns\n", .{result.total_time_ns});
    std.debug.print("  Avg per iter: {} ns\n", .{result.avg_time_per_iter_ns});
    std.debug.print("  Ops/sec: {d:.2}\n", .{result.operations_per_second});
}

pub fn compareBackends(results: []const BenchmarkResult) void {
    if (results.len < 2) return;

    std.debug.print("\n=== Backend Comparison ===\n", .{});

    // Group results by name
    var i: usize = 0;
    while (i < results.len) : (i += 1) {
        const name = results[i].name;
        var j = i;
        while (j < results.len and std.mem.eql(u8, results[j].name, name)) : (j += 1) {}

        std.debug.print("\n{s}:\n", .{name});
        std.debug.print("{s:<15} {s:>15} {s:>12}\n", .{ "Backend", "Ops/sec", "Speedup" });

        // Find best ops (GPU preferred)
        var best_ops: f64 = 0;
        for (results[i..j]) |r| {
            if (r.operations_per_second > best_ops) best_ops = r.operations_per_second;
        }

        for (results[i..j]) |r| {
            const speedup = r.operations_per_second / best_ops;
            std.debug.print("{s:<15} {d:>15.2} {d:>12.2}x\n", .{ r.backend, r.operations_per_second, speedup });
        }

        i = j;
    }

    // Summary
    std.debug.print("\n=== Summary ===\n", .{});
    std.debug.print("GPU (Metal) vs CPU performance comparison:\n", .{});
    var cpu_total: f64 = 0;
    var gpu_total: f64 = 0;
    var cpu_count: usize = 0;
    var gpu_count: usize = 0;
    for (results) |r| {
        if (std.mem.eql(u8, r.backend, "CPU")) {
            cpu_total += r.operations_per_second;
            cpu_count += 1;
        } else if (std.mem.eql(u8, r.backend, "Metal")) {
            gpu_total += r.operations_per_second;
            gpu_count += 1;
        }
    }
    if (cpu_count > 0 and gpu_count > 0) {
        const cpu_avg = cpu_total / @as(f64, @floatFromInt(cpu_count));
        const gpu_avg = gpu_total / @as(f64, @floatFromInt(gpu_count));
        const gpu_speedup = gpu_avg / cpu_avg;
        std.debug.print("  CPU average:  {d:.2} ops/sec\n", .{cpu_avg});
        std.debug.print("  GPU average:  {d:.2} ops/sec\n", .{gpu_avg});
        std.debug.print("  GPU speedup:  {d:.2}x\n", .{gpu_speedup});
    }
}

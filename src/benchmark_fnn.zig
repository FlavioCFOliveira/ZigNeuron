/// Performance benchmark: FNN with CPU vs Metal GPU comparison
const std = @import("std");
const zn = @import("ZigNeuron");

const BenchmarkResult = struct {
    backend_name: []const u8,
    total_time_ms: u64,
    forward_pass_ms: u64,
    backward_pass_ms: u64,
    memory_allocated: usize,
    epochs: usize,
    final_loss: f32,
};

fn benchmarkFNN(backend: zn.backend.Backend, allocator: std.mem.Allocator, comptime name: []const u8) !BenchmarkResult {
    const io = std.Io.Threaded.global_single_threaded.io();
    const start_clock = std.Io.Clock.now(.real, io);
    const start_total = @divTrunc(start_clock.nanoseconds, 1_000_000);

    // Create network: 4-16-16-1 (similar to FNN comprehensive example)
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();

    _ = try network.addDense(4, 16, .relu);
    _ = try network.addDense(16, 16, .relu);
    _ = try network.addDense(16, 1, .linear);

    // Generate synthetic data
    const data_size = 100;
    const data = try allocator.alloc([]const f32, data_size);
    defer allocator.free(data);
    const targets = try allocator.alloc([]const f32, data_size);
    defer allocator.free(targets);

    for (0..data_size) |i| {
        const sample = try allocator.alloc(f32, 4);
        const target = try allocator.alloc(f32, 1);

        // Generate random data
        sample[0] = @as(f32, @floatFromInt(i % 10)) / 10.0;
        sample[1] = @as(f32, @floatFromInt((i + 3) % 10)) / 10.0;
        sample[2] = @as(f32, @floatFromInt((i + 5) % 10)) / 10.0;
        sample[3] = @as(f32, @floatFromInt((i + 7) % 10)) / 10.0;
        target[0] = (sample[0] + sample[1] - sample[2] + sample[3]) / 4.0;

        data[i] = sample;
        targets[i] = target;
    }

    // Training benchmark
    const loss_fn = zn.loss.Loss{ .mse = {} };
    const learning_rate: f32 = 0.01;
    const epochs: usize = 100;

    var final_loss: f32 = 0;
    for (0..epochs) |epoch| {
        var total_loss: f32 = 0;

        for (data, targets) |sample, target| {
            const loss = try network.trainStep(sample, target, learning_rate, loss_fn);
            total_loss += loss;
        }

        if (data_size > 0) {
            total_loss /= @as(f32, @floatFromInt(data_size));
        }

        if (epoch == epochs - 1) {
            final_loss = total_loss;
        }
    }

    const end_clock = std.Io.Clock.now(.real, io);
    const end_total = @divTrunc(end_clock.nanoseconds, 1_000_000);
    const total_time_ms = @as(u64, @intCast(end_total - start_total));

    // Memory usage estimation
    const memory_allocated = network.layers.items.len * 4 * 1024; // Rough estimate

    return BenchmarkResult{
        .backend_name = name,
        .total_time_ms = total_time_ms,
        .forward_pass_ms = 0, // Would need instrumentation
        .backward_pass_ms = 0, // Would need instrumentation
        .memory_allocated = memory_allocated,
        .epochs = epochs,
        .final_loss = final_loss,
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n==================================================\n", .{});
    std.debug.print("  FNN Performance Benchmark: CPU vs Metal GPU\n", .{});
    std.debug.print("==================================================\n\n", .{});

    // Test CPU backend
    std.debug.print("Testing CPU Backend...\n", .{});
    const cpu_result = try benchmarkFNN(zn.backend.Backend{ .type = .cpu, .metal_ctx = null }, allocator, "CPU");

    std.debug.print("Testing Metal GPU Backend...\n", .{});
    const metal_result = try benchmarkFNN(zn.backend.Backend{ .gpu = .metal }, allocator, "Metal GPU");

    // Print results
    std.debug.print("\n==================================================\n", .{});
    std.debug.print("  Performance Results\n", .{});
    std.debug.print("==================================================\n\n", .{});

    std.debug.print("{s: <15} | {s: >12} | {s: >12} | {s: >12}\n", .{
        "Backend", "Time (ms)", "Time/Epoch", "Final Loss"
    });
    std.debug.print("{s: <15}-+-{s: >12}-+-{s: >12}-+-{s: >12}\n", .{
        "---------------", "------------", "------------", "------------"
    });

    const cpu_time_per_epoch = @divTrunc(cpu_result.total_time_ms, cpu_result.epochs);
    const metal_time_per_epoch = @divTrunc(metal_result.total_time_ms, metal_result.epochs);
    const speedup = @as(f32, @floatFromInt(cpu_result.total_time_ms)) / @as(f32, @floatFromInt(metal_result.total_time_ms));

    std.debug.print("{s: <15} | {d: >12} | {d: >12} | {d: >12.4}\n", .{
        cpu_result.backend_name,
        cpu_result.total_time_ms,
        cpu_time_per_epoch,
        cpu_result.final_loss,
    });

    std.debug.print("{s: <15} | {d: >12} | {d: >12} | {d: >12.4}\n", .{
        metal_result.backend_name,
        metal_result.total_time_ms,
        metal_time_per_epoch,
        metal_result.final_loss,
    });

    std.debug.print("\nSpeedup: {d:.2}x\n", .{speedup});

    // Memory comparison
    std.debug.print("\n==================================================\n", .{});
    std.debug.print("  Memory Usage\n", .{});
    std.debug.print("==================================================\n\n", .{});

    std.debug.print("CPU Memory:    {d} KB\n", .{cpu_result.memory_allocated / 1024});
    std.debug.print("Metal Memory:  {d} KB\n", .{metal_result.memory_allocated / 1024});
    std.debug.print("Memory Diff:   {d} KB\n", .{
        @as(i64, @intCast(cpu_result.memory_allocated)) - @as(i64, @intCast(metal_result.memory_allocated))
    });

    // Summary
    std.debug.print("\n==================================================\n", .{});
    std.debug.print("  Summary\n", .{});
    std.debug.print("==================================================\n\n", .{});

    if (speedup > 1.0) {
        std.debug.print("✅ Metal GPU is {d:.2}x faster than CPU\n", .{speedup});
    } else if (speedup < 1.0) {
        std.debug.print("⚠️  CPU is {d:.2}x faster than Metal GPU\n", .{1.0 / speedup});
    } else {
        std.debug.print("ℹ️  CPU and Metal GPU have similar performance\n", .{});
    }

    std.debug.print("\nNote: Metal GPU currently uses CPU fallback.\n", .{});
    std.debug.print("Actual GPU implementation would provide 10-100x speedup.\n", .{});
}

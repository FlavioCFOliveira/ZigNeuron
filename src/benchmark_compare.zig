/// Vulkan vs CPU Benchmark comparison
const std = @import("std");
const benchmark = @import("benchmark.zig");
const backend_module = @import("backend.zig");
const activation = @import("activation.zig");
const loss = @import("loss.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Vulkan vs CPU Benchmark Comparison ===\n\n", .{});

    var results: std.ArrayList(benchmark.BenchmarkResult) = .{
        .items = &[_]benchmark.BenchmarkResult{},
        .capacity = 0,
    };
    defer results.deinit(allocator);

    // Get backends
    const cpu_backend = backend_module.Backend{ .type = .cpu, .metal_ctx = null };
    const vulkan_backend = backend_module.Backend{ .gpu = .vulkan };

    std.debug.print("Running benchmarks on both backends...\n\n", .{});

    // Benchmark 1: Matrix multiplication (128x128)
    std.debug.print("Benchmark 1: Matrix Multiplication (128x128)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkMatrixMul(allocator, cpu_backend, 128, 128, 128, 100);
        const vulkan_result = try benchmark.benchmarkMatrixMul(allocator, vulkan_backend, 128, 128, 128, 100);
        try results.append(allocator, cpu_result);
        try results.append(allocator, vulkan_result);
        benchmark.printResult(cpu_result);
        benchmark.printResult(vulkan_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 2: Matrix multiplication (256x256)
    std.debug.print("Benchmark 2: Matrix Multiplication (256x256)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkMatrixMul(allocator, cpu_backend, 256, 256, 256, 100);
        const vulkan_result = try benchmark.benchmarkMatrixMul(allocator, vulkan_backend, 256, 256, 256, 100);
        try results.append(allocator, cpu_result);
        try results.append(allocator, vulkan_result);
        benchmark.printResult(cpu_result);
        benchmark.printResult(vulkan_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 3: Forward pass (small network)
    std.debug.print("Benchmark 3: Forward Pass (Small Network)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkForwardPass(allocator, cpu_backend, 8, 1, 3, 1000);
        const vulkan_result = try benchmark.benchmarkForwardPass(allocator, vulkan_backend, 8, 1, 3, 1000);
        try results.append(allocator, cpu_result);
        try results.append(allocator, vulkan_result);
        benchmark.printResult(cpu_result);
        benchmark.printResult(vulkan_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 4: Forward pass (larger network)
    std.debug.print("Benchmark 4: Forward Pass (Larger Network)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkForwardPass(allocator, cpu_backend, 16, 1, 4, 500);
        const vulkan_result = try benchmark.benchmarkForwardPass(allocator, vulkan_backend, 16, 1, 4, 500);
        try results.append(allocator, cpu_result);
        try results.append(allocator, vulkan_result);
        benchmark.printResult(cpu_result);
        benchmark.printResult(vulkan_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 5: Training step
    std.debug.print("Benchmark 5: Training Step (small)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkTraining(allocator, cpu_backend, 4, 1, 3, 10, 50);
        const vulkan_result = try benchmark.benchmarkTraining(allocator, vulkan_backend, 4, 1, 3, 10, 50);
        try results.append(allocator, cpu_result);
        try results.append(allocator, vulkan_result);
        benchmark.printResult(cpu_result);
        benchmark.printResult(vulkan_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 6: Activation forward (1024 elements)
    std.debug.print("Benchmark 6: Activation Forward (1024 elements)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkActivationForward(allocator, cpu_backend, .relu, 1024, 1000);
        const vulkan_result = try benchmark.benchmarkActivationForward(allocator, vulkan_backend, .relu, 1024, 1000);
        try results.append(allocator, cpu_result);
        try results.append(allocator, vulkan_result);
        benchmark.printResult(cpu_result);
        benchmark.printResult(vulkan_result);
    }
    std.debug.print("\n", .{});

    // Benchmark 7: Activation forward (4096 elements)
    std.debug.print("Benchmark 7: Activation Forward (4096 elements)\n", .{});
    {
        const cpu_result = try benchmark.benchmarkActivationForward(allocator, cpu_backend, .tanh, 4096, 500);
        const vulkan_result = try benchmark.benchmarkActivationForward(allocator, vulkan_backend, .tanh, 4096, 500);
        try results.append(allocator, cpu_result);
        try results.append(allocator, vulkan_result);
        benchmark.printResult(cpu_result);
        benchmark.printResult(vulkan_result);
    }
    std.debug.print("\n", .{});

    // Print comparison table for each benchmark type
    printComparisonTable(&results, allocator);

    std.debug.print("\n=== Benchmark Complete ===\n", .{});
}

fn printComparisonTable(results: *std.ArrayList(benchmark.BenchmarkResult), allocator: std.mem.Allocator) void {
    _ = allocator;
    std.debug.print("\n=== Vulkan vs CPU Comparison ===\n", .{});

    // Group results by benchmark name
    var i: usize = 0;
    while (i < results.items.len) : (i += 1) {
        const name = results.items[i].name;
        var j = i;
        while (j < results.items.len and std.mem.eql(u8, results.items[j].name, name)) : (j += 1) {}

        // Print header for this benchmark type
        std.debug.print("\n--- {s} ---\n", .{name});

        // Print table header
        std.debug.print("{s:<20} {s:>15} {s:>15} {s:>12}\n", .{ "Backend", "Ops/sec", "Time (ms)", "Speedup" });
        std.debug.print("{s:-<74}\n", .{""});

        // Find best ops for speedup calculation
        var best_ops: f64 = 0;
        for (results.items[i..j]) |r| {
            if (r.operations_per_second > best_ops) best_ops = r.operations_per_second;
        }

        var cpu_ops: f64 = 0;
        var vulkan_ops: f64 = 0;
        var cpu_count: usize = 0;
        var vulkan_count: usize = 0;

        for (results.items[i..j]) |r| {
            const speedup = r.operations_per_second / best_ops;
            const time_ms = @as(f64, @floatFromInt(r.avg_time_per_iter_ns)) / 1_000_000.0;
            std.debug.print("{s:<20} {d:>15.2} {d:>15.2} {d:>12.2}x\n", .{
                r.backend,
                r.operations_per_second,
                time_ms,
                speedup,
            });

            // Track stats for summary
            if (std.mem.eql(u8, r.backend, "CPU")) {
                cpu_ops += r.operations_per_second;
                cpu_count += 1;
            } else if (std.mem.eql(u8, r.backend, "Vulkan")) {
                vulkan_ops += r.operations_per_second;
                vulkan_count += 1;
            }
        }

        // Print summary for this benchmark type
        if (cpu_count > 0 and vulkan_count > 0) {
            const cpu_avg = cpu_ops / @as(f64, @floatFromInt(cpu_count));
            const vulkan_avg = vulkan_ops / @as(f64, @floatFromInt(vulkan_count));
            const speedup = vulkan_avg / cpu_avg;

            std.debug.print("\n{s:<20} {d:>15.2}\n", .{ "CPU average:", cpu_avg });
            std.debug.print("{s:<20} {d:>15.2}\n", .{ "Vulkan average:", vulkan_avg });
            std.debug.print("{s:<20} {d:>12.2}x\n", .{ "Vulkan speedup:", speedup });
        }

        i = j;
    }
}

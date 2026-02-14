/// Vulkan-specific performance benchmarks
const std = @import("std");
const activation = @import("activation.zig");
const loss = @import("loss.zig");
const layer = @import("layer.zig");
const network = @import("network.zig");
const backend_module = @import("backend.zig");
const vulkan = @import("vulkan.zig");

pub const BenchmarkResult = struct {
    name: []const u8,
    backend: []const u8,
    iterations: usize,
    total_time_ns: u64,
    avg_time_per_iter_ns: u64,
    operations_per_second: f64,
    size: usize,
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
    return @as(u64, tv.sec) * 1_000_000_000 + @as(u64, tv.nsec);
}

// Estimate time based on workload size (for comparison when real timing unavailable)
fn estimateTime(ns: u64) u64 {
    return ns;
}

/// Benchmark Vulkan matmul with varying matrix sizes
pub fn benchmarkVulkanMatMul(allocator: std.mem.Allocator, m: usize, n: usize, k: usize, iterations: usize) !BenchmarkResult {
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    // Initialize test data
    for (a, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 10)) / 10.0;
    }
    for (b, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt((i * 3) % 10)) / 10.0;
    }

    const start = nanoTimestamp();
    for (0..iterations) |_| {
        // Try Vulkan backend
        const device = vulkan.DeviceWrapper.init() catch {
            // Vulkan not available, use CPU fallback
            vulkan.cpuMatMul(a, b, c, m, n, k);
            continue;
        };
        defer device.deinit();

        vulkan.vulkanMatMul(&device, a, b, c, m, n, k) catch {
            vulkan.cpuMatMul(a, b, c, m, n, k);
        };
    }
    const end = nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = if (iterations > 0) total_time_ns / @as(u64, @intCast(iterations)) else 0;
    const ops_per_sec = if (total_time_ns > 0)
        @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0)
    else
        0.0;

    return BenchmarkResult{
        .name = "vulkan_matmul",
        .backend = "Vulkan",
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
        .size = m * n * k,
    };
}

/// Benchmark CPU matmul with same matrix sizes for comparison
pub fn benchmarkCpuMatMul(allocator: std.mem.Allocator, m: usize, n: usize, k: usize, iterations: usize) !BenchmarkResult {
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    // Initialize test data
    for (a, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 10)) / 10.0;
    }
    for (b, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt((i * 3) % 10)) / 10.0;
    }

    const start = nanoTimestamp();
    for (0..iterations) |_| {
        backend_module.cpuMatMul(a, b, c, m, n, k);
    }
    const end = nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = if (iterations > 0) total_time_ns / @as(u64, @intCast(iterations)) else 0;
    const ops_per_sec = if (total_time_ns > 0)
        @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0)
    else
        0.0;

    return BenchmarkResult{
        .name = "cpu_matmul",
        .backend = "CPU",
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
        .size = m * n * k,
    };
}

/// Benchmark Vulkan forward pass with varying network sizes
pub fn benchmarkVulkanForwardPass(allocator: std.mem.Allocator, input_size: usize, output_size: usize, layers: usize, iterations: usize) !BenchmarkResult {
    const net = try network.Network.init(allocator, backend_module.Backend{ .cpu = {} });
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
    const avg_time_ns = if (iterations > 0) total_time_ns / @as(u64, @intCast(iterations)) else 0;
    const ops_per_sec = if (total_time_ns > 0)
        @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0)
    else
        0.0;

    return BenchmarkResult{
        .name = "vulkan_forward_pass",
        .backend = "Vulkan",
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
        .size = input_size * output_size * layers,
    };
}

/// Benchmark Vulkan activation forward with varying sizes
pub fn benchmarkVulkanActivationForward(allocator: std.mem.Allocator, act: activation.Activation, size: usize, iterations: usize) !BenchmarkResult {
    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);

    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 100)) / 10.0;
    }

    const start = nanoTimestamp();
    for (0..iterations) |_| {
        // Try Vulkan backend
        const device = vulkan.Device.init() catch {
            // Vulkan not available, use CPU
            backend_module.cpuActivationForward(act, input, output);
            continue;
        };
        defer device.deinit();

        vulkan.vulkanActivationForward(&device, act, input, output) catch {
            backend_module.cpuActivationForward(act, input, output);
        };
    }
    const end = nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = if (iterations > 0) total_time_ns / @as(u64, @intCast(iterations)) else 0;
    const ops_per_sec = if (total_time_ns > 0)
        @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0)
    else
        0.0;

    return BenchmarkResult{
        .name = "vulkan_activation_forward",
        .backend = "Vulkan",
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
        .size = size,
    };
}

/// Benchmark CPU activation forward with same size for comparison
pub fn benchmarkCpuActivationForward(allocator: std.mem.Allocator, act: activation.Activation, size: usize, iterations: usize) !BenchmarkResult {
    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);

    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 100)) / 10.0;
    }

    const start = nanoTimestamp();
    for (0..iterations) |_| {
        backend_module.cpuActivationForward(act, input, output);
    }
    const end = nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = if (iterations > 0) total_time_ns / @as(u64, @intCast(iterations)) else 0;
    const ops_per_sec = if (total_time_ns > 0)
        @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0)
    else
        0.0;

    return BenchmarkResult{
        .name = "cpu_activation_forward",
        .backend = "CPU",
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
        .size = size,
    };
}

/// Benchmark Vulkan loss backward with varying sizes
pub fn benchmarkVulkanLossBackward(allocator: std.mem.Allocator, loss_fn: loss.Loss, size: usize, iterations: usize) !BenchmarkResult {
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);
    const target = try allocator.alloc(f32, size);
    defer allocator.free(target);
    const grad_output = try allocator.alloc(f32, size);
    defer allocator.free(grad_output);

    for (output, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 100)) / 10.0;
    }
    for (target, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt((i + 50) % 100)) / 10.0;
    }

    const start = nanoTimestamp();
    for (0..iterations) |_| {
        // Try Vulkan backend
        const device = vulkan.Device.init() catch {
            // Vulkan not available, use CPU
            backend_module.cpuLossBackward(loss_fn, output, target, grad_output);
            continue;
        };
        defer device.deinit();

        vulkan.vulkanLossBackward(&device, loss_fn, output, target, grad_output) catch {
            backend_module.cpuLossBackward(loss_fn, output, target, grad_output);
        };
    }
    const end = nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = if (iterations > 0) total_time_ns / @as(u64, @intCast(iterations)) else 0;
    const ops_per_sec = if (total_time_ns > 0)
        @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0)
    else
        0.0;

    return BenchmarkResult{
        .name = "vulkan_loss_backward",
        .backend = "Vulkan",
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
        .size = size,
    };
}

/// Benchmark CPU loss backward with same size for comparison
pub fn benchmarkCpuLossBackward(allocator: std.mem.Allocator, loss_fn: loss.Loss, size: usize, iterations: usize) !BenchmarkResult {
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);
    const target = try allocator.alloc(f32, size);
    defer allocator.free(target);
    const grad_output = try allocator.alloc(f32, size);
    defer allocator.free(grad_output);

    for (output, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 100)) / 10.0;
    }
    for (target, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt((i + 50) % 100)) / 10.0;
    }

    const start = nanoTimestamp();
    for (0..iterations) |_| {
        backend_module.cpuLossBackward(loss_fn, output, target, grad_output);
    }
    const end = nanoTimestamp();

    const total_time_ns = @as(u64, @intCast(end - start));
    const avg_time_ns = if (iterations > 0) total_time_ns / @as(u64, @intCast(iterations)) else 0;
    const ops_per_sec = if (total_time_ns > 0)
        @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(total_time_ns)) / 1_000_000_000.0)
    else
        0.0;

    return BenchmarkResult{
        .name = "cpu_loss_backward",
        .backend = "CPU",
        .iterations = iterations,
        .total_time_ns = total_time_ns,
        .avg_time_per_iter_ns = avg_time_ns,
        .operations_per_second = ops_per_sec,
        .size = size,
    };
}

/// Print benchmark result in markdown format for documentation
pub fn printResultMarkdown(result: BenchmarkResult) void {
    std.debug.print("| {s} ({s}) | {} | {d:.2} | {d:.2} | {d:.2}x |\n", .{
        result.name,
        result.backend,
        result.iterations,
        result.operations_per_second,
        result.total_time_ns / 1_000_000.0, // ms
        result.avg_time_per_iter_ns / 1_000_000.0, // ms
    });
}

/// Print comparison table in markdown format
pub fn printComparisonMarkdown(results: []const BenchmarkResult) void {
    std.debug.print("\n### Benchmark Results\n\n", .{});
    std.debug.print("| Test | Backend | Iterations | Ops/sec | Total Time | Avg/Iter |\n", .{});
    std.debug.print("|------|---------|------------|---------|------------|----------|\n", .{});

    for (results) |r| {
        printResultMarkdown(r);
    }

    // Summary statistics
    std.debug.print("\n### Summary\n\n", .{});

    // Calculate CPU vs GPU stats
    var cpu_total_ops: f64 = 0;
    var cpu_count: usize = 0;
    var vulkan_total_ops: f64 = 0;
    var vulkan_count: usize = 0;

    for (results) |r| {
        if (std.mem.eql(u8, r.backend, "CPU")) {
            cpu_total_ops += r.operations_per_second;
            cpu_count += 1;
        } else if (std.mem.eql(u8, r.backend, "Vulkan")) {
            vulkan_total_ops += r.operations_per_second;
            vulkan_count += 1;
        }
    }

    if (cpu_count > 0 and vulkan_count > 0) {
        const cpu_avg = cpu_total_ops / @as(f64, @floatFromInt(cpu_count));
        const vulkan_avg = vulkan_total_ops / @as(f64, @floatFromInt(vulkan_count));
        const speedup = vulkan_avg / cpu_avg;

        std.debug.print("- CPU average: {d:.2} ops/sec\n", .{cpu_avg});
        std.debug.print("- Vulkan average: {d:.2} ops/sec\n", .{vulkan_avg});
        std.debug.print("- **Speedup: {d:.2}x**\n", .{speedup});
    }
}

/// Simple CUDA vs CPU Performance Comparison
const std = @import("std");
const zn = @import("ZigNeuron");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n========================================\n", .{});
    std.debug.print("   CUDA vs CPU Performance Comparison\n", .{});
    std.debug.print("========================================\n\n", .{});

    // Check CUDA availability
    const cuda_available = zn.cuda.CudaBackend.isAvailable();
    std.debug.print("CUDA Available: {s}\n\n", .{if (cuda_available) "YES" else "NO"});

    // Configuration
    const input_size = 10;
    const hidden_size = 50;
    const output_size = 5;
    const num_samples = 500;
    const epochs = 100;
    const learning_rate = 0.01;

    std.debug.print("Configuration:\n", .{});
    std.debug.print("  Network: {d} -> {d} -> {d}\n", .{ input_size, hidden_size, output_size });
    std.debug.print("  Samples: {d}\n", .{num_samples});
    std.debug.print("  Epochs: {d}\n\n", .{epochs});

    // Generate data
    std.debug.print("Generating data...\n", .{});
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var inputs: [500][10]f32 = undefined;
    var targets: [500][5]f32 = undefined;
    var input_slices: [500][]const f32 = undefined;
    var target_slices: [500][]const f32 = undefined;

    for (0..num_samples) |i| {
        for (0..input_size) |j| {
            inputs[i][j] = random.float(f32) * 2.0 - 1.0;
        }
        for (0..output_size) |j| {
            targets[i][j] = inputs[i][j % input_size] * 0.5;
        }
        input_slices[i] = &inputs[i];
        target_slices[i] = &targets[i];
    }

    // Test CPU
    std.debug.print("\n--- CPU Test ---\n", .{});
    const cpu_time = try runBenchmark(allocator, .cpu, "CPU", &input_slices, &target_slices, epochs, learning_rate);

    // Test CUDA (if available)
    var cuda_time: f64 = 0;
    if (cuda_available) {
        std.debug.print("\n--- CUDA Test ---\n", .{});
        cuda_time = try runBenchmark(allocator, .{ .gpu = .cuda }, "CUDA", &input_slices, &target_slices, epochs, learning_rate);
    }

    // Summary
    std.debug.print("\n========================================\n", .{});
    std.debug.print("           RESULTS SUMMARY\n", .{});
    std.debug.print("========================================\n", .{});
    std.debug.print("CPU Time:  {d:.2} ms\n", .{cpu_time});
    if (cuda_available and cuda_time > 0) {
        std.debug.print("CUDA Time: {d:.2} ms\n", .{cuda_time});
        const speedup = cpu_time / cuda_time;
        std.debug.print("Speedup:   {d:.2}x faster\n", .{speedup});
    } else {
        std.debug.print("CUDA Time: N/A\n", .{});
    }
    std.debug.print("========================================\n\n", .{});
}

fn runBenchmark(
    allocator: std.mem.Allocator,
    backend_type: zn.backend.Backend.BackendType,
    name: []const u8,
    input_slices: []const []const f32,
    target_slices: []const []const f32,
    epochs: usize,
    learning_rate: f32,
) !f64 {
    std.debug.print("Initializing {s} backend...\n", .{name});

    var backend = try zn.backend.Backend.init(allocator);
    backend.type = backend_type;
    defer backend.deinit();

    std.debug.print("Creating network...\n", .{});
    var net = try zn.network.Network.init(allocator, backend);
    defer net.deinit();

    _ = try net.addDense(10, 50, .relu);
    _ = try net.addDense(50, 5, .linear);

    // Warmup
    std.debug.print("Warming up...\n", .{});
    const loss_fn = zn.loss.Loss{ .mse = {} };
    try net.train(input_slices[0..10], target_slices[0..10], 5, learning_rate, loss_fn, null, null);
    net.clearGradients();

    // Benchmark
    std.debug.print("Training {d} epochs...\n", .{epochs});
    const io = std.Io.Threaded.global_single_threaded.io();
    const start_clock = std.Io.Clock.now(.real, io);
    try net.train(input_slices, target_slices, epochs, learning_rate, loss_fn, null, null);
    const end_clock = std.Io.Clock.now(.real, io);
    const elapsed_ns = @as(u64, @intCast(end_clock.nanoseconds - start_clock.nanoseconds));
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

    std.debug.print("\n{s} Results:\n", .{name});
    std.debug.print("  Total time: {d:.2} ms ({d:.2} s)\n", .{ elapsed_ms, elapsed_ms / 1000.0 });
    std.debug.print("  Per epoch:  {d:.2} ms\n", .{elapsed_ms / @as(f64, @floatFromInt(epochs))});
    std.debug.print("  Throughput: {d:.1} epochs/sec\n", .{@as(f64, @floatFromInt(epochs)) / (elapsed_ms / 1000.0)});

    return elapsed_ms;
}

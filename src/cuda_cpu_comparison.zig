/// CUDA vs CPU Performance Comparison
/// Compares neural network training performance between CUDA and CPU backends
const std = @import("std");
const zn = @import("ZigNeuron");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("\n=== CUDA vs CPU Performance Comparison ===\n\n", .{});

    // Check which backends are available
    std.debug.print("Detecting backends...\n", .{});
    const detected = zn.backend.Backend.detect();
    const cuda_available = zn.cuda.CudaBackend.isAvailable();
    std.debug.print("  CUDA Available: {}\n", .{cuda_available});
    std.debug.print("  Detected Backend: ", .{});
    switch (detected) {
        .gpu => |gpu| {
            switch (gpu) {
                .metal => std.debug.print("Metal GPU\n", .{}),
                .cuda => std.debug.print("CUDA GPU\n", .{}),
                .cpu => std.debug.print("CPU (GPU enum)\n", .{}),
            }
        },
        .cpu => std.debug.print("CPU\n", .{}),
        .multi_gpu => std.debug.print("Multi-GPU\n", .{}),
    }
    std.debug.print("\n", .{});

    // Test configuration
    const input_size = 10;
    const hidden_size = 50;
    const output_size = 5;
    const num_samples = 500;
    const epochs = 100;
    const learning_rate = 0.01;

    std.debug.print("Network Configuration:\n", .{});
    std.debug.print("  Architecture: {d} -> {d} -> {d}\n", .{ input_size, hidden_size, output_size });
    std.debug.print("  Samples: {d}\n", .{num_samples});
    std.debug.print("  Epochs: {d}\n", .{epochs});
    std.debug.print("  Learning Rate: {d:.3}\n\n", .{learning_rate});

    // Generate synthetic data
    std.debug.print("Generating synthetic data...\n", .{});
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
        // Simple function: sum of first 5 inputs
        for (0..output_size) |j| {
            targets[i][j] = inputs[i][j % input_size] * 0.5 + inputs[i][(j + 1) % input_size] * 0.5;
        }
        input_slices[i] = &inputs[i];
        target_slices[i] = &targets[i];
    }
    std.debug.print("Data ready.\n\n", .{});

    // Test 1: CPU Backend
    std.debug.print("--- Test 1: CPU Backend ---\n", .{});
    var cpu_backend = try zn.backend.Backend.init(allocator);
    cpu_backend.type = .cpu; // Force CPU mode
    try benchmarkTraining(allocator, cpu_backend, "CPU", &input_slices, &target_slices, epochs, learning_rate);
    cpu_backend.deinit();

    // Test 2: CUDA Backend (if available)
    if (cuda_available) {
        std.debug.print("\n--- Test 2: CUDA Backend ---\n", .{});
        var cuda_backend = try zn.backend.Backend.init(allocator);
        // CUDA should be automatically detected and initialized
        try benchmarkTraining(allocator, cuda_backend, "CUDA", &input_slices, &target_slices, epochs, learning_rate);
        cuda_backend.deinit();

        // Print speedup summary
        std.debug.print("\n=== Speedup Summary ===\n", .{});
        // Note: We don't have the times stored, but user can compare manually
        std.debug.print("Compare the times above to see CUDA speedup!\n", .{});
    } else {
        std.debug.print("\n--- CUDA Not Available ---\n", .{});
        std.debug.print("Skipping CUDA tests (no NVIDIA GPU detected)\n", .{});
    }

    std.debug.print("\n=== All tests completed ===\n", .{});
}

fn benchmarkTraining(
    allocator: std.mem.Allocator,
    backend: zn.backend.Backend,
    name: []const u8,
    input_slices: []const []const f32,
    target_slices: []const []const f32,
    epochs: usize,
    learning_rate: f32,
) !void {
    std.debug.print("Backend: {s}\n", .{name});

    // Create network
    var net = try zn.network.Network.init(allocator, backend);
    defer net.deinit();

    // Build network: 10 -> 50 -> 5
    _ = try net.addDense(10, 50, .relu);
    _ = try net.addDense(50, 5, .linear);

    // Warmup
    std.debug.print("  Warmup...\n", .{});
    const loss_fn = zn.loss.Loss{ .mse = {} };
    try net.train(input_slices[0..10], target_slices[0..10], 5, learning_rate, loss_fn, null, null);

    // Clear gradients
    net.clearGradients();

    // Benchmark
    std.debug.print("  Training {d} epochs...\n", .{epochs});
    const timer = try std.time.Timer.start();
    try net.train(input_slices, target_slices, epochs, learning_rate, loss_fn, null, null);
    const elapsed_ns = timer.read();

    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const ms_per_epoch = elapsed_ms / @as(f64, @floatFromInt(epochs));

    std.debug.print("  Results:\n", .{});
    std.debug.print("    Total time: {d:.2} ms ({d:.2} s)\n", .{ elapsed_ms, elapsed_ms / 1000.0 });
    std.debug.print("    Time per epoch: {d:.3} ms\n", .{ms_per_epoch});
    std.debug.print("    Epochs per second: {d:.1}\n", .{@as(f64, @floatFromInt(epochs)) / (elapsed_ms / 1000.0)});
}

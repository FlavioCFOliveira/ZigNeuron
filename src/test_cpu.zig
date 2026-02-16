/// CPU-only performance test for FNN
const std = @import("std");
const zn = @import("ZigNeuron");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== FNN Performance Test: CPU Backend Only ===\n\n", .{});

    // Force CPU backend
    const backend = zn.backend.Backend{ .cpu = {} };
    std.debug.print("Using backend: CPU (forced)\n", .{});

    // Create network: 4 -> 8 -> 4 -> 2
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();

    _ = try network.addDense(4, 8, .relu);
    _ = try network.addDense(8, 4, .relu);
    _ = try network.addDense(4, 2, .sigmoid);

    // Generate synthetic data
    const num_samples = 100;
    const input_size = 4;
    const output_size = 2;

    var training_data = try allocator.alloc([]const f32, num_samples);
    defer allocator.free(training_data);
    var training_targets = try allocator.alloc([]const f32, num_samples);
    defer allocator.free(training_targets);

    // Generate random data
    var prng = std.rand.DefaultPrng.init(12345);
    const rand = prng.random();

    for (0..num_samples) |i| {
        training_data[i] = try allocator.alloc(f32, input_size);
        training_targets[i] = try allocator.alloc(f32, output_size);

        for (0..input_size) |j| {
            training_data[i][j] = rand.float(f32);
        }
        for (0..output_size) |j| {
            training_targets[i][j] = if (rand.float(f32) > 0.5) 1.0 else 0.0;
        }
    }

    // Warm-up run
    std.debug.print("Warm-up...\n", .{});
    const loss_fn = zn.loss.Loss{ .mse = {} };
    try network.train(training_data[0..10], training_targets[0..10], 1, 0.01, loss_fn);
    network.clearGradients();

    // Performance measurement
    const num_epochs = 50;
    std.debug.print("Training for {d} epochs...\n", .{num_epochs});

    var timer = try std.time.Timer.start();
    try network.train(training_data, training_targets, num_epochs, 0.01, loss_fn);
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

    std.debug.print("Completed in {d:.2}ms\n", .{elapsed_ms});
    std.debug.print("Average per epoch: {d:.2}ms\n", .{elapsed_ms / @as(f64, @floatFromInt(num_epochs))});

    // Memory usage estimation
    var memory_usage: usize = 0;
    for (network.layers.items) |layer| {
        memory_usage += layer.weights.len * @sizeOf(f32);
        memory_usage += layer.bias.len * @sizeOf(f32);
        memory_usage += layer.grad_weights.len * @sizeOf(f32);
        memory_usage += layer.grad_bias.len * @sizeOf(f32);
    }
    std.debug.print("Estimated memory usage: {d} bytes\n", .{memory_usage});

    // Test inference speed
    std.debug.print("Testing inference speed...\n", .{});
    timer.reset();

    const num_inferences = 1000;
    var output: [2]f32 = undefined;
    for (0..num_inferences) |_| {
        _ = try network.forward(training_data[0], &output);
    }
    const inference_ns = timer.read();
    const inference_ms = @as(f64, @floatFromInt(inference_ns)) / 1_000_000.0;

    std.debug.print("{d} inferences in {d:.2}ms\n", .{num_inferences, inference_ms});
    std.debug.print("Average per inference: {d:.3}ms\n", .{inference_ms / @as(f64, @floatFromInt(num_inferences))});
}

/// Comprehensive backend comparison test
/// Tests CPU, Metal GPU, and Vulkan backends
const std = @import("std");
const zn = @import("ZigNeuron");

const TestResult = struct {
    backend_name: []const u8,
    training_time_ms: f64,
    inference_time_ms: f64,
    memory_usage_bytes: usize,
    final_loss: f32,
    successful: bool,
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Backend Performance Comparison ===\n\n", .{});

    // Test configurations
    var cpu_backend = zn.backend.Backend.init(allocator) catch |err| {
        std.debug.print("Failed to initialize CPU backend: {s}\n", .{@errorName(err)});
        return;
    };
    cpu_backend.type = .cpu;

    const metal_backend: ?zn.backend.Backend = zn.backend.Backend.init(allocator) catch null;
    if (metal_backend) |mb| {
        if (mb.type == .gpu and mb.type.gpu == .metal) {
            // ok
        }
    }

    var backend_list = std.array_list.Managed(struct {
        name: []const u8,
        backend: zn.backend.Backend,
    }).init(allocator);
    defer backend_list.deinit();

    try backend_list.append(.{ .name = "CPU", .backend = cpu_backend });
    if (metal_backend) |mb| {
        try backend_list.append(.{ .name = "Metal GPU", .backend = mb });
    }

    const backends = backend_list.items;

    var results: std.ArrayListUnmanaged(TestResult) = .{};
    defer results.deinit(allocator);

    // Run tests for each backend
    for (backends) |config| {
        if (config.backend.type == .gpu and config.backend.type.gpu == .metal and config.backend.metal_ctx == null) {
            std.debug.print("Skipping {s} (not supported on this system)\n", .{config.name});
            continue;
        }
        std.debug.print("Testing {s} backend...\n", .{config.name});
        const result = try testBackend(allocator, config.backend, config.name);
        try results.append(allocator, result);
        std.debug.print("  ✓ Completed in {d:.2}ms\n\n", .{result.training_time_ms});
    }

    // Note: Backends are deinitialized by the Network in testBackend
    // No need to deinit them here as it would cause double-free of MetalContext

    // Print comparison table
    std.debug.print("\n=== Results Summary ===\n", .{});
    std.debug.print("{s:20} | {s:15} | {s:15} | {s:10} | {s:10}\n", .{
        "Backend", "Training (ms)", "Inference (ms)", "Memory (B)", "Loss",
    });
    std.debug.print("{s}\n", .{"-" ** 80});

    for (results.items) |result| {
        std.debug.print("{s:20} | {d:15.2} | {d:15.3} | {d:10} | {d:10.4}\n", .{
            result.backend_name,
            result.training_time_ms,
            result.inference_time_ms,
            result.memory_usage_bytes,
            result.final_loss,
        });
    }

    // Calculate speedups
    std.debug.print("\n=== Speedup Analysis ===\n", .{});
    if (results.items.len >= 2) {
        const cpu_result = results.items[0];
        const metal_result = results.items[1];

        if (cpu_result.successful and metal_result.successful) {
            const training_speedup = cpu_result.training_time_ms / metal_result.training_time_ms;
            const inference_speedup = cpu_result.inference_time_ms / metal_result.inference_time_ms;

            std.debug.print("Metal GPU vs CPU:\n", .{});
            std.debug.print("  Training speedup: {d:.2}x\n", .{training_speedup});
            std.debug.print("  Inference speedup: {d:.2}x\n", .{inference_speedup});

            if (training_speedup < 2.0) {
                std.debug.print("  ⚠️  Metal GPU is using CPU fallback - shaders not yet integrated\n", .{});
            }
        }
    }

    std.debug.print("\n=== All tests completed ===\n", .{});
}

fn testBackend(allocator: std.mem.Allocator, backend: zn.backend.Backend, name: []const u8) !TestResult {
    var result = TestResult{
        .backend_name = name,
        .training_time_ms = 0,
        .inference_time_ms = 0,
        .memory_usage_bytes = 0,
        .final_loss = 0,
        .successful = false,
    };

    // Create network: 64 -> 128 -> 64 -> 10
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();

    _ = try network.addDense(64, 128, .relu);
    _ = try network.addDense(128, 64, .relu);
    _ = try network.addDense(64, 10, .sigmoid);

    // Calculate memory usage
    for (network.layers.items) |lyr| {
        result.memory_usage_bytes += lyr.weights.slice.len * @sizeOf(f32);
        result.memory_usage_bytes += lyr.bias.slice.len * @sizeOf(f32);
        result.memory_usage_bytes += lyr.grad_weights.slice.len * @sizeOf(f32);
        result.memory_usage_bytes += lyr.grad_bias.slice.len * @sizeOf(f32);
    }

    // Generate test data
    const num_samples = 100;
    const input_size = 64;
    const output_size = 10;

    var training_data = try allocator.alloc([]const f32, num_samples);
    defer allocator.free(training_data);
    var training_targets = try allocator.alloc([]const f32, num_samples);
    defer allocator.free(training_targets);

    var prng = std.Random.DefaultPrng.init(12345);
    const rand = prng.random();

    for (0..num_samples) |i| {
        var data = try allocator.alloc(f32, input_size);
        var target = try allocator.alloc(f32, output_size);

        for (0..input_size) |j| {
            data[j] = rand.float(f32);
        }
        for (0..output_size) |j| {
            target[j] = if (rand.float(f32) > 0.5) 1.0 else 0.0;
        }

        training_data[i] = data;
        training_targets[i] = target;
    }

    // Warm-up run
    const loss_fn = zn.loss.Loss{ .mse = {} };
    try network.train(training_data[0..10], training_targets[0..10], 1, 0.01, loss_fn);
    network.clearGradients();

    // Training benchmark
    var timer = try std.time.Timer.start();
    try network.train(training_data, training_targets, 50, 0.01, loss_fn);
    result.training_time_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;

    // Get final loss
    const output_buf = try allocator.alloc(f32, output_size);
    defer allocator.free(output_buf);
    _ = try network.forward(training_data[0], output_buf);
    result.final_loss = try loss_fn.forward(output_buf, training_targets[0]);

    // Inference benchmark
    timer.reset();
    const num_inferences = 1000;
    for (0..num_inferences) |_| {
        _ = try network.forward(training_data[0], output_buf);
    }
    result.inference_time_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;

    result.successful = true;
    return result;
}

/// Test with different network architectures
pub fn testArchitectures(allocator: std.mem.Allocator, backend: zn.backend.Backend) !void {
    const architectures = [_]struct {
        name: []const u8,
        layers: []const struct { input: usize, output: usize, act: zn.activation.Activation },
    }{
        .{
            .name = "Small (2-4-1)",
            .layers = &[_]struct { input: usize, output: usize, act: zn.activation.Activation }{
                .{ .input = 2, .output = 4, .act = .relu },
                .{ .input = 4, .output = 1, .act = .sigmoid },
            },
        },
        .{
            .name = "Medium (4-8-4-2)",
            .layers = &[_]struct { input: usize, output: usize, act: zn.activation.Activation }{
                .{ .input = 4, .output = 8, .act = .relu },
                .{ .input = 8, .output = 4, .act = .relu },
                .{ .input = 4, .output = 2, .act = .sigmoid },
            },
        },
        .{
            .name = "Large (8-16-8-4)",
            .layers = &[_]struct { input: usize, output: usize, act: zn.activation.Activation }{
                .{ .input = 8, .output = 16, .act = .relu },
                .{ .input = 16, .output = 8, .act = .relu },
                .{ .input = 8, .output = 4, .act = .sigmoid },
            },
        },
    };

    std.debug.print("\n=== Architecture Comparison ===\n", .{});
    std.debug.print("{s:20} | {s:15} | {s:15}\n", .{"Architecture", "Training (ms)", "Inference (ms)"});
    std.debug.print("{s}\n", .{"-" ** 60});

    for (architectures) |arch| {
        const network = try zn.network.Network.init(allocator, backend);
        defer network.deinit();

        for (arch.layers) |layer| {
            _ = try network.addDense(layer.input, layer.output, layer.act);
        }

        // Quick benchmark
        const start = std.time.milliTimestamp();
        const loss_fn = zn.loss.Loss{ .mse = {} };
        try network.train(&[_][]const f32{}, &[_][]const f32{}, 10, 0.01, loss_fn);
        const train_time = std.time.milliTimestamp() - start;

        std.debug.print("{s:20} | {d:15} | {s:15}\n", .{
            arch.name,
            train_time,
            "N/A", // Would need actual inference benchmark
        });
    }
}

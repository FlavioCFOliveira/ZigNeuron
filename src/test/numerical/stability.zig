/// Numerical stability tests

const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const layer = zn.layer;
const activation = zn.activation;
const loss = zn.loss;
const network = zn.network;
const backend = zn.backend;

fn expectNear(actual: f32, expected: f32, tolerance: f32) !void {
    const diff = if (actual > expected) actual - expected else expected - actual;
    if (diff > tolerance) return error.ApproximationFailed;
}

test "numerical stability: very small inputs" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 1e-10 },
        &.{ 1e-8 },
        &.{ 1e-6 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1e-10 },
        &.{ 1e-8 },
        &.{ 1e-6 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 5e-7 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical stability: very large inputs" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 1e10 },
        &.{ 1e12 },
        &.{ 1e14 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1e10 },
        &.{ 1e12 },
        &.{ 1e14 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.0000001, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 5e13 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical stability: extreme learning rates" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    // Very high learning rate
    const net1 = try network.Network.init(allocator, be);
    defer net1.deinit();
    _ = try net1.addDense(1, 4, .relu);
    _ = try net1.addDense(4, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Very high learning rate
    try net1.train(training_data, training_targets, 100, 1.0, loss_fn);

    var output1: [1]f32 = undefined;
    _ = try net1.forward(&.{ 0.5 }, &output1);
    try testing.expect(std.math.isFinite(output1[0]));

    // Very low learning rate
    const net2 = try network.Network.init(allocator, be);
    defer net2.deinit();
    _ = try net2.addDense(1, 4, .relu);
    _ = try net2.addDense(4, 1, .linear);

    try net2.train(training_data, training_targets, 100, 1e-10, loss_fn);

    var output2: [1]f32 = undefined;
    _ = try net2.forward(&.{ 0.5 }, &output2);
    try testing.expect(std.math.isFinite(output2[0]));
}

test "numerical stability: activation functions at extremes" {
    // ReLU at negative values
    const relu = activation.Activation{ .relu = {} };
    try expectNear(relu.forward(-1000.0), 0.0, 0.0001);
    try expectNear(relu.forward(1000.0), 1000.0, 0.0001);

    // Sigmoid at extremes
    const sigmoid = activation.Activation{ .sigmoid = {} };
    try expectNear(sigmoid.forward(-1000.0), 0.0, 0.0001);
    try expectNear(sigmoid.forward(1000.0), 1.0, 0.0001);

    // Tanh at extremes
    const tanh = activation.Activation{ .tanh = {} };
    try expectNear(tanh.forward(-1000.0), -1.0, 0.0001);
    try expectNear(tanh.forward(1000.0), 1.0, 0.0001);
}

test "numerical stability: derivative at extremes" {
    // ReLU derivative
    const relu = activation.Activation{ .relu = {} };
    try expectNear(relu.backward(-1000.0, 1.0), 0.0, 0.0001);  // derivative is 0 for negative
    try expectNear(relu.backward(1000.0, 1.0), 1.0, 0.0001);   // derivative is 1 for positive

    // Sigmoid derivative at extremes
    const sigmoid = activation.Activation{ .sigmoid = {} };
    try expectNear(sigmoid.backward(-1000.0, 1.0), 0.0, 0.0001);  // derivative approaches 0
    try expectNear(sigmoid.backward(1000.0, 1.0), 0.0, 0.0001);   // derivative approaches 0
}

test "numerical stability: MSE loss at extremes" {
    const loss_fn = loss.Loss{ .mse = {} };

    // Small values
    var output_small: [2]f32 = .{ 1e-10, 1e-10 };
    var target_small: [2]f32 = .{ 0.0, 0.0 };
    const loss_small = try loss_fn.forward(&output_small, &target_small);
    try testing.expect(std.math.isFinite(loss_small));
    try testing.expect(loss_small >= 0);

    // Large values
    var output_large: [2]f32 = .{ 1e10, 1e10 };
    var target_large: [2]f32 = .{ 0.0, 0.0 };
    const loss_large = try loss_fn.forward(&output_large, &target_large);
    try testing.expect(std.math.isFinite(loss_large));
    try testing.expect(loss_large >= 0);
}

test "numerical stability: gradient at extremes" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Large input
    const input: []const f32 = &.{ 1e5 };
    const target: []const f32 = &.{ 1e5 };
    const loss_fn = loss.Loss{ .mse = {} };

    const loss_value = try net.trainStep(input, target, 0.000001, loss_fn);
    try testing.expect(std.math.isFinite(loss_value));

    // Check gradients are finite
    for (net.layers.items) |lyr| {
        for (lyr.grad_weights.slice) |g| {
            try testing.expect(std.math.isFinite(g));
        }
    }
}

test "numerical stability: weight initialization range" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    // Test with various weight scales
    for ([_]f32{ 1e-6, 1e-3, 1.0, 1e3, 1e6 }) |scale| {
        const net = try network.Network.init(allocator, be);
        defer net.deinit();

        _ = try net.addDense(1, 4, .relu);
        _ = try net.addDense(4, 1, .linear);

        // Scale the initial weights
        for (net.layers.items) |lyr| {
            for (lyr.weights.slice) |*w| {
                w.* *= scale;
            }
        }

        const training_data = &[_][]const f32{
            &.{ 0.0 },
            &.{ 1.0 },
        };
        const training_targets = &[_][]const f32{
            &.{ 0.0 },
            &.{ 1.0 },
        };

        const loss_fn = loss.Loss{ .mse = {} };
        try net.train(training_data, training_targets, 100, 0.01, loss_fn);

        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.5 }, &output);
        try testing.expect(std.math.isFinite(output[0]));
    }
}

test "numerical stability: batch training with large batches" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Create a large batch
    var training_data: [100][]const f32 = undefined;
    var training_targets: [100][]const f32 = undefined;

    for (0..100) |i| {
        training_data[i] = &.{ @as(f32, @floatFromInt(i)) / 50.0 };
        training_targets[i] = &.{ training_data[i][0] * training_data[i][0] };
    }

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(&training_data, &training_targets, 10, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical stability: multi-layer stability" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Deep network: 1-32-32-32-32-1
    _ = try net.addDense(1, 32, .relu);
    _ = try net.addDense(32, 32, .relu);
    _ = try net.addDense(32, 32, .relu);
    _ = try net.addDense(32, 32, .relu);
    _ = try net.addDense(32, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ -1.0 },
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1.0 },
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);
    try testing.expect(std.math.isFinite(output[0]));

    // All weights should be finite
    for (net.layers.items) |lyr| {
        for (lyr.weights.slice) |w| {
            try testing.expect(std.math.isFinite(w));
        }
        for (lyr.bias.slice) |b| {
            try testing.expect(std.math.isFinite(b));
        }
    }
}

test "numerical stability: NaN propagation prevention" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Train and check for NaN at each step
    for (0..100) |_| {
        for (training_data, training_targets) |data, target| {
            const loss_value = try net.trainStep(data, target, 0.01, loss_fn);
            try testing.expect(std.math.isFinite(loss_value));
        }
    }

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);
    try testing.expect(!std.math.isNan(output[0]));
}

test "numerical stability: Inf propagation prevention" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Train
    try net.train(training_data, training_targets, 100, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);

    try testing.expect(!std.math.isInf(output[0]));
}

test "numerical stability: gradient clipping simulation" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Train with reasonable learning rate
    try net.train(training_data, training_targets, 200, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);

    // Output should be valid
    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical stability: very shallow network" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Minimal: 1-1-1
    _ = try net.addDense(1, 1, .relu);
    _ = try net.addDense(1, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical stability: multiple input dimensions" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // 10 inputs
    _ = try net.addDense(10, 16, .relu);
    _ = try net.addDense(16, 1, .linear);

    // Training data with 10-dimensional inputs
    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        &.{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 10.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical stability: precision under multiple operations" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Run many training steps and check numerical drift
    var initial_weight_sum: f32 = 0;
    var final_weight_sum: f32 = 0;

    // Record initial weights
    for (net.layers.items) |lyr| {
        for (lyr.weights.slice) |w| {
            initial_weight_sum += w;
        }
    }

    // Train for many epochs
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    // Record final weights
    for (net.layers.items) |lyr| {
        for (lyr.weights.slice) |w| {
            final_weight_sum += w;
        }
    }

    // Weights should have changed but be finite
    try testing.expect(std.math.isFinite(final_weight_sum));
    try testing.expect(final_weight_sum != initial_weight_sum);
}

test "numerical stability: forward pass consistency" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.01, loss_fn);

    // Run many forward passes - should get same results
    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;
    var output3: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.5 }, &output1);
    _ = try net.forward(&.{ 0.5 }, &output2);
    _ = try net.forward(&.{ 0.5 }, &output3);

    try expectNear(output1[0], output2[0], 0.0001);
    try expectNear(output2[0], output3[0], 0.0001);
}

test "numerical stability: backward pass consistency" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Run multiple backward passes
    var initial_loss: f32 = 0;
    var second_loss: f32 = 0;

    initial_loss = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);
    second_loss = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);

    try testing.expect(std.math.isFinite(initial_loss));
    try testing.expect(std.math.isFinite(second_loss));
    try testing.expect(second_loss <= initial_loss);
}

test "numerical stability: layerwise gradients" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 4, .relu);
    _ = try net.addDense(4, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    const input: []const f32 = &.{ 1.0 };
    const target: []const f32 = &.{ 1.0 };
    const loss_fn = loss.Loss{ .mse = {} };

    _ = try net.trainStep(input, target, 0.1, loss_fn);

    // Check that gradients exist and are finite for all layers
    for (net.layers.items) |lyr| {
        for (lyr.grad_weights.slice) |g| {
            try testing.expect(std.math.isFinite(g));
        }
        for (lyr.grad_bias.slice) |g| {
            try testing.expect(std.math.isFinite(g));
        }
    }
}

test "numerical stability: loss function边界" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Large loss scenario
    const training_data = &[_][]const f32{
        &.{ 0.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1e6 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Train with very small learning rate to avoid overflow
    try net.train(training_data, training_targets, 100, 1e-10, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical stability: training with constant loss" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Perfectly learnable data
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Train until loss is very small
    var loss_val: f32 = 1.0;
    for (0..1000) |_| {
        loss_val = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);
        if (loss_val < 1e-10) break;
    }

    // Loss should be very small but finite
    try testing.expect(loss_val < 1e-6);
    try testing.expect(std.math.isFinite(loss_val));
}

test "numerical stability: exponential scale inputs" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Inputs spanning many orders of magnitude
    const training_data = &[_][]const f32{
        &.{ 1e-5 },
        &.{ 1e-3 },
        &.{ 1e-1 },
        &.{ 1e1 },
        &.{ 1e3 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1e-5 },
        &.{ 1e-3 },
        &.{ 1e-1 },
        &.{ 1e1 },
        &.{ 1e3 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1e2 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

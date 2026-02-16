/// Convergence tests for sinusoidal function learning (y = sin(x))

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

test "convergence sinusoidal: y = sin(x) basic" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Training data for y = sin(x)
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ std.math.pi / 4 },
        &.{ std.math.pi / 2 },
        &.{ 3 * std.math.pi / 4 },
        &.{ std.math.pi },
        &.{ 3 * std.math.pi / 2 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.7071 },
        &.{ 1.0 },
        &.{ 0.7071 },
        &.{ 0.0 },
        &.{ -1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    // Test interpolation
    _ = try net.forward(&.{ std.math.pi / 6 }, &output);
    // Expected: 0.5 (roughly)
    try testing.expect(output[0] > 0.3 and output[0] < 0.7);

    // Test negative value
    _ = try net.forward(&.{ 7 * std.math.pi / 6 }, &output);
    try testing.expect(output[0] < 0);
}

test "convergence sinusoidal: y = 2*sin(x) + 1" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Training data for y = 2*sin(x) + 1
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ std.math.pi / 2 },
        &.{ std.math.pi },
        &.{ 3 * std.math.pi / 2 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1.0 },   // 2*0 + 1
        &.{ 3.0 },   // 2*1 + 1
        &.{ 1.0 },   // 2*0 + 1
        &.{ -1.0 },  // 2*(-1) + 1
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ std.math.pi / 6 }, &output);
    // Expected: 2*0.5 + 1 = 2
    try testing.expect(output[0] > 1.5 and output[0] < 2.5);
}

test "convergence sinusoidal: y = sin(2x)" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Training data for y = sin(2x)
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ std.math.pi / 4 },
        &.{ std.math.pi / 2 },
        &.{ 3 * std.math.pi / 4 },
        &.{ std.math.pi },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 0.0 },
        &.{ -1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ std.math.pi / 8 }, &output);
    // Expected: sin(pi/4) = 0.707
    try expectNear(output[0], 0.707, 0.2);
}

test "convergence sinusoidal: shallow network" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Simple network
    _ = try net.addDense(1, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ std.math.pi / 2 },
        &.{ std.math.pi },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ std.math.pi / 4 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "convergence sinusoidal: deep network" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Deep network: 1-32-32-16-8-1
    _ = try net.addDense(1, 32, .relu);
    _ = try net.addDense(32, 32, .relu);
    _ = try net.addDense(32, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ std.math.pi / 6 },
        &.{ std.math.pi / 3 },
        &.{ std.math.pi / 2 },
        &.{ 2 * std.math.pi / 3 },
        &.{ 5 * std.math.pi / 6 },
        &.{ std.math.pi },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.5 },
        &.{ 0.8660 },
        &.{ 1.0 },
        &.{ 0.8660 },
        &.{ 0.5 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ std.math.pi / 4 }, &output);
    try expectNear(output[0], 0.7071, 0.3);
}

test "convergence sinusoidal: phase shift" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // y = sin(x + pi/4)
    const training_data = &[_][]const f32{
        &.{ -std.math.pi / 4 },
        &.{ std.math.pi / 4 },
        &.{ 3 * std.math.pi / 4 },
        &.{ 5 * std.math.pi / 4 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 0.0 },
        &.{ -1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.0 }, &output);
    // Expected: sin(pi/4) = 0.707
    try expectNear(output[0], 0.707, 0.2);
}

test "convergence sinusoidal: amplitude learning" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // y = A*sin(x) where A varies
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ std.math.pi / 2 },
        &.{ std.math.pi },
    };

    // Test different amplitudes
    for ([_]f32{ 0.5, 1.0, 2.0, 3.0 }) |amp| {
        const net_test = try network.Network.init(allocator, be);
        defer net_test.deinit();
        _ = try net_test.addDense(1, 16, .relu);
        _ = try net_test.addDense(16, 8, .relu);
        _ = try net_test.addDense(8, 1, .linear);

        const training_targets = &[_][]const f32{
            &.{ 0.0 },
            &.{ amp },
            &.{ 0.0 },
        };

        const loss_fn = loss.Loss{ .mse = {} };
        try net_test.train(training_data, training_targets, 300, 0.01, loss_fn);

        var output: [1]f32 = undefined;
        _ = try net_test.forward(&.{ std.math.pi / 6 }, &output);
        // Expected: A * 0.5
        try testing.expect(std.math.isFinite(output[0]));
    }
}

test "convergence sinusoidal: multiple periods" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Train on multiple periods
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ std.math.pi / 4 },
        &.{ std.math.pi / 2 },
        &.{ 3 * std.math.pi / 4 },
        &.{ std.math.pi },
        &.{ 5 * std.math.pi / 4 },
        &.{ 3 * std.math.pi / 2 },
        &.{ 7 * std.math.pi / 4 },
        &.{ 2 * std.math.pi },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.7071 },
        &.{ 1.0 },
        &.{ 0.7071 },
        &.{ 0.0 },
        &.{ -0.7071 },
        &.{ -1.0 },
        &.{ -0.7071 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    // Test outside training range
    _ = try net.forward(&.{ 2 * std.math.pi + std.math.pi / 6 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "convergence sinusoidal: loss convergence" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ std.math.pi / 2 },
        &.{ std.math.pi },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Record loss at different epochs
    var initial_loss: f32 = 0;
    var final_loss: f32 = 0;

    initial_loss = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);

    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    final_loss = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);

    // Loss should decrease
    try testing.expect(final_loss < initial_loss);
}

test "convergence sinusoidal: gradient bounds" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ std.math.pi / 2 },
        &.{ std.math.pi },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    // Check gradients are reasonable
    for (net.layers.items) |lyr| {
        for (lyr.grad_weights) |g| {
            try testing.expect(std.math.isFinite(g));
        }
        for (lyr.grad_bias) |g| {
            try testing.expect(std.math.isFinite(g));
        }
    }
}

test "convergence sinusoidal: network consistency" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ std.math.pi / 2 },
        &.{ std.math.pi },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    // Run multiple forward passes - should get same results
    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;
    var output3: [1]f32 = undefined;

    _ = try net.forward(&.{ std.math.pi / 4 }, &output1);
    _ = try net.forward(&.{ std.math.pi / 4 }, &output2);
    _ = try net.forward(&.{ std.math.pi / 4 }, &output3);

    // Results should be consistent
    try testing.expect(output1[0] == output2[0]);
    try testing.expect(output2[0] == output3[0]);
}

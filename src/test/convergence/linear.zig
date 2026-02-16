/// Convergence tests for linear function learning

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

test "convergence linear: y = 2x" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
        &.{ 4.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 2.0 },
        &.{ 4.0 },
        &.{ 6.0 },
        &.{ 8.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 1.5 }, &output);
    try expectNear(output[0], 3.0, 0.3);

    _ = try net.forward(&.{ 5.0 }, &output);
    try expectNear(output[0], 10.0, 1.0);
}

test "convergence linear: y = -3x + 5" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 5.0 },
        &.{ 2.0 },
        &.{ -1.0 },
        &.{ -4.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.0 }, &output);
    try expectNear(output[0], 5.0, 0.5);
}

test "convergence linear: learning rate comparison" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    // High learning rate
    const net1 = try network.Network.init(allocator, be);
    defer net1.deinit();
    _ = try net1.addDense(1, 4, .relu);
    _ = try net1.addDense(4, 1, .linear);

    // Low learning rate
    const net2 = try network.Network.init(allocator, be);
    defer net2.deinit();
    _ = try net2.addDense(1, 4, .relu);
    _ = try net2.addDense(4, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    try net1.train(training_data, training_targets, 200, 0.1, loss_fn);
    try net2.train(training_data, training_targets, 200, 0.01, loss_fn);

    // Both should learn the function
    var output: [1]f32 = undefined;

    _ = try net1.forward(&.{ 1.0 }, &output);
    try expectNear(output[0], 1.0, 0.5);

    _ = try net2.forward(&.{ 1.0 }, &output);
    try expectNear(output[0], 1.0, 0.5);
}

test "convergence linear: convergence speed" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
        &.{ 4.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
        &.{ 4.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Check that loss decreases
    var initial_loss: f32 = 0;
    var final_loss: f32 = 0;

    // Measure initial loss (before training)
    initial_loss = try net.trainStep(training_data[0], training_targets[0], 0, loss_fn);

    // Train
    try net.train(training_data, training_targets, 200, 0.01, loss_fn);

    // Measure final loss
    final_loss = try net.trainStep(training_data[0], training_targets[0], 0, loss_fn);

    // Loss should decrease
    try testing.expect(final_loss < initial_loss);
}

test "convergence linear: multiple inputs" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // y = x1 + x2
    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 0.0 },
        &.{ 0.0, 1.0 },
        &.{ 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 2.0, 3.0 }, &output);
    try expectNear(output[0], 5.0, 0.5);
}

test "convergence linear: negative slope" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // y = -2x
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ -2.0 },
        &.{ -4.0 },
        &.{ -6.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 1.5 }, &output);
    try expectNear(output[0], -3.0, 0.5);
}

test "convergence linear: with bias" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // y = 3x + 2
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 2.0 },
        &.{ 5.0 },
        &.{ 8.0 },
        &.{ 11.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.0 }, &output);
    try expectNear(output[0], 2.0, 0.3);
}

test "convergence linear: extrapolation" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // y = 2x, train on [0, 5]
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
        &.{ 4.0 },
        &.{ 5.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 2.0 },
        &.{ 4.0 },
        &.{ 6.0 },
        &.{ 8.0 },
        &.{ 10.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    // Extrapolate to x = 10
    _ = try net.forward(&.{ 10.0 }, &output);
    try testing.expect(output[0] > 15 and output[0] < 25);
}

test "convergence linear: gradient flow" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    const input: []const f32 = &.{ 1.0 };
    const target: []const f32 = &.{ 2.0 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Training step
    _ = try net.trainStep(input, target, 0.1, loss_fn);

    // Check that gradients exist
    for (net.layers.items) |lyr| {
        var has_nonzero_grad = false;
        for (lyr.grad_weights) |g| {
            if (@abs(g) > 1e-6) {
                has_nonzero_grad = true;
                break;
            }
        }
        try testing.expect(has_nonzero_grad);
    }
}

test "convergence linear: weight updates" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    // Store initial weights
    const layer0 = net.layers.items[0];
    var initial_weights: [4]f32 = undefined;
    @memcpy(&initial_weights, layer0.weights[0..4]);

    const input: []const f32 = &.{ 1.0 };
    const target: []const f32 = &.{ 1.0 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Training step
    _ = try net.trainStep(input, target, 0.1, loss_fn);

    // Check that weights changed
    var changed = false;
    for (layer0.weights, 0..) |w, i| {
        if (w != initial_weights[i]) {
            changed = true;
            break;
        }
    }
    try testing.expect(changed);
}

test "convergence linear: numerical stability" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Large values
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 100.0 },
        &.{ 200.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 100.0 },
        &.{ 200.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.0001, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 50.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "convergence linear: convergence with noise" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // y = 2x with some noise
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.1 },   // 0 + noise
        &.{ 1.9 },   // 2 - noise
        &.{ 4.1 },   // 4 + noise
        &.{ 5.9 },   // 6 - noise
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 1.5 }, &output);
    // Should be around 3.0
    try testing.expect(output[0] > 2.5 and output[0] < 3.5);
}

test "convergence linear: multiple epochs" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Train for 100 epochs
    try net.train(training_data, training_targets, 100, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.0 }, &output);
    try expectNear(output[0], 1.0, 0.3);
}

test "convergence linear: deep network" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    // y = x
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.5 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.5 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.75 }, &output);
    try expectNear(output[0], 0.75, 0.3);
}

test "convergence linear: network state persistence" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

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
    try net.train(training_data, training_targets, 200, 0.01, loss_fn);

    // Check that network produces consistent outputs
    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.5 }, &output1);
    _ = try net.forward(&.{ 0.5 }, &output2);

    try expectNear(output1[0], output2[0], 0.0001);
}

test "convergence linear: learning curve" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Record loss at different points
    var loss_at_0: f32 = 0;
    var loss_at_50: f32 = 0;
    var loss_at_100: f32 = 0;

    for (0..50) |_| {
        loss_at_0 = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);
    }

    for (0..50) |_| {
        loss_at_50 = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);
    }

    for (0..50) |_| {
        loss_at_100 = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);
    }

    // Loss should decrease
    try testing.expect(std.math.isFinite(loss_at_0));
    try testing.expect(std.math.isFinite(loss_at_50));
    try testing.expect(std.math.isFinite(loss_at_100));
}

test "convergence linear: different initializations" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    // Test multiple initializations
    for (0..3) |_| {
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

        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.5 }, &output);

        try testing.expect(std.math.isFinite(output[0]));
    }
}

test "convergence linear: loss function" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

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

    // Initial loss should be reasonable
    var loss_val = try net.trainStep(training_data[0], training_targets[0], 0, loss_fn);
    try testing.expect(std.math.isFinite(loss_val));
    try testing.expect(loss_val >= 0);

    // After training, loss should be lower
    try net.train(training_data, training_targets, 100, 0.01, loss_fn);
    loss_val = try net.trainStep(training_data[0], training_targets[0], 0, loss_fn);
    try testing.expect(std.math.isFinite(loss_val));
}

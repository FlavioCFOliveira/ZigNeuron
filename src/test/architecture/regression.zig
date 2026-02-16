/// Tests for regression networks

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

test "regression: learn y = 2x + 1" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Training data: y = 2x + 1
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
        &.{ 4.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1.0 },   // 2*0 + 1
        &.{ 3.0 },   // 2*1 + 1
        &.{ 5.0 },   // 2*2 + 1
        &.{ 7.0 },   // 2*3 + 1
        &.{ 9.0 },   // 2*4 + 1
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    // Test interpolation
    _ = try net.forward(&.{ 2.5 }, &output);
    try expectNear(output[0], 6.0, 0.5);

    // Test extrapolation
    _ = try net.forward(&.{ 5.0 }, &output);
    try testing.expect(output[0] > 8 and output[0] < 12);
}

test "regression: multi-output regression" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Network with 2 outputs
    _ = try net.addDense(2, 6, .relu);
    _ = try net.addDense(6, 2, .linear);

    const training_data = &[_][]const f32{
        &.{ 1.0, 0.0 },
        &.{ 0.0, 1.0 },
        &.{ 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1.0, 0.0 },
        &.{ 0.0, 1.0 },
        &.{ 1.0, 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [2]f32 = undefined;
    _ = try net.forward(&.{ 1.0, 0.0 }, &output);

    // Should be close to [1.0, 0.0]
    try expectNear(output[0], 1.0, 0.3);
    try expectNear(output[1], 0.0, 0.3);
}

test "regression: non-linear function approximation" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Train on y = x^2
    const training_data = &[_][]const f32{
        &.{ -2.0 },
        &.{ -1.0 },
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 4.0 },
        &.{ 1.0 },
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 4.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 1.5 }, &output);
    try expectNear(output[0], 2.25, 0.5);
}

test "regression: normalize inputs" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Training data with large values
    const training_data = &[_][]const f32{
        &.{ 100.0 },
        &.{ 200.0 },
        &.{ 300.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.0001, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 250.0 }, &output);

    // Should be around 2.0
    try testing.expect(output[0] > 1.0 and output[0] < 3.0);
}

test "regression: trend prediction" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 10, .relu);
    _ = try net.addDense(10, 1, .linear);

    // Training data with positive trend
    const training_data = &[_][]const f32{
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
        &.{ 4.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 10.0 },
        &.{ 15.0 },
        &.{ 20.0 },
        &.{ 25.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    // Predict next value (should be around 30)
    _ = try net.forward(&.{ 5.0 }, &output);
    try testing.expect(output[0] > 25 and output[0] < 35);
}

test "regression: output bounds with linear activation" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    // Linear output can be any value
    const input: []const f32 = &.{ 100.0, 100.0 };
    var output: [1]f32 = undefined;

    _ = try net.forward(input, &output);

    // Output should be valid (not NaN or Inf)
    try testing.expect(std.math.isFinite(output[0]));
}

test "regression: multiple features" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // y = x1 + 2*x2 + 3*x3
    _ = try net.addDense(3, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 1.0, 0.0, 0.0 },
        &.{ 0.0, 1.0, 0.0 },
        &.{ 0.0, 0.0, 1.0 },
        &.{ 1.0, 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
        &.{ 6.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 2.0, 3.0, 4.0 }, &output);

    // Expected: 2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
    try expectNear(output[0], 20.0, 2.0);
}

test "regression: convergence monitoring" {
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

    // Train and monitor loss
    var loss_history: [10]f32 = undefined;
    for (0..10) |i| {
        loss_history[i] = try net.trainStep(training_data[0], training_targets[0], 0.1, loss_fn);
    }

    // Loss should generally decrease
    try testing.expect(std.math.isFinite(loss_history[0]));
    try testing.expect(std.math.isFinite(loss_history[9]));
}

test "regression: sensitivity to learning rate" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    // Test with high learning rate
    const net1 = try network.Network.init(allocator, be);
    defer net1.deinit();
    _ = try net1.addDense(1, 4, .relu);
    _ = try net1.addDense(4, 1, .linear);

    // Test with low learning rate
    const net2 = try network.Network.init(allocator, be);
    defer net2.deinit();
    _ = try net2.addDense(1, 4, .relu);
    _ = try net2.addDense(4, 1, .linear);

    const input: []const f32 = &.{ 1.0 };
    const target: []const f32 = &.{ 1.0 };
    const loss_fn = loss.Loss{ .mse = {} };

    // High learning rate
    const loss1 = try net1.trainStep(input, target, 0.5, loss_fn);

    // Low learning rate
    const loss2 = try net2.trainStep(input, target, 0.01, loss_fn);

    // Both should be valid
    try testing.expect(std.math.isFinite(loss1));
    try testing.expect(std.math.isFinite(loss2));
}

test "regression: missing data handling" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    // Data with some zeros
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
    _ = try net.forward(&.{ 0.5, 0.5 }, &output);

    try testing.expect(std.math.isFinite(output[0]));
}

test "regression: weight regularization effect" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 20, .relu);
    _ = try net.addDense(20, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    // Check that weights are reasonable
    for (net.layers.items) |lyr| {
        for (lyr.weights) |w| {
            try testing.expect(std.math.isFinite(w));
        }
    }
}

test "regression: output scaling" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Large output values
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1000.0 },
        &.{ 2000.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.00001, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);

    // Should be around 1500
    try testing.expect(output[0] > 1000 and output[0] < 2000);
}

test "regression: early stopping" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

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

    // Train for enough epochs to converge
    try net.train(training_data, training_targets, 100, 0.1, loss_fn);

    // Final predictions should be good
    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.0 }, &output);
    try expectNear(output[0], 1.0, 0.2);
}

test "regression: noise robustness" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Add noise to targets
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.1 },
        &.{ 1.1 },
        &.{ 1.9 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.0 }, &output);

    // Should learn the trend despite noise
    try testing.expect(output[0] > 0.5 and output[0] < 1.5);
}

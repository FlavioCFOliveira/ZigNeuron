/// Tests for simple perceptron architectures (2-layer networks)

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

test "perceptron: simple binary classifier" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // 2 input -> 2 hidden (ReLU) -> 1 output (Sigmoid)
    _ = try net.addDense(2, 2, .relu);
    _ = try net.addDense(2, 1, .sigmoid);

    // Test forward pass
    const input: []const f32 = &.{ 0.5, 0.5 };
    var output: [1]f32 = undefined;
    _ = try net.forward(input, &output);

    try testing.expect(output[0] >= 0);
    try testing.expect(output[0] <= 1);
}

test "perceptron: learn AND gate" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 0.0, 1.0 },
        &.{ 1.0, 0.0 },
        &.{ 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.0 },
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    const learning_rate: f32 = 0.1;

    // Train the network
    try net.train(training_data, training_targets, 1000, learning_rate, loss_fn);

    // Check final predictions
    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.0, 0.0 }, &output);
    try testing.expect(output[0] < 0.2);  // Should be close to 0

    _ = try net.forward(&.{ 1.0, 1.0 }, &output);
    try testing.expect(output[0] > 0.8);  // Should be close to 1
}

test "perceptron: learn OR gate" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 0.0, 1.0 },
        &.{ 1.0, 0.0 },
        &.{ 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 1000, 0.1, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.0, 0.0 }, &output);
    try testing.expect(output[0] < 0.2);

    _ = try net.forward(&.{ 1.0, 0.0 }, &output);
    try testing.expect(output[0] > 0.8);
}

test "perceptron: single layer linear" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Linear layer: no activation (or linear activation)
    _ = try net.addDense(1, 1, .linear);

    // Train y = 2x
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 2.0 },
        &.{ 4.0 },
        &.{ 6.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.5 }, &output);

    // Should be close to 3.0
    try testing.expect(output[0] > 2.5);
    try testing.expect(output[0] < 3.5);
}

test "perceptron: weights are initialized" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(10, 5, .relu);
    _ = try net.addDense(5, 1, .sigmoid);

    // Check that weights are not all zero
    const layer1 = net.layers.items[0];
    var has_nonzero = false;
    for (layer1.weights) |w| {
        if (w != 0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}

test "perceptron: layer sizes correct" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(3, 4, .relu);
    _ = try net.addDense(4, 2, .sigmoid);

    try testing.expect(net.layers.items[0].input_size == 3);
    try testing.expect(net.layers.items[0].output_size == 4);
    try testing.expect(net.layers.items[1].input_size == 4);
    try testing.expect(net.layers.items[1].output_size == 2);
}

test "perceptron: batch forward pass" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 3, .relu);
    _ = try net.addDense(3, 2, .sigmoid);

    // Multiple forward passes
    for (0..10) |_| {
        const input: []const f32 = &.{ 0.5, 0.5 };
        var output: [2]f32 = undefined;
        _ = try net.forward(input, &output);

        try testing.expect(output[0] >= 0 and output[0] <= 1);
        try testing.expect(output[1] >= 0 and output[1] <= 1);
    }
}

test "perceptron: gradient flow" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 3, .relu);
    _ = try net.addDense(3, 1, .sigmoid);

    const input: []const f32 = &.{ 0.5, 0.5 };
    const target: []const f32 = &.{ 0.5 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Store initial weights
    const layer0 = net.layers.items[0];
    var initial_weights: [6]f32 = undefined;
    @memcpy(&initial_weights, layer0.weights[0..6]);

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

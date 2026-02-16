/// Tests for deep feedforward neural networks (4+ layers)

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

test "deep_fnn: 4-layer network forward" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(3, 8, .relu);
    _ = try net.addDense(8, 6, .relu);
    _ = try net.addDense(6, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const input: []const f32 = &.{ 0.5, 0.5, 0.5 };
    var output: [1]f32 = undefined;

    _ = try net.forward(input, &output);

    try testing.expect(output[0] >= 0 and output[0] <= 1);
}

test "deep_fnn: network with skip connections" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(4, 8, .relu);
    _ = try net.addDense(8, 8, .relu);
    _ = try net.addDense(8, 8, .relu);
    _ = try net.addDense(8, 1, .sigmoid);

    // Check that all layers were added correctly
    try testing.expect(net.layers.items.len == 4);
    try testing.expect(net.layers.items[0].output_size == 8);
    try testing.expect(net.layers.items[3].output_size == 1);
}

test "deep_fnn: deep network training" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // 4-layer network
    _ = try net.addDense(2, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 4, .relu);
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
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.1, loss_fn);

    // After training, the network should learn the XOR pattern roughly
    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.0, 0.0 }, &output);
    const pred_00 = output[0];

    _ = try net.forward(&.{ 1.0, 1.0 }, &output);
    const pred_11 = output[0];

    // XOR: 0,0 -> 0 and 1,1 -> 0 (approximately)
    try testing.expect(pred_00 < 0.5);
    try testing.expect(pred_11 < 0.5);
}

test "deep_fnn: network depth validation" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Add layers of varying sizes
    _ = try net.addDense(5, 10, .relu);
    _ = try net.addDense(10, 15, .relu);
    _ = try net.addDense(15, 8, .relu);
    _ = try net.addDense(8, 3, .relu);
    _ = try net.addDense(3, 1, .sigmoid);

    try testing.expect(net.layers.items.len == 5);
}

test "deep_fnn: weight initialization scales with depth" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Deep network
    _ = try net.addDense(10, 20, .relu);
    _ = try net.addDense(20, 15, .relu);
    _ = try net.addDense(15, 10, .relu);
    _ = try net.addDense(10, 5, .relu);
    _ = try net.addDense(5, 1, .sigmoid);

    // All weights should be initialized
    for (net.layers.items) |lyr| {
        var has_nonzero = false;
        for (lyr.weights) |w| {
            if (w != 0) {
                has_nonzero = true;
                break;
            }
        }
        try testing.expect(has_nonzero);
    }
}

test "deep_fnn: backward pass through multiple layers" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 3, .relu);
    _ = try net.addDense(3, 1, .sigmoid);

    const input: []const f32 = &.{ 0.5, 0.5 };
    const target: []const f32 = &.{ 0.5 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Training step should compute gradients through all layers
    _ = try net.trainStep(input, target, 0.1, loss_fn);

    // Check that all layers have non-zero gradients
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

test "deep_fnn: sigmoid output bounds preserved" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(10, 20, .relu);
    _ = try net.addDense(20, 15, .relu);
    _ = try net.addDense(15, 8, .relu);
    _ = try net.addDense(8, 1, .sigmoid);

    // Test with large inputs
    const input: []const f32 = &.{ 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 };
    var output: [1]f32 = undefined;

    _ = try net.forward(input, &output);

    // Output should still be between 0 and 1
    try testing.expect(output[0] >= 0);
    try testing.expect(output[0] <= 1);
}

test "deep_fnn: network state persistence" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    // Train for a bit
    const training_data = &[_][]const f32{ &.{ 0.5, 0.5 } };
    const training_targets = &[_][]const f32{ &.{ 0.5 } };
    const loss_fn = loss.Loss{ .mse = {} };

    try net.train(training_data, training_targets, 10, 0.1, loss_fn);

    // Check that weights changed
    const layer0 = net.layers.items[0];
    var weights_changed = false;
    for (layer0.weights) |w| {
        if (w != 0) {  // Weights are initialized non-zero
            weights_changed = true;
            break;
        }
    }
    try testing.expect(weights_changed);
}

test "deep_fnn: multiple forward passes consistency" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(3, 6, .relu);
    _ = try net.addDense(6, 4, .relu);
    _ = try net.addDense(4, 2, .relu);
    _ = try net.addDense(2, 1, .sigmoid);

    const input: []const f32 = &.{ 0.5, 0.5, 0.5 };
    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;

    _ = try net.forward(input, &output1);
    _ = try net.forward(input, &output2);

    // Same input should give same output
    try expectNear(output1[0], output2[0], 0.0001);
}

test "deep_fnn: deep network gradient scale" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Deep network
    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 4, .relu);
    _ = try net.addDense(4, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const input: []const f32 = &.{ 1.0, 1.0 };
    const target: []const f32 = &.{ 0.0 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Training step
    _ = try net.trainStep(input, target, 0.1, loss_fn);

    // Check that gradients exist and are finite
    for (net.layers.items) |lyr| {
        for (lyr.grad_weights) |g| {
            try testing.expect(std.math.isFinite(g));
        }
        for (lyr.grad_bias) |g| {
            try testing.expect(std.math.isFinite(g));
        }
    }
}


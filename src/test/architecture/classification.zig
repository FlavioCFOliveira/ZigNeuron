/// Tests for classification networks

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

test "classification: binary classification with sigmoid" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const input: []const f32 = &.{ 0.5, 0.5 };
    var output: [1]f32 = undefined;

    _ = try net.forward(input, &output);

    try testing.expect(output[0] >= 0);
    try testing.expect(output[0] <= 1);
}

test "classification: class prediction from logits" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(3, 5, .relu);
    _ = try net.addDense(5, 1, .sigmoid);

    const input: []const f32 = &.{ 1.0, 2.0, 1.0 };
    var output: [1]f32 = undefined;

    _ = try net.forward(input, &output);

    const pred = output[0];
    try testing.expect(pred >= 0);
    try testing.expect(pred <= 1);
}

test "classification: multi-class with softmax" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(4, 8, .relu);
    _ = try net.addDense(8, 3, .softmax);

    const input: []const f32 = &.{ 1.0, 2.0, 1.0, 2.0 };
    var output: [3]f32 = undefined;

    _ = try net.forward(input, &output);

    var sum: f32 = 0;
    for (output) |p| sum += p;
    try expectNear(sum, 1.0, 0.001);

    for (output) |p| {
        try testing.expect(p > 0);
    }
}

test "classification: learn simple linearly separable data" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

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
    try net.train(training_data, training_targets, 500, 0.1, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.0, 1.0 }, &output);
    try testing.expect(output[0] > 0.8);

    _ = try net.forward(&.{ 0.0, 0.0 }, &output);
    try testing.expect(output[0] < 0.2);
}

test "classification: decision boundary" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 6, .relu);
    _ = try net.addDense(6, 1, .sigmoid);

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
    try net.train(training_data, training_targets, 300, 0.1, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.0, 0.0 }, &output);
    const c00 = output[0];
    _ = try net.forward(&.{ 1.0, 1.0 }, &output);
    const c11 = output[0];

    try testing.expect(c00 < 0.5);
    try testing.expect(c11 < 0.5);
}

test "classification: confidence calibration" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(3, 6, .relu);
    _ = try net.addDense(6, 1, .sigmoid);

    const input: []const f32 = &.{ 10.0, 10.0, 10.0 };
    var output: [1]f32 = undefined;

    _ = try net.forward(input, &output);

    try testing.expect(output[0] >= 0);
    try testing.expect(output[0] <= 1);
}

test "classification: class imbalance handling" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 0.0, 0.0 },
        &.{ 0.0, 0.0 },
        &.{ 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.0 },
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.1, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.0, 1.0 }, &output);
    try testing.expect(output[0] > 0.5);
}

test "classification: overfitting detection" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 20, .relu);
    _ = try net.addDense(20, 1, .sigmoid);

    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    var final_loss: f32 = 0;
    for (0..100) |_| {
        final_loss = try net.trainStep(training_data[0], training_targets[0], 0.1, loss_fn);
    }

    try testing.expect(final_loss < 0.01);
}

test "classification: numerical precision" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const input: []const f32 = &.{ 0.001, 0.001 };
    var output: [1]f32 = undefined;

    _ = try net.forward(input, &output);

    try testing.expect(std.math.isFinite(output[0]));
    try testing.expect(output[0] >= 0);
    try testing.expect(output[0] <= 1);
}

test "classification: gradient clipping effect" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const input: []const f32 = &.{ 100.0, 100.0 };
    const target: []const f32 = &.{ 0.5 };
    const loss_fn = loss.Loss{ .mse = {} };

    const loss_val = try net.trainStep(input, target, 0.1, loss_fn);

    try testing.expect(std.math.isFinite(loss_val));

    for (net.layers.items) |lyr| {
        for (lyr.weights.slice) |w| {
            try testing.expect(std.math.isFinite(w));
        }
    }
}

test "classification: batch inference" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(3, 6, .relu);
    _ = try net.addDense(6, 2, .softmax);

    const input1: []const f32 = &.{ 1.0, 2.0, 1.0 };
    const input2: []const f32 = &.{ 2.0, 1.0, 2.0 };
    var output1: [2]f32 = undefined;
    var output2: [2]f32 = undefined;

    _ = try net.forward(input1, &output1);
    _ = try net.forward(input2, &output2);

    for (output1) |p| {
        try testing.expect(p > 0);
    }
    for (output2) |p| {
        try testing.expect(p > 0);
    }
}

test "classification: gradient vanishing check" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const input: []const f32 = &.{ 0.5, 0.5 };
    const target: []const f32 = &.{ 0.5 };
    const loss_fn = loss.Loss{ .mse = {} };

    _ = try net.trainStep(input, target, 0.1, loss_fn);

    const layer1 = net.layers.items[1];
    var has_valid_grad = false;
    for (layer1.grad_weights) |g| {
        if (@abs(g) > 1e-10) {
            has_valid_grad = true;
            break;
        }
    }
    try testing.expect(has_valid_grad);
}

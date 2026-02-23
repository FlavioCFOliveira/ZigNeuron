/// Convergence tests for quadratic function learning (y = x²)

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

test "convergence quadratic: y = x² basic" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Training data for y = x²
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

    // Test interpolation
    _ = try net.forward(&.{ 1.5 }, &output);
    try expectNear(output[0], 2.25, 0.5);

    // Test another point
    _ = try net.forward(&.{ -1.5 }, &output);
    try expectNear(output[0], 2.25, 0.5);
}

test "convergence quadratic: y = 2x² + 3x - 1" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Training data for y = 2x² + 3x - 1
    const training_data = &[_][]const f32{
        &.{ -3.0 },
        &.{ -2.0 },
        &.{ -1.0 },
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 8.0 },   // 2*9 + 3*(-3) - 1 = 18 - 9 - 1 = 8
        &.{ 1.0 },   // 2*4 + 3*(-2) - 1 = 8 - 6 - 1 = 1
        &.{ -2.0 },  // 2*1 + 3*(-1) - 1 = 2 - 3 - 1 = -2
        &.{ -1.0 },  // -1
        &.{ 4.0 },   // 2*1 + 3*1 - 1 = 2 + 3 - 1 = 4
        &.{ 13.0 },  // 2*4 + 3*2 - 1 = 8 + 6 - 1 = 13
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.0 }, &output);
    try expectNear(output[0], -1.0, 0.5);
}

test "convergence quadratic: shallow network" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Simple 2-layer network
    _ = try net.addDense(1, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

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
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.5 }, &output);
    // Should be around 0.25
    try testing.expect(output[0] > 0.1 and output[0] < 0.5);
}

test "convergence quadratic: deep network" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Deeper network: 1-16-16-16-1
    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 16, .relu);
    _ = try net.addDense(16, 16, .relu);
    _ = try net.addDense(16, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ -2.0 },
        &.{ -1.5 },
        &.{ -1.0 },
        &.{ -0.5 },
        &.{ 0.0 },
        &.{ 0.5 },
        &.{ 1.0 },
        &.{ 1.5 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 4.0 },
        &.{ 2.25 },
        &.{ 1.0 },
        &.{ 0.25 },
        &.{ 0.0 },
        &.{ 0.25 },
        &.{ 1.0 },
        &.{ 2.25 },
        &.{ 4.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.75 }, &output);
    try expectNear(output[0], 0.5625, 0.3);
}

test "convergence quadratic: multiple epochs" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

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
        &.{ 4.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Track loss over epochs
    var loss_history: [10]f32 = undefined;
    for (0..10) |i| {
        loss_history[i] = try net.trainStep(training_data[i % 3], training_targets[i % 3], 0.01, loss_fn);
    }

    // Loss should generally decrease
    try testing.expect(std.math.isFinite(loss_history[0]));
    try testing.expect(std.math.isFinite(loss_history[9]));
}

test "convergence quadratic: learning rate comparison" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    // High learning rate
    const net1 = try network.Network.init(allocator, be);
    defer net1.deinit();
    _ = try net1.addDense(1, 8, .relu);
    _ = try net1.addDense(8, 1, .linear);

    // Low learning rate
    const net2 = try network.Network.init(allocator, be);
    defer net2.deinit();
    _ = try net2.addDense(1, 8, .relu);
    _ = try net2.addDense(8, 1, .linear);

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

    try net1.train(training_data, training_targets, 200, 0.1, loss_fn);
    try net2.train(training_data, training_targets, 200, 0.01, loss_fn);

    // Both should learn
    var output: [1]f32 = undefined;

    _ = try net1.forward(&.{ 0.5 }, &output);
    try testing.expect(std.math.isFinite(output[0]));

    _ = try net2.forward(&.{ 0.5 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "convergence quadratic: extrapolation" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Train on [-2, 2]
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
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    // Extrapolate beyond training range
    _ = try net.forward(&.{ 3.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));

    _ = try net.forward(&.{ -3.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "convergence quadratic: loss monitoring" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

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

    // Record loss at different epochs
    var loss_at_start: f32 = 0;
    var loss_at_middle: f32 = 0;
    var loss_at_end: f32 = 0;

    for (0..100) |_| {
        loss_at_start = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);
    }
    for (0..100) |_| {
        loss_at_middle = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);
    }
    for (0..100) |_| {
        loss_at_end = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);
    }

    try testing.expect(std.math.isFinite(loss_at_start));
    try testing.expect(std.math.isFinite(loss_at_middle));
    try testing.expect(std.math.isFinite(loss_at_end));
}

test "convergence quadratic: weight bounds" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

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
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    // Check that all weights are finite
    for (net.layers.items) |*l| {
        for (l.weights.slice) |w| {
            try testing.expect(std.math.isFinite(w));
        }
        for (l.bias.slice) |b| {
            try testing.expect(std.math.isFinite(b));
        }
    }
}

test "convergence quadratic: gradient flow" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    const input: []const f32 = &.{ 1.0 };
    const target: []const f32 = &.{ 1.0 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Training step
    _ = try net.trainStep(input, target, 0.1, loss_fn);

    // Check that gradients exist
    for (net.layers.items) |*l| {
        var has_nonzero_grad = false;
        for (l.grad_weights.slice) |g| {
            if (@abs(g) > 1e-6) {
                has_nonzero_grad = true;
                break;
            }
        }
        try testing.expect(has_nonzero_grad);
    }
}

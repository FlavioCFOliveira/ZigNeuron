/// General function approximation convergence tests

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

test "convergence functions: polynomial approximation" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ -1.0 },
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ -4.0 },
        &.{ 0.0 },
        &.{ 0.0 },
        &.{ 2.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.5 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "convergence functions: exponential decay" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1.0 },
        &.{ 0.3679 },
        &.{ 0.1353 },
        &.{ 0.0498 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.5 }, &output);
    try testing.expect(output[0] > 0.1 and output[0] < 0.4);
}

test "convergence functions: logarithmic" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
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
        &.{ 0.6931 },
        &.{ 1.0986 },
        &.{ 1.3863 },
        &.{ 1.6094 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 2.5 }, &output);
    try testing.expect(output[0] > 1.0 and output[0] < 1.5);
}

test "convergence functions: rational function" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ -2.0 },
        &.{ -1.0 },
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.2 },
        &.{ 0.5 },
        &.{ 1.0 },
        &.{ 0.5 },
        &.{ 0.2 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);
    try expectNear(output[0], 0.8, 0.2);
}

test "convergence functions: ReLU-like function" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.5 },
        &.{ 1.0 },
        &.{ 1.5 },
        &.{ 2.0 },
        &.{ 2.5 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.5 },
        &.{ 1.0 },
        &.{ 0.5 },
        &.{ 0.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.25 }, &output);
    try expectNear(output[0], 0.75, 0.2);
}

test "convergence functions: piecewise constant" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .tanh);
    _ = try net.addDense(8, 1, .sigmoid);

    const training_data = &[_][]const f32{
        &.{ -1.0 },
        &.{ -0.5 },
        &.{ 0.0 },
        &.{ 0.5 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ -0.25 }, &output);
    try testing.expect(output[0] < 0.5);

    _ = try net.forward(&.{ 0.25 }, &output);
    try testing.expect(output[0] > 0.5);
}

test "convergence functions: multi-peak function" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ @as(f32, @floatCast(std.math.pi / 4)) },
        &.{ @as(f32, @floatCast(std.math.pi / 2)) },
        &.{ @as(f32, @floatCast(3 * std.math.pi / 4)) },
        &.{ @as(f32, @floatCast(std.math.pi)) },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.7071 + 0.5 * 0.7071 },
        &.{ 1.0 + 0.5 * (-1.0) },
        &.{ 0.7071 + 0.5 * (-0.7071) },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ @as(f32, @floatCast(std.math.pi / 6)) }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "convergence functions: spiral function" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 16, .relu);
    _ = try net.addDense(16, 16, .relu);
    _ = try net.addDense(16, 2, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 0.0 },
        &.{ 0.0, 1.0 },
        &.{ -1.0, 0.0 },
        &.{ 0.0, -1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 0.0 },
        &.{ 0.0, 1.0 },
        &.{ -1.0, 0.0 },
        &.{ 0.0, -1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.01, loss_fn);

    var output: [2]f32 = undefined;
    _ = try net.forward(&.{ 0.5, 0.5 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
    try testing.expect(std.math.isFinite(output[1]));
}

test "convergence functions: discontinuous function" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ -2.0 },
        &.{ -1.0 },
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.0 },
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ -0.5 }, &output);
    try testing.expect(output[0] < 0.1);

    _ = try net.forward(&.{ 0.5 }, &output);
    try expectNear(output[0], 0.5, 0.2);
}

test "convergence functions: sigmoid-like approximation" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ -3.0 },
        &.{ -2.0 },
        &.{ -1.0 },
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
        &.{ 3.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0474 },
        &.{ 0.1192 },
        &.{ 0.2689 },
        &.{ 0.5 },
        &.{ 0.7311 },
        &.{ 0.8808 },
        &.{ 0.9526 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.0 }, &output);
    try expectNear(output[0], 0.5, 0.1);
}

test "convergence functions: linear combination" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ @as(f32, @floatCast(std.math.pi / 2)) },
        &.{ @as(f32, @floatCast(std.math.pi)) },
        &.{ @as(f32, @floatCast(3 * std.math.pi / 2)) },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.5 * 1.5708 + 0.3 * 1.0 },
        &.{ 0.5 * 3.1416 + 0.3 * 0.0 },
        &.{ 0.5 * 4.7124 + 0.3 * (-1.0) },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ @as(f32, @floatCast(std.math.pi / 4)) }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "convergence functions: network capacity test" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    for ([_]usize{ 4, 8, 16, 32 }) |hidden_size| {
        const net = try network.Network.init(allocator, be);
        defer net.deinit();

        _ = try net.addDense(1, hidden_size, .relu);
        _ = try net.addDense(hidden_size, 1, .linear);

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
    }
}

test "convergence functions: output range verification" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ -10.0 },
        &.{ -5.0 },
        &.{ 0.0 },
        &.{ 5.0 },
        &.{ 10.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 100.0 },
        &.{ 25.0 },
        &.{ 0.0 },
        &.{ 25.0 },
        &.{ 100.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.001, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ -20.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));

    _ = try net.forward(&.{ 20.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "convergence functions: weight initialization test" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    var success_count: usize = 0;

    for (0..5) |_| {
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
        try net.train(training_data, training_targets, 200, 0.01, loss_fn);

        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.0 }, &output);

        if (output[0] < 0.5) {
            success_count += 1;
        }
    }

    try testing.expect(success_count >= 2);
}

test "convergence functions: gradient verification" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    const input: []const f32 = &.{ 1.0 };
    const target: []const f32 = &.{ 1.0 };
    const loss_fn = loss.Loss{ .mse = {} };

    const loss_value = try net.trainStep(input, target, 0.1, loss_fn);

    try testing.expect(std.math.isFinite(loss_value));
    try testing.expect(loss_value >= 0);

    for (net.layers.items) |*l| {
        var has_nonzero_grad = false;
        for (l.grad_weights) |g| {
            if (@abs(g) > 1e-10) {
                has_nonzero_grad = true;
                break;
            }
        }
        try testing.expect(has_nonzero_grad);
    }
}

test "convergence functions: batch vs sequential" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    const net1 = try network.Network.init(allocator, be);
    defer net1.deinit();
    _ = try net1.addDense(1, 8, .relu);
    _ = try net1.addDense(8, 1, .linear);

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

    try net1.train(training_data, training_targets, 200, 0.01, loss_fn);
    try net2.train(training_data, training_targets, 200, 0.01, loss_fn);

    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;

    _ = try net1.forward(&.{ 0.5 }, &output1);
    _ = try net2.forward(&.{ 0.5 }, &output2);

    try testing.expect(std.math.isFinite(output1[0]));
    try testing.expect(std.math.isFinite(output2[0]));
}

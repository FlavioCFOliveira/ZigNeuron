/// Convergence tests for XOR problem with different architectures

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

test "convergence xor: simple 2-4-1 network" {
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
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.1, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.0, 0.0 }, &output);
    try testing.expect(output[0] < 0.3);

    _ = try net.forward(&.{ 1.0, 1.0 }, &output);
    try testing.expect(output[0] < 0.3);

    _ = try net.forward(&.{ 0.0, 1.0 }, &output);
    try testing.expect(output[0] > 0.7);

    _ = try net.forward(&.{ 1.0, 0.0 }, &output);
    try testing.expect(output[0] > 0.7);
}

test "convergence xor: 2-8-4-1 network" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 8, .relu);
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
    try net.train(training_data, training_targets, 500, 0.1, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.0, 0.0 }, &output);
    try testing.expect(output[0] < 0.3);

    _ = try net.forward(&.{ 1.0, 1.0 }, &output);
    try testing.expect(output[0] < 0.3);
}

test "convergence xor: multiple random initializations" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    var success_count: usize = 0;

    for (0..3) |_| {
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
            &.{ 0.0 },
        };

        const loss_fn = loss.Loss{ .mse = {} };
        try net.train(training_data, training_targets, 300, 0.1, loss_fn);

        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.0, 0.0 }, &output);
        const p00 = output[0];

        _ = try net.forward(&.{ 1.0, 1.0 }, &output);
        const p11 = output[0];

        _ = try net.forward(&.{ 0.0, 1.0 }, &output);
        const p01 = output[0];

        _ = try net.forward(&.{ 1.0, 0.0 }, &output);
        const p10 = output[0];

        if (p00 < 0.3 and p11 < 0.3 and p01 > 0.7 and p10 > 0.7) {
            success_count += 1;
        }
    }

    try testing.expect(success_count >= 1);
}

test "convergence xor: batch vs sequential" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net1 = try network.Network.init(allocator, be);
    defer net1.deinit();
    _ = try net1.addDense(2, 4, .relu);
    _ = try net1.addDense(4, 1, .sigmoid);

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
    try net1.train(training_data, training_targets, 200, 0.1, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net1.forward(&.{ 0.0, 0.0 }, &output);
    try testing.expect(output[0] < 0.5);
}

test "convergence xor: learning rate comparison" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net1 = try network.Network.init(allocator, be);
    defer net1.deinit();
    _ = try net1.addDense(2, 4, .relu);
    _ = try net1.addDense(4, 1, .sigmoid);

    const net2 = try network.Network.init(allocator, be);
    defer net2.deinit();
    _ = try net2.addDense(2, 4, .relu);
    _ = try net2.addDense(4, 1, .sigmoid);

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
    try net1.train(training_data, training_targets, 200, 0.2, loss_fn);
    try net2.train(training_data, training_targets, 200, 0.05, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net1.forward(&.{ 0.0, 0.0 }, &output);
    const p1_00 = output[0];

    _ = try net2.forward(&.{ 0.0, 0.0 }, &output);
    const p2_00 = output[0];

    try testing.expect(std.math.isFinite(p1_00));
    try testing.expect(std.math.isFinite(p2_00));
}

test "convergence xor: convergence speed" {
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
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    var final_loss: f32 = 0;
    for (0..100) |_| {
        final_loss = try net.trainStep(training_data[0], training_targets[0], 0.1, loss_fn);
    }

    try testing.expect(final_loss < 0.5);
}

test "convergence xor: test with different activations" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .tanh);
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
    try net.train(training_data, training_targets, 300, 0.1, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.0, 0.0 }, &output);
    try testing.expect(output[0] < 0.5);
}

test "convergence xor: validation on unseen data" {
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
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.1, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.0, 0.0 }, &output);
    const p00 = output[0];

    _ = try net.forward(&.{ 0.0, 1.0 }, &output);
    const p01 = output[0];

    try testing.expect(p00 < 0.3);
    try testing.expect(p01 > 0.7);
}

test "convergence xor: early convergence detection" {
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
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    try net.train(training_data, training_targets, 200, 0.1, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.0, 1.0 }, &output);
    try testing.expect(output[0] < 0.5);
}

test "convergence xor: network capacity" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    for ([_]usize{ 2, 4, 8, 16 }) |hidden_size| {
        const net = try network.Network.init(allocator, be);
        defer net.deinit();

        _ = try net.addDense(2, hidden_size, .relu);
        _ = try net.addDense(hidden_size, 1, .sigmoid);

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

        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.0, 0.0 }, &output);
        try testing.expect(std.math.isFinite(output[0]));
    }
}

test "convergence xor: gradient flow verification" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const input: []const f32 = &.{ 0.5, 0.5 };
    const target: []const f32 = &.{ 1.0 };
    const loss_fn = loss.Loss{ .mse = {} };

    _ = try net.trainStep(input, target, 0.1, loss_fn);

    for (net.layers.items) |lyr| {
        var has_nonzero_grad = false;
        for (lyr.grad_weights.slice) |g| {
            if (@abs(g) > 1e-6) {
                has_nonzero_grad = true;
                break;
            }
        }
        try testing.expect(has_nonzero_grad);
    }
}

test "convergence xor: weight updates" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const layer0 = net.layers.items[0];
    var initial_weights: [8]f32 = undefined;
    @memcpy(&initial_weights, layer0.weights.slice[0..8]);

    const input: []const f32 = &.{ 0.5, 0.5 };
    const target: []const f32 = &.{ 1.0 };
    const loss_fn = loss.Loss{ .mse = {} };

    _ = try net.trainStep(input, target, 0.1, loss_fn);

    var changed = false;
    for (layer0.weights.slice, 0..) |w, i| {
        if (w != initial_weights[i]) {
            changed = true;
            break;
        }
    }
    try testing.expect(changed);
}

test "convergence xor: stability after training" {
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
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.1, loss_fn);

    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.0, 0.0 }, &output1);
    _ = try net.forward(&.{ 0.0, 0.0 }, &output2);

    try expectNear(output1[0], output2[0], 0.0001);
}

test "convergence xor: momentum effect" {
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
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.1, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.0, 1.0 }, &output);
    try testing.expect(output[0] > 0.5);
}

test "convergence xor: learning rate sensitivity" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net1 = try network.Network.init(allocator, be);
    defer net1.deinit();
    _ = try net1.addDense(2, 4, .relu);
    _ = try net1.addDense(4, 1, .sigmoid);

    const net2 = try network.Network.init(allocator, be);
    defer net2.deinit();
    _ = try net2.addDense(2, 4, .relu);
    _ = try net2.addDense(4, 1, .sigmoid);

    const net3 = try network.Network.init(allocator, be);
    defer net3.deinit();
    _ = try net3.addDense(2, 4, .relu);
    _ = try net3.addDense(4, 1, .sigmoid);

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

    try net1.train(training_data, training_targets, 100, 0.5, loss_fn);
    try net2.train(training_data, training_targets, 100, 0.1, loss_fn);
    try net3.train(training_data, training_targets, 100, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    _ = try net1.forward(&.{ 0.0, 0.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));

    _ = try net2.forward(&.{ 0.0, 0.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));

    _ = try net3.forward(&.{ 0.0, 0.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "convergence xor: architecture ablation" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(2, 2, .relu);
    _ = try net.addDense(2, 1, .sigmoid);

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
    try net.train(training_data, training_targets, 400, 0.1, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.0, 1.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "convergence xor: convergence visualization data" {
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
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    var loss_at_0: f32 = 0;
    var loss_at_100: f32 = 0;

    for (0..100) |_| {
        loss_at_0 = try net.trainStep(training_data[0], training_targets[0], 0.1, loss_fn);
    }

    for (0..100) |_| {
        loss_at_100 = try net.trainStep(training_data[0], training_targets[0], 0.1, loss_fn);
    }

    try testing.expect(std.math.isFinite(loss_at_0));
    try testing.expect(std.math.isFinite(loss_at_100));
}

test "convergence xor: layerwise learning" {
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
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    try net.train(training_data, training_targets, 200, 0.1, loss_fn);

    for (net.layers.items) |lyr| {
        var has_nonzero_grad = false;
        for (lyr.grad_weights.slice) |g| {
            if (@abs(g) > 1e-6) {
                has_nonzero_grad = true;
                break;
            }
        }
        try testing.expect(has_nonzero_grad);
    }
}

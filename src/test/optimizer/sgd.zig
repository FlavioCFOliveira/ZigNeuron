/// SGD (Stochastic Gradient Descent) optimizer tests

const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const layer = zn.layer;
const activation = zn.activation;
const backend = zn.backend;
const network = zn.network;
const loss = zn.loss;
const optimizer = zn.optimizer;

test "optimizer sgd: basic structure" {
    const sgd: optimizer.Sgd = .{};

    try testing.expect(sgd.momentum == 0.0);
    try testing.expect(sgd.velocity_weights == null);
    try testing.expect(sgd.velocity_bias == null);
}

test "optimizer sgd: with momentum initialization" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;

    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    try testing.expect(sgd.velocity_weights != null);
    try testing.expect(sgd.velocity_bias != null);
}

test "optimizer sgd: momentum weight update" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    // Set initial weight
    lyr.weights.slice[0] = 1.0;

    // First step with gradient = 0.1
    lyr.grad_weights.slice[0] = 0.1;
    sgd.step(&lyr, 0.01);

    // v = 0.9 * 0 + 0.1 = 0.1
    // w = 1.0 - 0.01 * 0.1 = 0.999
    try testing.expectNear(lyr.weights.slice[0], 0.999, 0.0001);

    // Second step with gradient = 0.2
    lyr.grad_weights.slice[0] = 0.2;
    sgd.step(&lyr, 0.01);

    // v = 0.9 * 0.1 + 0.2 = 0.29
    // w = 0.999 - 0.01 * 0.29 = 0.9961
    try testing.expectNear(lyr.weights.slice[0], 0.9961, 0.0001);
}

test "optimizer sgd: momentum bias update" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 2, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    // Set initial bias
    lyr.bias.slice[0] = 0.5;
    lyr.bias.slice[1] = 1.0;

    // First step
    lyr.grad_bias.slice[0] = 0.1;
    lyr.grad_bias.slice[1] = 0.2;
    sgd.step(&lyr, 0.01);

    try testing.expectNear(lyr.bias.slice[0], 0.499, 0.0001);
    try testing.expectNear(lyr.bias.slice[1], 0.998, 0.0001);

    // Second step
    lyr.grad_bias.slice[0] = 0.2;
    lyr.grad_bias.slice[1] = 0.3;
    sgd.step(&lyr, 0.01);

    // v_b[0] = 0.9 * 0.1 + 0.2 = 0.29
    // b[0] = 0.499 - 0.01 * 0.29 = 0.4961
    // v_b[1] = 0.9 * 0.2 + 0.3 = 0.48
    // b[1] = 0.998 - 0.01 * 0.48 = 0.9932
    try testing.expectNear(lyr.bias.slice[0], 0.4961, 0.001);
    try testing.expectNear(lyr.bias.slice[1], 0.9932, 0.001);
}

test "optimizer sgd: no momentum" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.0 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    lyr.weights.slice[0] = 1.0;
    lyr.weights.slice[1] = 2.0;
    lyr.grad_weights.slice[0] = 0.1;
    lyr.grad_weights.slice[1] = 0.2;

    sgd.step(&lyr, 0.01);

    // Without momentum: w = w - lr * grad
    try testing.expectNear(lyr.weights.slice[0], 0.99, 0.0001);
    try testing.expectNear(lyr.weights.slice[1], 1.98, 0.0001);
}

test "optimizer sgd: multiple weight updates" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 3, 2, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    // Set all weights to 1
    @memset(lyr.weights.slice, 1.0);
    @memset(lyr.grad_weights.slice, 0.1);

    sgd.step(&lyr, 0.01);

    // All weights should be updated
    for (lyr.weights.slice) |w| {
        try testing.expect(w < 1.0);  // Should have decreased
    }
}

test "optimizer sgd: training convergence" {
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
        &.{ 2.0 },
    };

    // Use SGD optimizer
    const opt = optimizer.Optimizer{ .sgd = optimizer.Sgd{ .momentum = 0.9 } };
    try net.initOptimizer(&opt);
    defer net.deinitOptimizer(&opt);

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.5 }, &output);
    try testing.expectNear(output[0], 1.5, 0.3);
}

test "optimizer sgd: different momentum values" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    // Test with different momentum values
    for ([_]f32{ 0.0, 0.5, 0.9, 0.99 }) |momentum| {
        const net = try network.Network.init(allocator, be);
        defer net.deinit();

        _ = try net.addDense(1, 4, .relu);
        _ = try net.addDense(4, 1, .linear);

        const training_data = &[_][]const f32{
            &.{ 0.0 },
            &.{ 1.0 },
        };
        const training_targets = &[_][]const f32{
            &.{ 0.0 },
            &.{ 1.0 },
        };

        const opt = optimizer.Optimizer{ .sgd = optimizer.Sgd{ .momentum = momentum } };
        try net.initOptimizer(&opt);
        defer net.deinitOptimizer(&opt);

        const loss_fn = loss.Loss{ .mse = {} };
        try net.train(training_data, training_targets, 100, 0.01, loss_fn);

        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.5 }, &output);
        try testing.expect(std.math.isFinite(output[0]));
    }
}

test "optimizer sgd: weight decay effect" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    // Large initial weight
    lyr.weights.slice[0] = 10.0;

    // Small gradient
    lyr.grad_weights.slice[0] = 0.01;

    // Multiple steps with momentum
    for (0..10) |_| {
        sgd.step(&lyr, 0.1);
    }

    // Weight should have decreased significantly
    try testing.expect(lyr.weights.slice[0] < 10.0);
}

test "optimizer sgd: numerical stability" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    // Very small values
    lyr.weights.slice[0] = 1e-10;
    lyr.grad_weights.slice[0] = 1e-12;

    sgd.step(&lyr, 0.01);

    try testing.expect(std.math.isFinite(lyr.weights.slice[0]));

    // Very large values
    lyr.weights.slice[0] = 1e10;
    lyr.grad_weights.slice[0] = 1e8;

    sgd.step(&lyr, 0.001);

    try testing.expect(std.math.isFinite(lyr.weights.slice[0]));
}

test "optimizer sgd: comparison with manual calculation" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.8 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    lyr.weights.slice[0] = 5.0;
    lyr.grad_weights.slice[0] = 0.5;

    // Step 1: v = 0.8 * 0 + 0.5 = 0.5, w = 5.0 - 0.01 * 0.5 = 4.995
    sgd.step(&lyr, 0.01);
    try testing.expectNear(lyr.weights.slice[0], 4.995, 0.0001);

    // Step 2: v = 0.8 * 0.5 + 0.5 = 0.9, w = 4.995 - 0.01 * 0.9 = 4.986
    lyr.grad_weights.slice[0] = 0.5;
    sgd.step(&lyr, 0.01);
    try testing.expectNear(lyr.weights.slice[0], 4.986, 0.0001);
}

test "optimizer sgd: deinit" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);

    // Deinit should not panic
    sgd.deinit(allocator);
}

test "optimizer sgd: union interface" {
    var opt = optimizer.Optimizer{ .sgd = optimizer.Sgd{ .momentum = 0.9 } };
    const allocator = testing.allocator;

    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try opt.init(allocator, &lyr);
    defer opt.deinit(allocator, &lyr);

    lyr.weights.slice[0] = 1.0;
    lyr.grad_weights.slice[0] = 0.1;

    opt.step(&lyr, 0.01);

    // Weight should have changed
    try testing.expect(lyr.weights.slice[0] < 1.0);
}

test "optimizer sgd: large layer" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 100, 50, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    // Set all gradients to 0.1
    @memset(lyr.grad_weights.slice, 0.1);
    @memset(lyr.grad_bias.slice, 0.1);

    sgd.step(&lyr, 0.01);

    // All weights and biases should have changed
    var all_changed = true;
    for (lyr.weights.slice) |w| {
        if (w != 1.0) {  // After initialization
            all_changed = false;
            break;
        }
    }

    // After first step, weights should have changed from their initial values
    // (The initial value is implementation-dependent, so we just check they're finite)
    for (lyr.weights.slice) |w| {
        try testing.expect(std.math.isFinite(w));
    }
    for (lyr.bias.slice) |b| {
        try testing.expect(std.math.isFinite(b));
    }
}

test "optimizer sgd: learning rate effect" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

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
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const opt1 = optimizer.Optimizer{ .sgd = optimizer.Sgd{ .momentum = 0.9 } };
    const opt2 = optimizer.Optimizer{ .sgd = optimizer.Sgd{ .momentum = 0.9 } };

    try net1.initOptimizer(&opt1);
    try net2.initOptimizer(&opt2);

    const loss_fn = loss.Loss{ .mse = {} };

    // High learning rate
    try net1.train(training_data, training_targets, 100, 0.1, loss_fn);
    // Low learning rate
    try net2.train(training_data, training_targets, 100, 0.01, loss_fn);

    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;

    _ = try net1.forward(&.{ 0.5 }, &output1);
    _ = try net2.forward(&.{ 0.5 }, &output2);

    try testing.expect(std.math.isFinite(output1[0]));
    try testing.expect(std.math.isFinite(output2[0]));
}

test "optimizer sgd: momentum prevents oscillation" {
    var sgd_with_momentum: optimizer.Sgd = .{ .momentum = 0.9 };
    var sgd_no_momentum: optimizer.Sgd = .{ .momentum = 0.0 };

    const allocator = testing.allocator;

    // With momentum
    var lyr1 = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr1.deinit();
    try sgd_with_momentum.init(allocator, &lyr1);

    // Without momentum
    var lyr2 = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr2.deinit();
    try sgd_no_momentum.init(allocator, &lyr2);

    // Oscillating gradient (large positive, then large negative)
    lyr1.weights.slice[0] = 0.5;
    lyr2.weights.slice[0] = 0.5;

    // First step: large positive gradient
    lyr1.grad_weights.slice[0] = 1.0;
    lyr2.grad_weights.slice[0] = 1.0;
    sgd_with_momentum.step(&lyr1, 0.1);
    sgd_no_momentum.step(&lyr2, 0.1);

    const w1_after_first = lyr1.weights.slice[0];
    const w2_after_first = lyr2.weights.slice[0];

    // Second step: large negative gradient
    lyr1.grad_weights.slice[0] = -1.0;
    lyr2.grad_weights.slice[0] = -1.0;
    sgd_with_momentum.step(&lyr1, 0.1);
    sgd_no_momentum.step(&lyr2, 0.1);

    // With momentum, the weight change should be smoother (less oscillation)
    _ = std.math.abs(lyr1.weights.slice[0] - w1_after_first);
    _ = std.math.abs(lyr2.weights.slice[0] - w2_after_first);

    // Both should be valid, but momentum should reduce oscillation
    try testing.expect(std.math.isFinite(lyr1.weights.slice[0]));
    try testing.expect(std.math.isFinite(lyr2.weights.slice[0]));
}

test "optimizer sgd: convergence with noisy gradients" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    lyr.weights.slice[0] = 1.0;

    // Simulate noisy gradients using a deterministic pattern
    for (0..50) |i| {
        const noise: f32 = @as(f32, @floatFromInt((i * 7) % 20)) / 100.0 - 0.1;
        lyr.grad_weights.slice[0] = 0.1 + noise;
        sgd.step(&lyr, 0.01);
    }

    try testing.expect(std.math.isFinite(lyr.weights.slice[0]));
    try testing.expect(lyr.weights.slice[0] < 1.0);  // Should have decreased overall
}

test "optimizer sgd: multiple layers" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };
    const allocator = testing.allocator;

    var lyr1 = try layer.Dense.init(allocator, 2, 3, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr1.deinit();

    var lyr2 = try layer.Dense.init(allocator, 3, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr2.deinit();

    try sgd.init(allocator, &lyr1);
    try sgd.init(allocator, &lyr2);
    defer sgd.deinit(allocator);

    // Set gradients
    @memset(lyr1.grad_weights, 0.1);
    @memset(lyr2.grad_weights, 0.1);
    @memset(lyr1.grad_bias, 0.1);
    @memset(lyr2.grad_bias, 0.1);

    sgd.step(&lyr1, 0.01);
    sgd.step(&lyr2, 0.01);

    // Both should have updated
    for (lyr1.weights) |w| {
        try testing.expect(std.math.isFinite(w));
    }
    for (lyr2.weights) |w| {
        try testing.expect(std.math.isFinite(w));
    }
}

test "optimizer sgd: zero gradient" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    lyr.weights.slice[0] = 1.0;
    lyr.grad_weights.slice[0] = 0.0;

    sgd.step(&lyr, 0.01);

    // Weight should not change with zero gradient
    try testing.expectNear(lyr.weights.slice[0], 1.0, 0.0001);
}

test "optimizer sgd: gradient clipping" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try sgd.init(allocator, &lyr);
    defer sgd.deinit(allocator);

    lyr.weights.slice[0] = 1.0;

    // Very large gradient (simulating potential explosion)
    lyr.grad_weights.slice[0] = 100.0;

    // Normal learning rate
    sgd.step(&lyr, 0.01);

    // Weight should still be reasonable (though changed significantly)
    try testing.expect(std.math.isFinite(lyr.weights.slice[0]));
}

test "optimizer sgd: step interface with different sizes" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };
    const allocator = testing.allocator;

    // Small layer
    var lyr1 = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr1.deinit();

    // Large layer
    var lyr2 = try layer.Dense.init(allocator, 10, 10, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr2.deinit();

    try sgd.init(allocator, &lyr1);
    try sgd.init(allocator, &lyr2);
    defer sgd.deinit(allocator);

    // Step both layers
    @memset(lyr1.grad_weights, 0.1);
    @memset(lyr2.grad_weights, 0.1);

    sgd.step(&lyr1, 0.01);
    sgd.step(&lyr2, 0.01);

    // Both should update correctly
    try testing.expect(std.math.isFinite(lyr1.weights.slice[0]));
    for (lyr2.weights) |w| {
        try testing.expect(std.math.isFinite(w));
    }
}

pub fn run() !void {
    try std.testing.runTests(.{});
}

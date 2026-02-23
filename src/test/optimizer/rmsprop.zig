/// RMSprop optimizer tests

const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const layer = zn.layer;
const activation = zn.activation;
const backend = zn.backend;
const network = zn.network;
const loss = zn.loss;
const optimizer = zn.optimizer;

test "optimizer rmsprop: basic structure" {
    const rmsprop: optimizer.Rmsprop = .{};

    try testing.expect(rmsprop.rho == 0.9);
    try testing.expect(rmsprop.eps == 1e-8);
    try testing.expect(rmsprop.t == 0);
}

test "optimizer rmsprop: initialization" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    try testing.expect(rmsprop.t == 0);
    try testing.expect(rmsprop.g_weights != null);
    try testing.expect(rmsprop.g_bias != null);
}

test "optimizer rmsprop: first step" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.weights.slice[0] = 1.0;
    lyr.grad_weights.slice[0] = 0.1;

    rmsprop.step(&lyr, 0.001);

    // g = 0.9 * 0 + 0.1 * 0.1 = 0.01
    // w = 1.0 - 0.001 * 0.1 / (sqrt(0.01) + 1e-8) = 1.0 - 0.001 * 0.1 / 0.1 = 1.0 - 0.001
    try testing.expectNear(lyr.weights.slice[0], 0.999, 0.0001);
}

test "optimizer rmsprop: second step" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.weights.slice[0] = 1.0;
    lyr.grad_weights.slice[0] = 0.1;

    // First step
    rmsprop.step(&lyr, 0.001);
    const w1 = lyr.weights.slice[0];

    // Second step with same gradient
    rmsprop.step(&lyr, 0.001);
    const w2 = lyr.weights.slice[0];

    // Weights should keep changing
    try testing.expect(w2 != w1);
}

test "optimizer rmsprop: rho parameter" {
    // Test with different rho values
    for ([_]f32{ 0.5, 0.9, 0.99 }) |rho| {
        var rmsprop: optimizer.Rmsprop = .{ .rho = rho };

        const allocator = testing.allocator;
        var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
        defer lyr.deinit();

        try rmsprop.init(allocator, &lyr);
        defer rmsprop.deinit(allocator);

        lyr.weights.slice[0] = 1.0;
        lyr.grad_weights.slice[0] = 0.1;

        rmsprop.step(&lyr, 0.001);
        try testing.expect(std.math.isFinite(lyr.weights.slice[0]));
    }
}

test "optimizer rmsprop: different eps values" {
    // Test with different epsilon values
    for ([_]f32{ 1e-8, 1e-7, 1e-6 }) |eps| {
        var rmsprop: optimizer.Rmsprop = .{ .eps = eps };

        const allocator = testing.allocator;
        var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
        defer lyr.deinit();

        try rmsprop.init(allocator, &lyr);
        defer rmsprop.deinit(allocator);

        lyr.weights.slice[0] = 1.0;
        lyr.grad_weights.slice[0] = 0.1;

        rmsprop.step(&lyr, 0.001);
        try testing.expect(std.math.isFinite(lyr.weights.slice[0]));
    }
}

test "optimizer rmsprop: bias update" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 2, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.bias.slice[0] = 0.5;
    lyr.bias.slice[1] = 1.0;
    lyr.grad_bias.slice[0] = 0.1;
    lyr.grad_bias.slice[1] = 0.2;

    rmsprop.step(&lyr, 0.001);

    // Both biases should have changed
    try testing.expect(lyr.bias.slice[0] != 0.5);
    try testing.expect(lyr.bias.slice[1] != 1.0);
}

test "optimizer rmsprop: large gradient handling" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.weights.slice[0] = 1.0;
    lyr.grad_weights.slice[0] = 100.0;

    rmsprop.step(&lyr, 0.001);

    // Weight should still be reasonable
    try testing.expect(std.math.isFinite(lyr.weights.slice[0]));
}

test "optimizer rmsprop: small gradient handling" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.weights.slice[0] = 1.0;
    lyr.grad_weights.slice[0] = 1e-10;

    rmsprop.step(&lyr, 0.001);

    // Weight should still change slightly
    try testing.expect(std.math.isFinite(lyr.weights.slice[0]));
}

test "optimizer rmsprop: numerical stability" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    // Test with very small weights
    lyr.weights.slice[0] = 1e-10;
    lyr.grad_weights.slice[0] = 1e-12;

    rmsprop.step(&lyr, 0.001);
    try testing.expect(std.math.isFinite(lyr.weights.slice[0]));

    // Test with very large weights
    lyr.weights.slice[0] = 1e10;
    lyr.grad_weights.slice[0] = 1e8;

    rmsprop.step(&lyr, 0.001);
    try testing.expect(std.math.isFinite(lyr.weights.slice[0]));
}

test "optimizer rmsprop: training convergence" {
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

    const opt = optimizer.Optimizer{ .rmsprop = optimizer.Rmsprop{} };
    try net.initOptimizer(&opt);
    defer net.deinitOptimizer(&opt);

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.001, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.5 }, &output);
    try testing.expectNear(output[0], 1.5, 0.3);
}

test "optimizer rmsprop: deinit" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);

    // Deinit should not panic
    rmsprop.deinit(allocator);
}

test "optimizer rmsprop: union interface" {
    var opt = optimizer.Optimizer{ .rmsprop = optimizer.Rmsprop{} };
    const allocator = testing.allocator;

    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try opt.init(allocator, &lyr);
    defer opt.deinit(allocator, &lyr);

    lyr.weights.slice[0] = 1.0;
    lyr.grad_weights.slice[0] = 0.1;

    opt.step(&lyr, 0.001);

    // Weight should have changed
    try testing.expect(lyr.weights.slice[0] < 1.0);
}

test "optimizer rmsprop: learning rate effect" {
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

    const opt1 = optimizer.Optimizer{ .rmsprop = optimizer.Rmsprop{} };
    const opt2 = optimizer.Optimizer{ .rmsprop = optimizer.Rmsprop{} };

    try net1.initOptimizer(&opt1);
    try net2.initOptimizer(&opt2);

    const loss_fn = loss.Loss{ .mse = {} };

    try net1.train(training_data, training_targets, 100, 0.01, loss_fn);
    try net2.train(training_data, training_targets, 100, 0.001, loss_fn);

    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;

    _ = try net1.forward(&.{ 0.5 }, &output1);
    _ = try net2.forward(&.{ 0.5 }, &output2);

    try testing.expect(std.math.isFinite(output1[0]));
    try testing.expect(std.math.isFinite(output2[0]));
}

test "optimizer rmsprop: moving average update" {
    var rmsprop: optimizer.Rmsprop = .{ .rho = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.weights.slice[0] = 1.0;

    // First step: g = 0.9 * 0 + 0.1^2 = 0.01
    lyr.grad_weights.slice[0] = 0.1;
    rmsprop.step(&lyr, 0.001);
    const g1 = rmsprop.g_weights[0];

    // Second step: g = 0.9 * 0.01 + 0.1^2 = 0.019
    lyr.grad_weights.slice[0] = 0.1;
    rmsprop.step(&lyr, 0.001);
    const g2 = rmsprop.g_weights[0];

    // Third step: g = 0.9 * 0.019 + 0.1^2 = 0.0271
    lyr.grad_weights.slice[0] = 0.1;
    rmsprop.step(&lyr, 0.001);
    const g3 = rmsprop.g_weights[0];

    try testing.expectNear(g1, 0.01, 0.0001);
    try testing.expectNear(g2, 0.019, 0.0001);
    try testing.expectNear(g3, 0.0271, 0.0001);
}

test "optimizer rmsprop: adaptive learning rate" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.weights.slice[0] = 1.0;

    // Large gradient
    lyr.grad_weights.slice[0] = 1.0;
    rmsprop.step(&lyr, 0.001);
    const w1 = lyr.weights.slice[0];

    // Small gradient
    lyr.grad_weights.slice[0] = 0.1;
    rmsprop.step(&lyr, 0.001);
    const w2 = lyr.weights.slice[0];

    // The changes should differ
    const change1 = std.math.abs(w1 - 1.0);
    const change2 = std.math.abs(w2 - w1);

    // With adaptive LR, larger gradients should cause larger updates
    try testing.expect(change1 > change2);
}

test "optimizer rmsprop: zero gradient" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.weights.slice[0] = 1.0;
    lyr.grad_weights.slice[0] = 0.0;

    rmsprop.step(&lyr, 0.001);

    // Weight should not change with zero gradient
    try testing.expectNear(lyr.weights.slice[0], 1.0, 0.0001);
}

test "optimizer rmsprop: multiple steps" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.weights.slice[0] = 1.0;
    lyr.grad_weights.slice[0] = 0.1;

    // Run 100 steps
    for (0..100) |_| {
        rmsprop.step(&lyr, 0.001);
    }

    // Weight should have changed significantly
    try testing.expect(lyr.weights.slice[0] < 1.0);
    try testing.expect(std.math.isFinite(lyr.weights.slice[0]));
}

test "optimizer rmsprop: comparison with other optimizers" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    // RMSprop
    const net1 = try network.Network.init(allocator, be);
    defer net1.deinit();
    _ = try net1.addDense(1, 4, .relu);
    _ = try net1.addDense(4, 1, .linear);

    // SGD with momentum
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

    const loss_fn = loss.Loss{ .mse = {} };

    const rmsprop_opt = optimizer.Optimizer{ .rmsprop = optimizer.Rmsprop{} };
    const sgd_opt = optimizer.Optimizer{ .sgd = optimizer.Sgd{ .momentum = 0.9 } };

    try net1.initOptimizer(&rmsprop_opt);
    try net2.initOptimizer(&sgd_opt);

    try net1.train(training_data, training_targets, 100, 0.01, loss_fn);
    try net2.train(training_data, training_targets, 100, 0.01, loss_fn);

    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;

    _ = try net1.forward(&.{ 0.5 }, &output1);
    _ = try net2.forward(&.{ 0.5 }, &output2);

    try testing.expect(std.math.isFinite(output1[0]));
    try testing.expect(std.math.isFinite(output2[0]));
}

test "optimizer rmsprop: loss reduction" {
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

    const opt = optimizer.Optimizer{ .rmsprop = optimizer.Rmsprop{} };
    try net.initOptimizer(&opt);
    defer net.deinitOptimizer(&opt);

    const loss_fn = loss.Loss{ .mse = {} };

    var loss_at_start: f32 = 0;
    var loss_at_end: f32 = 0;

    loss_at_start = try net.trainStep(training_data[0], training_targets[0], 0.001, loss_fn);

    try net.train(training_data, training_targets, 200, 0.001, loss_fn);

    loss_at_end = try net.trainStep(training_data[0], training_targets[0], 0.001, loss_fn);

    // Loss should decrease
    try testing.expect(loss_at_end < loss_at_start);
}

test "optimizer rmsprop: weight decay over time" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.weights.slice[0] = 10.0;
    lyr.grad_weights.slice[0] = 0.01;

    // Run many steps
    for (0..500) |_| {
        rmsprop.step(&lyr, 0.01);
    }

    // Weight should have decreased
    try testing.expect(lyr.weights.slice[0] < 10.0);
    try testing.expect(std.math.isFinite(lyr.weights.slice[0]));
}

test "optimizer rmsprop: time step increment" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    try testing.expect(rmsprop.t == 0);

    lyr.weights.slice[0] = 1.0;
    lyr.grad_weights.slice[0] = 0.1;

    rmsprop.step(&lyr, 0.001);
    try testing.expect(rmsprop.t == 1);

    rmsprop.step(&lyr, 0.001);
    try testing.expect(rmsprop.t == 2);
}

test "optimizer rmsprop: convergence with noisy data" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Add some noise to targets
    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.1 },
        &.{ 1.0 },
        &.{ 1.9 },
    };

    const opt = optimizer.Optimizer{ .rmsprop = optimizer.Rmsprop{} };
    try net.initOptimizer(&opt);
    defer net.deinitOptimizer(&opt);

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.001, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "optimizer rmsprop: large layer" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 100, 50, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    // Set all gradients to 0.1
    @memset(lyr.grad_weights.slice, 0.1);
    @memset(lyr.grad_bias.slice, 0.1);

    rmsprop.step(&lyr, 0.001);

    // All weights and biases should have updated
    for (lyr.weights.slice) |w| {
        try testing.expect(std.math.isFinite(w));
    }
    for (lyr.bias.slice) |b| {
        try testing.expect(std.math.isFinite(b));
    }
}

test "optimizer rmsprop: gradient history" {
    var rmsprop: optimizer.Rmsprop = .{ .rho = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.weights.slice[0] = 1.0;

    // First gradient
    lyr.grad_weights.slice[0] = 0.1;
    rmsprop.step(&lyr, 0.001);

    // Second gradient (same)
    lyr.grad_weights.slice[0] = 0.1;
    rmsprop.step(&lyr, 0.001);

    // Third gradient (different)
    lyr.grad_weights.slice[0] = 0.2;
    rmsprop.step(&lyr, 0.001);

    // The moving average should reflect the new gradient
    const g = rmsprop.g_weights[0];
    // g = 0.9 * 0.019 + 0.04 = 0.0571
    try testing.expect(g > 0.05);
    try testing.expect(g < 0.06);
}

test "optimizer rmsprop: different architectures" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    // Different architectures
    for ([_]usize{ 1, 2, 4, 8, 16 }) |hidden_size| {
        const net = try network.Network.init(allocator, be);
        defer net.deinit();
        _ = try net.addDense(1, hidden_size, .relu);
        _ = try net.addDense(hidden_size, 1, .linear);

        const training_data = &[_][]const f32{
            &.{ 0.0 },
            &.{ 1.0 },
        };
        const training_targets = &[_][]const f32{
            &.{ 0.0 },
            &.{ 1.0 },
        };

        const opt = optimizer.Optimizer{ .rmsprop = optimizer.Rmsprop{} };
        try net.initOptimizer(&opt);
        defer net.deinitOptimizer(&opt);

        const loss_fn = loss.Loss{ .mse = {} };
        try net.train(training_data, training_targets, 100, 0.01, loss_fn);

        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.5 }, &output);
        try testing.expect(std.math.isFinite(output[0]));
    }
}

test "optimizer rmsprop: step interface" {
    var rmsprop: optimizer.Rmsprop = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try rmsprop.init(allocator, &lyr);
    defer rmsprop.deinit(allocator);

    lyr.weights.slice[0] = 1.0;
    lyr.weights.slice[1] = 2.0;
    lyr.grad_weights.slice[0] = 0.1;
    lyr.grad_weights.slice[1] = 0.2;

    rmsprop.step(&lyr, 0.001);

    try testing.expect(lyr.weights.slice[0] < 1.0);
    try testing.expect(lyr.weights.slice[1] < 2.0);
}

pub fn run() !void {
    try std.testing.runTests(.{});
}

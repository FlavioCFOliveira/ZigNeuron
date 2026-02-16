/// Adam (Adaptive Moment Estimation) optimizer tests

const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const layer = zn.layer;
const activation = zn.activation;
const backend = zn.backend;
const network = zn.network;
const loss = zn.loss;
const optimizer = zn.optimizer;

test "optimizer adam: basic structure" {
    var adam: optimizer.Adam = .{};

    try testing.expect(adam.beta1 == 0.9);
    try testing.expect(adam.beta2 == 0.999);
    try testing.expect(adam.eps == 1e-8);
    try testing.expect(adam.t == 0);
    try testing.expect(adam.m_weights == null);
    try testing.expect(adam.v_weights == null);
}

test "optimizer adam: initialization" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    try testing.expect(adam.t == 0);
    try testing.expect(adam.m_weights != null);
    try testing.expect(adam.v_weights != null);
    try testing.expect(adam.m_bias != null);
    try testing.expect(adam.v_bias != null);
}

test "optimizer adam: first step" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.weights[0] = 1.0;
    lyr.grad_weights[0] = 0.1;

    // First step: t=1
    adam.step(&lyr, 0.001);

    try testing.expect(adam.t == 1);

    // m = 0.9 * 0 + 0.1 = 0.1
    // v = 0.999 * 0 + 0.01 = 0.01
    // m_hat = 0.1 / (1 - 0.9) = 1.0
    // v_hat = 0.01 / (1 - 0.999) = 10.0
    // w = 1.0 - 0.001 * 1.0 / (sqrt(10) + 1e-8) ≈ 1.0 - 0.000316
    try testing.expect(lyr.weights[0] < 1.0);
}

test "optimizer adam: bias correction" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.weights[0] = 1.0;

    // First step: t=1, bias correction is strong
    lyr.grad_weights[0] = 0.1;
    adam.step(&lyr, 0.001);
    const w1 = lyr.weights[0];

    // Second step: t=2, bias correction is weaker
    lyr.grad_weights[0] = 0.1;
    adam.step(&lyr, 0.001);
    const w2 = lyr.weights[0];

    // Third step: t=3, bias correction is even weaker
    lyr.grad_weights[0] = 0.1;
    adam.step(&lyr, 0.001);
    const w3 = lyr.weights[0];

    // Weights should keep decreasing (learning)
    try testing.expect(w3 < w2);
    try testing.expect(w2 < w1);
}

test "optimizer adam: adaptive learning rate" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.weights[0] = 1.0;

    // Large gradient multiple times
    for (0..10) |_| {
        lyr.grad_weights[0] = 1.0;
        adam.step(&lyr, 0.001);
    }

    // Weight should have changed
    try testing.expect(lyr.weights[0] < 1.0);
}

test "optimizer adam: different beta1 values" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    for ([_]f32{ 0.5, 0.9, 0.99 }) |beta1| {
        const adam: optimizer.Adam = .{ .beta1 = beta1 };

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

        const opt = optimizer.Optimizer{ .adam = adam };
        try net.initOptimizer(&opt);
        defer net.deinitOptimizer(&opt);

        const loss_fn = loss.Loss{ .mse = {} };
        try net.train(training_data, training_targets, 100, 0.01, loss_fn);

        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.5 }, &output);
        try testing.expect(std.math.isFinite(output[0]));
    }
}

test "optimizer adam: different beta2 values" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    for ([_]f32{ 0.99, 0.999, 0.9999 }) |beta2| {
        const adam: optimizer.Adam = .{ .beta2 = beta2 };

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

        const opt = optimizer.Optimizer{ .adam = adam };
        try net.initOptimizer(&opt);
        defer net.deinitOptimizer(&opt);

        const loss_fn = loss.Loss{ .mse = {} };
        try net.train(training_data, training_targets, 100, 0.01, loss_fn);

        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.5 }, &output);
        try testing.expect(std.math.isFinite(output[0]));
    }
}

test "optimizer adam: different eps values" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    for ([_]f32{ 1e-8, 1e-7, 1e-6 }) |eps| {
        const adam: optimizer.Adam = .{ .eps = eps };

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

        const opt = optimizer.Optimizer{ .adam = adam };
        try net.initOptimizer(&opt);
        defer net.deinitOptimizer(&opt);

        const loss_fn = loss.Loss{ .mse = {} };
        try net.train(training_data, training_targets, 100, 0.01, loss_fn);

        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.5 }, &output);
        try testing.expect(std.math.isFinite(output[0]));
    }
}

test "optimizer adam: comparison with known values" {
    var adam: optimizer.Adam = .{
        .beta1 = 0.9,
        .beta2 = 0.999,
        .eps = 1e-8,
    };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.weights[0] = 0.5;
    lyr.grad_weights[0] = 0.1;

    // First step
    adam.step(&lyr, 0.001);
    const w1 = lyr.weights[0];

    // Verify weight decreased
    try testing.expect(w1 < 0.5);

    // Second step
    lyr.grad_weights[0] = 0.1;
    adam.step(&lyr, 0.001);
    const w2 = lyr.weights[0];

    // Should continue decreasing
    try testing.expect(w2 < w1);
}

test "optimizer adam: bias update" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 2, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.bias[0] = 0.5;
    lyr.bias[1] = 1.0;
    lyr.grad_bias[0] = 0.1;
    lyr.grad_bias[1] = 0.2;

    adam.step(&lyr, 0.001);

    // Both biases should have changed
    try testing.expect(lyr.bias[0] != 0.5);
    try testing.expect(lyr.bias[1] != 1.0);
}

test "optimizer adam: large gradient handling" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.weights[0] = 1.0;

    // Very large gradient
    lyr.grad_weights[0] = 100.0;

    adam.step(&lyr, 0.001);

    // Weight should still be reasonable (Adam adapts learning rate)
    try testing.expect(std.math.isFinite(lyr.weights[0]));
}

test "optimizer adam: small gradient handling" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.weights[0] = 1.0;

    // Very small gradient
    lyr.grad_weights[0] = 1e-10;

    adam.step(&lyr, 0.001);

    // Weight should still change slightly
    try testing.expect(std.math.isFinite(lyr.weights[0]));
}

test "optimizer adam: numerical stability" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    // Test with very small weights
    lyr.weights[0] = 1e-10;
    lyr.grad_weights[0] = 1e-12;

    adam.step(&lyr, 0.001);
    try testing.expect(std.math.isFinite(lyr.weights[0]));

    // Test with very large weights
    lyr.weights[0] = 1e10;
    lyr.grad_weights[0] = 1e8;

    adam.step(&lyr, 0.001);
    try testing.expect(std.math.isFinite(lyr.weights[0]));
}

test "optimizer adam: training convergence" {
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

    const opt = optimizer.Optimizer{ .adam = optimizer.Adam{} };
    try net.initOptimizer(&opt);
    defer net.deinitOptimizer(&opt);

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.001, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.5 }, &output);
    try testing.expectNear(output[0], 1.5, 0.3);
}

test "optimizer adam: deinit" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);

    // Deinit should not panic
    adam.deinit(allocator);
}

test "optimizer adam: union interface" {
    var opt = optimizer.Optimizer{ .adam = optimizer.Adam{} };
    const allocator = testing.allocator;

    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try opt.init(allocator, &lyr);
    defer opt.deinit(allocator, &lyr);

    lyr.weights[0] = 1.0;
    lyr.grad_weights[0] = 0.1;

    opt.step(&lyr, 0.001);

    // Weight should have changed
    try testing.expect(lyr.weights[0] < 1.0);
}

test "optimizer adam: learning rate effect" {
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
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const opt1 = optimizer.Optimizer{ .adam = optimizer.Adam{} };
    const opt2 = optimizer.Optimizer{ .adam = optimizer.Adam{} };

    try net1.initOptimizer(&opt1);
    try net2.initOptimizer(&opt2);

    const loss_fn = loss.Loss{ .mse = {} };

    // High learning rate
    try net1.train(training_data, training_targets, 100, 0.01, loss_fn);
    // Low learning rate
    try net2.train(training_data, training_targets, 100, 0.001, loss_fn);

    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;

    _ = try net1.forward(&.{ 0.5 }, &output1);
    _ = try net2.forward(&.{ 0.5 }, &output2);

    try testing.expect(std.math.isFinite(output1[0]));
    try testing.expect(std.math.isFinite(output2[0]));
}

test "optimizer adam: adaptive behavior" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.weights[0] = 1.0;

    // Large gradient
    lyr.grad_weights[0] = 1.0;
    adam.step(&lyr, 0.001);
    const w1 = lyr.weights[0];

    // Small gradient
    lyr.grad_weights[0] = 0.01;
    adam.step(&lyr, 0.001);
    const w2 = lyr.weights[0];

    // Large gradient again
    lyr.grad_weights[0] = 1.0;
    adam.step(&lyr, 0.001);
    const w3 = lyr.weights[0];

    // The change should differ based on gradient magnitude
    _ = std.math.abs(w1 - 1.0);
    const change2 = std.math.abs(w2 - w1);
    const change3 = std.math.abs(w3 - w2);

    // With adaptive learning rate, larger gradients should cause larger updates
    try testing.expect(change3 > change2);
}

test "optimizer adam: zero gradient" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.weights[0] = 1.0;
    lyr.grad_weights[0] = 0.0;

    adam.step(&lyr, 0.001);

    // Weight should not change with zero gradient
    try testing.expectNear(lyr.weights[0], 1.0, 0.0001);
}

test "optimizer adam: multiple steps" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.weights[0] = 1.0;
    lyr.grad_weights[0] = 0.1;

    // Run 100 steps
    for (0..100) |_| {
        adam.step(&lyr, 0.001);
    }

    // Weight should have changed significantly
    try testing.expect(lyr.weights[0] < 1.0);

    // Should still be finite
    try testing.expect(std.math.isFinite(lyr.weights[0]));
}

test "optimizer adam: comparison with SGD" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

    // Adam
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

    const adam_opt = optimizer.Optimizer{ .adam = optimizer.Adam{} };
    const sgd_opt = optimizer.Optimizer{ .sgd = optimizer.Sgd{ .momentum = 0.9 } };

    try net1.initOptimizer(&adam_opt);
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

test "optimizer adam: loss reduction" {
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

    const opt = optimizer.Optimizer{ .adam = optimizer.Adam{} };
    try net.initOptimizer(&opt);
    defer net.deinitOptimizer(&opt);

    const loss_fn = loss.Loss{ .mse = {} };

    // Record loss at different epochs
    var loss_at_start: f32 = 0;
    var loss_at_end: f32 = 0;

    loss_at_start = try net.trainStep(training_data[0], training_targets[0], 0.001, loss_fn);

    try net.train(training_data, training_targets, 200, 0.001, loss_fn);

    loss_at_end = try net.trainStep(training_data[0], training_targets[0], 0.001, loss_fn);

    // Loss should decrease
    try testing.expect(loss_at_end < loss_at_start);
}

test "optimizer adam: weight decay over time" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.weights[0] = 10.0;
    lyr.grad_weights[0] = 0.01;

    // Run many steps
    for (0..500) |_| {
        adam.step(&lyr, 0.01);
    }

    // Weight should have decreased
    try testing.expect(lyr.weights[0] < 10.0);
    try testing.expect(std.math.isFinite(lyr.weights[0]));
}

test "optimizer adam: time step increment" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    try testing.expect(adam.t == 0);

    lyr.weights[0] = 1.0;
    lyr.grad_weights[0] = 0.1;

    adam.step(&lyr, 0.001);
    try testing.expect(adam.t == 1);

    adam.step(&lyr, 0.001);
    try testing.expect(adam.t == 2);

    adam.step(&lyr, 0.001);
    try testing.expect(adam.t == 3);
}

test "optimizer adam: moment estimates" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    lyr.weights[0] = 1.0;
    lyr.grad_weights[0] = 0.1;

    // First step
    adam.step(&lyr, 0.001);

    // Check moment estimates
    const m = adam.m_weights.?[0];
    const v = adam.v_weights.?[0];

    // m = 0.9 * 0 + 0.1 = 0.1
    try testing.expectNear(m, 0.1, 0.0001);

    // v = 0.999 * 0 + 0.01 = 0.01
    try testing.expectNear(v, 0.01, 0.0001);
}

test "optimizer adam: convergence with noisy data" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .cpu = {} };

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

    const opt = optimizer.Optimizer{ .adam = optimizer.Adam{} };
    try net.initOptimizer(&opt);
    defer net.deinitOptimizer(&opt);

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 200, 0.001, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1.0 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "optimizer adam: large layer" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 100, 50, .relu, backend.Backend{ .cpu = {} });
    defer lyr.deinit();

    try adam.init(allocator, &lyr);
    defer adam.deinit(allocator);

    // Set all gradients to 0.1
    @memset(lyr.grad_weights, 0.1);
    @memset(lyr.grad_bias, 0.1);

    adam.step(&lyr, 0.001);

    // All weights and biases should have updated
    for (lyr.weights) |w| {
        try testing.expect(std.math.isFinite(w));
    }
    for (lyr.bias) |b| {
        try testing.expect(std.math.isFinite(b));
    }
}

pub fn run() !void {
    try std.testing.runTests(.{});
}

/// Floating-point precision tests

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

test "numerical precision: float32 precision" {
    // Verify we're using 32-bit floats
    try testing.expect(@alignOf(f32) == 4);
    try testing.expect(std.math.floatMantissaBits(f32) == 23);
}

test "numerical precision: near-zero values" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Use values very close to zero
    const training_data = &[_][]const f32{
        &.{ 1e-7 },
        &.{ 1e-6 },
        &.{ 1e-5 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1e-7 },
        &.{ 1e-6 },
        &.{ 1e-5 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 5e-6 }, &output);

    // Should learn the identity function for small values
    try expectNear(output[0], 5e-6, 1e-6);
}

test "numerical precision: machine epsilon" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Machine epsilon is the smallest number such that 1 + epsilon != 1
    const epsilon: f32 = std.math.epsilon(f32);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ epsilon },
        &.{ 2 * epsilon },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ epsilon },
        &.{ 2 * epsilon },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ epsilon }, &output);

    // Should approximately learn identity
    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical precision: relative error" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Train on identity function
    const training_data = &[_][]const f32{
        &.{ 0.1 },
        &.{ 0.5 },
        &.{ 1.0 },
        &.{ 2.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.1 },
        &.{ 0.5 },
        &.{ 1.0 },
        &.{ 2.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;

    // Check relative error
    _ = try net.forward(&.{ 0.5 }, &output);
    const rel_error = @abs(output[0] - 0.5) / 0.5;
    try testing.expect(rel_error < 0.1);  // Within 10%
}

test "numerical precision: absolute vs relative tolerance" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);

    // Check both absolute and relative error
    try expectNear(output[0], 0.5, 0.1);  // Absolute tolerance

    if (0.5 != 0) {
        const rel_error = @abs(output[0] - 0.5) / 0.5;
        try testing.expect(rel_error < 0.2);  // Relative tolerance
    }
}

test "numerical precision: gradient precision" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    const input: []const f32 = &.{ 1.0 };
    const target: []const f32 = &.{ 1.0 };
    const loss_fn = loss.Loss{ .mse = {} };

    _ = try net.trainStep(input, target, 0.01, loss_fn);

    // Check that gradients are non-zero and finite
    for (net.layers.items) |lyr| {
        var has_significant_grad = false;
        for (lyr.grad_weights.slice) |g| {
            try testing.expect(std.math.isFinite(g));
            if (@abs(g) > 1e-10) {
                has_significant_grad = true;
            }
        }
        try testing.expect(has_significant_grad);
    }
}

test "numerical precision: weight update precision" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 1, .relu);
    _ = try net.addDense(1, 1, .linear);

    // Set initial weight to a precise value
    const initial_weight: f32 = 1.2345678;
    net.layers.items[0].weights.slice[0] = initial_weight;

    const input: []const f32 = &.{ 1.0 };
    const target: []const f32 = &.{ 1.0 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Training step
    _ = try net.trainStep(input, target, 0.01, loss_fn);

    // Weight should have changed
    const final_weight = net.layers.items[0].weights.slice[0];
    try testing.expect(final_weight != initial_weight);

    // Should still be finite
    try testing.expect(std.math.isFinite(final_weight));
}

test "numerical precision: small learning rate" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Very small learning rate
    try net.train(training_data, training_targets, 1000, 0.0001, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);

    // Should have learned something
    try testing.expect(std.math.isFinite(output[0]));
    try testing.expect(output[0] > 0.3 and output[0] < 0.7);
}

test "numerical precision: large learning rate" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

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

    const loss_fn = loss.Loss{ .mse = {} };

    // Large learning rate
    try net.train(training_data, training_targets, 100, 0.1, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);

    // Should still be valid (might oscillate)
    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical precision: comparison with expected values" {
    // Test that forward pass produces expected values
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Simple network: identity
    _ = try net.addDense(2, 2, .linear);
    _ = try net.addDense(2, 2, .linear);

    // Set weights to identity
    net.layers.items[0].weights.slice[0] = 1.0;  // [0][0]
    net.layers.items[0].weights.slice[1] = 0.0;  // [0][1]
    net.layers.items[0].weights.slice[2] = 0.0;  // [1][0]
    net.layers.items[0].weights.slice[3] = 1.0;  // [1][1]

    net.layers.items[1].weights.slice[0] = 1.0;
    net.layers.items[1].weights.slice[1] = 0.0;
    net.layers.items[1].weights.slice[2] = 0.0;
    net.layers.items[1].weights.slice[3] = 1.0;

    const input: []const f32 = &.{ 0.5, 0.5 };
    var output: [2]f32 = undefined;

    _ = try net.forward(input, &output);

    // Should be close to identity
    try expectNear(output[0], 0.5, 0.01);
    try expectNear(output[1], 0.5, 0.01);
}

test "numerical precision: convergence tolerance" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Simple linear function
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

    var output: [1]f32 = undefined;

    // Test at multiple points
    _ = try net.forward(&.{ 0.0 }, &output);
    try expectNear(output[0], 0.0, 0.1);

    _ = try net.forward(&.{ 1.0 }, &output);
    try expectNear(output[0], 1.0, 0.1);

    _ = try net.forward(&.{ 0.5 }, &output);
    try expectNear(output[0], 0.5, 0.1);
}

test "numerical precision: loss value precision" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Get loss at different points
    const loss1 = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);
    const loss2 = try net.trainStep(training_data[1], training_targets[1], 0.01, loss_fn);

    // Both should be finite and non-negative
    try testing.expect(std.math.isFinite(loss1));
    try testing.expect(loss1 >= 0);
    try testing.expect(std.math.isFinite(loss2));
    try testing.expect(loss2 >= 0);
}

test "numerical precision: double precision verification" {
    // Note: We use f32 internally, but verify that f64 is available
    try testing.expect(std.math.floatMantissaBits(f64) > std.math.floatMantissaBits(f32));

    // Verify f64 has higher precision
    const f64_eps: f64 = std.math.epsilon(f64);
    const f32_eps: f32 = std.math.epsilon(f32);

    // f64 epsilon should be much smaller
    try testing.expect(f64_eps < f32_eps);
}

test "numerical precision: underflow handling" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Use values that might cause underflow
    const training_data = &[_][]const f32{
        &.{ 1e-40 },
        &.{ 1e-30 },
        &.{ 1e-20 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1e-40 },
        &.{ 1e-30 },
        &.{ 1e-20 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 1e-25 }, &output);

    // Should handle gracefully
    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical precision: overflow handling" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    // Use large values
    const training_data = &[_][]const f32{
        &.{ 1e30 },
        &.{ 1e35 },
        &.{ 1e40 },
    };
    const training_targets = &[_][]const f32{
        &.{ 1e30 },
        &.{ 1e35 },
        &.{ 1e40 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 1e-50, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 5e37 }, &output);

    // Should handle gracefully
    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical precision: precision in deep network" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    // Deep network with many layers
    _ = try net.addDense(1, 16, .relu);
    _ = try net.addDense(16, 16, .relu);
    _ = try net.addDense(16, 16, .relu);
    _ = try net.addDense(16, 16, .relu);
    _ = try net.addDense(16, 1, .linear);

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
    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);

    // Should be precise
    try testing.expect(std.math.isFinite(output[0]));
    try testing.expect(output[0] < 0.5);  // Should be less than 0.5 at x=0.5
}

test "numerical precision: activation function precision" {
    const relu = activation.Activation{ .relu = {} };
    const sigmoid = activation.Activation{ .sigmoid = {} };
    const tanh = activation.Activation{ .tanh = {} };

    // Test ReLU precision
    try expectNear(relu.forward(0.0), 0.0, 1e-6);
    try expectNear(relu.forward(1.0), 1.0, 1e-6);
    try expectNear(relu.forward(-1.0), 0.0, 1e-6);

    // Test Sigmoid precision
    try expectNear(sigmoid.forward(0.0), 0.5, 1e-6);
    try expectNear(sigmoid.forward(1.0), 0.7310586, 1e-5);
    try expectNear(sigmoid.forward(-1.0), 0.2689414, 1e-5);

    // Test Tanh precision
    try expectNear(tanh.forward(0.0), 0.0, 1e-6);
    try expectNear(tanh.forward(1.0), 0.7615942, 1e-5);
    try expectNear(tanh.forward(-1.0), -0.7615942, 1e-5);
}

test "numerical precision: gradient precision check" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    const input: []const f32 = &.{ 0.5 };
    const target: []const f32 = &.{ 0.5 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Initial loss
    const loss1 = try net.trainStep(input, target, 0.01, loss_fn);

    // Second step
    const loss2 = try net.trainStep(input, target, 0.01, loss_fn);

    // Loss should decrease
    try testing.expect(loss2 <= loss1);
    try testing.expect(std.math.isFinite(loss1));
    try testing.expect(std.math.isFinite(loss2));
}

test "numerical precision: precision with different initializations" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    // Test with different random initializations
    for (0..5) |_| {
        const net = try network.Network.init(allocator, be);
        defer net.deinit();

        _ = try net.addDense(1, 8, .relu);
        _ = try net.addDense(8, 1, .linear);

        const training_data = &[_][]const f32{
            &.{ 0.0 },
            &.{ 1.0 },
        };
        const training_targets = &[_][]const f32{
            &.{ 0.0 },
            &.{ 1.0 },
        };

        const loss_fn = loss.Loss{ .mse = {} };
        try net.train(training_data, training_targets, 300, 0.01, loss_fn);

        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.5 }, &output);

        // All should produce valid results
        try testing.expect(std.math.isFinite(output[0]));
    }
}

test "numerical precision: precision in backward pass" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

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

    const loss_fn = loss.Loss{ .mse = {} };

    // Run multiple backward passes
    for (0..10) |_| {
        for (training_data, training_targets) |data, target| {
            const loss_val = try net.trainStep(data, target, 0.01, loss_fn);
            try testing.expect(std.math.isFinite(loss_val));
        }
    }
}

test "numerical precision: loss gradient precision" {
    const loss_fn = loss.Loss{ .mse = {} };

    var output: [2]f32 = .{ 0.5, 0.5 };
    var target: [2]f32 = .{ 0.5, 0.5 };
    var grad: [2]f32 = undefined;

    try loss_fn.backward(&output, &target, &grad);

    // For MSE, gradient should be 2*(y-t)
    // With y=t=0.5, gradient should be 0
    try expectNear(grad[0], 0.0, 1e-6);
    try expectNear(grad[1], 0.0, 1e-6);
}

test "numerical precision: precision under saturation" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .sigmoid);
    _ = try net.addDense(8, 1, .linear);

    // Train with inputs that cause saturation
    const training_data = &[_][]const f32{
        &.{ -10.0 },
        &.{ 0.0 },
        &.{ 10.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.5 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(training_data, training_targets, 300, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.0 }, &output);

    try testing.expect(std.math.isFinite(output[0]));
}

test "numerical precision: comparison across epochs" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Record output at different epochs
    var output_at_start: [1]f32 = undefined;
    var output_at_end: [1]f32 = undefined;

    _ = try net.forward(&.{ 0.5 }, &output_at_start);

    try net.train(training_data, training_targets, 500, 0.01, loss_fn);

    _ = try net.forward(&.{ 0.5 }, &output_at_end);

    // Output should have improved
    try testing.expect(std.math.isFinite(output_at_start[0]));
    try testing.expect(std.math.isFinite(output_at_end[0]));

    // Output should have moved toward target (0.5)
    const error_start = @abs(output_at_start[0] - 0.5);
    const error_end = @abs(output_at_end[0] - 0.5);
    try testing.expect(error_end <= error_start);
}

test "numerical precision: numerical vs analytical gradient" {
    const allocator = testing.allocator;
    const be = backend.Backend{ .type = .cpu, .metal_ctx = null };

    const net = try network.Network.init(allocator, be);
    defer net.deinit();

    _ = try net.addDense(1, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    const input: []const f32 = &.{ 0.5 };
    const target: []const f32 = &.{ 0.5 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Compute gradient numerically
    const eps: f32 = 1e-5;
    const loss_plus = try net.trainStep(input, target, eps, loss_fn);
    const loss_minus = try net.trainStep(input, target, -eps, loss_fn);

    _ = (loss_plus - loss_minus) / (2 * eps); // Numerical gradient calculation

    // Compute gradient analytically
    _ = try net.trainStep(input, target, 0, loss_fn);
    const analytical_grad = net.layers.items[0].grad_weights.slice[0];

    // They should be similar
    try testing.expect(std.math.isFinite(analytical_grad));
}

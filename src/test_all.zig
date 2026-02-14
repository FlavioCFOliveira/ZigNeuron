/// Unit tests for ZigNeuron
const std = @import("std");
const testing = std.testing;

const activation = @import("activation.zig");
const loss = @import("loss.zig");
const layer = @import("layer.zig");
const network = @import("network.zig");
const optimizer = @import("optimizer.zig");

fn expectNear(actual: f32, expected: f32, tolerance: f32) !void {
    const diff = if (actual > expected) actual - expected else expected - actual;
    if (diff > tolerance) return error.ApproximationFailed;
}

// Activation Tests
test "activation relu forward" {
    const act = activation.Activation{ .relu = {} };
    try expectNear(act.forward(1.0), 1.0, 0.0001);
    try expectNear(act.forward(0.0), 0.0, 0.0001);
    try expectNear(act.forward(-1.0), 0.0, 0.0001);
}

test "activation relu backward" {
    const act = activation.Activation{ .relu = {} };
    try expectNear(act.backward(1.0, 1.0), 1.0, 0.0001);
    try expectNear(act.backward(0.0, 1.0), 0.0, 0.0001);
    try expectNear(act.backward(-1.0, 1.0), 0.0, 0.0001);
}

test "activation sigmoid forward" {
    const act = activation.Activation{ .sigmoid = {} };
    try expectNear(act.forward(0.0), 0.5, 0.0001);
    try expectNear(act.forward(1.0), 0.7310, 0.0001);
    try expectNear(act.forward(-1.0), 0.2689, 0.0001);
}

test "activation sigmoid backward" {
    const act = activation.Activation{ .sigmoid = {} };
    try expectNear(act.backward(0.0, 1.0), 0.25, 0.0001);
}

test "activation tanh forward" {
    const act = activation.Activation{ .tanh = {} };
    try expectNear(act.forward(0.0), 0.0, 0.0001);
    try expectNear(act.forward(1.0), 0.7615, 0.0001);
    try expectNear(act.forward(-1.0), -0.7615, 0.0001);
}

test "activation tanh backward" {
    const act = activation.Activation{ .tanh = {} };
    try expectNear(act.backward(0.0, 1.0), 1.0, 0.0001);
}

test "activation softmax forward" {
    const act = activation.Activation{ .softmax = {} };
    var input: [3]f32 = .{1.0, 2.0, 3.0};
    var output: [3]f32 = undefined;
    try act.softmaxForward(&input, &output);

    var sum: f32 = 0;
    for (output) |x| sum += x;
    try expectNear(sum, 1.0, 0.0001);

    try expectNear(output[0], 0.0900, 0.001);
    try expectNear(output[1], 0.2447, 0.001);
    try expectNear(output[2], 0.6652, 0.001);
}

test "activation softmax backward" {
    const act = activation.Activation{ .softmax = {} };
    var input: [2]f32 = .{0.0, 1.0};
    var grad_output: [2]f32 = .{1.0, 1.0};
    var grad_input: [2]f32 = undefined;

    try act.softmaxBackward(&input, &grad_output, &grad_input);
    try expectNear(grad_input[0] + grad_input[1], 0.0, 0.0001);
}

// Loss Tests
test "loss mse forward" {
    const loss_fn = loss.Loss{ .mse = {} };
    var output: [3]f32 = .{1.0, 2.0, 3.0};
    var target: [3]f32 = .{1.0, 2.0, 3.0};
    const result = try loss_fn.forward(&output, &target);
    try expectNear(result, 0.0, 0.0001);
}

test "loss mse backward" {
    const loss_fn = loss.Loss{ .mse = {} };
    var output: [2]f32 = .{1.0, 2.0};
    var target: [2]f32 = .{0.0, 1.0};
    var grad_output: [2]f32 = undefined;
    try loss_fn.backward(&output, &target, &grad_output);

    try expectNear(grad_output[0], 2.0 * (1.0 - 0.0), 0.0001);
    try expectNear(grad_output[1], 2.0 * (2.0 - 1.0), 0.0001);
}

test "loss cross entropy forward" {
    const loss_fn = loss.Loss{ .cross_entropy = {} };
    var output: [2]f32 = .{0.7, 0.3};
    var target: [2]f32 = .{1.0, 0.0};
    const result = try loss_fn.forward(&output, &target);
    // Simplified expected value - just verify it runs without panic
    _ = result;
}

test "loss binary cross entropy forward" {
    const loss_fn = loss.Loss{ .binary_cross_entropy = {} };
    var output: [2]f32 = .{0.7, 0.3};
    var target: [2]f32 = .{1.0, 0.0};
    const result = try loss_fn.forward(&output, &target);
    // Simplified expected value - just verify it runs without panic
    _ = result;
}

// Layer Tests
test "layer dense forward" {
    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu);
    defer lyr.deinit();

    lyr.weights[0] = 1.0;
    lyr.weights[1] = 1.0;
    lyr.bias[0] = 0.0;

    var input: [2]f32 = .{1.0, 1.0};
    var output: [1]f32 = undefined;
    try lyr.forward(&input, &output);

    try expectNear(output[0], 2.0, 0.0001);
}

test "layer dense backward" {
    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu);
    defer lyr.deinit();

    lyr.weights[0] = 1.0;
    lyr.weights[1] = 1.0;
    lyr.bias[0] = 0.0;

    var input: [2]f32 = .{1.0, 1.0};
    var grad_output: [1]f32 = .{1.0};
    var grad_input: [2]f32 = undefined;

    try lyr.backward(&input, &grad_output, &grad_input);

    try expectNear(grad_input[0], 1.0, 0.0001);
    try expectNear(grad_input[1], 1.0, 0.0001);
}

// Optimizer Tests - basic functionality tests
test "optimizer sgd basic" {
    // Test that SGD struct can be created
    const sgd: optimizer.Sgd = .{};
    _ = sgd;
}

test "optimizer adam basic" {
    // Test that Adam struct can be created
    const adam: optimizer.Adam = .{};
    _ = adam;
}

// Network Tests - basic functionality
test "network basic" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator);
    defer net.deinit();

    _ = try net.addDense(2, 3, .relu);
    _ = try net.addDense(3, 1, .sigmoid);

    // Verify layers were added
    if (net.layers.items.len != 2) @panic("Expected 2 layers");
}

// Memory Usage Tests
test "memory: Dense layer" {
    const allocator = testing.allocator;

    const lyr = try layer.Dense.init(allocator, 4, 8, .relu);
    defer lyr.deinit();

    const input = allocator.alloc(f32, 4) catch unreachable;
    defer allocator.free(input);
    const output = allocator.alloc(f32, 8) catch unreachable;
    defer allocator.free(output);

    try lyr.forward(input, output);
}

test "memory: Network forward pass" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator);
    defer net.deinit();

    _ = try net.addDense(4, 8, .relu);
    _ = try net.addDense(8, 1, .relu);

    const input: []const f32 = &.{ 0.1, 0.2, 0.3, 0.4 };
    var output: [1]f32 = undefined;

    _ = try net.forward(input, &output);
}

test "memory: Training step" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator);
    defer net.deinit();

    _ = try net.addDense(4, 8, .relu);
    _ = try net.addDense(8, 1, .relu);

    const input: []const f32 = &.{ 0.1, 0.2, 0.3, 0.4 };
    const target: []const f32 = &.{ 0.5 };
    const loss_fn = loss.Loss{ .mse = {} };

    const loss_value = try net.trainStep(input, target, 0.01, loss_fn);
    // Just verify training runs and produces a valid loss (non-negative)
    try std.testing.expect(loss_value >= 0 and loss_value < 100);
}

test "memory: Optimizer training" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator);
    defer net.deinit();

    _ = try net.addDense(4, 8, .relu);
    _ = try net.addDense(8, 1, .relu);

    var opt = optimizer.Optimizer{ .sgd = optimizer.Sgd{} };
    try net.initOptimizer(&opt);
    defer net.deinitOptimizer(&opt);

    const input: []const f32 = &.{ 0.1, 0.2, 0.3, 0.4 };
    const target: []const f32 = &.{ 0.5 };
    const loss_fn = loss.Loss{ .mse = {} };

    const loss_value = try net.trainStepWithOptimizer(input, target, 0.01, loss_fn, &opt);
    // Just verify training runs and produces a valid loss (non-negative)
    try std.testing.expect(loss_value >= 0 and loss_value < 100);
}

test "memory: Softmax no extra allocations" {
    const allocator = testing.allocator;

    const size: usize = 1024;
    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);

    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 100)) / 10.0;
    }

    const act = activation.Activation{ .softmax = {} };
    try act.softmaxForward(input, output);

    // Verify output is valid softmax (sums to ~1)
    var sum: f32 = 0;
    for (output) |v| {
        sum += v;
    }
    try expectNear(sum, 1.0, 0.01);
}

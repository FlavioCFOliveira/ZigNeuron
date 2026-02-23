/// Unit tests for loss functions

const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const loss = zn.loss;

fn expectNear(actual: f32, expected: f32, tolerance: f32) !void {
    const diff = if (actual > expected) actual - expected else expected - actual;
    if (diff > tolerance) return error.ApproximationFailed;
}

test "loss mse forward" {
    const loss_fn = loss.Loss{ .mse = {} };

    // Perfect prediction
    var output: [3]f32 = .{1.0, 2.0, 3.0};
    var target: [3]f32 = .{1.0, 2.0, 3.0};
    const result = try loss_fn.forward(&output, &target);
    try expectNear(result, 0.0, 0.0001);

    // Some error
    output = .{1.0, 2.0, 3.0};
    target = .{0.0, 1.0, 2.0};
    const result2 = try loss_fn.forward(&output, &target);
    // MSE = ((1-0)^2 + (2-1)^2 + (3-2)^2) / 3 = 3/3 = 1
    try expectNear(result2, 1.0, 0.0001);
}

test "loss mse backward" {
    const loss_fn = loss.Loss{ .mse = {} };

    var output: [2]f32 = .{1.0, 2.0};
    var target: [2]f32 = .{0.0, 1.0};
    var grad_output: [2]f32 = undefined;
    try loss_fn.backward(&output, &target, &grad_output);

    // MSE gradient: dL/dy = 2(y - t) / n
    try expectNear(grad_output[0], 2.0 * (1.0 - 0.0) / 2.0, 0.0001);
    try expectNear(grad_output[1], 2.0 * (2.0 - 1.0) / 2.0, 0.0001);
}

test "loss mse gradient direction" {
    const loss_fn = loss.Loss{ .mse = {} };

    // If output > target, gradient should be positive (push output down)
    var output: [1]f32 = .{2.0};
    var target: [1]f32 = .{1.0};
    var grad: [1]f32 = undefined;
    try loss_fn.backward(&output, &target, &grad);

    try testing.expect(grad[0] > 0);

    // If output < target, gradient should be negative (push output up)
    output = .{1.0};
    target = .{2.0};
    try loss_fn.backward(&output, &target, &grad);

    try testing.expect(grad[0] < 0);
}

test "loss cross_entropy forward" {
    const loss_fn = loss.Loss{ .cross_entropy = {} };

    // Logits that match one-hot target
    var output: [2]f32 = .{1.0, 0.0};  // logit for class 0 is higher
    var target: [2]f32 = .{1.0, 0.0};  // class 0 is target
    const result = try loss_fn.forward(&output, &target);
    // CE should be low when prediction matches target
    try testing.expect(result >= 0);
    try testing.expect(result < 1.0);
}

test "loss cross_entropy backward" {
    const loss_fn = loss.Loss{ .cross_entropy = {} };

    // For cross-entropy with logits, gradient is (softmax(logits) - target)
    var output: [2]f32 = .{0.0, 0.0};  // softmax = [0.5, 0.5]
    var target: [2]f32 = .{1.0, 0.0};  // one-hot
    var grad_output: [2]f32 = undefined;
    try loss_fn.backward(&output, &target, &grad_output);

    // gradient should be [0.5 - 1, 0.5 - 0] = [-0.5, 0.5]
    try expectNear(grad_output[0], -0.5, 0.001);
    try expectNear(grad_output[1], 0.5, 0.001);
}

test "loss binary_cross_entropy forward" {
    const loss_fn = loss.Loss{ .binary_cross_entropy = {} };

    // Perfect prediction
    var output: [2]f32 = .{0.01, 0.99};  // close to 0 and 1
    var target: [2]f32 = .{0.0, 1.0};
    const result = try loss_fn.forward(&output, &target);
    // BCE should be very low for perfect predictions
    try testing.expect(result >= 0);
    try testing.expect(result < 0.1);
}

test "loss binary_cross_entropy backward" {
    const loss_fn = loss.Loss{ .binary_cross_entropy = {} };

    // For BCE with sigmoid, gradient is (p - t) / n
    var output: [2]f32 = .{0.7, 0.3};
    var target: [2]f32 = .{1.0, 0.0};
    var grad_output: [2]f32 = undefined;
    try loss_fn.backward(&output, &target, &grad_output);

    // gradient should be [(0.7 - 1) / 2, (0.3 - 0) / 2] = [-0.15, 0.15]
    try expectNear(grad_output[0], -0.15, 0.001);
    try expectNear(grad_output[1], 0.15, 0.001);
}

test "loss isLogitsGradient" {
    const loss_ce = loss.Loss{ .cross_entropy = {} };
    try testing.expect(loss_ce.isLogitsGradient());

    const loss_mse = loss.Loss{ .mse = {} };
    try testing.expect(!loss_mse.isLogitsGradient());

    const loss_bce = loss.Loss{ .binary_cross_entropy = {} };
    try testing.expect(!loss_bce.isLogitsGradient());
}

test "loss numerical stability" {
    const loss_fn = loss.Loss{ .mse = {} };

    // Very large values
    var output: [2]f32 = .{1e6, 1e6};
    var target: [2]f32 = .{0.0, 0.0};
    const result = try loss_fn.forward(&output, &target);
    try testing.expect(std.math.isFinite(result));

    // Very small values
    output = .{1e-10, 1e-10};
    target = .{1e-10, 1e-10};
    const result2 = try loss_fn.forward(&output, &target);
    try expectNear(result2, 0.0, 1e-20);
}

test "loss: MSE gradient scale" {
    const loss_fn = loss.Loss{ .mse = {} };

    // Test that gradient scales correctly
    var output1: [1]f32 = .{2.0};
    var target1: [1]f32 = .{1.0};
    var grad1: [1]f32 = undefined;
    try loss_fn.backward(&output1, &target1, &grad1);

    // Double the error
    var output2: [1]f32 = .{3.0};
    var target2: [1]f32 = .{1.0};
    var grad2: [1]f32 = undefined;
    try loss_fn.backward(&output2, &target2, &grad2);

    // Gradient should be twice as large
    try expectNear(grad2[0], 2 * grad1[0], 0.0001);
}

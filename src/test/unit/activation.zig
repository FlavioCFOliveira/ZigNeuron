/// Unit tests for activation functions

const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const activation = zn.activation;

fn expectNear(actual: f32, expected: f32, tolerance: f32) !void {
    const diff = if (actual > expected) actual - expected else expected - actual;
    if (diff > tolerance) return error.ApproximationFailed;
}

test "activation relu forward" {
    const act = activation.Activation{ .relu = {} };
    try expectNear(act.forward(1.0), 1.0, 0.0001);
    try expectNear(act.forward(0.0), 0.0, 0.0001);
    try expectNear(act.forward(-1.0), 0.0, 0.0001);
    try expectNear(act.forward(100.0), 100.0, 0.0001);
    try expectNear(act.forward(-100.0), 0.0, 0.0001);
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
    // backward(y, grad) where y = sigmoid(x). For x=0, y=0.5
    try expectNear(act.backward(0.5, 1.0), 0.25, 0.0001);
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
    try expectNear(sum, 1.0, 0.001);
}

test "activation softmax backward" {
    const act = activation.Activation{ .softmax = {} };
    var input: [2]f32 = .{0.0, 1.0};
    var grad_output: [2]f32 = .{1.0, 1.0};
    var grad_input: [2]f32 = undefined;

    try act.softmaxBackward(&input, &grad_output, &grad_input);
    var grad_sum: f32 = 0;
    for (grad_input) |g| grad_sum += g;
    try expectNear(grad_sum, 0.0, 0.0001);
}

test "activation linear forward" {
    const act = activation.Activation{ .linear = {} };
    try expectNear(act.forward(1.0), 1.0, 0.0001);
    try expectNear(act.forward(-1.0), -1.0, 0.0001);
}

test "activation linear backward" {
    const act = activation.Activation{ .linear = {} };
    try expectNear(act.backward(1.0, 1.0), 1.0, 0.0001);
}

test "activation derivatives at boundaries" {
    const relu = activation.Activation{ .relu = {} };
    const sigmoid = activation.Activation{ .sigmoid = {} };
    const tanh = activation.Activation{ .tanh = {} };

    // ReLU at zero
    try expectNear(relu.backward(0.0, 1.0), 0.0, 0.0001);

    // Sigmoid at extreme values
    try expectNear(sigmoid.backward(1.0, 1.0), 0.0, 0.001);
    try expectNear(sigmoid.backward(0.0, 1.0), 0.0, 0.001);

    // Tanh at extreme values
    try expectNear(tanh.backward(1.0, 1.0), 0.0, 0.001);
    try expectNear(tanh.backward(-1.0, 1.0), 0.0, 0.001);
}

test "activation numeric precision" {
    const act = activation.Activation{ .sigmoid = {} };

    // Very small input
    const small: f32 = 1e-10;
    const output_small = act.forward(small);
    try testing.expect(std.math.isFinite(output_small));
    try testing.expect(output_small >= 0 and output_small <= 1);

    // Very large input
    const large: f32 = 1e10;
    const output_large = act.forward(large);
    try testing.expect(std.math.isFinite(output_large));
    try expectNear(output_large, 1.0, 0.0001);
}

test "activation cache behavior" {
    const act = activation.Activation{ .relu = {} };

    // Multiple calls with same input should give same output
    const input: f32 = 1.5;
    const out1 = act.forward(input);
    const out2 = act.forward(input);
    try expectNear(out1, out2, 0.0001);
}

test "activation backprop gradient magnitude" {
    const act = activation.Activation{ .sigmoid = {} };

    // Gradient should be max at output=0.5 (input=0)
    const grad_at_0 = act.backward(0.5, 1.0);
    try expectNear(grad_at_0, 0.25, 0.0001);

    // Gradient should be smaller at extreme outputs
    const grad_at_1 = act.backward(1.0, 1.0);
    try testing.expect(grad_at_1 < grad_at_0);
}

test "activation linear derivative" {
    const act = activation.Activation{ .linear = {} };
    try expectNear(act.backward(123.456, 1.0), 1.0, 0.0001);
}

test "activation softmax temperature" {
    const act = activation.Activation{ .softmax = {} };
    var input: [3]f32 = .{1.0, 2.0, 3.0};
    var output: [3]f32 = undefined;
    try act.softmaxForward(&input, &output);

    // Higher input should give higher probability
    try testing.expect(output[2] > output[1]);
    try testing.expect(output[1] > output[0]);
}

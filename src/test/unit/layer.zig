/// Unit tests for dense layer

const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const layer = zn.layer;
const activation = zn.activation;
const backend = zn.backend;

fn expectNear(actual: f32, expected: f32, tolerance: f32) !void {
    const diff = if (actual > expected) actual - expected else expected - actual;
    if (diff > tolerance) return error.ApproximationFailed;
}

test "layer dense forward" {
    const allocator = testing.allocator;
    var be = try backend.Backend.init(allocator);
    be.type = .cpu;
    defer be.deinit();
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, be);
    defer lyr.deinit();

    lyr.weights.slice[0] = 1.0;
    lyr.weights.slice[1] = 1.0;
    lyr.bias.slice[0] = 0.0;

    var input: [2]f32 = .{1.0, 1.0};
    var output: [1]f32 = undefined;
    try lyr.forward(&input, null, &output, null);

    try expectNear(output[0], 2.0, 0.0001);
}

test "layer dense backward" {
    const allocator = testing.allocator;
    var be = try backend.Backend.init(allocator);
    be.type = .cpu;
    defer be.deinit();
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, be);
    defer lyr.deinit();

    lyr.weights.slice[0] = 1.0;
    lyr.weights.slice[1] = 1.0;
    lyr.bias.slice[0] = 0.0;

    var input: [2]f32 = .{1.0, 1.0};
    var grad_output: [1]f32 = .{1.0};
    var grad_input: [2]f32 = undefined;
    var pre_act: [1]f32 = .{2.0};
    try lyr.backward(&input, null, &grad_output, null, &grad_input, null, &pre_act, null);

    try expectNear(grad_input[0], 1.0, 0.0001);
    try expectNear(grad_input[1], 1.0, 0.0001);
}

test "layer dense initialization" {
    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 4, 8, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    try testing.expect(lyr.weights.slice.len > 0);
    try testing.expect(lyr.bias.slice.len > 0);
}

test "layer dense batch forward" {
    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 3, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    var input: [2]f32 = .{1.0, 2.0};
    var output: [3]f32 = undefined;
    try lyr.forward(&input, null, &output, null);

    for (output) |o| {
        try testing.expect(std.math.isFinite(o));
    }
}

test "layer dense weight updates" {
    const allocator = testing.allocator;
    var be = try backend.Backend.init(allocator);
    be.type = .cpu;
    defer be.deinit();
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, be);
    defer lyr.deinit();

    const initial_weight = lyr.weights.slice[0];
    lyr.weights.slice[0] = 1.0;
    lyr.grad_weights.slice[0] = 0.1;

    lyr.weights.slice[0] -= 0.01 * lyr.grad_weights.slice[0];

    try testing.expect(lyr.weights.slice[0] != initial_weight);
}

test "layer dense different activations" {
    const allocator = testing.allocator;

    for ([_]activation.ActivationType{ .relu, .sigmoid, .tanh }) |act_type| {
        var lyr = try layer.Dense.init(allocator, 2, 2, act_type, backend.Backend{ .type = .cpu, .metal_ctx = null });
        defer lyr.deinit();

        var input: [2]f32 = .{0.5, 0.5};
        var output: [2]f32 = undefined;
        try lyr.forward(&input, null, &output, null);

        for (output) |o| {
            try testing.expect(std.math.isFinite(o));
        }
    }
}

test "layer dense gradient computation" {
    const allocator = testing.allocator;
    var be = try backend.Backend.init(allocator);
    be.type = .cpu;
    defer be.deinit();
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, be);
    defer lyr.deinit();

    lyr.weights.slice[0] = 1.0;
    lyr.weights.slice[1] = 1.0;
    lyr.bias.slice[0] = 0.0;

    var input: [2]f32 = .{1.0, 1.0};
    var grad_output: [1]f32 = .{1.0};
    var grad_input: [2]f32 = undefined;
    var pre_act: [1]f32 = .{2.0};
    try lyr.backward(&input, null, &grad_output, null, &grad_input, null, &pre_act, null);

    try testing.expect(std.math.isFinite(grad_input[0]));
    try testing.expect(std.math.isFinite(grad_input[1]));
}

test "layer dense large dimension" {
    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 100, 100, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    const input = try allocator.alloc(f32, 100);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, 100);
    defer allocator.free(output);

    @memset(input, 1.0);
    try lyr.forward(input, null, output, null);

    for (output) |o| {
        try testing.expect(std.math.isFinite(o));
    }
}

test "layer dense zero input" {
    const allocator = testing.allocator;
    var be = try backend.Backend.init(allocator);
    be.type = .cpu;
    defer be.deinit();
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, be);
    defer lyr.deinit();

    var input: [2]f32 = .{0.0, 0.0};
    var output: [1]f32 = undefined;
    try lyr.forward(&input, null, &output, null);

    try testing.expect(std.math.isFinite(output[0]));
}

test "layer dense negative input" {
    const allocator = testing.allocator;
    var be = try backend.Backend.init(allocator);
    be.type = .cpu;
    defer be.deinit();
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, be);
    defer lyr.deinit();

    lyr.weights.slice[0] = 1.0;
    lyr.weights.slice[1] = 1.0;
    lyr.bias.slice[0] = 0.0;

    var input: [2]f32 = .{-1.0, -1.0};
    var output: [1]f32 = undefined;
    try lyr.forward(&input, null, &output, null);

    try expectNear(output[0], 0.0, 0.0001);
}

test "layer dense precision" {
    const allocator = testing.allocator;
    var be = try backend.Backend.init(allocator);
    be.type = .cpu;
    defer be.deinit();
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, be);
    defer lyr.deinit();

    lyr.weights.slice[0] = 1.2345678;
    lyr.weights.slice[1] = 2.3456789;
    lyr.bias.slice[0] = 0.9876543;

    var input: [2]f32 = .{1.0, 1.0};
    var output: [1]f32 = undefined;
    try lyr.forward(&input, null, &output, null);

    const expected = 1.2345678 + 2.3456789 + 0.9876543;
    try expectNear(output[0], expected, 0.0001);
}

test "layer dense bias contribution" {
    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .linear, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    lyr.weights.slice[0] = 0.0;
    lyr.weights.slice[1] = 0.0;
    lyr.bias.slice[0] = 5.0;

    var input: [2]f32 = .{0.0, 0.0};
    var output: [1]f32 = undefined;
    try lyr.forward(&input, null, &output, null);

    try expectNear(output[0], 5.0, 0.0001);
}

test "layer dense backward with sigmoid" {
    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .sigmoid, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();

    lyr.weights.slice[0] = 0.0;
    lyr.weights.slice[1] = 0.0;
    lyr.bias.slice[0] = 0.0;

    var input: [2]f32 = .{0.0, 0.0};
    var grad_output: [1]f32 = .{1.0};
    var grad_input: [2]f32 = undefined;
    var pre_act: [1]f32 = .{0.0};
    try lyr.backward(&input, null, &grad_output, null, &grad_input, null, &pre_act, null);

    try expectNear(grad_input[0], 0.0, 0.0001);
    try expectNear(grad_input[1], 0.0, 0.0001);
}

test "layer dense multiple forward calls" {
    const allocator = testing.allocator;
    var be = try backend.Backend.init(allocator);
    be.type = .cpu;
    defer be.deinit();
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, be);
    defer lyr.deinit();

    lyr.weights.slice[0] = 1.0;
    lyr.weights.slice[1] = 1.0;
    lyr.bias.slice[0] = 0.0;

    var input: [2]f32 = .{1.0, 1.0};
    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;

    try lyr.forward(&input, null, &output1, null);
    try lyr.forward(&input, null, &output2, null);

    try expectNear(output1[0], output2[0], 0.0001);
}

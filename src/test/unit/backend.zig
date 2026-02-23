/// Unit tests for backend (CPU implementation)

const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const layer = zn.layer;
const activation = zn.activation;
const loss = zn.loss;
const backend = zn.backend;

fn expectNear(actual: f32, expected: f32, tolerance: f32) !void {
    const diff = if (actual > expected) actual - expected else expected - actual;
    if (diff > tolerance) return error.ApproximationFailed;
}

test "backend matmul cpu" {
    const allocator = testing.allocator;

    const m: usize = 2;
    const n: usize = 3;
    const k: usize = 2;

    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    for (a, 0..) |*v, i| {
        v.* = @as(f32, @floatCast(i % 10)) / 10.0;
    }
    for (b, 0..) |*v, i| {
        v.* = @as(f32, @floatCast((i + 1) % 10)) / 10.0;
    }

    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();
    try cpu_backend.matMul(a, b, c, m, n, k);

    var expected: f32 = 0;
    for (0..k) |p| {
        expected += a[0 * k + p] * b[p * n + 0];
    }

    const tolerance: f32 = 0.0001;
    const diff = if (c[0] > expected) c[0] - expected else expected - c[0];
    try testing.expect(diff < tolerance);
}

test "backend activation forward relu" {
    var cpu_backend = try backend.Backend.init(testing.allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();
    const act = activation.Activation{ .relu = {} };

    const allocator = testing.allocator;
    const input = try allocator.alloc(f32, 10);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, 10);
    defer allocator.free(output);

    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatCast(i - 5)) / 2.0;
    }

    try cpu_backend.activationForward(act, input, output);

    for (input, output) |in, out| {
        const expected = if (in > 0) in else 0;
        try expectNear(out, expected, 0.0001);
    }
}

test "backend activation forward sigmoid" {
    var cpu_backend = try backend.Backend.init(testing.allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();
    const act = activation.Activation{ .sigmoid = {} };

    const allocator = testing.allocator;
    const input = try allocator.alloc(f32, 5);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, 5);
    defer allocator.free(output);

    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatCast(i)) / 2.0 - 1.0;
    }

    try cpu_backend.activationForward(act, input, output);

    for (output) |o| {
        try testing.expect(o >= 0);
        try testing.expect(o <= 1);
    }

    try expectNear(output[2], 0.5, 0.0001);
}

test "backend activation backward" {
    var cpu_backend = try backend.Backend.init(testing.allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();
    const act = activation.Activation{ .sigmoid = {} };

    const allocator = testing.allocator;
    const input = try allocator.alloc(f32, 5);
    defer allocator.free(input);
    const grad_output = try allocator.alloc(f32, 5);
    defer allocator.free(grad_output);
    const grad_input = try allocator.alloc(f32, 5);
    defer allocator.free(grad_input);

    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatCast(i)) / 2.0 - 1.0;
    }
    @memset(grad_output, 1.0);

    try cpu_backend.activationBackward(act, input, grad_output, grad_input);

    for (input, grad_input) |in, gi| {
        const s = act.forward(in);
        const expected = s * (1 - s);
        try expectNear(gi, expected, 0.0001);
    }
}

test "backend loss backward mse" {
    var cpu_backend = try backend.Backend.init(testing.allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();
    const loss_fn = loss.Loss{ .mse = {} };

    const allocator = testing.allocator;
    const output = try allocator.alloc(f32, 5);
    defer allocator.free(output);
    const target = try allocator.alloc(f32, 5);
    defer allocator.free(target);
    const grad_output = try allocator.alloc(f32, 5);
    defer allocator.free(grad_output);

    for (output, 0..) |*v, i| {
        v.* = @as(f32, @floatCast(i)) / 2.0;
    }
    for (target, 0..) |*v, i| {
        v.* = @as(f32, @floatCast(i + 5)) / 2.0;
    }

    try cpu_backend.lossBackward(loss_fn, output, target, grad_output);

    for (output, target, grad_output) |o, t, g| {
        const expected = 2 * (o - t);
        try expectNear(g, expected, 0.0001);
    }
}

test "backend loss backward cross_entropy" {
    var cpu_backend = try backend.Backend.init(testing.allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();
    const loss_fn = loss.Loss{ .cross_entropy = {} };

    const allocator = testing.allocator;
    const output = try allocator.alloc(f32, 3);
    defer allocator.free(output);
    const target = try allocator.alloc(f32, 3);
    defer allocator.free(target);
    const grad_output = try allocator.alloc(f32, 3);
    defer allocator.free(grad_output);

    output[0] = 1.0;
    output[1] = 2.0;
    output[2] = 0.0;

    target[0] = 0.0;
    target[1] = 1.0;
    target[2] = 0.0;

    try cpu_backend.lossBackward(loss_fn, output, target, grad_output);

    try testing.expect(grad_output[1] < 0);
}

test "backend default detection" {
    const backend_type = backend.Backend.default();
    _ = backend_type;
}

test "backend matmul correctness" {
    var cpu_backend = try backend.Backend.init(testing.allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    const a_data = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const b_data = [_]f32{
        7, 8,
        9, 10,
        11, 12,
    };
    const expected = [_]f32{
        58, 64,
        139, 154,
    };

    const a = &a_data;
    const b = &b_data;
    const c = try testing.allocator.alloc(f32, 4);
    defer testing.allocator.free(c);

    try cpu_backend.matMul(a, b, c, 2, 2, 3);

    for (c, expected) |actual, exp| {
        try expectNear(actual, exp, 0.0001);
    }
}

test "backend matmul large matrix" {
    var cpu_backend = try backend.Backend.init(testing.allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    const allocator = testing.allocator;

    const size: usize = 64;
    const a_data = try allocator.alloc(f32, size * size);
    defer allocator.free(a_data);
    const b_data = try allocator.alloc(f32, size * size);
    defer allocator.free(b_data);
    const c_data = try allocator.alloc(f32, size * size);
    defer allocator.free(c_data);

    @memset(a_data, 1.0);
    @memset(b_data, 1.0);

    try cpu_backend.matMul(a_data, b_data, c_data, size, size, size);

    for (c_data) |val| {
        try expectNear(val, @as(f32, @floatCast(size)), 0.0001);
    }
}

test "backend activation forward batch" {
    var cpu_backend = try backend.Backend.init(testing.allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();
    const act = activation.Activation{ .relu = {} };

    const allocator = testing.allocator;
    const input = try allocator.alloc(f32, 100);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, 100);
    defer allocator.free(output);

    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatCast(i % 100)) / 10.0 - 5.0;
    }

    try cpu_backend.activationForward(act, input, output);

    for (input, output) |in, out| {
        const expected = if (in > 0) in else 0;
        try expectNear(out, expected, 0.0001);
    }
}

test "backend numerical stability" {
    var cpu_backend = try backend.Backend.init(testing.allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();
    const act = activation.Activation{ .sigmoid = {} };

    const allocator = testing.allocator;
    const input = try allocator.alloc(f32, 4);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, 4);
    defer allocator.free(output);

    input[0] = 1000.0;
    input[1] = -1000.0;
    input[2] = 0.0;
    input[3] = 10.0;

    try cpu_backend.activationForward(act, input, output);

    try expectNear(output[0], 1.0, 0.0001);
    try expectNear(output[1], 0.0, 0.0001);
    try expectNear(output[2], 0.5, 0.0001);
    try expectNear(output[3], 0.999, 0.001);
}

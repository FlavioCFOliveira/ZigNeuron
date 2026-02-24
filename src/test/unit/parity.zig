const std = @import("std");
const testing = std.testing;
const zn = @import("ZigNeuron");
const Tensor = zn.tensor.Tensor;

fn expectNear(actual: f32, expected: f32, tolerance: f32) !void {
    const diff = if (actual > expected) actual - expected else expected - actual;
    if (diff > tolerance) {
        std.debug.print("Expected near: actual={d}, expected={d}, diff={d}, tolerance={d}\n", .{ actual, expected, diff, tolerance });
        return error.ApproximationFailed;
    }
}

fn compareSlices(cpu: []const f32, metal: []const f32, epsilon: f32) !void {
    for (cpu, metal, 0..) |c, m, i| {
        const diff = @abs(c - m);
        if (diff > epsilon) {
            std.debug.print("Parity failure at index {}: cpu={d:.6}, metal={d:.6}, diff={d:.6}\n", .{ i, c, m, diff });
            return error.ParityCheckFailed;
        }
    }
}

test "backend parity: matmul" {
    const allocator = testing.allocator;

    var cpu_backend = try zn.backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var metal_backend = try zn.backend.Backend.init(allocator);
    defer metal_backend.deinit();

    if (metal_backend.type != .gpu or metal_backend.type.gpu != .metal) return;

    const m = 16;
    const n = 16;
    const k = 16;

    var a = try Tensor.init(allocator, &.{ m, k }, cpu_backend); defer a.deinit();
    var b = try Tensor.init(allocator, &.{ k, n }, cpu_backend); defer b.deinit();
    var a_m = try Tensor.init(allocator, &.{ m, k }, metal_backend); defer a_m.deinit();
    var b_m = try Tensor.init(allocator, &.{ k, n }, metal_backend); defer b_m.deinit();

    var c_cpu = try Tensor.init(allocator, &.{ m, n }, cpu_backend); defer c_cpu.deinit();
    var c_metal = try Tensor.init(allocator, &.{ m, n }, metal_backend); defer c_metal.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (a.slice) |*v| v.* = rand.float(f32);
    for (b.slice) |*v| v.* = rand.float(f32);
    @memcpy(a_m.slice, a.slice);
    @memcpy(b_m.slice, b.slice);

    try cpu_backend.matMul(a.slice, a.getMtlBuffer(), b.slice, b.getMtlBuffer(), c_cpu.slice, c_cpu.getMtlBuffer(), m, n, k, false);
    try metal_backend.matMul(a_m.slice, a_m.getMtlBuffer(), b_m.slice, b_m.getMtlBuffer(), c_metal.slice, c_metal.getMtlBuffer(), m, n, k, false);

    try compareSlices(c_cpu.slice, c_metal.slice, 1e-4);
}

test "backend parity: activation" {
    const allocator = testing.allocator;

    var cpu_backend = try zn.backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var metal_backend = try zn.backend.Backend.init(allocator);
    defer metal_backend.deinit();

    if (metal_backend.type != .gpu or metal_backend.type.gpu != .metal) return;

    const size = 100;
    var input = try Tensor.init(allocator, &.{size}, cpu_backend); defer input.deinit();
    var input_m = try Tensor.init(allocator, &.{size}, metal_backend); defer input_m.deinit();
    var out_cpu = try Tensor.init(allocator, &.{size}, cpu_backend); defer out_cpu.deinit();
    var out_metal = try Tensor.init(allocator, &.{size}, metal_backend); defer out_metal.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (input.slice) |*v| v.* = rand.float(f32) * 10.0 - 5.0;
    @memcpy(input_m.slice, input.slice);

    const acts = [_]zn.activation.Activation{ .relu, .sigmoid, .tanh };
    for (acts) |act| {
        try cpu_backend.activationForward(act, input.slice, input.getMtlBuffer(), out_cpu.slice, out_cpu.getMtlBuffer());
        try metal_backend.activationForward(act, input_m.slice, input_m.getMtlBuffer(), out_metal.slice, out_metal.getMtlBuffer());
        try compareSlices(out_cpu.slice, out_metal.slice, 1e-5);
    }
}

test "backend parity: loss" {
    const allocator = testing.allocator;

    var cpu_backend = try zn.backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var metal_backend = try zn.backend.Backend.init(allocator);
    defer metal_backend.deinit();

    if (metal_backend.type != .gpu or metal_backend.type.gpu != .metal) return;

    const size = 10;
    var pred = try Tensor.init(allocator, &.{size}, cpu_backend); defer pred.deinit();
    var pred_m = try Tensor.init(allocator, &.{size}, metal_backend); defer pred_m.deinit();
    var target = try Tensor.init(allocator, &.{size}, cpu_backend); defer target.deinit();
    var target_m = try Tensor.init(allocator, &.{size}, metal_backend); defer target_m.deinit();
    var grad_cpu = try Tensor.init(allocator, &.{size}, cpu_backend); defer grad_cpu.deinit();
    var grad_metal = try Tensor.init(allocator, &.{size}, metal_backend); defer grad_metal.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    const losses = [_]zn.loss.Loss{ .{ .mse = {} }, .{ .binary_cross_entropy = {} }, .{ .cross_entropy = {} } };
    for (losses) |l| {
        if (l == .cross_entropy) {
            for (pred.slice) |*v| v.* = rand.float(f32);
            @memcpy(pred_m.slice, pred.slice);
            @memset(target.slice, 0);
            target.slice[rand.uintLessThan(usize, size)] = 1.0;
            @memcpy(target_m.slice, target.slice);
        } else {
            for (pred.slice) |*v| v.* = rand.float(f32);
            @memcpy(pred_m.slice, pred.slice);
            for (target.slice) |*v| v.* = if (rand.float(f32) > 0.5) 1.0 else 0.0;
            @memcpy(target_m.slice, target.slice);
        }

        try cpu_backend.lossBackward(l, pred.slice, pred.getMtlBuffer(), target.slice, target.getMtlBuffer(), grad_cpu.slice, grad_cpu.getMtlBuffer());
        try metal_backend.lossBackward(l, pred_m.slice, pred_m.getMtlBuffer(), target_m.slice, target_m.getMtlBuffer(), grad_metal.slice, grad_metal.getMtlBuffer());
        try compareSlices(grad_cpu.slice, grad_metal.slice, 1e-5);
    }
}

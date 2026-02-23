/// Unit tests for Phase 4 layers (Conv1D, LayerNorm, Dropout)
const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const layer = zn.layer;
const backend = zn.backend;

test "conv1d forward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    // 1 in_channel, 1 out_channel, kernel_size 2, input_len 3
    var conv = try layer.Conv1D.init(allocator, 1, 1, 2, 3, .linear, cpu_backend);
    defer conv.deinit();

    @memset(conv.weights.slice, 1.0);
    @memset(conv.bias.slice, 0.0);

    const input = [_]f32{ 1.0, 2.0, 3.0 };
    var output = [_]f32{ 0, 0 }; // out_len = (3-2)/1 + 1 = 2

    try conv.forward(&input, null, &output, null);

    // output[0] = 1*1 + 2*1 = 3
    // output[1] = 2*1 + 3*1 = 5
    try testing.expectEqual(@as(f32, 3), output[0]);
    try testing.expectEqual(@as(f32, 5), output[1]);
}

test "layer_norm forward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var ln = try layer.LayerNorm.init(allocator, 3, cpu_backend);
    defer ln.deinit();

    const input = [_]f32{ 1.0, 2.0, 3.0 }; // mean=2, var=((1-2)^2+(2-2)^2+(3-2)^2)/3 = 2/3
    var output = [_]f32{ 0, 0, 0 };

    try ln.forward(&input, null, &output, null);

    // output sums to ~0
    var sum: f32 = 0;
    for (output) |x| sum += x;
    try testing.expectApproxEqAbs(@as(f32, 0), sum, 1e-5);
}

test "dropout training vs inference" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var dr = try layer.Dropout.init(allocator, 100, 0.5, cpu_backend);
    defer dr.deinit();

    const input = [_]f32{ 1.0 } ** 100;
    var output = [_]f32{ 0 } ** 100;

    // Inference mode
    dr.training = false;
    try dr.forward(&input, null, &output, null);
    for (output) |o| try testing.expectEqual(@as(f32, 1.0), o);

    // Training mode
    dr.training = true;
    try dr.forward(&input, null, &output, null);
    var zero_count: usize = 0;
    for (output) |o| if (o == 0) { zero_count += 1; };
    try testing.expect(zero_count > 0);
}

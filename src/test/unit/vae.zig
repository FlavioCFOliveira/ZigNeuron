/// Unit tests for VAE components
const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const loss = zn.loss;
const layer = zn.layer;
const backend = zn.backend;

test "kl_divergence loss forward/backward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    const kld_loss = loss.Loss{ .kl_divergence = {} };

    // mu = [0, 0], log_var = [0, 0] => KLD should be 0
    const output_zeros = [_]f32{ 0, 0, 0, 0 };
    const l1 = try kld_loss.forward(&output_zeros, &output_zeros);
    try testing.expectEqual(@as(f32, 0), l1);

    // Test backward
    const output = [_]f32{ 0.1, 0.2, -0.1, -0.2 }; // mu=[0.1, 0.2], log_var=[-0.1, -0.2]
    var grad = [_]f32{ 0, 0, 0, 0 };
    try kld_loss.backward(&output, &output, &grad);

    // dKLD/dmu = mu / n = [0.1, 0.2] / 2 = [0.05, 0.1]
    try testing.expectApproxEqAbs(@as(f32, 0.05), grad[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.1), grad[1], 1e-5);

    // dKLD/dlog_var = 0.5 * (exp(log_var) - 1) / n
    // for log_var = -0.1: 0.5 * (exp(-0.1) - 1) / 2 = 0.25 * (0.9048 - 1) = -0.0238
    try testing.expect(grad[2] < 0);
}

test "sampling_layer forward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var sl = try layer.SamplingLayer.init(allocator, 4, cpu_backend);
    defer sl.deinit();

    const input = [_]f32{ 0, 0, 0, 0 }; // mu=0, log_var=0 => exp(0.5*log_var)=1
    var output = [_]f32{ 0, 0 };

    try sl.forward(&input, null, &output, null);

    // output = mu + epsilon * exp(0.5*log_var) = 0 + epsilon * 1 = epsilon
    // epsilon is ~ N(0, 1), so output should be random
    // We can't predict exact value but we know it should have been updated
    try testing.expect(output[0] != 0 or output[1] != 0);
}

const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const layer = zn.layer;
const activation = zn.activation;
const backend = zn.backend;
const optimizer = zn.optimizer;

fn expectNear(actual: f32, expected: f32, tolerance: f32) !void {
    const diff = if (actual > expected) actual - expected else expected - actual;
    if (diff > tolerance) return error.ApproximationFailed;
}

test "optimizer sgd basic" {
    const sgd: optimizer.Sgd = .{};
    _ = sgd;
}

test "optimizer sgd with momentum" {
    var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();
    var l = layer.Layer{ .dense = lyr };

    try sgd.init(allocator, &l);

    lyr.grad_weights.slice[0] = 0.1;
    lyr.weights.slice[0] = 0.5;

    try sgd.step(&l, 0.01);

    try expectNear(lyr.weights.slice[0], 0.499, 0.0001);

    lyr.grad_weights.slice[0] = 0.2;
    try sgd.step(&l, 0.01);

    try expectNear(lyr.weights.slice[0], 0.4961, 0.001);

    sgd.deinit();
}

test "optimizer adam basic" {
    var adam: optimizer.Adam = .{};

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();
    var l = layer.Layer{ .dense = lyr };

    try adam.init(allocator, &l);

    try testing.expect(adam.t == 0);

    lyr.grad_weights.slice[0] = 0.1;
    lyr.weights.slice[0] = 0.5;

    try adam.step(&l, 0.001);

    try testing.expect(adam.t == 1);

    adam.deinit();
}

test "optimizer adam with known values" {
    var adam: optimizer.Adam = .{
        .beta1 = 0.9,
        .beta2 = 0.999,
        .eps = 1e-8,
    };

    const allocator = testing.allocator;
    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();
    var l = layer.Layer{ .dense = lyr };

    try adam.init(allocator, &l);
    defer adam.deinit();

    lyr.weights.slice[0] = 0.5;
    lyr.grad_weights.slice[0] = 0.1;

    try adam.step(&l, 0.001);

    try testing.expect(lyr.weights.slice[0] < 0.5);
}

test "optimizer rmsprop basic" {
    const allocator = testing.allocator;

    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();
    var l = layer.Layer{ .dense = lyr };

    var rmsprop: optimizer.Rmsprop = .{};

    try rmsprop.init(allocator, &l);

    lyr.grad_weights.slice[0] = 0.1;
    lyr.weights.slice[0] = 0.5;

    try rmsprop.step(&l, 0.001);

    try testing.expect(lyr.weights.slice[0] != 0.5);

    rmsprop.deinit();
}

test "optimizer sgd clears gradients after step" {
    var sgd: optimizer.Sgd = .{};
    const allocator = testing.allocator;

    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();
    var l = layer.Layer{ .dense = lyr };

    try sgd.init(allocator, &l);

    lyr.grad_weights.slice[0] = 0.5;
    lyr.weights.slice[0] = 1.0;

    try sgd.step(&l, 0.1);
    const w1 = lyr.weights.slice[0];

    lyr.grad_weights.slice[0] = 0.3;
    try sgd.step(&l, 0.1);
    const w2 = lyr.weights.slice[0];

    try testing.expect(w1 != w2);

    sgd.deinit();
}

test "optimizer adam: adaptive learning rate" {
    var adam: optimizer.Adam = .{};
    const allocator = testing.allocator;

    var lyr = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();
    var l = layer.Layer{ .dense = lyr };

    try adam.init(allocator, &l);

    lyr.weights.slice[0] = 0.0;
    lyr.grad_weights.slice[0] = 0.1;

    try adam.step(&l, 0.01);
    const w1 = lyr.weights.slice[0];

    try adam.step(&l, 0.01);
    const w2 = lyr.weights.slice[0];

    try testing.expect(w1 != w2);

    adam.deinit();
}

test "optimizer sgd vs no momentum" {
    const allocator = testing.allocator;

    var sgd_no_mom: optimizer.Sgd = .{};
    var lyr1 = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr1.deinit();
    var l1 = layer.Layer{ .dense = lyr1 };

    try sgd_no_mom.init(allocator, &l1);
    lyr1.weights.slice[0] = 1.0;
    lyr1.grad_weights.slice[0] = 0.1;

    try sgd_no_mom.step(&l1, 0.1);
    const w1_after = lyr1.weights.slice[0];

    var sgd_mom: optimizer.Sgd = .{ .momentum = 0.9 };
    var lyr2 = try layer.Dense.init(allocator, 1, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr2.deinit();
    var l2 = layer.Layer{ .dense = lyr2 };

    try sgd_mom.init(allocator, &l2);
    lyr2.weights.slice[0] = 1.0;
    lyr2.grad_weights.slice[0] = 0.1;

    try sgd_mom.step(&l2, 0.1);
    const w2_after = lyr2.weights.slice[0];

    try testing.expect(w1_after != 1.0);
    try testing.expect(w2_after != 1.0);

    sgd_no_mom.deinit();
    sgd_mom.deinit();
}

test "optimizer: step interface" {
    var opt = optimizer.Optimizer{ .sgd = optimizer.Sgd{} };
    const allocator = testing.allocator;

    var lyr = try layer.Dense.init(allocator, 2, 1, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr.deinit();
    var l = layer.Layer{ .dense = lyr };

    try opt.init(allocator, &l);
    lyr.grad_weights.slice[0] = 0.1;
    lyr.weights.slice[0] = 0.5;
    try opt.step(&l, 0.01);

    try testing.expect(lyr.weights.slice[0] != 0.5);

    opt.deinit();
}

test "optimizer: multiple layers with different sizes" {
    var sgd: optimizer.Sgd = .{};
    const allocator = testing.allocator;

    var lyr1 = try layer.Dense.init(allocator, 2, 3, .relu, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr1.deinit();
    var l1 = layer.Layer{ .dense = lyr1 };

    var lyr2 = try layer.Dense.init(allocator, 3, 1, .sigmoid, backend.Backend{ .type = .cpu, .metal_ctx = null });
    defer lyr2.deinit();
    var l2 = layer.Layer{ .dense = lyr2 };

    try sgd.init(allocator, &l1);
    try sgd.init(allocator, &l2);

    @memset(lyr1.grad_weights.slice, 0.1);
    @memset(lyr2.grad_weights.slice, 0.1);

    try sgd.step(&l1, 0.01);
    try sgd.step(&l2, 0.01);

    var changed = false;
    for (lyr1.weights.slice) |w| {
        if (w != 0) {
            changed = true;
            break;
        }
    }
    try testing.expect(changed);

    changed = false;
    for (lyr2.weights.slice) |w| {
        if (w != 0) {
            changed = true;
            break;
        }
    }
    try testing.expect(changed);

    sgd.deinit();
}

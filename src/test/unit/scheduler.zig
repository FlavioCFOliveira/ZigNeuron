/// Tests for learning rate schedulers
const std = @import("std");
const zn = @import("ZigNeuron");
const optimizer = zn.optimizer;

const LRScheduler = optimizer.LRScheduler;

// Test constant scheduler (no change)
test "LRScheduler constant" {
    const scheduler = LRScheduler{ .constant = {} };
    const initial_lr: f32 = 0.01;

    // Learning rate should remain constant
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 0), initial_lr, 1e-7);
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 10), initial_lr, 1e-7);
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 100), initial_lr, 1e-7);
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 1000), initial_lr, 1e-7);
}

// Test step decay scheduler
test "LRScheduler step decay" {
    const scheduler = LRScheduler{
        .step = .{ .step_size = 10, .decay_rate = 0.5 },
    };
    const initial_lr: f32 = 0.01;

    // First 10 epochs: no decay
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 0), 0.01, 1e-6);
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 5), 0.01, 1e-6);
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 9), 0.01, 1e-6);

    // After 10 epochs: decay by 0.5
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 10), 0.005, 1e-6);
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 19), 0.005, 1e-6);

    // After 20 epochs: decay by 0.5 again
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 20), 0.0025, 1e-6);
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 29), 0.0025, 1e-6);

    // After 30 epochs
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 30), 0.00125, 1e-6);
}

// Test exponential decay scheduler
test "LRScheduler exponential decay" {
    const scheduler = LRScheduler{
        .exponential = .{ .decay_rate = 0.95 },
    };
    const initial_lr: f32 = 0.01;

    // Epoch 0: no decay
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 0), 0.01, 1e-6);

    // Epoch 1: decay by 0.95
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 1), 0.0095, 1e-6);

    // Epoch 10: decay^10
    const expected_10 = 0.01 * std.math.pow(f32, 0.95, 10);
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 10), expected_10, 1e-6);

    // Learning rate should decrease monotonically
    var prev_lr: f32 = initial_lr;
    for (1..100) |epoch| {
        const lr = scheduler.getLearningRate(initial_lr, epoch);
        try std.testing.expect(lr < prev_lr);
        prev_lr = lr;
    }
}

// Test cosine annealing scheduler
test "LRScheduler cosine annealing" {
    const min_lr: f32 = 0.0001;
    const max_epochs: usize = 100;
    const scheduler = LRScheduler{
        .cosine = .{ .min_lr = min_lr, .max_epochs = max_epochs },
    };
    const initial_lr: f32 = 0.01;

    // Epoch 0: start at initial_lr
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, 0), initial_lr, 1e-6);

    // Epoch max_epochs/2: should be near min_lr (cosine of π/2 ≈ 0)
    const mid_lr = scheduler.getLearningRate(initial_lr, max_epochs / 2);
    try std.testing.expect(mid_lr < initial_lr);
    try std.testing.expect(mid_lr > min_lr);

    // Epoch max_epochs: should be at min_lr
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, max_epochs), min_lr, 1e-6);

    // Beyond max_epochs: should stay at min_lr
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, max_epochs + 10), min_lr, 1e-6);
    try std.testing.expectApproxEqAbs(scheduler.getLearningRate(initial_lr, max_epochs + 100), min_lr, 1e-6);

    // Learning rate should decrease (or stay same) over epochs
    var prev_lr: f32 = initial_lr;
    for (1..max_epochs) |epoch| {
        const lr = scheduler.getLearningRate(initial_lr, epoch);
        try std.testing.expect(lr <= prev_lr);
        prev_lr = lr;
    }
}

// Test cosine annealing with different parameters
test "LRScheduler cosine annealing variations" {
    // Test with high min_lr
    const scheduler1 = LRScheduler{
        .cosine = .{ .min_lr = 0.001, .max_epochs = 50 },
    };
    const lr_start = scheduler1.getLearningRate(0.01, 0);
    const lr_end = scheduler1.getLearningRate(0.01, 50);
    try std.testing.expectApproxEqAbs(lr_start, 0.01, 1e-6);
    try std.testing.expectApproxEqAbs(lr_end, 0.001, 1e-6);

    // Test with very long schedule
    const scheduler2 = LRScheduler{
        .cosine = .{ .min_lr = 0.00001, .max_epochs = 10000 },
    };
    // At epoch 5000 (halfway), cosine should be near min (cos(π) = -1, so result should be min_lr)
    // But let's just verify it's decreasing
    const lr_start2 = scheduler2.getLearningRate(0.1, 0);
    const lr_mid2 = scheduler2.getLearningRate(0.1, 5000);
    try std.testing.expect(lr_mid2 < lr_start2);
}

// Test step decay with different step sizes
test "LRScheduler step decay variations" {
    // Small step size
    const scheduler1 = LRScheduler{
        .step = .{ .step_size = 1, .decay_rate = 0.9 },
    };
    try std.testing.expectApproxEqAbs(scheduler1.getLearningRate(0.01, 0), 0.01, 1e-6);
    try std.testing.expectApproxEqAbs(scheduler1.getLearningRate(0.01, 1), 0.009, 1e-6);
    try std.testing.expectApproxEqAbs(scheduler1.getLearningRate(0.01, 2), 0.0081, 1e-6);

    // Large step size
    const scheduler2 = LRScheduler{
        .step = .{ .step_size = 100, .decay_rate = 0.5 },
    };
    try std.testing.expectApproxEqAbs(scheduler2.getLearningRate(0.01, 99), 0.01, 1e-6);
    try std.testing.expectApproxEqAbs(scheduler2.getLearningRate(0.01, 100), 0.005, 1e-6);
}

// Test exponential decay edge cases
test "LRScheduler exponential decay edge cases" {
    // Very slow decay: 0.999^1000 ≈ 0.368, so lr ≈ 0.00368
    const scheduler1 = LRScheduler{
        .exponential = .{ .decay_rate = 0.999 },
    };
    const lr_1000 = scheduler1.getLearningRate(0.01, 1000);
    const expected_lr = 0.01 * std.math.pow(f32, 0.999, 1000);
    try std.testing.expect(lr_1000 < 0.01);
    try std.testing.expectApproxEqAbs(lr_1000, expected_lr, 1e-5);

    // Fast decay
    const scheduler2 = LRScheduler{
        .exponential = .{ .decay_rate = 0.5 },
    };
    const lr_10 = scheduler2.getLearningRate(0.01, 10);
    try std.testing.expectApproxEqAbs(lr_10, 0.01 * std.math.pow(f32, 0.5, 10), 1e-7);
}

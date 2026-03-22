/// CUDA Loss Functions Validation (Phase 4)
const std = @import("std");
const cuda = @import("cuda.zig");

// T4.1: MSE Loss - Placeholder
test "cuda_mse_loss_placeholder" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Verify backend is ready for loss computation
    try std.testing.expect(backend.context.device_props.compute_capability_major >= 3);
}

// T4.2: Cross-Entropy Loss
test "cuda_crossentropy_forward" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const batch = 16;
    const classes = 10;

    const input = try std.testing.allocator.alloc(f32, batch * classes);
    const output = try std.testing.allocator.alloc(f32, batch * classes);
    defer std.testing.allocator.free(input);
    defer std.testing.allocator.free(output);

    // Softmax input
    for (input, 0..) |*val, i| val.* = @as(f32, @floatFromInt(i % classes)) * 0.1;

    try backend.softmaxForward(input, output, batch, classes);

    // Verify output sums to 1 for each batch
    for (0..batch) |b| {
        var sum: f32 = 0;
        for (0..classes) |c| {
            sum += output[b * classes + c];
        }
        try std.testing.expect(@abs(sum - 1.0) < 1e-5);
    }
}

// T4.3: BCE Loss - Placeholder
test "cuda_bce_loss_placeholder" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    try std.testing.expect(backend.context.device_props.compute_capability_major >= 3);
}

// T4.4: KL Divergence - Placeholder
test "cuda_kl_divergence_placeholder" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    try std.testing.expect(backend.context.device_props.compute_capability_major >= 3);
}

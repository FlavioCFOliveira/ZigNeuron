/// Gradient checking utilities for validating backward passes
/// Uses finite differences to verify gradients computed by backpropagation
const std = @import("std");
const network = @import("network.zig");
const layer = @import("layer.zig");
const loss = @import("loss.zig");

/// Epsilon for finite difference approximation
const DEFAULT_EPSILON: f32 = 1e-4;

/// Tolerance for gradient comparison
const DEFAULT_TOLERANCE: f32 = 1e-4;

/// Gradient check result
pub const GradientCheckResult = struct {
    /// Whether all gradients passed the check
    passed: bool,
    /// Maximum absolute difference found
    max_diff: f32,
    /// Average absolute difference
    avg_diff: f32,
    /// Number of parameters checked
    num_params: usize,
    /// Number of parameters that failed
    num_failed: usize,

    pub fn print(self: GradientCheckResult) void {
        std.debug.print("\n=== Gradient Check Results ===\n", .{});
        std.debug.print("Passed: {s}\n", .{if (self.passed) "YES" else "NO"});
        std.debug.print("Max difference: {e:.6}\n", .{self.max_diff});
        std.debug.print("Avg difference: {e:.6}\n", .{self.avg_diff});
        std.debug.print("Parameters checked: {}\n", .{self.num_params});
        std.debug.print("Parameters failed: {}\n", .{self.num_failed});
        std.debug.print("===============================\n", .{});
    }
};

/// Check gradients for a single dense layer using finite differences
pub fn checkLayerGradients(
    allocator: std.mem.Allocator,
    lyr: *layer.Dense,
    input: []const f32,
    target: []const f32,
    loss_fn: loss.Loss,
    epsilon: f32,
) !GradientCheckResult {
    var result = GradientCheckResult{
        .passed = true,
        .max_diff = 0.0,
        .avg_diff = 0.0,
        .num_params = 0,
        .num_failed = 0,
    };

    const total_diffs = try allocator.alloc(f32, lyr.weights.slice.len + lyr.bias.slice.len);
    defer allocator.free(total_diffs);
    var diff_idx: usize = 0;

    // Check weight gradients
    for (0..lyr.weights.slice.len) |i| {
        const original_val = lyr.weights.slice[i];

        // Forward pass with w + epsilon
        lyr.weights.slice[i] = original_val + epsilon;
        var output_plus = try allocator.alloc(f32, lyr.output_size);
        defer allocator.free(output_plus);
        try lyr.forward(input, null, output_plus, null);
        const loss_plus = computeLoss(loss_fn, output_plus, target);

        // Forward pass with w - epsilon
        lyr.weights.slice[i] = original_val - epsilon;
        var output_minus = try allocator.alloc(f32, lyr.output_size);
        defer allocator.free(output_minus);
        try lyr.forward(input, null, output_minus, null);
        const loss_minus = computeLoss(loss_fn, output_minus, target);

        // Restore original value
        lyr.weights.slice[i] = original_val;

        // Numerical gradient
        const numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);

        // Analytical gradient from backprop
        const analytical_grad = lyr.grad_weights.slice[i];

        // Compare
        const diff = @abs(numerical_grad - analytical_grad);
        total_diffs[diff_idx] = diff;
        diff_idx += 1;
        result.num_params += 1;

        if (diff > epsilon) {
            result.num_failed += 1;
            result.passed = false;
        }
        if (diff > result.max_diff) {
            result.max_diff = diff;
        }
    }

    // Check bias gradients
    for (0..lyr.bias.slice.len) |i| {
        const original_val = lyr.bias.slice[i];

        // Forward pass with b + epsilon
        lyr.bias.slice[i] = original_val + epsilon;
        var output_plus = try allocator.alloc(f32, lyr.output_size);
        defer allocator.free(output_plus);
        try lyr.forward(input, null, output_plus, null);
        const loss_plus = computeLoss(loss_fn, output_plus, target);

        // Forward pass with b - epsilon
        lyr.bias.slice[i] = original_val - epsilon;
        var output_minus = try allocator.alloc(f32, lyr.output_size);
        defer allocator.free(output_minus);
        try lyr.forward(input, null, output_minus, null);
        const loss_minus = computeLoss(loss_fn, output_minus, target);

        // Restore original value
        lyr.bias.slice[i] = original_val;

        // Numerical gradient
        const numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);

        // Analytical gradient
        const analytical_grad = lyr.grad_bias.slice[i];

        // Compare
        const diff = @abs(numerical_grad - analytical_grad);
        total_diffs[diff_idx] = diff;
        diff_idx += 1;
        result.num_params += 1;

        if (diff > epsilon) {
            result.num_failed += 1;
            result.passed = false;
        }
        if (diff > result.max_diff) {
            result.max_diff = diff;
        }
    }

    // Compute average
    var sum_diff: f32 = 0.0;
    for (total_diffs[0..diff_idx]) |d| {
        sum_diff += d;
    }
    result.avg_diff = sum_diff / @as(f32, @floatFromInt(diff_idx));

    return result;
}

/// Compute loss for a single sample
fn computeLoss(loss_fn: loss.Loss, output: []const f32, target: []const f32) f32 {
    return switch (loss_fn) {
        .mse => {
            var sum: f32 = 0.0;
            for (output, target) |o, t| {
                const diff = o - t;
                sum += diff * diff;
            }
            return sum / @as(f32, @floatFromInt(output.len));
        },
        .binary_cross_entropy => {
            var sum: f32 = 0.0;
            for (output, target) |o, t| {
                const p = std.math.clamp(o, 1e-7, 1.0 - 1e-7);
                sum += t * std.math.log(p) + (1.0 - t) * std.math.log(1.0 - p);
            }
            return -sum / @as(f32, @floatFromInt(output.len));
        },
        .cross_entropy => {
            // Assume output is already softmax probabilities
            var sum: f32 = 0.0;
            for (output, target) |o, t| {
                const p = std.math.clamp(o, 1e-7, 1.0);
                sum += t * std.math.log(p);
            }
            return -sum;
        },
        else => @panic("Loss function not supported for gradient check"),
    };
}

/// Check gradients for an entire network
pub fn checkNetworkGradients(
    allocator: std.mem.Allocator,
    net: *network.Network,
    input: []const f32,
    target: []const f32,
    loss_fn: loss.Loss,
    epsilon: f32,
) !GradientCheckResult {
    var result = GradientCheckResult{
        .passed = true,
        .max_diff = 0.0,
        .avg_diff = 0.0,
        .num_params = 0,
        .num_failed = 0,
    };

    // Run forward and backward pass to compute analytical gradients
    _ = try net.forward(input, &[_]f32{});
    try net.computeGradients(target, loss_fn);

    // Check each dense layer
    for (net.layers.items) |lyr| {
        switch (lyr) {
            .dense => |d| {
                const layer_result = try checkLayerGradients(allocator, d, input, target, loss_fn, epsilon);

                result.num_params += layer_result.num_params;
                result.num_failed += layer_result.num_failed;
                if (layer_result.max_diff > result.max_diff) {
                    result.max_diff = layer_result.max_diff;
                }
                if (!layer_result.passed) {
                    result.passed = false;
                }
            },
            else => {}, // Skip non-dense layers for now
        }
    }

    return result;
}

test "gradient check on simple dense layer" {
    const allocator = std.testing.allocator;

    // Create a simple network
    const backend = try @import("backend.zig").Backend.init(allocator);
    defer backend.deinit();

    var net = try network.Network.init(allocator, backend);
    defer net.deinit();

    _ = try net.addDense(2, 2, .sigmoid);

    // Simple test data
    const input = [_]f32{ 0.5, 0.3 };
    const target = [_]f32{ 0.8, 0.2 };

    const result = try checkNetworkGradients(allocator, net, &input, &target, .{ .mse = {} }, 1e-4);

    // Should pass with small tolerance
    try std.testing.expect(result.passed);
}

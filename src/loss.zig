/// Loss functions for neural network training
const std = @import("std");

pub const Loss = union(enum) {
    mse,  // Mean Squared Error
    cross_entropy,
    binary_cross_entropy,

    /// Compute loss value
    pub fn forward(self: Loss, output: []const f32, target: []const f32) !f32 {
        if (output.len != target.len) return error.ShapeMismatch;

        return switch (self) {
            .mse => self.mseForward(output, target),
            .cross_entropy => @panic("cross_entropy not implemented"),
            .binary_cross_entropy => @panic("binary_cross_entropy not implemented"),
        };
    }

    /// Compute gradient of loss w.r.t output
    pub fn backward(self: Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        if (output.len != target.len) return error.ShapeMismatch;
        if (output.len != grad_output.len) return error.ShapeMismatch;

        for (output, target, grad_output, 0..) |o, t, _, i| {
            _ = self;
            grad_output[i] = 2 * (o - t);
        }
    }

    fn mseForward(self: Loss, output: []const f32, target: []const f32) f32 {
        _ = self;
        var sum: f32 = 0;
        for (output, target) |o, t| {
            const diff = o - t;
            sum += diff * diff;
        }
        return sum / @as(f32, @floatFromInt(output.len));
    }
};

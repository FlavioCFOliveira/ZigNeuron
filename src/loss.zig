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
            .cross_entropy => self.crossEntropyForward(output, target),
            .binary_cross_entropy => self.binaryCrossEntropyForward(output, target),
        };
    }

    /// Compute gradient of loss w.r.t output
    pub fn backward(self: Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        if (output.len != target.len) return error.ShapeMismatch;
        if (output.len != grad_output.len) return error.ShapeMismatch;

        switch (self) {
            .mse => try self.mseBackward(output, target, grad_output),
            .cross_entropy => try self.crossEntropyBackward(output, target, grad_output),
            .binary_cross_entropy => try self.binaryCrossEntropyBackward(output, target, grad_output),
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

    fn mseBackward(self: Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        _ = self;
        for (output, target, grad_output, 0..) |o, t, _, i| {
            grad_output[i] = 2 * (o - t);
        }
    }

    fn crossEntropyForward(self: Loss, output: []const f32, target: []const f32) !f32 {
        _ = self;
        var sum: f32 = 0;
        const eps: f32 = 1e-8;
        for (output, target) |o, t| {
            // Clamp output for numerical stability
            var p = o;
            if (p < eps) p = eps;
            if (p > 1 - eps) p = 1 - eps;
            sum -= t * @log(p);
        }
        return sum / @as(f32, @floatFromInt(output.len));
    }

    fn crossEntropyBackward(self: Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        _ = self;
        const eps: f32 = 1e-8;
        for (output, target, grad_output, 0..) |o, t, _, i| {
            // Clamp output for numerical stability
            var p = o;
            if (p < eps) p = eps;
            if (p > 1 - eps) p = 1 - eps;
            grad_output[i] = -t / p;
        }
    }

    fn binaryCrossEntropyForward(self: Loss, output: []const f32, target: []const f32) !f32 {
        _ = self;
        var sum: f32 = 0;
        const eps: f32 = 1e-8;
        for (output, target) |o, t| {
            // Clamp output for numerical stability
            var p = o;
            if (p < eps) p = eps;
            if (p > 1 - eps) p = 1 - eps;
            sum -= t * @log(p) + (1 - t) * @log(1 - p);
        }
        return sum / @as(f32, @floatFromInt(output.len));
    }

    fn binaryCrossEntropyBackward(self: Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        _ = self;
        const eps: f32 = 1e-8;
        for (output, target, grad_output, 0..) |o, t, _, i| {
            // Clamp output for numerical stability
            var p = o;
            if (p < eps) p = eps;
            if (p > 1 - eps) p = 1 - eps;
            grad_output[i] = (p - t) / (p * (1 - p));
        }
    }
};

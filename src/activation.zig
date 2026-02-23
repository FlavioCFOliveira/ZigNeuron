/// Activation functions and their derivatives
const std = @import("std");

pub const Activation = union(enum) {
    relu,
    sigmoid,
    tanh,
    softmax,
    linear, // Identity activation (no transformation)

    pub fn forward(self: Activation, x: f32) f32 {
        return switch (self) {
            .relu => reluForward(x),
            .sigmoid => sigmoidForward(x),
            .tanh => tanhForward(x),
            .softmax => @panic("softmax forward not implemented for single value"),
            .linear => x, // Identity: f(x) = x
        };
    }

    pub fn backward(self: Activation, y: f32, grad: f32) f32 {
        return switch (self) {
            .relu => reluBackward(y) * grad,
            .sigmoid => sigmoidBackward(y) * grad,
            .tanh => tanhBackward(y) * grad,
            .softmax => @panic("softmax backward not implemented for single value"),
            .linear => grad, // Derivative of identity is 1
        };
    }

    /// Softmax forward pass for vector input
    pub fn softmaxForward(self: Activation, input: []const f32, output: []f32) !void {
        _ = self;
        if (input.len != output.len) return error.ShapeMismatch;

        // Find max value for numerical stability
        var max_val: f32 = -std.math.inf(f32);
        for (input) |x| {
            if (x > max_val) max_val = x;
        }

        // Compute exponentials
        var sum: f32 = 0;
        for (input, 0..) |x, i| {
            output[i] = std.math.exp(x - max_val);
            sum += output[i];
        }

        // Normalize
        for (0..output.len) |i| {
            output[i] /= sum;
        }
    }

    /// Softmax backward pass (efficient O(N) implementation)
    /// Assumes 'softmax_output' contains the already computed softmax values
    pub fn softmaxBackward(self: Activation, softmax_output: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        _ = self;
        if (softmax_output.len != grad_output.len or softmax_output.len != grad_input.len) return error.ShapeMismatch;

        // Efficient gradient computation: grad_input_j = softmax_j * (grad_output_j - sum(grad_output_i * softmax_i))
        var sum_grad_softmax: f32 = 0;
        for (grad_output, softmax_output) |go, s| {
            sum_grad_softmax += go * s;
        }

        for (grad_input, grad_output, softmax_output) |*gi, go, s| {
            gi.* = s * (go - sum_grad_softmax);
        }
    }

    fn reluForward(x: f32) f32 {
        return if (x > 0) x else 0;
    }

    fn reluBackward(y: f32) f32 {
        return if (y > 0) 1 else 0;
    }

    fn sigmoidForward(x: f32) f32 {
        if (x >= 0) {
            return 1 / (1 + std.math.exp(-x));
        } else {
            const exp_x = std.math.exp(x);
            return exp_x / (1 + exp_x);
        }
    }

    fn sigmoidBackward(y: f32) f32 {
        return y * (1 - y);
    }

    fn tanhForward(x: f32) f32 {
        return std.math.tanh(x);
    }

    fn tanhBackward(y: f32) f32 {
        return 1 - y * y;
    }
};

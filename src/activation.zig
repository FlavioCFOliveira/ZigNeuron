/// Activation functions and their derivatives
const std = @import("std");

pub const Activation = union(enum) {
    relu,
    sigmoid,
    tanh,
    softmax,

    pub fn forward(self: Activation, x: f32) f32 {
        return switch (self) {
            .relu => reluForward(x),
            .sigmoid => sigmoidForward(x),
            .tanh => tanhForward(x),
            .softmax => @panic("softmax forward not implemented for single value"),
        };
    }

    pub fn backward(self: Activation, x: f32, grad: f32) f32 {
        return switch (self) {
            .relu => reluBackward(x) * grad,
            .sigmoid => sigmoidBackward(x) * grad,
            .tanh => tanhBackward(x) * grad,
            .softmax => @panic("softmax backward not implemented for single value"),
        };
    }

    fn reluForward(x: f32) f32 {
        return if (x > 0) x else 0;
    }

    fn reluBackward(x: f32) f32 {
        return if (x > 0) 1 else 0;
    }

    fn sigmoidForward(x: f32) f32 {
        if (x >= 0) {
            return 1 / (1 + std.math.exp(-x));
        } else {
            const exp_x = std.math.exp(x);
            return exp_x / (1 + exp_x);
        }
    }

    fn sigmoidBackward(x: f32) f32 {
        const s = sigmoidForward(x);
        return s * (1 - s);
    }

    fn tanhForward(x: f32) f32 {
        return std.math.tanh(x);
    }

    fn tanhBackward(x: f32) f32 {
        const t = tanhForward(x);
        return 1 - t * t;
    }
};

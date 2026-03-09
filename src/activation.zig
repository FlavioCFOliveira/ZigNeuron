/// Activation functions and their derivatives
///
/// References:
/// - ReLU: Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted
///   Boltzmann machines. ICML.
/// - Sigmoid & Tanh: Traditional activation functions, see Haykin, S. (1998). Neural
///   Networks: A Comprehensive Foundation (2nd ed.).
/// - Softmax: Bridle, J. S. (1990). Probabilistic interpretation of feedforward
///   classification network outputs. Neurocomputing.
/// - GELU: Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs).
///   arXiv:1606.08415.
const std = @import("std");

pub const Activation = union(enum) {
    relu,
    sigmoid,
    tanh,
    softmax,
    linear, // Identity activation (no transformation)
    gelu, // Gaussian Error Linear Unit
    leaky_relu, // Leaky ReLU: f(x) = x if x > 0, else alpha * x (alpha = 0.01)
    elu, // Exponential Linear Unit: f(x) = x if x > 0, else alpha * (exp(x) - 1) (alpha = 1.0)

    pub fn forward(self: Activation, x: f32) f32 {
        return switch (self) {
            .relu => reluForward(x),
            .sigmoid => sigmoidForward(x),
            .tanh => tanhForward(x),
            .softmax => @panic("softmax forward not implemented for single value"),
            .linear => x, // Identity: f(x) = x
            .gelu => geluForward(x),
            .leaky_relu => leakyReluForward(x),
            .elu => eluForward(x),
        };
    }

    pub fn backward(self: Activation, y: f32, grad: f32) f32 {
        return switch (self) {
            .relu => reluBackward(y) * grad,
            .sigmoid => sigmoidBackward(y) * grad,
            .tanh => tanhBackward(y) * grad,
            .softmax => @panic("softmax backward not implemented for single value"),
            .linear => grad, // Derivative of identity is 1
            .gelu => geluBackward(y) * grad,
            .leaky_relu => leakyReluBackward(y) * grad,
            .elu => eluBackward(y) * grad,
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

        // Normalize with protection against division by zero
        if (sum > 0) {
            for (0..output.len) |i| {
                output[i] /= sum;
            }
        } else {
            // If sum is 0 (e.g., all inputs were -inf), set uniform distribution
            const uniform_val = 1.0 / @as(f32, @floatFromInt(output.len));
            @memset(output, uniform_val);
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

    /// GELU forward: GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
    /// Using the approximation: 0.5 * x * (1 + tanh[√(2/π) * (x + 0.044715 * x³)])
    fn geluForward(x: f32) f32 {
        const sqrt_2_over_pi: f32 = 0.7978845608; // sqrt(2/π)
        const coeff: f32 = 0.044715;
        const x_cubed = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        return 0.5 * x * (1.0 + std.math.tanh(inner));
    }

    /// GELU backward derivative
    /// For GELU, we approximate x from y (activated output) for backward compatibility
    /// dGELU/dx = Φ(x) + x * φ(x) where φ is the PDF of standard normal
    fn geluBackward(y: f32) f32 {
        // Approximate x ≈ y for computing derivative (good approximation for small values)
        // For more accuracy, we'd need to store x during forward pass
        const x = y;
        const sqrt_2_over_pi: f32 = 0.7978845608;
        const coeff: f32 = 0.044715;
        const x_cubed = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        const tanh_inner = std.math.tanh(inner);
        const sech2_inner = 1.0 - tanh_inner * tanh_inner;
        return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x);
    }

    /// LeakyReLU forward: f(x) = x if x > 0, else alpha * x
    /// Default alpha = 0.01 (common value)
    /// Reference: Maas, A. L., et al. (2013). Rectifier nonlinearities improve neural network acoustic models. ICML.
    fn leakyReluForward(x: f32) f32 {
        const alpha: f32 = 0.01;
        return if (x > 0) x else alpha * x;
    }

    /// LeakyReLU backward derivative
    /// f'(x) = 1 if x > 0, else alpha
    fn leakyReluBackward(y: f32) f32 {
        const alpha: f32 = 0.01;
        // y > 0 means x was > 0 (since leaky_relu preserves sign for positive x)
        return if (y > 0) 1.0 else alpha;
    }

    /// ELU forward: f(x) = x if x > 0, else alpha * (exp(x) - 1)
    /// Default alpha = 1.0
    /// Reference: Clevert, D. A., et al. (2015). Fast and accurate deep network learning by exponential linear units. ICLR.
    fn eluForward(x: f32) f32 {
        const alpha: f32 = 1.0;
        return if (x > 0) x else alpha * (std.math.exp(x) - 1.0);
    }

    /// ELU backward derivative
    /// f'(x) = 1 if x > 0, else alpha * exp(x) = f(x) + alpha
    fn eluBackward(y: f32) f32 {
        const alpha: f32 = 1.0;
        // For ELU, if y > 0 then x was > 0 and derivative is 1
        // If y <= 0, then x was <= 0 and derivative is y + alpha
        return if (y > 0) 1.0 else y + alpha;
    }
};

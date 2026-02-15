/// Loss functions for neural network training
const std = @import("std");

pub const Loss = union(enum) {
    mse,  // Mean Squared Error
    cross_entropy,  // Cross-entropy (expects pre-computed probabilities)
    cross_entropy_logits,  // Cross-entropy with logits (combined softmax + cross-entropy)
    binary_cross_entropy,

    /// Compute loss value
    pub fn forward(self: Loss, output: []const f32, target: []const f32) !f32 {
        if (output.len != target.len) return error.ShapeMismatch;

        return switch (self) {
            .mse => self.mseForward(output, target),
            .cross_entropy => self.crossEntropyForward(output, target),
            .cross_entropy_logits => self.crossEntropyLogitsForward(output, target),
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
            .cross_entropy_logits => try self.crossEntropyLogitsBackward(output, target, grad_output),
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
        // This computes cross-entropy assuming 'output' are logits (not probabilities)
        // For numerical stability, we use log-sum-exp trick
        // CE = -sum(t * log(softmax(o)))
        // Gradient with softmax is simply (o - t) when using logits

        // Find max logit for numerical stability
        var max_logit: f32 = -std.math.inf(f32);
        for (output) |o| {
            if (o > max_logit) max_logit = o;
        }

        // If all logits are -inf, return a default loss
        if (max_logit == -std.math.inf(f32)) {
            return 0;
        }

        // Compute log sum exp with overflow protection
        var log_sum_exp: f32 = 0;
        for (output) |o| {
            const diff = o - max_logit;
            // Clamp the difference to avoid overflow
            const clamped_diff = if (diff > 50) 50 else if (diff < -50) -50 else diff;
            const exp_val = std.math.exp(clamped_diff);
            log_sum_exp += exp_val;
        }

        // If log_sum_exp is 0 or invalid, return a default loss
        if (log_sum_exp <= 0 or !std.math.isFinite(log_sum_exp)) {
            return 0;
        }

        log_sum_exp = @log(log_sum_exp) + max_logit;

        // Clamp log_sum_exp for numerical stability
        if (log_sum_exp > 50) log_sum_exp = 50;
        if (log_sum_exp < -50) log_sum_exp = -50;

        // Compute loss: -t * log(softmax(o)) = -t * (o - log_sum_exp)
        // For one-hot encoding, we just take the target class
        var sum: f32 = 0;
        for (target, output) |t, o| {
            if (t > 0.5) {
                const loss_val = -(o - log_sum_exp);
                // Clamp loss to avoid extreme values
                if (std.math.isFinite(loss_val)) {
                    sum += loss_val;
                }
                break;
            }
        }
        return sum / @as(f32, @floatFromInt(output.len));
    }

    fn crossEntropyBackward(self: Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        _ = self;
        // For cross-entropy with softmax output, the gradient simplifies to (p - t)
        // This is the derivative of the loss with respect to the logits
        for (output, target, grad_output, 0..) |o, t, _, i| {
            grad_output[i] = o - t;
        }
    }

    /// Cross-entropy loss with logits (numerically stable)
    /// This combines softmax + cross-entropy for numerical stability
    /// Like PyTorch's nn.CrossEntropyLoss or TensorFlow's softmax_cross_entropy_with_logits
    fn crossEntropyLogitsForward(self: Loss, logits: []const f32, target: []const f32) !f32 {
        _ = self;
        
        // Find max logit for numerical stability (log-sum-exp trick)
        var max_logit: f32 = logits[0];
        for (logits[1..]) |logit| {
            if (logit > max_logit) max_logit = logit;
        }

        // Compute log-sum-exp: log(sum(exp(logits - max)))
        var sum_exp: f32 = 0;
        for (logits) |logit| {
            sum_exp += std.math.exp(logit - max_logit);
        }
        const log_sum_exp = @log(sum_exp) + max_logit;

        // Compute cross-entropy: -sum(target * (logit - log_sum_exp))
        // For one-hot targets, only the true class contributes
        var loss: f32 = 0;
        for (target, logits) |t, logit| {
            if (t > 0.5) {  // One-hot encoding check
                loss = -(logit - log_sum_exp);
                break;
            }
        }

        return loss;
    }

    /// Gradient of cross-entropy with logits
    /// The beautiful property: d(softmax_cross_entropy)/d(logits) = softmax(logits) - target
    fn crossEntropyLogitsBackward(self: Loss, logits: []const f32, target: []const f32, grad_output: []f32) !void {
        _ = self;
        
        // Compute softmax(logits)
        var max_logit: f32 = logits[0];
        for (logits[1..]) |logit| {
            if (logit > max_logit) max_logit = logit;
        }

        var sum_exp: f32 = 0;
        for (logits) |logit| {
            sum_exp += std.math.exp(logit - max_logit);
        }

        // Gradient is: softmax(logits) - target
        for (logits, target, grad_output, 0..) |logit, t, _, i| {
            const prob = std.math.exp(logit - max_logit) / sum_exp;
            grad_output[i] = prob - t;
        }
    }

    fn binaryCrossEntropyForward(self: Loss, output: []const f32, target: []const f32) !f32 {
        _ = self;
        var sum: f32 = 0;
        const eps: f32 = 1e-7;
        for (output, target) |o, t| {
            // Clamp output for numerical stability
            var p = o;
            if (p < eps) p = eps;
            if (p > 1 - eps) p = 1 - eps;
            // Clamp again for 1-p to avoid NaN
            var one_minus_p = 1 - p;
            if (one_minus_p < eps) one_minus_p = eps;
            if (one_minus_p > 1 - eps) one_minus_p = 1 - eps;
            // Use safe log with clamped values
            sum -= t * @log(p) + (1 - t) * @log(one_minus_p);
        }
        return sum / @as(f32, @floatFromInt(output.len));
    }

    fn binaryCrossEntropyBackward(self: Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        _ = self;
        // For BCE with sigmoid output, the gradient simplifies to (p - t)
        // because the sigmoid derivative cancels with the denominator
        // Use clipping for numerical stability
        const eps: f32 = 1e-7;
        for (output, target, grad_output, 0..) |o, t, _, i| {
            var p = o;
            if (p < eps) p = eps;
            if (p > 1 - eps) p = 1 - eps;
            // Gradient of BCE with sigmoid is simply (prediction - target)
            grad_output[i] = p - t;
        }
    }
};

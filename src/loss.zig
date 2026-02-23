/// Loss functions for neural network training
const std = @import("std");

pub const Loss = union(enum) {
    mse, // Mean Squared Error
    cross_entropy, // Cross-entropy (expects pre-computed probabilities)
    cross_entropy_logits, // Cross-entropy with logits (combined softmax + cross-entropy)
    binary_cross_entropy,
    kl_divergence, // KL Divergence for VAE

    /// Whether the loss gradient is already computed w.r.t. pre-activation (logits)
    /// For cross-entropy with log-softmax, the gradient is (softmax - target) which is w.r.t. logits
    /// For MSE and BCE, the gradient needs to be multiplied by activation derivative
    pub fn isLogitsGradient(self: Loss) bool {
        return switch (self) {
            .cross_entropy, .kl_divergence => true,
            else => false,
        };
    }

    /// Compute loss value
    pub fn forward(self: Loss, output: []const f32, target: []const f32) !f32 {
        // target is unused for kl_divergence
        if (self != .kl_divergence and output.len != target.len) return error.ShapeMismatch;

        return switch (self) {
            .mse => self.mseForward(output, target),
            .cross_entropy => self.crossEntropyForward(output, target),
            .cross_entropy_logits => self.crossEntropyLogitsForward(output, target),
            .binary_cross_entropy => self.binaryCrossEntropyForward(output, target),
            .kl_divergence => self.klDivergenceForward(output),
        };
    }

    /// Compute gradient of loss w.r.t output
    pub fn backward(self: Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        if (self != .kl_divergence and output.len != target.len) return error.ShapeMismatch;
        if (output.len != grad_output.len) return error.ShapeMismatch;

        switch (self) {
            .mse => try self.mseBackward(output, target, grad_output),
            .cross_entropy => try self.crossEntropyBackward(output, target, grad_output),
            .cross_entropy_logits => try self.crossEntropyLogitsBackward(output, target, grad_output),
            .binary_cross_entropy => try self.binaryCrossEntropyBackward(output, target, grad_output),
            .kl_divergence => try self.klDivergenceBackward(output, grad_output),
        }
    }

    fn klDivergenceForward(self: Loss, output: []const f32) f32 {
        _ = self;
        // output = [mu, log_var]
        const n = output.len / 2;
        const mu = output[0..n];
        const log_var = output[n..];
        var kld: f32 = 0;
        for (0..n) |i| {
            kld += -0.5 * (1.0 + log_var[i] - mu[i] * mu[i] - std.math.exp(log_var[i]));
        }
        return kld / @as(f32, @floatFromInt(n));
    }

    fn klDivergenceBackward(self: Loss, output: []const f32, grad_output: []f32) !void {
        _ = self;
        const n = output.len / 2;
        const mu = output[0..n];
        const log_var = output[n..];
        const grad_mu = grad_output[0..n];
        const grad_log_var = grad_output[n..];
        const n_f = @as(f32, @floatFromInt(n));

        for (mu, grad_mu) |m, *gm| {
            // dKLD/dmu = mu
            gm.* = m / n_f;
        }
        for (log_var, grad_log_var) |lv, *glv| {
            // dKLD/dlog_var = 0.5 * (exp(log_var) - 1)
            glv.* = 0.5 * (std.math.exp(lv) - 1.0) / n_f;
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
        const n: f32 = @floatFromInt(output.len);
        for (output, target, grad_output) |o, t, *go| {
            go.* = 2 * (o - t) / n;
        }
    }

    fn crossEntropyForward(self: Loss, output: []const f32, target: []const f32) !f32 {
        _ = self;
        // This computes cross-entropy assuming 'output' are logits (not probabilities)
        // For numerical stability, we use log-sum-exp trick
        // CE = -sum(t * log(softmax(o))) = -sum(t * (o - log_sum_exp))
        // Gradient with softmax is simply (softmax - target) when using logits

        const n = output.len;
        if (n == 0) return 0;

        // Find max logit for numerical stability
        var max_logit: f32 = -std.math.inf(f32);
        for (output) |o| {
            if (o > max_logit) max_logit = o;
        }

        // If all logits are -inf, return a default loss
        if (max_logit == -std.math.inf(f32)) {
            return 0;
        }

        // Compute log sum exp using log-sum-exp trick for stability
        // log_sum_exp = log(sum(exp(o - max_logit))) + max_logit
        var log_sum_exp: f32 = 0;
        for (output) |o| {
            const diff = o - max_logit;
            // Use log1p(exp(diff)) for numerical stability when diff is small
            // But for simplicity and stability, we just clamp exp
            const clamped_diff = if (diff > 50) 50 else if (diff < -50) -50 else diff;
            const exp_val = std.math.exp(clamped_diff);
            log_sum_exp += exp_val;
        }

        // If log_sum_exp is invalid, return a default loss
        if (!std.math.isFinite(log_sum_exp) or log_sum_exp <= 0) {
            return 0;
        }

        log_sum_exp = @log(log_sum_exp) + max_logit;

        // Compute loss: -sum(t * (o - log_sum_exp))
        // For one-hot encoding, only the target class contributes
        // CE = -log(softmax(target_class)) = log_sum_exp - target_class_logit
        var loss_sum: f32 = 0;
        var sample_count: f32 = 0;

        for (target, output) |t, o| {
            if (t > 0.5) {
                // Loss for this sample: log_sum_exp - output[target_class]
                // This is: -log(softmax(output[target_class]))
                const sample_loss = log_sum_exp - o;
                loss_sum += sample_loss;
                sample_count += 1;
            }
        }

        // Average over number of samples (not output dimension)
        if (sample_count == 0) return 0;
        return loss_sum / sample_count;
    }

    fn crossEntropyBackward(self: Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        _ = self;
        // For cross-entropy with log-softmax, the gradient is (softmax(output) - target)
        // This is the derivative of the loss with respect to the logits

        const n = output.len;
        if (n == 0) return;

        // Find max value for numerical stability
        var max_val: f32 = output[0];
        for (output[1..]) |x| {
            if (x > max_val) max_val = x;
        }

        // Compute exponentials into grad_output to avoid allocation
        var sum: f32 = 0;
        for (output, grad_output) |o, *go| {
            go.* = std.math.exp(o - max_val);
            sum += go.*;
        }

        // Normalize and subtract target: grad = softmax - target
        for (grad_output, target) |*go, t| {
            go.* = (go.* / sum) - t;
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
            if (t > 0.5) { // One-hot encoding check
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

        if (logits.len == 0) return;

        // Compute softmax(logits)
        var max_logit: f32 = logits[0];
        for (logits[1..]) |logit| {
            if (logit > max_logit) max_logit = logit;
        }

        var sum_exp: f32 = 0;
        for (logits, grad_output) |logit, *go| {
            go.* = std.math.exp(logit - max_logit);
            sum_exp += go.*;
        }

        // Gradient is: softmax(logits) - target
        for (grad_output, target) |*go, t| {
            go.* = (go.* / sum_exp) - t;
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
        const n: f32 = @floatFromInt(output.len);
        const eps: f32 = 1e-7;
        for (output, target, grad_output) |o, t, *go| {
            var p = o;
            if (p < eps) p = eps;
            if (p > 1.0 - eps) p = 1.0 - eps;
            // Gradient of BCE with sigmoid is simply (prediction - target)
            go.* = (p - t) / n;
        }
    }
};

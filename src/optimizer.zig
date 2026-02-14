/// Optimizers for neural network training
/// Provides different optimization algorithms for gradient descent
const std = @import("std");
const layer_module = @import("layer.zig");

/// Base optimizer interface
pub const Optimizer = union(enum) {
    sgd: Sgd,
    adam: Adam,
    rmsprop: Rmsprop,

    /// Initialize the optimizer for a layer
    pub fn init(self: *Optimizer, allocator: std.mem.Allocator, lyr: *const layer_module.Dense) !void {
        switch (self.*) {
            .sgd => |*opt| try opt.init(allocator, lyr),
            .adam => |*opt| try opt.init(allocator, lyr),
            .rmsprop => |*opt| try opt.init(allocator, lyr),
        }
    }

    /// Deinitialize the optimizer for a layer
    pub fn deinit(self: *Optimizer, allocator: std.mem.Allocator, lyr: *const layer_module.Dense) void {
        _ = lyr;
        switch (self.*) {
            .sgd => |*opt| opt.deinit(allocator),
            .adam => |*opt| opt.deinit(allocator),
            .rmsprop => |*opt| opt.deinit(allocator),
        }
    }

    /// Update weights for a specific layer
    pub fn step(self: *Optimizer, lyr: *layer_module.Dense, learning_rate: f32) void {
        switch (self.*) {
            .sgd => |*opt| opt.step(lyr, learning_rate),
            .adam => |*opt| opt.step(lyr, learning_rate),
            .rmsprop => |*opt| opt.step(lyr, learning_rate),
        }
    }
};

/// Stochastic Gradient Descent
pub const Sgd = struct {
    /// Momentum term for SGD with momentum
    momentum: f32 = 0.0,
    /// Velocity for momentum
    velocity_weights: ?[]f32 = null,
    velocity_bias: ?[]f32 = null,

    pub fn init(self: *Sgd, allocator: std.mem.Allocator, lyr: *const layer_module.Dense) !void {
        if (self.momentum > 0) {
            const n_weights = lyr.input_size * lyr.output_size;
            const n_bias = lyr.output_size;

            self.velocity_weights = try allocator.alloc(f32, n_weights);
            self.velocity_bias = try allocator.alloc(f32, n_bias);

            @memset(self.velocity_weights.?, 0);
            @memset(self.velocity_bias.?, 0);
        }
    }

    pub fn deinit(self: *Sgd, allocator: std.mem.Allocator) void {
        if (self.velocity_weights) |v| {
            allocator.free(v);
        }
        if (self.velocity_bias) |v| {
            allocator.free(v);
        }
        self.velocity_weights = null;
        self.velocity_bias = null;
    }

    pub fn step(self: *Sgd, lyr: *layer_module.Dense, learning_rate: f32) void {
        if (self.momentum > 0) {
            if (self.velocity_weights) |vel_w| {
                // SGD with momentum
                for (lyr.weights, 0..) |_, i| {
                    const grad = lyr.grad_weights[i];
                    const v_old = vel_w[i];
                    const v_new = self.momentum * v_old + grad;
                    vel_w[i] = v_new;
                    lyr.weights[i] -= learning_rate * v_new;
                }

                for (lyr.bias, 0..) |_, i| {
                    const grad = lyr.grad_bias[i];
                    const v_old = self.velocity_bias.?[i];
                    const v_new = self.momentum * v_old + grad;
                    self.velocity_bias.?[i] = v_new;
                    lyr.bias[i] -= learning_rate * v_new;
                }
                return;
            }
        }

        // Basic SGD
        for (lyr.weights, 0..) |_, i| {
            lyr.weights[i] -= learning_rate * lyr.grad_weights[i];
        }

        for (lyr.bias, 0..) |_, i| {
            lyr.bias[i] -= learning_rate * lyr.grad_bias[i];
        }
    }
};

/// Adam optimizer (Adaptive Moment Estimation)
pub const Adam = struct {
    /// First moment estimate (mean)
    m_weights: ?[]f32 = null,
    m_bias: ?[]f32 = null,
    /// Second moment estimate (uncentered variance)
    v_weights: ?[]f32 = null,
    v_bias: ?[]f32 = null,
    /// Decay rates
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    /// Small value for numerical stability
    eps: f32 = 1e-8,
    /// Time step
    t: usize = 0,

    pub fn init(self: *Adam, allocator: std.mem.Allocator, lyr: *const layer_module.Dense) !void {
        const n_weights = lyr.input_size * lyr.output_size;
        const n_bias = lyr.output_size;

        self.m_weights = try allocator.alloc(f32, n_weights);
        self.m_bias = try allocator.alloc(f32, n_bias);
        self.v_weights = try allocator.alloc(f32, n_weights);
        self.v_bias = try allocator.alloc(f32, n_bias);

        @memset(self.m_weights.?, 0);
        @memset(self.m_bias.?, 0);
        @memset(self.v_weights.?, 0);
        @memset(self.v_bias.?, 0);
    }

    pub fn deinit(self: *Adam, allocator: std.mem.Allocator) void {
        if (self.m_weights) |v| allocator.free(v);
        if (self.m_bias) |v| allocator.free(v);
        if (self.v_weights) |v| allocator.free(v);
        if (self.v_bias) |v| allocator.free(v);
        self.m_weights = null;
        self.m_bias = null;
        self.v_weights = null;
        self.v_bias = null;
    }

    /// Main step function
    pub fn step(self: *Adam, lyr: *layer_module.Dense, learning_rate: f32) void {
        self.t += 1;

        const t_f = @as(f32, @floatFromInt(self.t));

        // Bias correction factors
        const bias_corr1 = 1 - std.math.pow(f32, self.beta1, t_f);
        const bias_corr2 = 1 - std.math.pow(f32, self.beta2, t_f);

        const m_w = self.m_weights.?;
        const v_w = self.v_weights.?;
        const m_b = self.m_bias.?;
        const v_b = self.v_bias.?;

        for (lyr.weights, lyr.grad_weights, 0..) |_, grad, i| {
            // Update biased first moment estimate
            m_w[i] = self.beta1 * m_w[i] + (1 - self.beta1) * grad;

            // Update biased second raw moment estimate
            v_w[i] = self.beta2 * v_w[i] + (1 - self.beta2) * grad * grad;

            // Compute bias-corrected first moment estimate
            const m_hat = m_w[i] / bias_corr1;

            // Compute bias-corrected second raw moment estimate
            const v_hat = v_w[i] / bias_corr2;

            // Update weights
            lyr.weights[i] -= learning_rate * m_hat / (std.math.sqrt(v_hat) + self.eps);
        }

        for (lyr.bias, lyr.grad_bias, 0..) |_, grad, i| {
            // Update biased first moment estimate
            m_b[i] = self.beta1 * m_b[i] + (1 - self.beta1) * grad;

            // Update biased second raw moment estimate
            v_b[i] = self.beta2 * v_b[i] + (1 - self.beta2) * grad * grad;

            // Compute bias-corrected first moment estimate
            const m_hat = m_b[i] / bias_corr1;

            // Compute bias-corrected second raw moment estimate
            const v_hat = v_b[i] / bias_corr2;

            // Update bias
            lyr.bias[i] -= learning_rate * m_hat / (std.math.sqrt(v_hat) + self.eps);
        }
    }
};

/// RMSprop optimizer (Root Mean Square Propagation)
pub const Rmsprop = struct {
    /// Square gradient moving average
    g_weights: []f32,
    g_bias: []f32,
    /// Decay rate
    rho: f32 = 0.9,
    /// Small value for numerical stability
    eps: f32 = 1e-8,
    /// Time step
    t: usize = 0,

    pub fn init(self: *Rmsprop, allocator: std.mem.Allocator, lyr: *const layer_module.Dense) !void {
        const n_weights = lyr.input_size * lyr.output_size;
        const n_bias = lyr.output_size;

        self.g_weights = try allocator.alloc(f32, n_weights);
        self.g_bias = try allocator.alloc(f32, n_bias);

        @memset(self.g_weights, 0);
        @memset(self.g_bias, 0);
    }

    pub fn deinit(self: *Rmsprop, allocator: std.mem.Allocator) void {
        allocator.free(self.g_weights);
        allocator.free(self.g_bias);
    }

    /// Main step function
    pub fn step(self: *Rmsprop, lyr: *layer_module.Dense, learning_rate: f32) void {
        for (lyr.weights, lyr.grad_weights, 0..) |_, grad, i| {
            // Update moving average of squared gradients
            self.g_weights[i] = self.rho * self.g_weights[i] + (1 - self.rho) * grad * grad;

            // Update weights
            lyr.weights[i] -= learning_rate * grad / (std.math.sqrt(self.g_weights[i]) + self.eps);
        }

        for (lyr.bias, lyr.grad_bias, 0..) |_, grad, i| {
            // Update moving average of squared gradients
            self.g_bias[i] = self.rho * self.g_bias[i] + (1 - self.rho) * grad * grad;

            // Update bias
            lyr.bias[i] -= learning_rate * grad / (std.math.sqrt(self.g_bias[i]) + self.eps);
        }
    }
};

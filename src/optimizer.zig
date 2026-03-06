/// Optimizers for neural network training
/// Provides different optimization algorithms for gradient descent
///
/// References:
/// - Adam: Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
///   arXiv preprint arXiv:1412.6980.
/// - SGD with Momentum: Sutskever, I., et al. (2013). On the importance of initialization
///   and momentum in deep learning. ICML.
/// - RMSprop: Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient
///   by a running average of its recent magnitude.
const std = @import("std");
const layer_module = @import("layer.zig");
const tensor = @import("tensor.zig");

/// Learning rate schedulers for adjusting learning rate during training
pub const LRScheduler = union(enum) {
    /// Constant learning rate (no change)
    constant: void,
    /// Step decay: lr = initial_lr * decay_rate ^ (epoch / step_size)
    step: struct { step_size: usize, decay_rate: f32 },
    /// Exponential decay: lr = initial_lr * decay_rate ^ epoch
    exponential: struct { decay_rate: f32 },
    /// Cosine annealing: lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * epoch / max_epochs))
    cosine: struct { min_lr: f32, max_epochs: usize },

    /// Get the learning rate for the current epoch
    pub fn getLearningRate(self: LRScheduler, initial_lr: f32, epoch: usize) f32 {
        return switch (self) {
            .constant => initial_lr,
            .step => |s| initial_lr * std.math.pow(f32, s.decay_rate, @as(f32, @floatFromInt(epoch / s.step_size))),
            .exponential => |e| initial_lr * std.math.pow(f32, e.decay_rate, @floatFromInt(epoch)),
            .cosine => |c| {
                if (epoch >= c.max_epochs) return c.min_lr;
                const progress = @as(f32, @floatFromInt(epoch)) / @as(f32, @floatFromInt(c.max_epochs));
                const cosine_val = std.math.cos(std.math.pi * progress);
                return c.min_lr + 0.5 * (initial_lr - c.min_lr) * (1.0 + cosine_val);
            },
        };
    }
};

/// Base optimizer interface
pub const Optimizer = union(enum) {
    sgd: Sgd,
    adam: Adam,
    rmsprop: Rmsprop,

    /// Initialize the optimizer for a layer
    pub fn init(self: *Optimizer, allocator: std.mem.Allocator, lyr: *const layer_module.Layer) !void {
        switch (self.*) {
            .sgd => |*opt| try opt.init(allocator, lyr),
            .adam => |*opt| try opt.init(allocator, lyr),
            .rmsprop => |*opt| try opt.init(allocator, lyr),
        }
    }

    /// Deinitialize the optimizer for a layer
    pub fn deinit(self: *Optimizer) void {
        switch (self.*) {
            .sgd => |*opt| opt.deinit(),
            .adam => |*opt| opt.deinit(),
            .rmsprop => |*opt| opt.deinit(),
        }
    }

    /// Update weights for a specific layer
    pub fn step(self: *Optimizer, lyr: *layer_module.Layer, learning_rate: f32) !void {
        switch (self.*) {
            .sgd => |*opt| try opt.step(lyr, learning_rate),
            .adam => |*opt| try opt.step(lyr, learning_rate),
            .rmsprop => |*opt| try opt.step(lyr, learning_rate),
        }
    }
};

/// Stochastic Gradient Descent
pub const Sgd = struct {
    /// Momentum term for SGD with momentum
    momentum: f32 = 0.0,
    /// Velocity for momentum
    velocity_weights: ?tensor.Tensor = null,
    velocity_bias: ?tensor.Tensor = null,

    pub fn init(self: *Sgd, allocator: std.mem.Allocator, lyr: *const layer_module.Layer) !void {
        if (self.momentum > 0) {
            const w = lyr.getWeights();
            const b = lyr.getBias();
            const backend = lyr.getBackendFromLayer();
            self.velocity_weights = try tensor.Tensor.init(allocator, w.shape, backend);
            self.velocity_bias = try tensor.Tensor.init(allocator, b.shape, backend);
        }
    }

    pub fn deinit(self: *Sgd) void {
        if (self.velocity_weights) |*v| v.deinit();
        if (self.velocity_bias) |*v| v.deinit();
        self.velocity_weights = null;
        self.velocity_bias = null;
    }

    pub fn step(self: *Sgd, lyr: *layer_module.Layer, learning_rate: f32) !void {
        const w = lyr.getWeights();
        const b = lyr.getBias();
        const gw = lyr.getGradWeights();
        const gb = lyr.getGradBias();
        const backend = lyr.getBackendFromLayer();

        if (self.momentum > 0) {
            // SGD with momentum: v = momentum * v - lr * grad; w = w + v
            const vw = self.velocity_weights.?;
            const vb = self.velocity_bias.?;
            const allocator = self.velocity_weights.?.allocator;

            // Allocate temp buffers for intermediate results
            const temp_vw = try allocator.alloc(f32, vw.slice.len);
            defer allocator.free(temp_vw);
            const temp_vb = try allocator.alloc(f32, vb.slice.len);
            defer allocator.free(temp_vb);

            // Step 1: Scale velocity by momentum: v = momentum * v
            @memcpy(temp_vw, vw.slice);
            try backend.scale(temp_vw, vw.getMtlBuffer(), self.momentum);

            @memcpy(temp_vb, vb.slice);
            try backend.scale(temp_vb, vb.getMtlBuffer(), self.momentum);

            // Step 2: Scale gradients: grad_scaled = lr * grad
            const grad_scaled_w = try allocator.alloc(f32, gw.slice.len);
            defer allocator.free(grad_scaled_w);
            @memcpy(grad_scaled_w, gw.slice);
            try backend.scale(grad_scaled_w, gw.getMtlBuffer(), learning_rate);

            const grad_scaled_b = try allocator.alloc(f32, gb.slice.len);
            defer allocator.free(grad_scaled_b);
            @memcpy(grad_scaled_b, gb.slice);
            try backend.scale(grad_scaled_b, gb.getMtlBuffer(), learning_rate);

            // Step 3: Update velocity: v = v - lr*grad (using element-wise subtraction)
            // Store result in velocity tensors
            try backend.elementWise(.sub, temp_vw, vw.getMtlBuffer(), grad_scaled_w, gw.getMtlBuffer(), vw.slice, vw.getMtlBuffer());
            try backend.elementWise(.sub, temp_vb, vb.getMtlBuffer(), grad_scaled_b, gb.getMtlBuffer(), vb.slice, vb.getMtlBuffer());

            // Step 4: Update weights: w = w + v
            const temp_w = try allocator.alloc(f32, w.slice.len);
            defer allocator.free(temp_w);
            @memcpy(temp_w, w.slice);

            const temp_b = try allocator.alloc(f32, b.slice.len);
            defer allocator.free(temp_b);
            @memcpy(temp_b, b.slice);

            try backend.elementWise(.add, temp_w, w.getMtlBuffer(), vw.slice, vw.getMtlBuffer(), w.slice, w.getMtlBuffer());
            try backend.elementWise(.add, temp_b, b.getMtlBuffer(), vb.slice, vb.getMtlBuffer(), b.slice, b.getMtlBuffer());
        } else {
            // Standard SGD without momentum
            try backend.sgdUpdate(w.slice, w.getMtlBuffer(), gw.slice, gw.getMtlBuffer(), learning_rate, 0.0);
            try backend.sgdUpdateBias(b.slice, b.getMtlBuffer(), gb.slice, gb.getMtlBuffer(), learning_rate);
        }
    }
};

/// Adam optimizer (Adaptive Moment Estimation)
pub const Adam = struct {
    /// First moment estimate (mean)
    m_weights: ?tensor.Tensor = null,
    m_bias: ?tensor.Tensor = null,
    /// Second moment estimate (uncentered variance)
    v_weights: ?tensor.Tensor = null,
    v_bias: ?tensor.Tensor = null,
    /// Decay rates
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    /// Small value for numerical stability
    eps: f32 = 1e-8,
    /// Time step
    t: usize = 0,

    pub fn init(self: *Adam, allocator: std.mem.Allocator, lyr: *const layer_module.Layer) !void {
        const w = lyr.getWeights();
        const b = lyr.getBias();
        const backend = lyr.getBackendFromLayer();

        self.m_weights = try tensor.Tensor.init(allocator, w.shape, backend);
        self.m_bias = try tensor.Tensor.init(allocator, b.shape, backend);
        self.v_weights = try tensor.Tensor.init(allocator, w.shape, backend);
        self.v_bias = try tensor.Tensor.init(allocator, b.shape, backend);
    }

    pub fn deinit(self: *Adam) void {
        if (self.m_weights) |*v| v.deinit();
        if (self.m_bias) |*v| v.deinit();
        if (self.v_weights) |*v| v.deinit();
        if (self.v_bias) |*v| v.deinit();
        self.m_weights = null;
        self.m_bias = null;
        self.v_weights = null;
        self.v_bias = null;
    }

    /// Main step function
    pub fn step(self: *Adam, lyr: *layer_module.Layer, learning_rate: f32) !void {
        self.t += 1;
        const t_f = @as(f32, @floatFromInt(self.t));
        const bias_corr1 = 1.0 - std.math.pow(f32, self.beta1, t_f);
        const bias_corr2 = 1.0 - std.math.pow(f32, self.beta2, t_f);

        const w = lyr.getWeights();
        const b = lyr.getBias();
        const gw = lyr.getGradWeights();
        const gb = lyr.getGradBias();
        const backend = lyr.getBackendFromLayer();

        try backend.adamUpdate(w.slice, w.getMtlBuffer(), gw.slice, gw.getMtlBuffer(), self.m_weights.?.slice, self.m_weights.?.getMtlBuffer(), self.v_weights.?.slice, self.v_weights.?.getMtlBuffer(), learning_rate, self.beta1, self.beta2, self.eps, bias_corr1, bias_corr2);
        try backend.adamUpdate(b.slice, b.getMtlBuffer(), gb.slice, gb.getMtlBuffer(), self.m_bias.?.slice, self.m_bias.?.getMtlBuffer(), self.v_bias.?.slice, self.v_bias.?.getMtlBuffer(), learning_rate, self.beta1, self.beta2, self.eps, bias_corr1, bias_corr2);
    }
};

/// RMSprop optimizer (Root Mean Square Propagation)
pub const Rmsprop = struct {
    /// Square gradient moving average
    g_weights: ?tensor.Tensor = null,
    g_bias: ?tensor.Tensor = null,
    /// Decay rate
    rho: f32 = 0.9,
    /// Small value for numerical stability
    eps: f32 = 1e-8,

    pub fn init(self: *Rmsprop, allocator: std.mem.Allocator, lyr: *const layer_module.Layer) !void {
        const w = lyr.getWeights();
        const b = lyr.getBias();
        const backend = lyr.getBackendFromLayer();

        self.g_weights = try tensor.Tensor.init(allocator, w.shape, backend);
        self.g_bias = try tensor.Tensor.init(allocator, b.shape, backend);
    }

    pub fn deinit(self: *Rmsprop) void {
        if (self.g_weights) |*v| v.deinit();
        if (self.g_bias) |*v| v.deinit();
        self.g_weights = null;
        self.g_bias = null;
    }

    /// Main step function
    pub fn step(self: *Rmsprop, lyr: *layer_module.Layer, learning_rate: f32) !void {
        const w = lyr.getWeights();
        const b = lyr.getBias();
        const gw = lyr.getGradWeights();
        const gb = lyr.getGradBias();
        const backend = lyr.getBackendFromLayer();

        try backend.rmspropUpdate(w.slice, w.getMtlBuffer(), gw.slice, gw.getMtlBuffer(), self.g_weights.?.slice, self.g_weights.?.getMtlBuffer(), learning_rate, self.rho, self.eps);
        try backend.rmspropUpdate(b.slice, b.getMtlBuffer(), gb.slice, gb.getMtlBuffer(), self.g_bias.?.slice, self.g_bias.?.getMtlBuffer(), learning_rate, self.rho, self.eps);
    }
};

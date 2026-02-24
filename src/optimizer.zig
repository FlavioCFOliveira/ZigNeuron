/// Optimizers for neural network training
/// Provides different optimization algorithms for gradient descent
const std = @import("std");
const layer_module = @import("layer.zig");
const tensor = @import("tensor.zig");

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
            // Simplified: use SGD update without momentum for now, or implement momentum kernel
            try backend.sgdUpdate(w.slice, w.getMtlBuffer(), gw.slice, gw.getMtlBuffer(), learning_rate, 0.0);
            try backend.sgdUpdateBias(b.slice, b.getMtlBuffer(), gb.slice, gb.getMtlBuffer(), learning_rate);
        } else {
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

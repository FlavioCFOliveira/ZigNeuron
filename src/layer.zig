/// Neural network layers
const std = @import("std");
const activation = @import("activation.zig");
const backend_module = @import("backend.zig");
const tensor = @import("tensor.zig");
const metal = @import("metal.zig");
const recurrent = @import("recurrent.zig");

pub const Layer = union(enum) {
    dense: *Dense,
    rnn: *recurrent.VanillaRNN,
    lstm: *recurrent.LSTM,
    gru: *recurrent.GRU,
    sampling: *SamplingLayer,
    conv1d: *Conv1D,
    layer_norm: *LayerNorm,
    dropout: *Dropout,
    attention: *Attention,
    bidirectional: *recurrent.Bidirectional,
    twopath: *recurrent.TwoPath,

    pub fn deinit(self: Layer) void {
        switch (self) {
            .dense => |d| d.deinit(),
            .rnn => |r| r.deinit(),
            .lstm => |l| l.deinit(),
            .gru => |g| g.deinit(),
            .sampling => |s| s.deinit(),
            .conv1d => |c| c.deinit(),
            .layer_norm => |ln| ln.deinit(),
            .dropout => |dr| dr.deinit(),
            .attention => |a| a.deinit(),
            .bidirectional => |b| b.deinit(),
            .twopath => |t| t.deinit(),
        }
    }

    pub fn forward(self: Layer, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        switch (self) {
            .dense => |d| try d.forward(input, input_buf, output, output_buf),
            .rnn => |r| try r.forward(input, input_buf, output, output_buf),
            .lstm => |l| try l.forward(input, input_buf, output, output_buf),
            .gru => |g| try g.forward(input, input_buf, output, output_buf),
            .sampling => |s| try s.forward(input, input_buf, output, output_buf),
            .conv1d => |c| try c.forward(input, input_buf, output, output_buf),
            .layer_norm => |ln| try ln.forward(input, input_buf, output, output_buf),
            .dropout => |dr| try dr.forward(input, input_buf, output, output_buf),
            .attention => |a| try a.forward(input, input_buf, output, output_buf),
            .bidirectional => |b| try b.forward(input, input_buf, output, output_buf),
            .twopath => |t| try t.forward(input, input_buf, output, output_buf),
        }
    }

    pub fn backward(self: Layer,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer,
        activated_output: []const f32, activated_output_buf: ?*const metal.MTLBuffer
    ) !void {
        switch (self) {
            .dense => |d| try d.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, activated_output, activated_output_buf),
            .rnn => |r| try r.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, activated_output, activated_output_buf),
            .lstm => |l| try l.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, activated_output, activated_output_buf),
            .gru => |g| try g.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, activated_output, activated_output_buf),
            .sampling => |s| try s.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, activated_output, activated_output_buf),
            .conv1d => |c| try c.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, activated_output, activated_output_buf),
            .layer_norm => |ln| try ln.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, activated_output, activated_output_buf),
            .dropout => |dr| try dr.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, activated_output, activated_output_buf),
            .attention => |a| try a.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, activated_output, activated_output_buf),
            .bidirectional => |b| try b.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, activated_output, activated_output_buf),
            .twopath => |t| try t.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, activated_output, activated_output_buf),
        }
    }

    pub fn inputSize(self: Layer) usize {
        switch (self) {
            .dense => |d| return d.input_size,
            .rnn => |r| return r.input_size,
            .lstm => |l| return l.input_size,
            .gru => |g| return g.input_size,
            .sampling => |s| return s.input_size,
            .conv1d => |c| return c.input_size,
            .layer_norm => |ln| return ln.size,
            .dropout => |dr| return dr.size,
            .attention => |a| return a.size,
            .bidirectional => |b| return b.input_size,
            .twopath => |t| return t.input_size,
        }
    }

    pub fn outputSize(self: Layer) usize {
        switch (self) {
            .dense => |d| return d.output_size,
            .rnn => |r| return r.hidden_size,
            .lstm => |l| return l.hidden_size,
            .gru => |g| return g.hidden_size,
            .sampling => |s| return s.input_size / 2,
            .conv1d => |c| return c.output_size,
            .layer_norm => |ln| return ln.size,
            .dropout => |dr| return dr.size,
            .attention => |a| return a.size,
            .bidirectional => |b| return 2 * b.hidden_size,
            .twopath => |t| return t.hidden_size1 + t.hidden_size2,
        }
    }

    pub fn getWeights(self: Layer) *tensor.Tensor {
        switch (self) {
            .dense => |d| return &d.weights,
            .rnn => |r| return &r.weights_ih,
            .lstm => |l| return &l.weights_ih,
            .gru => |g| return &g.weights_ih,
            .sampling => |s| return &s.epsilon,
            .conv1d => |c| return &c.weights,
            .layer_norm => |ln| return &ln.gamma,
            .dropout => |dr| return &dr.mask,
            .attention => |a| return &a.query_weights,
            .bidirectional => |b| return b.fw_layer.getWeights(),
            .twopath => |t| return t.path1.getWeights(),
        }
    }

    pub fn getBias(self: Layer) *tensor.Tensor {
        switch (self) {
            .dense => |d| return &d.bias,
            .rnn => |r| return &r.bias,
            .lstm => |l| return &l.bias,
            .gru => |g| return &g.bias,
            .sampling => |s| return &s.epsilon,
            .conv1d => |c| return &c.bias,
            .layer_norm => |ln| return &ln.beta,
            .dropout => |dr| return &dr.mask,
            .attention => |a| return &a.query_weights,
            .bidirectional => |b| return b.fw_layer.getBias(),
            .twopath => |t| return t.path1.getBias(),
        }
    }

    pub fn getGradWeights(self: Layer) *tensor.Tensor {
        switch (self) {
            .dense => |d| return &d.grad_weights,
            .rnn => |r| return &r.grad_weights_ih,
            .lstm => |l| return &l.grad_weights_ih,
            .gru => |g| return &g.grad_weights_ih,
            .sampling => |s| return &s.epsilon,
            .conv1d => |c| return &c.grad_weights,
            .layer_norm => |ln| return &ln.grad_gamma,
            .dropout => |dr| return &dr.mask,
            .attention => |a| return &a.query_weights,
            .bidirectional => |b| return b.fw_layer.getGradWeights(),
            .twopath => |t| return t.path1.getGradWeights(),
        }
    }

    pub fn getGradBias(self: Layer) *tensor.Tensor {
        switch (self) {
            .dense => |d| return &d.grad_bias,
            .rnn => |r| return &r.grad_bias,
            .lstm => |l| return &l.grad_bias,
            .gru => |g| return &g.grad_bias,
            .sampling => |s| return &s.epsilon,
            .conv1d => |c| return &c.grad_bias,
            .layer_norm => |ln| return &ln.grad_beta,
            .dropout => |dr| return &dr.mask,
            .attention => |a| return &a.query_weights,
            .bidirectional => |b| return b.fw_layer.getGradBias(),
            .twopath => |t| return t.path1.getGradBias(),
        }
    }

    pub fn getActivation(self: Layer) activation.Activation {
        switch (self) {
            .dense => |d| return d.act,
            .rnn => |r| return r.act,
            .lstm => return .tanh,
            .gru => return .tanh,
            .sampling => return .linear,
            .conv1d => |c| return c.act,
            .layer_norm => return .linear,
            .dropout => return .linear,
            .attention => return .linear,
            .bidirectional => return .tanh,
            .twopath => return .tanh,
        }
    }

    pub fn updateWeights(self: Layer, learning_rate: f32, weight_decay: f32) !void {
        switch (self) {
            .dense => |d| {
                try d.backend.sgdUpdate(
                    d.weights.slice,
                    d.weights.getMtlBuffer(),
                    d.grad_weights.slice,
                    d.grad_weights.getMtlBuffer(),
                    learning_rate,
                    weight_decay,
                );
                try d.backend.sgdUpdateBias(
                    d.bias.slice,
                    d.bias.getMtlBuffer(),
                    d.grad_bias.slice,
                    d.grad_bias.getMtlBuffer(),
                    learning_rate,
                );
            },
            .rnn => |r| {
                try r.backend.sgdUpdate(
                    r.weights_ih.slice,
                    r.weights_ih.getMtlBuffer(),
                    r.grad_weights_ih.slice,
                    r.grad_weights_ih.getMtlBuffer(),
                    learning_rate,
                    weight_decay,
                );
                try r.backend.sgdUpdate(
                    r.weights_hh.slice,
                    r.weights_hh.getMtlBuffer(),
                    r.grad_weights_hh.slice,
                    r.grad_weights_hh.getMtlBuffer(),
                    learning_rate,
                    weight_decay,
                );
                try r.backend.sgdUpdateBias(
                    r.bias.slice,
                    r.bias.getMtlBuffer(),
                    r.grad_bias.slice,
                    r.grad_bias.getMtlBuffer(),
                    learning_rate,
                );
            },
            .lstm => |l| {
                try l.backend.sgdUpdate(
                    l.weights_ih.slice,
                    l.weights_ih.getMtlBuffer(),
                    l.grad_weights_ih.slice,
                    l.grad_weights_ih.getMtlBuffer(),
                    learning_rate,
                    weight_decay,
                );
                try l.backend.sgdUpdate(
                    l.weights_hh.slice,
                    l.weights_hh.getMtlBuffer(),
                    l.grad_weights_hh.slice,
                    l.grad_weights_hh.getMtlBuffer(),
                    learning_rate,
                    weight_decay,
                );
                try l.backend.sgdUpdateBias(
                    l.bias.slice,
                    l.bias.getMtlBuffer(),
                    l.grad_bias.slice,
                    l.grad_bias.getMtlBuffer(),
                    learning_rate,
                );
            },
            .gru => |g| {
                try g.backend.sgdUpdate(
                    g.weights_ih.slice,
                    g.weights_ih.getMtlBuffer(),
                    g.grad_weights_ih.slice,
                    g.grad_weights_ih.getMtlBuffer(),
                    learning_rate,
                    weight_decay,
                );
                try g.backend.sgdUpdate(
                    g.weights_hh.slice,
                    g.weights_hh.getMtlBuffer(),
                    g.grad_weights_hh.slice,
                    g.grad_weights_hh.getMtlBuffer(),
                    learning_rate,
                    weight_decay,
                );
                try g.backend.sgdUpdateBias(
                    g.bias.slice,
                    g.bias.getMtlBuffer(),
                    g.grad_bias.slice,
                    g.grad_bias.getMtlBuffer(),
                    learning_rate,
                );
            },
            .sampling => {},
            .conv1d => |c| {
                try c.backend.sgdUpdate(c.weights.slice, c.weights.getMtlBuffer(), c.grad_weights.slice, c.grad_weights.getMtlBuffer(), learning_rate, weight_decay);
                try c.backend.sgdUpdateBias(c.bias.slice, c.bias.getMtlBuffer(), c.grad_bias.slice, c.grad_bias.getMtlBuffer(), learning_rate);
            },
            .layer_norm => |ln| {
                try ln.backend.sgdUpdate(ln.gamma.slice, ln.gamma.getMtlBuffer(), ln.grad_gamma.slice, ln.grad_gamma.getMtlBuffer(), learning_rate, weight_decay);
                try ln.backend.sgdUpdateBias(ln.beta.slice, ln.beta.getMtlBuffer(), ln.grad_beta.slice, ln.grad_beta.getMtlBuffer(), learning_rate);
            },
            .dropout => {},
            .attention => {},
            .bidirectional => |b| {
                try b.fw_layer.updateWeights(learning_rate, weight_decay);
                try b.bw_layer.updateWeights(learning_rate, weight_decay);
            },
            .twopath => |t| {
                try t.path1.updateWeights(learning_rate, weight_decay);
                try t.path2.updateWeights(learning_rate, weight_decay);
            },
        }
    }
};

pub const Dense = struct {
    weights: tensor.Tensor,
    bias: tensor.Tensor,
    grad_weights: tensor.Tensor, // Gradient buffer for weights
    grad_bias: tensor.Tensor, // Gradient buffer for bias
    grad_after_act: tensor.Tensor, // Reusable buffer for backward pass
    input_size: usize,
    output_size: usize,
    act: activation.Activation,
    backend: backend_module.Backend,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize, act: activation.Activation, backend: backend_module.Backend) !*Dense {
        const self = allocator.create(Dense) catch return error.OutOfMemory;
        errdefer allocator.destroy(self);

        // Initialize tensors with unified memory support
        self.weights = try tensor.Tensor.init(allocator, &.{ input_size, output_size }, backend);
        errdefer self.weights.deinit();

        self.bias = try tensor.Tensor.init(allocator, &.{output_size}, backend);
        errdefer self.bias.deinit();

        self.grad_weights = try tensor.Tensor.init(allocator, &.{ input_size, output_size }, backend);
        errdefer self.grad_weights.deinit();

        self.grad_bias = try tensor.Tensor.init(allocator, &.{output_size}, backend);
        errdefer self.grad_bias.deinit();

        self.grad_after_act = try tensor.Tensor.init(allocator, &.{output_size}, backend);
        errdefer self.grad_after_act.deinit();

        // Xavier/He initialization based on activation function
        var prng = std.Random.DefaultPrng.init(@intCast(@as(u64, @bitCast(std.time.timestamp())) +% input_size +% output_size));
        const random = prng.random();

        const scale = switch (act) {
            .relu => @sqrt(2.0 / @as(f32, @floatFromInt(input_size))),
            .sigmoid, .tanh, .softmax => @sqrt(2.0 / @as(f32, @floatFromInt(input_size + output_size))),
            .linear => @sqrt(1.0 / @as(f32, @floatFromInt(input_size))),
        };

        for (self.weights.slice) |*w| {
            w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        }
        @memset(self.bias.slice, 0);

        self.input_size = input_size;
        self.output_size = output_size;
        self.act = act;
        self.backend = backend;
        self.allocator = allocator;

        return self;
    }

    pub fn deinit(self: *Dense) void {
        self.weights.deinit();
        self.bias.deinit();
        self.grad_weights.deinit();
        self.grad_bias.deinit();
        self.grad_after_act.deinit();
        self.allocator.destroy(self);
    }

    /// Compute pre-activation values (linear part only, no activation)
    /// Used for caching during forward pass
    pub fn computePreActivation(self: *Dense, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        if (input.len != self.input_size) return error.InvalidInputSize;
        if (output.len != self.output_size) return error.InvalidOutputSize;

        const batch_size: usize = 1;
        try self.backend.matMul(
            input, input_buf,
            self.weights.slice, self.weights.getMtlBuffer(),
            output, output_buf,
            batch_size,
            self.output_size,
            self.input_size,
        );

        // Add bias
        try self.backend.addBias(output, output_buf, self.bias.slice, self.bias.getMtlBuffer());
    }

    pub fn forward(self: *Dense, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        const batch_size = input.len / self.input_size;
        if (output.len != batch_size * self.output_size) return error.InvalidOutputSize;

        try self.backend.matMul(
            input, input_buf,
            self.weights.slice, self.weights.getMtlBuffer(),
            output, output_buf,
            batch_size,
            self.output_size,
            self.input_size,
        );

        // Add bias (broadcasted over batch)
        try self.backend.addBias(output, output_buf, self.bias.slice, self.bias.getMtlBuffer());

        // Apply activation
        if (self.act == .softmax) {
            for (0..batch_size) |s| {
                const start = s * self.output_size;
                const end = (s + 1) * self.output_size;
                const out_sample = output[start..end];
                const out_buf_sample = if (output_buf) |b| b else null; // Metal needs offset support which we might need to add to activationForward
                try self.backend.activationForward(self.act, out_sample, out_buf_sample, out_sample, out_buf_sample);
            }
        } else {
            try self.backend.activationForward(self.act, output, output_buf, output, output_buf);
        }
    }

    pub fn backward(self: *Dense,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer,
        activated_output: []const f32, activated_output_buf: ?*const metal.MTLBuffer
    ) !void {
        const batch_size = input.len / self.input_size;

        // Ensure grad_after_act is large enough for the batch
        if (self.grad_after_act.slice.len < batch_size * self.output_size) {
            self.grad_after_act.deinit();
            self.grad_after_act = try tensor.Tensor.init(self.allocator, &.{ batch_size * self.output_size }, self.backend);
        }
        const gaa = self.grad_after_act.slice[0 .. batch_size * self.output_size];
        const gaa_buf = self.grad_after_act.getMtlBuffer();

        // Apply activation derivative to grad_output
        if (self.act == .softmax) {
            for (0..batch_size) |s| {
                const start = s * self.output_size;
                const end = (s + 1) * self.output_size;
                const act_sample = activated_output[start..end];
                const go_sample = grad_output[start..end];
                const gaa_sample = gaa[start..end];
                const gaa_buf_sample = gaa_buf; // getBuffer handles offset
                try self.backend.activationBackward(self.act, act_sample, activated_output_buf, go_sample, grad_output_buf, gaa_sample, gaa_buf_sample);
            }
        } else {
            try self.backend.activationBackward(self.act,
                activated_output, activated_output_buf,
                grad_output, grad_output_buf,
                gaa, gaa_buf
            );
        }

        // Compute grad_input = grad_after_act * weights^T
        try self.backend.matMulTransposeB(
            gaa, gaa_buf,
            self.weights.slice, self.weights.getMtlBuffer(),
            grad_input, grad_input_buf,
            batch_size,
            self.input_size,
            self.output_size,
        );

        // Accumulate bias gradient (sum over batch)
        try self.backend.accumulateBias(
            self.grad_bias.slice, self.grad_bias.getMtlBuffer(),
            gaa, gaa_buf
        );

        // Accumulate weight gradients
        // dW = input^T * grad_after_act
        try self.backend.matMulTransposeA(
            input, input_buf,
            gaa, gaa_buf,
            self.grad_weights.slice, self.grad_weights.getMtlBuffer(),
            self.input_size,
            self.output_size,
            batch_size
        );
    }
};

pub const SamplingLayer = struct {
    input_size: usize,
    epsilon: tensor.Tensor,
    exp_log_var: tensor.Tensor,
    backend: backend_module.Backend,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, backend: backend_module.Backend) !*SamplingLayer {
        const self = try allocator.create(SamplingLayer);
        const latent_dim = input_size / 2;
        self.input_size = input_size;
        self.epsilon = try tensor.Tensor.init(allocator, &.{latent_dim}, backend);
        self.exp_log_var = try tensor.Tensor.init(allocator, &.{latent_dim}, backend);
        self.backend = backend;
        self.allocator = allocator;
        return self;
    }

    pub fn deinit(self: *SamplingLayer) void {
        self.epsilon.deinit();
        self.exp_log_var.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *SamplingLayer, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        _ = input_buf; _ = output_buf;
        const latent_dim = self.input_size / 2;
        const mu = input[0..latent_dim];
        const log_var = input[latent_dim..];

        // 1. Generate epsilon ~ N(0, 1)
        try self.backend.fillRandomNormal(self.epsilon.slice, self.epsilon.getMtlBuffer(), 0.0, 1.0, @intCast(std.time.timestamp()));

        // 2. exp_log_var = exp(0.5 * log_var)
        // We use map to compute 0.5 * log_var and then exp, or just use scale + exp
        for (0..latent_dim) |i| {
            self.exp_log_var.slice[i] = std.math.exp(0.5 * log_var[i]);
        }
        // Sync to GPU if needed
        if (self.backend.type == .gpu) {
             // In a real implementation we'd have a kernel for this.
             // For now, let's use the CPU fallback for this specific step or implement a custom kernel if needed.
             // Actually, let's just use what we have.
        }

        // 3. z = mu + epsilon * exp_log_var
        for (0..latent_dim) |i| {
            output[i] = mu[i] + self.epsilon.slice[i] * self.exp_log_var.slice[i];
        }
    }

    pub fn backward(self: *SamplingLayer,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer,
        activated_output: []const f32, activated_output_buf: ?*const metal.MTLBuffer
    ) !void {
        _ = activated_output; _ = activated_output_buf; _ = input_buf; _ = grad_output_buf; _ = grad_input_buf;
        const latent_dim = self.input_size / 2;
        const log_var = input[latent_dim..];
        const grad_mu = grad_input[0..latent_dim];
        const grad_log_var = grad_input[latent_dim..];

        for (0..latent_dim) |i| {
            // dz/dmu = 1 => grad_mu = grad_output
            grad_mu[i] = grad_output[i];
            // dz/dlog_var = epsilon * exp(0.5 * log_var) * 0.5
            grad_log_var[i] = grad_output[i] * self.epsilon.slice[i] * std.math.exp(0.5 * log_var[i]) * 0.5;
        }
    }
};

pub const Conv1D = struct {
    weights: tensor.Tensor, // [out_channels, in_channels, kernel_size]
    bias: tensor.Tensor,    // [out_channels]
    grad_weights: tensor.Tensor,
    grad_bias: tensor.Tensor,
    grad_after_act: tensor.Tensor, // Reusable buffer
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    input_size: usize,
    output_size: usize,
    act: activation.Activation,
    backend: backend_module.Backend,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, in_channels: usize, out_channels: usize, kernel_size: usize, input_len: usize, act: activation.Activation, backend: backend_module.Backend) !*Conv1D {
        const self = try allocator.create(Conv1D);
        self.in_channels = in_channels;
        self.out_channels = out_channels;
        self.kernel_size = kernel_size;
        self.stride = 1;
        self.dilation = 1;
        self.input_size = in_channels * input_len;
        const out_len = (input_len - kernel_size) / self.stride + 1;
        self.output_size = out_channels * out_len;

        self.weights = try tensor.Tensor.init(allocator, &.{ out_channels, in_channels, kernel_size }, backend);
        self.bias = try tensor.Tensor.init(allocator, &.{out_channels}, backend);
        self.grad_weights = try tensor.Tensor.init(allocator, &.{ out_channels, in_channels, kernel_size }, backend);
        self.grad_bias = try tensor.Tensor.init(allocator, &.{out_channels}, backend);
        errdefer self.grad_bias.deinit();

        self.grad_after_act = try tensor.Tensor.init(allocator, &.{self.output_size}, backend);
        errdefer self.grad_after_act.deinit();

        // Kaiming initialization
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(in_channels * kernel_size)));
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        for (self.weights.slice) |*w| w.* = (prng.random().float(f32) * 2.0 - 1.0) * scale;
        @memset(self.bias.slice, 0);

        self.act = act;
        self.backend = backend;
        self.allocator = allocator;
        return self;
    }

    pub fn deinit(self: *Conv1D) void {
        self.weights.deinit();
        self.bias.deinit();
        self.grad_weights.deinit();
        self.grad_bias.deinit();
        self.grad_after_act.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Conv1D, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        _ = input_buf;
        const in_len = input.len / self.in_channels;
        const out_len = (in_len - self.kernel_size) / self.stride + 1;

        @memset(output, 0);
        for (0..self.out_channels) |oc| {
            for (0..out_len) |t| {
                var sum: f32 = 0;
                for (0..self.in_channels) |ic| {
                    for (0..self.kernel_size) |k| {
                        const in_idx = ic * in_len + (t * self.stride + k * self.dilation);
                        const w_idx = (oc * self.in_channels + ic) * self.kernel_size + k;
                        sum += input[in_idx] * self.weights.slice[w_idx];
                    }
                }
                output[oc * out_len + t] = sum + self.bias.slice[oc];
            }
        }
        try self.backend.activationForward(self.act, output, output_buf, output, output_buf);
    }

    pub fn backward(self: *Conv1D, input: []const f32, input_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, activated_output: []const f32, activated_output_buf: ?*const metal.MTLBuffer) !void {
        _ = input_buf; _ = grad_input_buf;
        const in_len = input.len / self.in_channels;
        const out_len = grad_output.len / self.out_channels;

        // 1. Activation backward using pre-allocated buffer
        try self.backend.activationBackward(self.act, activated_output, activated_output_buf, grad_output, grad_output_buf, self.grad_after_act.slice, self.grad_after_act.getMtlBuffer());
        const gaa = self.grad_after_act.slice;

        // 2. Compute grad_input and accumulate grad_weights/bias
        @memset(grad_input, 0);
        for (0..self.out_channels) |oc| {
            for (0..out_len) |t| {
                const go = gaa[oc * out_len + t];
                self.grad_bias.slice[oc] += go;
                for (0..self.in_channels) |ic| {
                    for (0..self.kernel_size) |k| {
                        const in_idx = ic * in_len + (t * self.stride + k * self.dilation);
                        const w_idx = (oc * self.in_channels + ic) * self.kernel_size + k;
                        self.grad_weights.slice[w_idx] += input[in_idx] * go;
                        grad_input[in_idx] += self.weights.slice[w_idx] * go;
                    }
                }
            }
        }
    }
};

pub const LayerNorm = struct {
    gamma: tensor.Tensor,
    beta: tensor.Tensor,
    grad_gamma: tensor.Tensor,
    grad_beta: tensor.Tensor,
    size: usize,
    eps: f32,
    backend: backend_module.Backend,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size: usize, backend: backend_module.Backend) !*LayerNorm {
        const self = try allocator.create(LayerNorm);
        self.size = size;
        self.eps = 1e-5;
        self.gamma = try tensor.Tensor.init(allocator, &.{size}, backend);
        self.beta = try tensor.Tensor.init(allocator, &.{size}, backend);
        self.grad_gamma = try tensor.Tensor.init(allocator, &.{size}, backend);
        self.grad_beta = try tensor.Tensor.init(allocator, &.{size}, backend);

        @memset(self.gamma.slice, 1.0);
        @memset(self.beta.slice, 0.0);

        self.backend = backend;
        self.allocator = allocator;
        return self;
    }

    pub fn deinit(self: *LayerNorm) void {
        self.gamma.deinit();
        self.beta.deinit();
        self.grad_gamma.deinit();
        self.grad_beta.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *LayerNorm, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        _ = input_buf; _ = output_buf;
        var mean: f32 = 0;
        for (input) |x| mean += x;
        mean /= @as(f32, @floatFromInt(input.len));

        var variance: f32 = 0;
        for (input) |x| variance += (x - mean) * (x - mean);
        variance /= @as(f32, @floatFromInt(input.len));

        const std_inv = 1.0 / @sqrt(variance + self.eps);
        for (input, 0..) |x, i| {
            output[i] = (x - mean) * std_inv * self.gamma.slice[i] + self.beta.slice[i];
        }
    }

    pub fn backward(self: *LayerNorm, input: []const f32, input_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, activated_output: []const f32, activated_output_buf: ?*const metal.MTLBuffer) !void {
        _ = input_buf; _ = grad_output_buf; _ = grad_input_buf; _ = activated_output; _ = activated_output_buf;
        // Simplified LN backward
        var mean: f32 = 0;
        for (input) |x| mean += x;
        mean /= @as(f32, @floatFromInt(input.len));
        var variance: f32 = 0;
        for (input) |x| variance += (x - mean) * (x - mean);
        variance /= @as(f32, @floatFromInt(input.len));
        const std_inv = 1.0 / @sqrt(variance + self.eps);

        for (input, grad_output, 0..) |x, go, i| {
            const x_hat = (x - mean) * std_inv;
            self.grad_gamma.slice[i] += go * x_hat;
            self.grad_beta.slice[i] += go;
            // Simple approximation for grad_input
            grad_input[i] = go * self.gamma.slice[i] * std_inv;
        }
    }
};

pub const Dropout = struct {
    size: usize,
    rate: f32,
    mask: tensor.Tensor,
    backend: backend_module.Backend,
    allocator: std.mem.Allocator,
    training: bool,

    pub fn init(allocator: std.mem.Allocator, size: usize, rate: f32, backend: backend_module.Backend) !*Dropout {
        const self = try allocator.create(Dropout);
        self.size = size;
        self.rate = rate;
        self.mask = try tensor.Tensor.init(allocator, &.{size}, backend);
        self.backend = backend;
        self.allocator = allocator;
        self.training = true;
        return self;
    }

    pub fn deinit(self: *Dropout) void {
        self.mask.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Dropout, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        _ = input_buf; _ = output_buf;
        if (!self.training) {
            @memcpy(output, input);
            return;
        }

        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const scale = 1.0 / (1.0 - self.rate);
        for (0..self.size) |i| {
            if (prng.random().float(f32) > self.rate) {
                self.mask.slice[i] = 1.0;
                output[i] = input[i] * scale;
            } else {
                self.mask.slice[i] = 0.0;
                output[i] = 0.0;
            }
        }
    }

    pub fn backward(self: *Dropout, input: []const f32, input_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, activated_output: []const f32, activated_output_buf: ?*const metal.MTLBuffer) !void {
        _ = input; _ = input_buf; _ = grad_output_buf; _ = grad_input_buf; _ = activated_output; _ = activated_output_buf;
        const scale = 1.0 / (1.0 - self.rate);
        for (0..self.size) |i| {
            grad_input[i] = grad_output[i] * self.mask.slice[i] * scale;
        }
    }
};

pub const Attention = struct {
    size: usize,
    query_weights: tensor.Tensor,
    key_weights: tensor.Tensor,
    value_weights: tensor.Tensor,
    backend: backend_module.Backend,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size: usize, backend: backend_module.Backend) !*Attention {
        const self = try allocator.create(Attention);
        self.size = size;
        self.query_weights = try tensor.Tensor.init(allocator, &.{ size, size }, backend);
        self.key_weights = try tensor.Tensor.init(allocator, &.{ size, size }, backend);
        self.value_weights = try tensor.Tensor.init(allocator, &.{ size, size }, backend);
        self.backend = backend;
        self.allocator = allocator;

        // Simple initialization
        @memset(self.query_weights.slice, 0.1);
        @memset(self.key_weights.slice, 0.1);
        @memset(self.value_weights.slice, 0.1);

        return self;
    }

    pub fn deinit(self: *Attention) void {
        self.query_weights.deinit();
        self.key_weights.deinit();
        self.value_weights.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Attention, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        _ = self; _ = input_buf; _ = output_buf;
        // Simplified Dot-Product Attention forward
        // In a real Transformer, this would be Q*K^T / sqrt(d) * V
        // For now, we'll just do a identity-like pass for the example
        @memcpy(output, input);
    }

    pub fn backward(self: *Attention, input: []const f32, input_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, activated_output: []const f32, activated_output_buf: ?*const metal.MTLBuffer) !void {
        _ = self; _ = input; _ = input_buf; _ = grad_output_buf; _ = grad_input_buf; _ = activated_output; _ = activated_output_buf;
        @memcpy(grad_input, grad_output);
    }
};

test "layer dense forward with backend" {
    const allocator = std.testing.allocator;
    var backend = try backend_module.Backend.init(allocator);
    backend.type = .cpu;
    defer backend.deinit();

    var lyr = try Dense.init(allocator, 2, 1, .relu, backend);
    defer lyr.deinit();

    lyr.weights.slice[0] = 1.0;
    lyr.weights.slice[1] = 1.0;
    lyr.bias.slice[0] = 0.0;

    var input: [2]f32 = .{ 1.0, 1.0 };
    var output: [1]f32 = undefined;
    try lyr.forward(&input, null, &output, null);

    // ReLU(1.0 * 1.0 + 1.0 * 1.0 + 0.0) = ReLU(2.0) = 2.0
    try std.testing.expect(output[0] > 1.9 and output[0] < 2.1);
}

test "layer dense backward with backend" {
    const allocator = std.testing.allocator;
    var backend = try backend_module.Backend.init(allocator);
    backend.type = .cpu;
    defer backend.deinit();

    var lyr = try Dense.init(allocator, 2, 1, .relu, backend);
    defer lyr.deinit();

    lyr.weights.slice[0] = 1.0;
    lyr.weights.slice[1] = 1.0;
    lyr.bias.slice[0] = 0.0;

    var input: [2]f32 = .{ 1.0, 1.0 };
    var grad_output: [1]f32 = .{1.0};
    var grad_input: [2]f32 = undefined;
    var pre_activation: [1]f32 = .{2.0};

    try lyr.backward(&input, null, &grad_output, null, &grad_input, null, &pre_activation, null);

    // With ReLU and positive pre-activation, gradient passes through
    // grad_input[0] = 1.0 * 1.0 = 1.0 (weight[0] * grad_output)
    // grad_input[1] = 1.0 * 1.0 = 1.0 (weight[1] * grad_output)
    try std.testing.expect(grad_input[0] > 0.9 and grad_input[0] < 1.1);
    try std.testing.expect(grad_input[1] > 0.9 and grad_input[1] < 1.1);
}

test "layer dense initialization" {
    const allocator = std.testing.allocator;
    var backend = try backend_module.Backend.init(allocator);
    backend.type = .cpu;
    defer backend.deinit();

    var lyr = try Dense.init(allocator, 4, 8, .sigmoid, backend);
    defer lyr.deinit();

    try std.testing.expect(lyr.weights.slice.len == 32); // 4 * 8
    try std.testing.expect(lyr.bias.slice.len == 8);
    try std.testing.expect(lyr.grad_weights.slice.len == 32);
    try std.testing.expect(lyr.grad_bias.slice.len == 8);
}

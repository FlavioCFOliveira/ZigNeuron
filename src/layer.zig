/// Neural network layers
const std = @import("std");
const activation = @import("activation.zig");
const backend_module = @import("backend.zig");
const tensor = @import("tensor.zig");
const metal = @import("metal.zig");

pub const Dense = struct {
    weights: tensor.Tensor,
    bias: tensor.Tensor,
    grad_weights: tensor.Tensor, // Gradient buffer for weights
    grad_bias: tensor.Tensor, // Gradient buffer for bias
    input_size: usize,
    output_size: usize,
    act: activation.Activation,
    backend: backend_module.Backend,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize, act: activation.Activation, backend: backend_module.Backend) !*Dense {
        const self = allocator.create(Dense) catch return error.OutOfMemory;
        errdefer allocator.destroy(self);

        // Initialize tensors with unified memory support
        const weight_count = input_size * output_size;
        self.weights = try tensor.Tensor.init(allocator, weight_count, backend);
        errdefer self.weights.deinit();

        self.bias = try tensor.Tensor.init(allocator, output_size, backend);
        errdefer self.bias.deinit();

        self.grad_weights = try tensor.Tensor.init(allocator, weight_count, backend);
        errdefer self.grad_weights.deinit();

        self.grad_bias = try tensor.Tensor.init(allocator, output_size, backend);
        errdefer self.grad_bias.deinit();

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
        for (0..self.output_size) |i| {
            output[i] += self.bias.slice[i];
        }
    }

    pub fn forward(self: *Dense, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
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
        for (0..self.output_size) |i| {
            output[i] += self.bias.slice[i];
        }

        // Apply activation
        try self.backend.activationForward(self.act, output, output_buf, output, output_buf);
    }

    pub fn backward(self: *Dense,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer,
        activated_output: []const f32, activated_output_buf: ?*const metal.MTLBuffer
    ) !void {
        _ = input;
        _ = input_buf;
        // Apply activation derivative to grad_output
        // We need a temporary buffer for grad_after_act.
        // In a real optimized network, this would be pre-allocated.
        var grad_after_act = try tensor.Tensor.init(self.allocator, self.output_size, self.backend);
        defer grad_after_act.deinit();

        try self.backend.activationBackward(self.act,
            activated_output, activated_output_buf,
            grad_output, grad_output_buf,
            grad_after_act.slice, grad_after_act.getMtlBuffer()
        );

        // Compute grad_input = grad_after_act * weights^T
        // grad_after_act: [batch_size x output_size]
        // weights: [input_size x output_size]
        // grad_input: [batch_size x input_size]
        const batch_size: usize = 1;
        try self.backend.matMulTransposeB(
            grad_after_act.slice, grad_after_act.getMtlBuffer(),
            self.weights.slice, self.weights.getMtlBuffer(),
            grad_input, grad_input_buf,
            batch_size,
            self.input_size,
            self.output_size,
        );
    }

    /// Accumulate gradients for weights and bias from a sample
    /// This is a helper function for backpropagation in networks
    pub fn accumulateGradients(self: *Dense, input: []const f32, grad_after_act: []const f32) void {
        // Accumulate bias gradient
        for (0..self.output_size) |i| {
            self.grad_bias.slice[i] += grad_after_act[i];
        }

        // Accumulate weight gradients
        // Weights are stored as [input_size, output_size] row-major
        // grad_weights[in_idx, out_idx] += input[in_idx] * grad_after_act[out_idx]
        for (0..self.input_size) |in_idx| {
            for (0..self.output_size) |out_idx| {
                const weight_idx = in_idx * self.output_size + out_idx;
                self.grad_weights.slice[weight_idx] += input[in_idx] * grad_after_act[out_idx];
            }
        }
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

/// Neural network layers
const std = @import("std");
const activation = @import("activation.zig");
const backend_module = @import("backend.zig");

pub const Dense = struct {
    weights: []f32,
    bias: []f32,
    grad_weights: []f32, // Gradient buffer for weights
    grad_bias: []f32, // Gradient buffer for bias
    input_size: usize,
    output_size: usize,
    act: activation.Activation,
    backend: backend_module.Backend,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize, act: activation.Activation, backend: backend_module.Backend) !*Dense {
        const self = allocator.create(Dense) catch return error.OutOfMemory;
        errdefer allocator.destroy(self);

        // Initialize weights with small random values (Xavier initialization)
        const weight_count = input_size * output_size;
        self.weights = try allocator.alloc(f32, weight_count);
        self.bias = try allocator.alloc(f32, output_size);
        self.grad_weights = try allocator.alloc(f32, weight_count);
        self.grad_bias = try allocator.alloc(f32, output_size);

        // Xavier/He initialization based on activation function
        // He initialization for ReLU: std = sqrt(2/fan_in)
        // Xavier initialization for others: std = sqrt(2/(fan_in + fan_out))
        var seed: u32 = 12345;

        const scale = switch (act) {
            .relu => @sqrt(2.0 / @as(f32, @floatFromInt(input_size))),
            .sigmoid, .tanh, .softmax => @sqrt(2.0 / @as(f32, @floatFromInt(input_size + output_size))),
            .linear => @sqrt(1.0 / @as(f32, @floatFromInt(input_size))),
        };

        for (self.weights, 0..) |*w, i| {
            _ = i;
            // Simple LCG random with overflow handled using wrap-around addition
            seed = seed +% 1664525 +% 1013904223;
            // Generate value in [-1, 1] range then scale
            const rand_val = (@as(f32, @floatFromInt(@as(u8, @intCast(seed & 0xFF)))) / 127.5) - 1.0;
            w.* = rand_val * scale;
        }
        for (self.bias) |*b| {
            b.* = 0;
        }

        self.input_size = input_size;
        self.output_size = output_size;
        self.act = act;
        self.backend = backend;
        self.allocator = allocator;

        return self;
    }

    pub fn deinit(self: *Dense) void {
        self.allocator.free(self.weights);
        self.allocator.free(self.bias);
        self.allocator.free(self.grad_weights);
        self.allocator.free(self.grad_bias);
        self.allocator.destroy(self);
    }

    /// Compute pre-activation values (linear part only, no activation)
    /// Used for caching during forward pass
    pub fn computePreActivation(self: *Dense, input: []const f32, output: []f32) !void {
        if (input.len != self.input_size) return error.InvalidInputSize;
        if (output.len != self.output_size) return error.InvalidOutputSize;

        const batch_size: usize = 1;
        try self.backend.matMul(
            input,
            self.weights,
            output,
            batch_size,
            self.output_size,
            self.input_size,
        );

        // Add bias
        for (0..self.output_size) |i| {
            output[i] += self.bias[i];
        }
    }

    pub fn forward(self: *Dense, input: []const f32, output: []f32) !void {
        if (input.len != self.input_size) return error.InvalidInputSize;
        if (output.len != self.output_size) return error.InvalidOutputSize;

        // Compute weighted sum + bias
        // This is a simple implementation; for large matrices, use backend.matMul

        // Create weight matrix view (output_size x input_size)
        // Input vector (input_size x 1)
        // Result (output_size x 1)

        const batch_size: usize = 1;
        try self.backend.matMul(
            input,
            self.weights,
            output,
            batch_size,
            self.output_size,
            self.input_size,
        );

        // Add bias
        for (0..self.output_size) |i| {
            output[i] += self.bias[i];
        }

        // Apply activation
        try self.backend.activationForward(self.act, output, output);
    }

    pub fn backward(self: *Dense, input: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        // Compute gradient w.r.t input
        // grad_input = grad_output * weights (with activation derivative)

        // First, compute pre-activation values for activation derivative
        const pre_activation = try self.allocator.alloc(f32, self.output_size);
        defer self.allocator.free(pre_activation);

        // Recompute pre-activation
        for (0..self.output_size) |i| {
            var sum: f32 = self.bias[i];
            for (0..self.input_size) |j| {
                sum += input[j] * self.weights[i * self.input_size + j];
            }
            pre_activation[i] = sum;
        }

        // Apply activation derivative to grad_output
        const grad_after_act = try self.allocator.alloc(f32, self.output_size);
        defer self.allocator.free(grad_after_act);

        try self.backend.activationBackward(self.act, pre_activation, grad_output, grad_after_act);

        // Compute grad_input = grad_after_act * weights^T
        const batch_size: usize = 1;
        try self.backend.matMul(
            grad_after_act,
            self.weights,
            grad_input,
            batch_size,
            self.input_size,
            self.output_size,
        );
    }

    /// Accumulate gradients for weights and bias from a batch
    /// This is a helper function for backpropagation in networks
    pub fn accumulateGradients(self: *Dense, input: []const f32, grad_after_act: []const f32) void {
        // Accumulate bias gradient
        for (0..self.output_size) |i| {
            self.grad_bias[i] += grad_after_act[i];
        }

        // Accumulate weight gradients
        for (0..self.output_size) |out_idx| {
            for (0..self.input_size) |in_idx| {
                const weight_idx = out_idx * self.input_size + in_idx;
                self.grad_weights[weight_idx] += grad_after_act[out_idx] * input[in_idx];
            }
        }
    }
};

test "layer dense forward with backend" {
    const allocator = std.testing.allocator;
    const backend = backend_module.Backend{ .cpu = {} };
    var lyr = try Dense.init(allocator, 2, 1, .relu, backend);
    defer lyr.deinit();

    lyr.weights[0] = 1.0;
    lyr.weights[1] = 1.0;
    lyr.bias[0] = 0.0;

    var input: [2]f32 = .{ 1.0, 1.0 };
    var output: [1]f32 = undefined;
    try lyr.forward(&input, &output);

    // ReLU(1.0 * 1.0 + 1.0 * 1.0 + 0.0) = ReLU(2.0) = 2.0
    try std.testing.expect(output[0] > 1.9 and output[0] < 2.1);
}

test "layer dense backward with backend" {
    const allocator = std.testing.allocator;
    const backend = backend_module.Backend{ .cpu = {} };
    var lyr = try Dense.init(allocator, 2, 1, .relu, backend);
    defer lyr.deinit();

    lyr.weights[0] = 1.0;
    lyr.weights[1] = 1.0;
    lyr.bias[0] = 0.0;

    var input: [2]f32 = .{ 1.0, 1.0 };
    var grad_output: [1]f32 = .{1.0};
    var grad_input: [2]f32 = undefined;

    try lyr.backward(&input, &grad_output, &grad_input);

    // With ReLU and positive pre-activation, gradient passes through
    // grad_input[0] = 1.0 * 1.0 = 1.0 (weight[0] * grad_output)
    // grad_input[1] = 1.0 * 1.0 = 1.0 (weight[1] * grad_output)
    try std.testing.expect(grad_input[0] > 0.9 and grad_input[0] < 1.1);
    try std.testing.expect(grad_input[1] > 0.9 and grad_input[1] < 1.1);
}

test "layer dense initialization" {
    const allocator = std.testing.allocator;
    const backend = backend_module.Backend{ .cpu = {} };
    var lyr = try Dense.init(allocator, 4, 8, .sigmoid, backend);
    defer lyr.deinit();

    try std.testing.expect(lyr.weights.len == 32); // 4 * 8
    try std.testing.expect(lyr.bias.len == 8);
    try std.testing.expect(lyr.grad_weights.len == 32);
    try std.testing.expect(lyr.grad_bias.len == 8);
}

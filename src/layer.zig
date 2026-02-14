/// Neural network layers
const std = @import("std");
const activation = @import("activation.zig");

pub const Dense = struct {
    weights: []f32,
    bias: []f32,
    grad_weights: []f32,  // Gradient buffer for weights
    grad_bias: []f32,     // Gradient buffer for bias
    input_size: usize,
    output_size: usize,
    act: activation.Activation,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize, act: activation.Activation) !*Dense {
        const self = allocator.create(Dense) catch return error.OutOfMemory;
        errdefer allocator.destroy(self);

        // Initialize weights with small random values (Xavier initialization)
        const weight_count = input_size * output_size;
        self.weights = try allocator.alloc(f32, weight_count);
        self.bias = try allocator.alloc(f32, output_size);
        self.grad_weights = try allocator.alloc(f32, weight_count);
        self.grad_bias = try allocator.alloc(f32, output_size);

        // Simple deterministic initialization - should be replaced with proper RNG
        var seed: u32 = 12345;
        for (self.weights, 0..) |*w, i| {
            _ = i;
            // Simple LCG random with overflow handled using wrap-around addition
            seed = seed +% 1664525 +% 1013904223;
            w.* = (@as(f32, @floatFromInt(@as(u8, @intCast(seed & 0xFF)))) / 255.0 * 0.2) - 0.1;
        }
        for (self.bias) |*b| {
            b.* = 0;
        }

        self.input_size = input_size;
        self.output_size = output_size;
        self.act = act;
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

    pub fn forward(self: *Dense, input: []const f32, output: []f32) !void {
        if (input.len != self.input_size) return error.InvalidInputSize;
        if (output.len != self.output_size) return error.InvalidOutputSize;

        // Compute weighted sum + bias
        for (0..self.output_size) |i| {
            var sum: f32 = self.bias[i];
            for (0..self.input_size) |j| {
                sum += input[j] * self.weights[i * self.input_size + j];
            }
            output[i] = self.act.forward(sum);
        }
    }

    pub fn backward(self: *Dense, input: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        // Compute gradient w.r.t input
        // grad_input = grad_output * weights (with activation derivative)
        for (0..self.input_size) |j| {
            var sum: f32 = 0;
            for (0..self.output_size) |i| {
                // Apply activation derivative to grad_output
                // Need to recompute pre-activation for backward pass
                var pre_act: f32 = self.bias[i];
                for (0..self.input_size) |k| {
                    pre_act += input[k] * self.weights[i * self.input_size + k];
                }
                const act_grad = self.act.backward(pre_act, grad_output[i]);
                sum += self.weights[i * self.input_size + j] * act_grad;
            }
            grad_input[j] = sum;
        }
    }
};

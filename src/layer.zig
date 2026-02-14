/// Neural network layers
const std = @import("std");
const activation = @import("activation.zig");

pub const Dense = struct {
    weights: []f32,
    bias: []f32,
    input_size: usize,
    output_size: usize,
    act: activation.Activation,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize, act: activation.Activation) !*Dense {
        const self = allocator.create(Dense) catch return error.OutOfMemory;
        errdefer allocator.destroy(self);

        // Initialize weights with small random values (Xavier initialization)
        const weight_count = input_size * output_size;
        self.weights = try allocator.alloc(f32, weight_count);
        self.bias = try allocator.alloc(f32, output_size);

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

        return self;
    }

    pub fn deinit(self: *Dense, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        allocator.free(self.bias);
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
        _ = input;
        // Compute gradient w.r.t input
        for (0..self.input_size) |j| {
            var sum: f32 = 0;
            for (0..self.output_size) |i| {
                sum += self.weights[i * self.input_size + j] * grad_output[i];
            }
            grad_input[j] = sum;
        }
    }
};

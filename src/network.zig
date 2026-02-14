/// Neural network composition with backpropagation training
const std = @import("std");
const layer = @import("layer.zig");
const activation = @import("activation.zig");
const loss = @import("loss.zig");
const optimizer = @import("optimizer.zig");

/// Cached values needed for backpropagation
const LayerCache = struct {
    pre_activation: []f32,  // Values before activation
    input: []const f32,     // Input to this layer
};

pub const Network = struct {
    layers: std.ArrayList(*layer.Dense),
    allocator: std.mem.Allocator,
    caches: std.ArrayList(?LayerCache),  // Cache for each layer's backprop values

    pub fn init(allocator: std.mem.Allocator) !*Network {
        const self = allocator.create(Network) catch return error.OutOfMemory;
        errdefer allocator.destroy(self);

        self.layers = std.ArrayList(*layer.Dense){ .items = &.{}, .capacity = 0 };
        self.allocator = allocator;
        self.caches = std.ArrayList(?LayerCache){ .items = &.{}, .capacity = 0 };

        return self;
    }

    pub fn deinit(self: *Network) void {
        const allocator = self.allocator;
        for (self.layers.items) |l| {
            l.deinit();
        }
        self.layers.deinit(self.allocator);
        for (self.caches.items) |cache| {
            if (cache) |c| {
                allocator.free(c.pre_activation);
                allocator.free(c.input);
            }
        }
        self.caches.deinit(self.allocator);
        allocator.destroy(self);
    }

    pub fn addDense(self: *Network, input_size: usize, output_size: usize, act: activation.Activation) !*layer.Dense {
        const l = try layer.Dense.init(self.allocator, input_size, output_size, act);
        try self.layers.append(self.allocator, l);
        try self.caches.append(self.allocator, null);  // No cache initially
        return l;
    }

    /// Initialize optimizer for all layers in the network
    pub fn initOptimizer(self: *Network, opt: *optimizer.Optimizer) !void {
        for (self.layers.items) |l| {
            try opt.init(self.allocator, l);
        }
    }

    /// Deinitialize optimizer for all layers in the network
    pub fn deinitOptimizer(self: *Network, opt: *optimizer.Optimizer) void {
        for (self.layers.items) |l| {
            opt.deinit(self.allocator, l);
        }
    }

    /// Forward pass with caching for backpropagation
    /// Returns the output buffer for use in backward pass
    pub fn forward(self: *Network, input: []const f32, output: []f32) ![]const f32 {
        var current = input;
        var cache_idx: usize = 0;

        for (self.layers.items, 0..) |l, i| {
            const temp_size = l.output_size;
            const temp = try self.allocator.alloc(f32, temp_size);
            errdefer self.allocator.free(temp);

            try l.forward(current, temp);

            // Cache values for backprop
            // Store input to this layer
            const cache_input = try self.allocator.alloc(f32, current.len);
            @memcpy(cache_input, current);

            // For the last layer, save the pre-activation values
            // For other layers, we need the activation output
            const cache_pre = if (i == self.layers.items.len - 1) temp else null;

            self.caches.items[i] = LayerCache{
                .pre_activation = if (cache_pre) |p| p else temp,
                .input = cache_input,
            };
            cache_idx += 1;

            current = temp;
        }

        // Copy final output
        @memcpy(output[0..current.len], current);
        return current;
    }

    /// Clear gradients for all layers
    pub fn clearGradients(self: *Network) void {
        for (self.layers.items) |l| {
            @memset(l.grad_weights, 0);
            @memset(l.grad_bias, 0);
        }
    }

    /// Compute gradients for all layers (without updating weights)
    fn computeGradients(self: *Network, target: []const f32, loss_fn: loss.Loss) !void {
        const output_size = self.layers.items[self.layers.items.len - 1].output_size;

        // Get cached output from forward pass
        const last_cache = self.caches.items[self.layers.items.len - 1] orelse return error.NoCache;
        const output = last_cache.pre_activation[0..output_size];

        // Find max layer size for gradient buffer
        var max_size = output_size;
        for (self.layers.items) |l| {
            if (l.output_size > max_size) max_size = l.output_size;
            if (l.input_size > max_size) max_size = l.input_size;
        }

        // Compute gradient of loss w.r.t. output (use max size to handle all layers)
        var grad = try self.allocator.alloc(f32, max_size);
        defer self.allocator.free(grad);
        try loss_fn.backward(output, target, grad[0..output_size]);

        // Backpropagate through layers in reverse order
        var i: usize = self.layers.items.len;
        while (i > 0) {
            i -= 1;
            const l = self.layers.items[i];
            const cache = self.caches.items[i] orelse return error.NoCache;

            // Compute gradient after activation (dL/dz = dL/da * da/dz)
            const grad_after_act = try self.allocator.alloc(f32, l.output_size);
            defer self.allocator.free(grad_after_act);

            for (0..l.output_size) |idx| {
                grad_after_act[idx] = l.act.backward(cache.pre_activation[idx], grad[idx]);
            }

            // Accumulate gradients for weights and bias
            for (0..l.output_size) |out_idx| {
                // Accumulate bias gradient
                l.grad_bias[out_idx] += grad_after_act[out_idx];

                // Accumulate weight gradients
                for (0..l.input_size) |in_idx| {
                    const weight_idx = out_idx * l.input_size + in_idx;
                    const in_val = cache.input[in_idx];
                    l.grad_weights[weight_idx] += grad_after_act[out_idx] * in_val;
                }
            }

            // Compute gradient for previous layer
            if (i > 0) {
                const prev_out_size = l.input_size;

                // Reuse grad buffer if large enough, otherwise allocate new
                if (prev_out_size <= grad.len) {
                    @memset(grad[0..prev_out_size], 0);
                    for (0..prev_out_size) |j| {
                        var sum: f32 = 0;
                        for (0..l.output_size) |k| {
                            const weight_idx = k * l.input_size + j;
                            sum += grad_after_act[k] * l.weights[weight_idx];
                        }
                        grad[j] = sum;
                    }
                } else {
                    const prev_grad = try self.allocator.alloc(f32, prev_out_size);
                    defer self.allocator.free(prev_grad);

                    for (0..prev_out_size) |j| {
                        var sum: f32 = 0;
                        for (0..l.output_size) |k| {
                            const weight_idx = k * l.input_size + j;
                            sum += grad_after_act[k] * l.weights[weight_idx];
                        }
                        prev_grad[j] = sum;
                    }
                    @memcpy(grad[0..prev_out_size], prev_grad);
                }
            }
        }
    }

    /// Train the network on a single sample using backpropagation with SGD
    /// Returns the loss value
    pub fn trainStep(self: *Network, input: []const f32, target: []const f32, learning_rate: f32, loss_fn: loss.Loss) !f32 {
        // Clear previous gradients
        self.clearGradients();

        // Forward pass stores output in the last layer's cache
        const output_size = self.layers.items[self.layers.items.len - 1].output_size;
        if (target.len != output_size) return error.OutputSizeMismatch;

        const output = try self.allocator.alloc(f32, output_size);
        defer self.allocator.free(output);

        _ = try self.forward(input, output);

        // Compute loss
        const sample_loss = try loss_fn.forward(output, target);

        // Compute gradients
        try self.computeGradients(target, loss_fn);

        // Update weights using simple gradient descent (SGD)
        for (self.layers.items) |l| {
            for (l.weights, l.grad_weights, 0..) |_, grad, i| {
                l.weights[i] -= learning_rate * grad;
            }
            for (l.bias, l.grad_bias, 0..) |_, grad, i| {
                l.bias[i] -= learning_rate * grad;
            }
        }

        return sample_loss;
    }

    /// Train the network on a single sample using backpropagation with optimizer
    /// Returns the loss value
    pub fn trainStepWithOptimizer(self: *Network, input: []const f32, target: []const f32, learning_rate: f32, loss_fn: loss.Loss, opt: *optimizer.Optimizer) !f32 {
        // Forward pass
        const output_size = self.layers.items[self.layers.items.len - 1].output_size;
        if (target.len != output_size) return error.OutputSizeMismatch;

        // Clear previous gradients
        self.clearGradients();

        // Forward pass stores output in the last layer's cache
        const output = try self.allocator.alloc(f32, output_size);
        defer self.allocator.free(output);

        _ = try self.forward(input, output);

        // Compute loss
        const sample_loss = try loss_fn.forward(output, target);

        // Compute gradients
        try self.computeGradients(target, loss_fn);

        // Update weights using optimizer
        for (self.layers.items) |l| {
            opt.step(l, learning_rate);
        }

        return sample_loss;
    }

    /// Train the network on multiple epochs
    pub fn train(self: *Network, data: []const []const f32, targets: []const []const f32, epochs: usize, learning_rate: f32, loss_fn: loss.Loss) !void {
        for (0..epochs) |epoch| {
            var total_loss: f32 = 0;
            var sample_count: usize = 0;

            for (data, targets) |sample, target| {
                const sample_loss = try self.trainStep(sample, target, learning_rate, loss_fn);
                total_loss += sample_loss;
                sample_count += 1;
            }

            if (sample_count > 0) {
                total_loss /= @as(f32, @floatFromInt(sample_count));
            }

            if (epoch % 100 == 0) {
                std.debug.print("Epoch {}: Loss = {d:.4}\n", .{ epoch, total_loss });
            }
        }
    }

    /// Train the network on multiple epochs using optimizer
    pub fn trainWithOptimizer(self: *Network, data: []const []const f32, targets: []const []const f32, epochs: usize, learning_rate: f32, loss_fn: loss.Loss, opt: *optimizer.Optimizer) !void {
        for (0..epochs) |epoch| {
            var total_loss: f32 = 0;
            var sample_count: usize = 0;

            for (data, targets) |sample, target| {
                const sample_loss = try self.trainStepWithOptimizer(sample, target, learning_rate, loss_fn, opt);
                total_loss += sample_loss;
                sample_count += 1;
            }

            if (sample_count > 0) {
                total_loss /= @as(f32, @floatFromInt(sample_count));
            }

            if (epoch % 100 == 0) {
                std.debug.print("Epoch {}: Loss = {d:.4}\n", .{ epoch, total_loss });
            }
        }
    }
};

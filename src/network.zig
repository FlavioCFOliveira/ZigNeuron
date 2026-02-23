/// Neural network composition with backpropagation training
///
/// GPU OPTIMIZATION FEATURES (Metal on Apple Silicon):
/// - Reduced GPU threshold (64 elements vs 256) to leverage parallelism more aggressively
/// - Pre-allocated work buffers to reduce memory allocation overhead during training
/// - Cache-optimized CPU fallback with loop tiling/blocking for large matrices
/// - Metal GPU automatically selected on macOS with Apple Silicon
/// - Automatic backend selection: Metal > Vulkan > CPU
///
/// PERFORMANCE TIPS:
/// - Larger layer sizes (>64 neurons) benefit more from GPU acceleration
/// - GPU overhead is amortized across multiple operations in each training step
/// - Pre-allocated buffers reduce memory allocation overhead in tight training loops
///
/// USAGE:
/// ```zig
/// const network = try Network.init(allocator, try Backend.init(allocator));
/// _ = try network.addDense(input_size, hidden_size, .relu);
/// _ = try network.addDense(hidden_size, output_size, .linear);
/// try network.train(data, targets, epochs, learning_rate, loss_fn);
/// ```
const std = @import("std");
const layer = @import("layer.zig");
const activation = @import("activation.zig");
const loss = @import("loss.zig");
const optimizer = @import("optimizer.zig");
const backend_module = @import("backend.zig");
const tensor = @import("tensor.zig");
const metal = @import("metal.zig");

/// Cached values needed for backpropagation
const LayerCache = struct {
    activated_output: tensor.Tensor, // Activated output of this layer
    input: tensor.Tensor, // Input to this layer
};

pub const Network = struct {
    layers: std.array_list.Managed(*layer.Dense),
    allocator: std.mem.Allocator,
    caches: std.array_list.Managed(?LayerCache), // Cache for each layer's backprop values
    backend: backend_module.Backend,
    /// Optimizer state per layer
    optimizers: std.array_list.Managed(?optimizer.Optimizer),
    /// Pre-allocated buffers for GPU efficiency
    work_buffer: ?[]f32, // Reusable buffer for intermediate computations
    max_layer_size: usize, // Track maximum layer size for buffer allocation

    // Pre-allocated work tensors for backprop to avoid repeated allocation/deallocation (crucial for Metal batching)
    grad_work_1: ?tensor.Tensor = null,
    grad_work_2: ?tensor.Tensor = null,
    grad_after_act_work: ?tensor.Tensor = null,

    pub fn init(allocator: std.mem.Allocator, backend: backend_module.Backend) !*Network {
        const self = try allocator.create(Network);
        errdefer allocator.destroy(self);

        self.layers = std.array_list.Managed(*layer.Dense).init(allocator);
        self.allocator = allocator;
        self.caches = std.array_list.Managed(?LayerCache).init(allocator);
        self.backend = backend;
        self.work_buffer = null;
        self.max_layer_size = 0;
        self.optimizers = std.array_list.Managed(?optimizer.Optimizer).init(allocator);

        self.grad_work_1 = null;
        self.grad_work_2 = null;
        self.grad_after_act_work = null;

        return self;
    }

    pub fn deinit(self: *Network) void {
        const allocator = self.allocator;
        for (self.layers.items) |l| {
            l.deinit();
        }
        self.layers.deinit();
        for (self.caches.items) |*cache| {
            if (cache.*) |*c| {
                c.activated_output.deinit();
                c.input.deinit();
            }
        }
        self.caches.deinit();
        self.optimizers.deinit();

        // Free work buffer if allocated
        if (self.work_buffer) |buf| {
            allocator.free(buf);
        }

        // Free work tensors
        if (self.grad_work_1) |*t| t.deinit();
        if (self.grad_work_2) |*t| t.deinit();
        if (self.grad_after_act_work) |*t| t.deinit();

        var mutable_backend = self.backend;
        mutable_backend.deinit();
        allocator.destroy(self);
    }

    pub fn addDense(self: *Network, input_size: usize, output_size: usize, act: activation.Activation) !*layer.Dense {
        const l = try layer.Dense.init(self.allocator, input_size, output_size, act, self.backend);
        try self.layers.append(l);
        try self.caches.append(null); // No cache initially

        // Update max layer size for buffer pre-allocation
        const layer_max = @max(input_size, output_size);
        if (layer_max > self.max_layer_size) {
            self.max_layer_size = layer_max;

            // Re-allocate work buffers if needed
            if (self.backend.type == .gpu) {
                if (self.work_buffer) |buf| self.allocator.free(buf);
                self.work_buffer = try self.allocator.alloc(f32, layer_max);

                if (self.grad_work_1) |*t| t.deinit();
                if (self.grad_work_2) |*t| t.deinit();
                if (self.grad_after_act_work) |*t| t.deinit();

                self.grad_work_1 = try tensor.Tensor.init(self.allocator, layer_max, self.backend);
                self.grad_work_2 = try tensor.Tensor.init(self.allocator, layer_max, self.backend);
                self.grad_after_act_work = try tensor.Tensor.init(self.allocator, layer_max, self.backend);
            }
        }

        // Initialize optimizer state for this layer (default SGD)
        try self.optimizers.append(null);

        return l;
    }

    pub fn forward(self: *Network, input: []const f32, output: []f32) !void {
        var current_slice = input;
        var current_buf: ?*const metal.MTLBuffer = null;

        for (self.layers.items, 0..) |l, i| {
            // Check if we already have a cache for this layer from a previous forward pass
            if (self.caches.items[i]) |*old_cache| {
                old_cache.activated_output.deinit();
                old_cache.input.deinit();
            }

            // Create new tensors for cache (MUST use Metal-backed tensors if GPU enabled)
            // Store the input to this layer
            var cache_input = try tensor.Tensor.init(self.allocator, current_slice.len, self.backend);
            errdefer cache_input.deinit();

            if (current_buf) |buf| {
                // GPU to GPU copy (using linear activation as a copy kernel)
                try self.backend.activationForward(.linear, current_slice, buf, cache_input.slice, cache_input.getMtlBuffer());
            } else {
                // CPU to GPU copy
                @memcpy(cache_input.slice, current_slice);
            }

            var activated_output = try tensor.Tensor.init(self.allocator, l.output_size, self.backend);
            errdefer activated_output.deinit();

            // Run layer forward pass
            try l.forward(cache_input.slice, cache_input.getMtlBuffer(), activated_output.slice, activated_output.getMtlBuffer());

            // Store in cache
            self.caches.items[i] = LayerCache{
                .activated_output = activated_output,
                .input = cache_input,
            };

            current_slice = activated_output.slice;
            current_buf = activated_output.getMtlBuffer();
        }

        // Copy final output to provided buffer
        const last_activated_output = self.caches.items[self.layers.items.len - 1].?.activated_output;
        const final_slice = last_activated_output.slice;
        @memcpy(output[0..final_slice.len], final_slice);
    }

    pub fn clearGradients(self: *Network) void {
        for (self.layers.items) |l| {
            @memset(l.grad_weights.slice, 0);
            @memset(l.grad_bias.slice, 0);
        }
    }

    pub fn computeGradients(self: *Network, target: []const f32, loss_fn: loss.Loss) !void {
        if (self.layers.items.len == 0) return;

        const last_layer_idx = self.layers.items.len - 1;
        const last_cache = self.caches.items[last_layer_idx] orelse return error.NoCache;
        const output = last_cache.activated_output.slice;
        const output_buf = last_cache.activated_output.getMtlBuffer();

        // Use pre-allocated work tensors
        if (self.grad_work_1 == null) {
            // If they weren't allocated in addDense (CPU mode), allocate them here
            self.grad_work_1 = try tensor.Tensor.init(self.allocator, self.max_layer_size, self.backend);
            self.grad_work_2 = try tensor.Tensor.init(self.allocator, self.max_layer_size, self.backend);
            self.grad_after_act_work = try tensor.Tensor.init(self.allocator, self.max_layer_size, self.backend);
        }

        var grad = self.grad_work_1.?;
        var next_grad = self.grad_work_2.?;
        var grad_after_act = self.grad_after_act_work.?;

        const output_size = self.layers.items[last_layer_idx].output_size;

        // Compute gradient of loss w.r.t output
        try self.backend.lossBackward(loss_fn, output, output_buf, target, null, grad.slice[0..output_size], grad.getMtlBuffer());

        // Backpropagate through layers in reverse order
        var i: usize = self.layers.items.len;
        while (i > 0) {
            i -= 1;
            const l = self.layers.items[i];
            const cache = self.caches.items[i] orelse return error.NoCache;

            // Compute gradient after activation
            try self.backend.activationBackward(l.act,
                cache.activated_output.slice, cache.activated_output.getMtlBuffer(),
                grad.slice[0..l.output_size], grad.getMtlBuffer(),
                grad_after_act.slice, grad_after_act.getMtlBuffer()
            );

            // Accumulate gradients for weights and bias
            try l.accumulateGradients(
                cache.input.slice, cache.input.getMtlBuffer(),
                grad_after_act.slice, grad_after_act.getMtlBuffer()
            );

            // Compute gradient for previous layer
            if (i > 0) {
                // Compute gradient for previous layer: grad_input = grad_after_act * weights^T
                try self.backend.matMulTransposeB(
                    grad_after_act.slice, grad_after_act.getMtlBuffer(),
                    l.weights.slice, l.weights.getMtlBuffer(),
                    next_grad.slice, next_grad.getMtlBuffer(),
                    1, // batch_size
                    l.input_size,
                    l.output_size
                );

                // Swap grad and next_grad for the next iteration
                const tmp = grad;
                grad = next_grad;
                next_grad = tmp;
            }
        }
    }

    pub fn trainStep(self: *Network, input: []const f32, target: []const f32, learning_rate: f32, loss_fn: loss.Loss) !f32 {
        // Start command batch for GPU
        try self.backend.beginCommandBatch();
        // Use errdefer to ensure batch is ended on failure
        errdefer self.backend.endCommandBatch() catch {};

        // Clear previous gradients
        self.clearGradients();

        const output_size = self.layers.items[self.layers.items.len - 1].output_size;
        if (target.len != output_size) return error.OutputSizeMismatch;

        const output = try self.allocator.alloc(f32, output_size);
        defer self.allocator.free(output);

        _ = try self.forward(input, output);

        // Compute gradients
        try self.computeGradients(target, loss_fn);

        // Update weights using GPU-accelerated SGD if available
        const weight_decay: f32 = 0.0;
        for (self.layers.items) |l| {
            try self.backend.sgdUpdate(
                l.weights.slice,
                l.weights.getMtlBuffer(),
                l.grad_weights.slice,
                l.grad_weights.getMtlBuffer(),
                learning_rate,
                weight_decay,
            );
            try self.backend.sgdUpdateBias(
                l.bias.slice,
                l.bias.getMtlBuffer(),
                l.grad_bias.slice,
                l.grad_bias.getMtlBuffer(),
                learning_rate,
            );
        }

        // End command batch and wait for completion BEFORE calculating loss
        // This ensures 'output' has the latest values from GPU
        try self.backend.endCommandBatch();

        // Sync output after batch completion (Network.forward's copy was too early during a batch)
        const last_activated_output = self.caches.items[self.layers.items.len - 1].?.activated_output;
        @memcpy(output, last_activated_output.slice);

        return try loss_fn.forward(output, target);
    }

    pub fn train(self: *Network, inputs: []const []const f32, targets: []const []const f32, epochs: usize, learning_rate: f32, loss_fn: loss.Loss) !void {
        for (0..epochs) |epoch| {
            var total_loss: f32 = 0;
            for (inputs, targets) |input, target| {
                total_loss += try self.trainStep(input, target, learning_rate, loss_fn);
            }
            if (epoch % 100 == 0 or epoch == epochs - 1) {
                std.debug.print("Epoch {}: Loss = {d:.4}\n", .{ epoch, total_loss / @as(f32, @floatFromInt(inputs.len)) });
            }
        }
    }
};

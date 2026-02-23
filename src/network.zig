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
const recurrent = @import("recurrent.zig");

/// Cached values needed for backpropagation
const LayerCache = struct {
    activated_output: tensor.Tensor, // Activated output of this layer
    input: tensor.Tensor, // Input to this layer
};

pub const Network = struct {
    layers: std.array_list.Managed(layer.Layer),
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

        self.layers = std.array_list.Managed(layer.Layer).init(allocator);
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
        try self.layers.append(.{ .dense = l });
        try self.caches.append(null);

        // Update max layer size for buffer pre-allocation
        const layer_max = @max(input_size, output_size);
        if (layer_max > self.max_layer_size) {
            self.max_layer_size = layer_max;
            try self.reallocateWorkBuffers(layer_max);
        }

        try self.optimizers.append(null);
        return l;
    }

    pub fn addSampling(self: *Network, input_size: usize) !*layer.SamplingLayer {
        const l = try layer.SamplingLayer.init(self.allocator, input_size, self.backend);
        try self.layers.append(.{ .sampling = l });
        try self.caches.append(null);
        try self.optimizers.append(null);
        return l;
    }

    pub fn addConv1D(self: *Network, in_channels: usize, out_channels: usize, kernel_size: usize, input_len: usize, act: activation.Activation) !*layer.Conv1D {
        const l = try layer.Conv1D.init(self.allocator, in_channels, out_channels, kernel_size, input_len, act, self.backend);
        try self.layers.append(.{ .conv1d = l });
        try self.caches.append(null);
        const layer_max = @max(in_channels * input_len, out_channels * ((input_len - kernel_size) + 1));
        if (layer_max > self.max_layer_size) {
            self.max_layer_size = layer_max;
            try self.reallocateWorkBuffers(layer_max);
        }
        try self.optimizers.append(null);
        return l;
    }

    pub fn addLayerNorm(self: *Network, size: usize) !*layer.LayerNorm {
        const l = try layer.LayerNorm.init(self.allocator, size, self.backend);
        try self.layers.append(.{ .layer_norm = l });
        try self.caches.append(null);
        if (size > self.max_layer_size) {
            self.max_layer_size = size;
            try self.reallocateWorkBuffers(size);
        }
        try self.optimizers.append(null);
        return l;
    }

    pub fn addDropout(self: *Network, size: usize, rate: f32) !*layer.Dropout {
        const l = try layer.Dropout.init(self.allocator, size, rate, self.backend);
        try self.layers.append(.{ .dropout = l });
        try self.caches.append(null);
        if (size > self.max_layer_size) {
            self.max_layer_size = size;
            try self.reallocateWorkBuffers(size);
        }
        try self.optimizers.append(null);
        return l;
    }

    pub fn addRNN(self: *Network, input_size: usize, hidden_size: usize, act: activation.Activation) !*recurrent.VanillaRNN {
        const l = try recurrent.VanillaRNN.init(self.allocator, input_size, hidden_size, act, self.backend);
        try self.layers.append(.{ .rnn = l });
        try self.caches.append(null);

        // Update max layer size for buffer pre-allocation
        const layer_max = @max(input_size, hidden_size);
        if (layer_max > self.max_layer_size) {
            self.max_layer_size = layer_max;
            try self.reallocateWorkBuffers(layer_max);
        }

        try self.optimizers.append(null);
        return l;
    }

    pub fn addLSTM(self: *Network, input_size: usize, hidden_size: usize, max_seq_len: usize) !*recurrent.LSTM {
        const l = try recurrent.LSTM.init(self.allocator, input_size, hidden_size, max_seq_len, self.backend);
        try self.layers.append(.{ .lstm = l });
        try self.caches.append(null);

        // Update max layer size for buffer pre-allocation
        // LSTM has internal gate activations of 4 * hidden_size
        const layer_max = @max(input_size, 4 * hidden_size);
        if (layer_max > self.max_layer_size) {
            self.max_layer_size = layer_max;
            try self.reallocateWorkBuffers(layer_max);
        }

        try self.optimizers.append(null);
        return l;
    }

    pub fn addGRU(self: *Network, input_size: usize, hidden_size: usize, max_seq_len: usize) !*recurrent.GRU {
        const l = try recurrent.GRU.init(self.allocator, input_size, hidden_size, max_seq_len, self.backend);
        try self.layers.append(.{ .gru = l });
        try self.caches.append(null);

        // Update max layer size for buffer pre-allocation
        // GRU has internal gate activations of 3 * hidden_size
        const layer_max = @max(input_size, 3 * hidden_size);
        if (layer_max > self.max_layer_size) {
            self.max_layer_size = layer_max;
            try self.reallocateWorkBuffers(layer_max);
        }

        try self.optimizers.append(null);
        return l;
    }

    pub fn addBidirectional(self: *Network, layer_type: recurrent.RecurrentLayerType, input_size: usize, hidden_size: usize, max_seq_len: usize, act: activation.Activation) !*recurrent.Bidirectional {
        const l = try recurrent.Bidirectional.init(self.allocator, layer_type, input_size, hidden_size, max_seq_len, act, self.backend);
        try self.layers.append(.{ .bidirectional = l });
        try self.caches.append(null);

        // Update max layer size for buffer pre-allocation
        // Bidirectional output is 2 * hidden_size. Internal layers might have larger gate activations.
        var layer_max = @max(input_size, 2 * hidden_size);
        switch (layer_type) {
            .rnn => layer_max = @max(layer_max, hidden_size),
            .lstm => layer_max = @max(layer_max, 4 * hidden_size),
            .gru => layer_max = @max(layer_max, 3 * hidden_size),
        }

        if (layer_max > self.max_layer_size) {
            self.max_layer_size = layer_max;
            try self.reallocateWorkBuffers(layer_max);
        }

        try self.optimizers.append(null);
        return l;
    }

    pub fn addTwoPath(self: *Network, layer_type1: recurrent.RecurrentLayerType, hidden_size1: usize, layer_type2: recurrent.RecurrentLayerType, hidden_size2: usize, input_size: usize, max_seq_len: usize, act: activation.Activation) !*recurrent.TwoPath {
        const l = try recurrent.TwoPath.init(self.allocator, layer_type1, hidden_size1, layer_type2, hidden_size2, input_size, max_seq_len, act, self.backend);
        try self.layers.append(.{ .twopath = l });
        try self.caches.append(null);

        // Update max layer size
        var layer_max = @max(input_size, hidden_size1 + hidden_size2);
        // Check internal gate sizes for path1
        switch (layer_type1) {
            .rnn => layer_max = @max(layer_max, hidden_size1),
            .lstm => layer_max = @max(layer_max, 4 * hidden_size1),
            .gru => layer_max = @max(layer_max, 3 * hidden_size1),
        }
        // Check internal gate sizes for path2
        switch (layer_type2) {
            .rnn => layer_max = @max(layer_max, hidden_size2),
            .lstm => layer_max = @max(layer_max, 4 * hidden_size2),
            .gru => layer_max = @max(layer_max, 3 * hidden_size2),
        }

        if (layer_max > self.max_layer_size) {
            self.max_layer_size = layer_max;
            try self.reallocateWorkBuffers(layer_max);
        }

        try self.optimizers.append(null);
        return l;
    }

    pub fn addAttention(self: *Network, size: usize) !layer.Layer {
        const l = try layer.Attention.init(self.allocator, size, self.backend);
        const layer_obj = layer.Layer{ .attention = l };
        try self.layers.append(layer_obj);
        try self.caches.append(null);

        if (size > self.max_layer_size) {
            self.max_layer_size = size;
            try self.reallocateWorkBuffers(size);
        }

        try self.optimizers.append(null);
        return layer_obj;
    }

    fn reallocateWorkBuffers(self: *Network, size: usize) !void {
        if (self.backend.type == .gpu) {
            if (self.work_buffer) |buf| self.allocator.free(buf);
            self.work_buffer = try self.allocator.alloc(f32, size);

            if (self.grad_work_1) |*t| t.deinit();
            if (self.grad_work_2) |*t| t.deinit();
            if (self.grad_after_act_work) |*t| t.deinit();

            self.grad_work_1 = try tensor.Tensor.init(self.allocator, &.{size}, self.backend);
            self.grad_work_2 = try tensor.Tensor.init(self.allocator, &.{size}, self.backend);
            self.grad_after_act_work = try tensor.Tensor.init(self.allocator, &.{size}, self.backend);
        }
    }

    pub fn forward(self: *Network, input: []const f32, output: []f32) !void {
        // Use batching for forward pass too to minimize synchronization
        try self.backend.beginCommandBatch();
        errdefer self.backend.endCommandBatch() catch {};

        var current_slice = input;
        var current_buf: ?*const metal.MTLBuffer = null;

        for (self.layers.items, 0..) |l, i| {
            // Reuse cache if it exists and has the correct size
            if (self.caches.items[i]) |*cache| {
                if (cache.input.slice.len != current_slice.len) {
                    cache.input.deinit();
                    cache.input = try tensor.Tensor.init(self.allocator, &.{current_slice.len}, self.backend);
                }
                if (cache.activated_output.slice.len != l.outputSize()) {
                    cache.activated_output.deinit();
                    cache.activated_output = try tensor.Tensor.init(self.allocator, &.{l.outputSize()}, self.backend);
                }
            } else {
                self.caches.items[i] = LayerCache{
                    .input = try tensor.Tensor.init(self.allocator, &.{current_slice.len}, self.backend),
                    .activated_output = try tensor.Tensor.init(self.allocator, &.{l.outputSize()}, self.backend),
                };
            }

            var cache = &self.caches.items[i].?;

            if (current_buf) |buf| {
                // GPU to GPU copy (using linear activation as a copy kernel)
                try self.backend.activationForward(.linear, current_slice, buf, cache.input.slice, cache.input.getMtlBuffer());
            } else {
                // CPU to GPU copy
                @memcpy(cache.input.slice, current_slice);
            }

            // Run layer forward pass
            try l.forward(cache.input.slice, cache.input.getMtlBuffer(), cache.activated_output.slice, cache.activated_output.getMtlBuffer());

            current_slice = cache.activated_output.slice;
            current_buf = cache.activated_output.getMtlBuffer();
        }

        // Copy final output to provided buffer
        const last_activated_output = self.caches.items[self.layers.items.len - 1].?.activated_output;
        const final_slice = last_activated_output.slice;
        @memcpy(output[0..final_slice.len], final_slice);
    }

    pub fn clearGradients(self: *Network) void {
        for (self.layers.items) |l| {
            @memset(l.getGradWeights().slice, 0);
            @memset(l.getGradBias().slice, 0);
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
            self.grad_work_1 = try tensor.Tensor.init(self.allocator, &.{self.max_layer_size}, self.backend);
            self.grad_work_2 = try tensor.Tensor.init(self.allocator, &.{self.max_layer_size}, self.backend);
            self.grad_after_act_work = try tensor.Tensor.init(self.allocator, &.{self.max_layer_size}, self.backend);
        }

        var grad = self.grad_work_1.?;
        var next_grad = self.grad_work_2.?;
        var grad_after_act = self.grad_after_act_work.?;

        const output_size = self.layers.items[last_layer_idx].outputSize();

        // Compute gradient of loss w.r.t output
        try self.backend.lossBackward(loss_fn, output, output_buf, target, null, grad.slice[0..output_size], grad.getMtlBuffer());

        // Backpropagate through layers in reverse order
        var i: usize = self.layers.items.len;
        while (i > 0) {
            i -= 1;
            const l = self.layers.items[i];
            const cache = self.caches.items[i] orelse return error.NoCache;

            // Compute gradient after activation
            try self.backend.activationBackward(l.getActivation(),
                cache.activated_output.slice, cache.activated_output.getMtlBuffer(),
                grad.slice[0..l.outputSize()], grad.getMtlBuffer(),
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
                const weights = l.getWeights();
                try self.backend.matMulTransposeB(
                    grad_after_act.slice, grad_after_act.getMtlBuffer(),
                    weights.slice, weights.getMtlBuffer(),
                    next_grad.slice, next_grad.getMtlBuffer(),
                    1, // batch_size
                    l.inputSize(),
                    l.outputSize()
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

        const output_size = self.layers.items[self.layers.items.len - 1].outputSize();
        if (target.len != output_size) return error.OutputSizeMismatch;

        const output = try self.allocator.alloc(f32, output_size);
        defer self.allocator.free(output);

        _ = try self.forward(input, output);

        // Compute gradients
        try self.computeGradients(target, loss_fn);

        // Update weights using GPU-accelerated SGD if available
        const weight_decay: f32 = 0.0;
        for (self.layers.items) |l| {
            try l.updateWeights(learning_rate, weight_decay);
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

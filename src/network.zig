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
    output_buffer: ?[]f32, // Pre-allocated output buffer for training
    max_layer_size: usize, // Track maximum layer size for buffer allocation

    // Pre-allocated work tensors for backprop to avoid repeated allocation/deallocation (crucial for Metal batching)
    grad_work_1: ?tensor.Tensor = null,
    grad_work_2: ?tensor.Tensor = null,

    pub fn init(allocator: std.mem.Allocator, backend: backend_module.Backend) !*Network {
        const self = try allocator.create(Network);
        errdefer allocator.destroy(self);

        self.layers = std.array_list.Managed(layer.Layer).init(allocator);
        self.allocator = allocator;
        self.caches = std.array_list.Managed(?LayerCache).init(allocator);
        self.backend = backend;
        self.work_buffer = null;
        self.output_buffer = null;
        self.max_layer_size = 0;
        self.optimizers = std.array_list.Managed(?optimizer.Optimizer).init(allocator);

        self.grad_work_1 = null;
        self.grad_work_2 = null;

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
        if (self.output_buffer) |buf| {
            allocator.free(buf);
        }

        // Free work tensors
        if (self.grad_work_1) |*t| t.deinit();
        if (self.grad_work_2) |*t| t.deinit();

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

    pub fn addBatchNorm(self: *Network, size: usize) !*layer.BatchNorm {
        const l = try layer.BatchNorm.init(self.allocator, size, self.backend);
        try self.layers.append(.{ .batch_norm = l });
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
        if (self.work_buffer) |buf| self.allocator.free(buf);
        self.work_buffer = try self.allocator.alloc(f32, size);

        if (self.output_buffer) |buf| self.allocator.free(buf);
        self.output_buffer = try self.allocator.alloc(f32, size);

        if (self.grad_work_1) |*t| t.deinit();
        if (self.grad_work_2) |*t| t.deinit();

        self.grad_work_1 = try tensor.Tensor.init(self.allocator, &.{size}, self.backend);
        self.grad_work_2 = try tensor.Tensor.init(self.allocator, &.{size}, self.backend);
    }

    pub fn forward(self: *Network, input: []const f32, output: []f32) !void {
        // Use batching for forward pass too to minimize synchronization
        try self.backend.beginCommandBatch();
        errdefer self.backend.endCommandBatch() catch {};

        var current_slice = input;
        var current_buf: ?*const metal.MTLBuffer = null;

        for (self.layers.items, 0..) |l, i| {
            const l_input_size = l.inputSize();
            const seq_len = current_slice.len / l_input_size;
            const l_output_size = l.outputSize();
            const total_output_size = seq_len * l_output_size;

            // Reuse cache if it exists and has the correct size
            if (self.caches.items[i]) |*cache| {
                if (cache.input.slice.len < current_slice.len) {
                    cache.input.deinit();
                    cache.input = try tensor.Tensor.init(self.allocator, &.{current_slice.len}, self.backend);
                }
                if (cache.activated_output.slice.len < total_output_size) {
                    cache.activated_output.deinit();
                    cache.activated_output = try tensor.Tensor.init(self.allocator, &.{total_output_size}, self.backend);
                }
            } else {
                self.caches.items[i] = LayerCache{
                    .input = try tensor.Tensor.init(self.allocator, &.{current_slice.len}, self.backend),
                    .activated_output = try tensor.Tensor.init(self.allocator, &.{total_output_size}, self.backend),
                };
            }

            var cache = &(self.caches.items[i] orelse return error.MissingCache);

            if (current_buf) |buf| {
                // GPU to GPU copy
                try self.backend.copyData(current_slice, buf, cache.input.slice[0..current_slice.len], cache.input.getMtlBuffer());
            } else {
                // CPU to GPU copy
                @memcpy(cache.input.slice[0..current_slice.len], current_slice);
                try self.backend.copyData(cache.input.slice[0..current_slice.len], null, cache.input.slice[0..current_slice.len], cache.input.getMtlBuffer());
            }

            // Run layer forward pass
            try l.forward(
                cache.input.slice[0..current_slice.len], cache.input.getMtlBuffer(),
                cache.activated_output.slice[0..total_output_size], cache.activated_output.getMtlBuffer()
            );

            current_slice = cache.activated_output.slice[0..total_output_size];
            current_buf = cache.activated_output.getMtlBuffer();
        }

        // Copy final output to provided buffer
        const last_cache = self.caches.items[self.layers.items.len - 1] orelse return error.MissingCache;
        const final_layer_output_size = self.layers.items[self.layers.items.len - 1].outputSize();
        const total_final_size = current_slice.len;
        const final_seq_len = total_final_size / final_layer_output_size;

        // Sync final output from GPU to CPU slice
        // Only wait for completion IF we are not in a larger batch
        if (self.backend.metal_ctx) |ctx| {
            if (ctx.active_command_buffer == null) {
                try self.backend.endCommandBatch();
                try self.backend.copyData(@constCast(current_slice), last_cache.activated_output.getMtlBuffer(), @constCast(current_slice), null);
            }
        } else {
            // CPU mode or already in a batch - the caller will call endCommandBatch
        }

        const out_len = @min(output.len, total_final_size);
        if (final_seq_len > 1 and output.len == final_layer_output_size) {
            // Many-to-one: only copy the last step of the sequence
            const last_step_start = (final_seq_len - 1) * final_layer_output_size;
            @memcpy(output, current_slice[last_step_start .. last_step_start + final_layer_output_size]);
        } else {
            @memcpy(output[0..out_len], current_slice[0..out_len]);
        }
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

        // Use pre-allocated work tensors, resizing if necessary
        // We need to accommodate the largest sequence across all layers
        var max_needed: usize = 0;
        for (self.caches.items) |cache_opt| {
            if (cache_opt) |c| {
                max_needed = @max(max_needed, c.input.slice.len);
                max_needed = @max(max_needed, c.activated_output.slice.len);
            }
        }

        if (self.grad_work_1 == null or self.grad_work_1.?.slice.len < max_needed) {
            if (self.grad_work_1) |*t| t.deinit();
            if (self.grad_work_2) |*t| t.deinit();
            self.grad_work_1 = try tensor.Tensor.init(self.allocator, &.{max_needed}, self.backend);
            self.grad_work_2 = try tensor.Tensor.init(self.allocator, &.{max_needed}, self.backend);
        }

        var grad = self.grad_work_1.?;
        var next_grad = self.grad_work_2.?;

        // Compute gradient of loss w.r.t output
        // Handle many-to-one vs sequence-to-sequence
        const final_layer = self.layers.items[last_layer_idx];
        const final_layer_output_size = final_layer.outputSize();
        const total_output_size = output.len;
        const final_seq_len = total_output_size / final_layer_output_size;

        if (final_seq_len > 1 and target.len == final_layer_output_size) {
            // Many-to-one: only compute gradient for the last step
            @memset(grad.slice[0..total_output_size], 0);
            const last_step_start = (final_seq_len - 1) * final_layer_output_size;
            try self.backend.lossBackward(
                loss_fn,
                output[last_step_start .. last_step_start + final_layer_output_size],
                output_buf, // Offset handled by getBuffer if possible, but wait...
                target,
                null,
                grad.slice[last_step_start .. last_step_start + final_layer_output_size],
                grad.getMtlBuffer()
            );
            // Sync back to GPU if necessary
            try self.backend.copyData(grad.slice[0..total_output_size], null, grad.slice[0..total_output_size], grad.getMtlBuffer());
        } else {
            // Sequence-to-sequence or single output
            try self.backend.lossBackward(loss_fn, output, output_buf, target, null, grad.slice[0..output.len], grad.getMtlBuffer());
        }

        // Backpropagate through layers in reverse order
        var i: usize = self.layers.items.len;
        while (i > 0) {
            i -= 1;
            const l = self.layers.items[i];
            const cache = self.caches.items[i] orelse return error.NoCache;

            const grad_in_size = cache.input.slice.len;
            const grad_input_slice = next_grad.slice[0..grad_in_size];

            // Use unified backward call - now handles activation derivative,
            // weight accumulation, and grad_input computation for all layer types.
            try l.backward(
                cache.input.slice, cache.input.getMtlBuffer(),
                grad.slice[0..cache.activated_output.slice.len], grad.getMtlBuffer(),
                grad_input_slice, next_grad.getMtlBuffer(),
                cache.activated_output.slice, cache.activated_output.getMtlBuffer()
            );

            // Swap grad and next_grad for the next iteration
            const tmp = grad;
            grad = next_grad;
            next_grad = tmp;
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

        // Use pre-allocated output buffer
        if (self.output_buffer == null or self.output_buffer.?.len < output_size) {
            if (self.output_buffer) |buf| self.allocator.free(buf);
            self.output_buffer = try self.allocator.alloc(f32, output_size);
        }
        const output = self.output_buffer.?[0..output_size];

        _ = try self.forward(input, output);

        // Compute gradients
        try self.computeGradients(target, loss_fn);

        // End command batch and wait for completion BEFORE gradient clipping
        try self.backend.endCommandBatch();

        // Optional: Gradient clipping to prevent explosion (common in RNNs)
        const max_norm: f32 = 1.0;
        var total_norm: f32 = 0.0;
        for (self.layers.items) |l| {
            const gw = l.getGradWeights();
            const gb = l.getGradBias();
            for (gw.slice) |g| total_norm += g * g;
            for (gb.slice) |g| total_norm += g * g;
        }
        total_norm = @sqrt(total_norm);

        if (total_norm > max_norm) {
            const clip_factor = max_norm / (total_norm + 1e-6);
            for (self.layers.items) |l| {
                const gw = l.getGradWeights();
                const gb = l.getGradBias();
                for (gw.slice) |*g| g.* *= clip_factor;
                for (gb.slice) |*g| g.* *= clip_factor;
                // Sync clipped gradients back to GPU if necessary
                try self.backend.copyData(gw.slice, null, gw.slice, gw.getMtlBuffer());
                try self.backend.copyData(gb.slice, null, gb.slice, gb.getMtlBuffer());
            }
        }

        // Start a new batch for weight updates
        try self.backend.beginCommandBatch();

        // Update weights using optimizers
        for (self.layers.items, 0..) |*l, i| {
            if (self.optimizers.items[i]) |*opt| {
                try opt.step(l, learning_rate);
            } else {
                // Default to SGD if no optimizer is set
                var opt = optimizer.Optimizer{ .sgd = .{ .momentum = 0.0 } };
                try opt.step(l, learning_rate);
            }
        }

        // End command batch and wait for completion BEFORE calculating loss
        try self.backend.endCommandBatch();

        // Sync final output from GPU to CPU to ensure loss calculation is accurate
        if (self.backend.type == .gpu and self.backend.type.gpu == .metal) {
            const last_cache = self.caches.items[self.layers.items.len - 1] orelse return error.MissingCache;
            const last_layer_output_size = self.layers.items[self.layers.items.len - 1].outputSize();
            const total_final_size = last_cache.activated_output.slice.len;
            const final_seq_len = total_final_size / last_layer_output_size;

            // Sync the GPU buffer to the Tensor's CPU slice
            try self.backend.copyData(last_cache.activated_output.slice, last_cache.activated_output.getMtlBuffer(), last_cache.activated_output.slice, null);

            // Update the output buffer provided to trainStep
            if (final_seq_len > 1 and output.len == last_layer_output_size) {
                const last_step_start = (final_seq_len - 1) * last_layer_output_size;
                @memcpy(output, last_cache.activated_output.slice[last_step_start .. last_step_start + last_layer_output_size]);
            } else {
                const out_len = @min(output.len, total_final_size);
                @memcpy(output[0..out_len], last_cache.activated_output.slice[0..out_len]);
            }
        }

        const l_val = try loss_fn.forward(output, target);
        if (std.math.isNan(l_val)) {
            std.debug.print("NaN loss detected!\n", .{});
        }
        return l_val;
    }

/// Early stopping configuration
pub const EarlyStopping = struct {
    patience: usize, // Number of epochs to wait before stopping
    min_delta: f32, // Minimum change to qualify as improvement
    restore_best_weights: bool, // Whether to restore weights from best epoch

    pub fn init(patience: usize, min_delta: f32, restore_best_weights: bool) EarlyStopping {
        return .{
            .patience = patience,
            .min_delta = min_delta,
            .restore_best_weights = restore_best_weights,
        };
    }
};

    pub fn train(self: *Network, inputs: []const []const f32, targets: []const []const f32, epochs: usize, learning_rate: f32, loss_fn: loss.Loss, scheduler: ?optimizer.LRScheduler, early_stopping: ?EarlyStopping) !void {
        const lr_scheduler = scheduler orelse optimizer.LRScheduler{ .constant = {} };

        // Early stopping state
        var best_loss: f32 = std.math.inf(f32);
        var best_weights: ?std.array_list.Managed(tensor.Tensor) = null;
        var epochs_without_improvement: usize = 0;
        const es = early_stopping;

        if (es != null and es.?.restore_best_weights) {
            // Pre-allocate storage for best weights
            best_weights = std.array_list.Managed(tensor.Tensor).init(self.allocator);
        }

        for (0..epochs) |epoch| {
            const current_lr = lr_scheduler.getLearningRate(learning_rate, epoch);
            var total_loss: f32 = 0;
            for (inputs, targets) |input, target| {
                total_loss += try self.trainStep(input, target, current_lr, loss_fn);
            }
            const avg_loss = total_loss / @as(f32, @floatFromInt(inputs.len));

            // Check early stopping
            if (es) |config| {
                const improvement = best_loss - avg_loss;
                if (improvement > config.min_delta) {
                    // Improvement found
                    best_loss = avg_loss;
                    epochs_without_improvement = 0;

                    // Save best weights if requested
                    if (config.restore_best_weights) {
                        // Clear previous best weights
                        if (best_weights) |*bw| {
                            for (bw.items) |*t| {
                                t.deinit();
                            }
                            bw.clearRetainingCapacity();
                        }

                        // Save current weights
                        if (best_weights) |*bw| {
                            for (self.layers.items) |l| {
                                const weights = l.getWeights();
                                const weights_copy = try tensor.Tensor.init(self.allocator, &.{weights.slice.len}, self.backend);
                                @memcpy(weights_copy.slice, weights.slice);
                                try bw.append(weights_copy);
                            }
                        }
                    }
                } else {
                    epochs_without_improvement += 1;
                    if (epochs_without_improvement >= config.patience) {
                        std.debug.print("Early stopping triggered after {} epochs\n", .{epoch + 1});

                        // Restore best weights if requested
                        if (config.restore_best_weights) {
                            if (best_weights) |*bw| {
                                var i: usize = 0;
                                for (self.layers.items) |l| {
                                    const layer_weights = l.getWeights();
                                    @memcpy(layer_weights.slice, bw.items[i].slice);
                                    i += 1;
                                }
                                std.debug.print("Restored weights from best epoch (loss: {d:.4})\n", .{best_loss});
                            }
                        }

                        // Clean up
                        if (best_weights) |*bw| {
                            for (bw.items) |*t| {
                                t.deinit();
                            }
                            bw.deinit();
                        }
                        return;
                    }
                }
            }

            if (epoch % 10 == 0 or epoch == epochs - 1) {
                std.debug.print("Epoch {}: Loss = {d:.4}, LR = {d:.6}\n", .{ epoch, avg_loss, current_lr });
            }
        }

        // Clean up best weights storage
        if (best_weights) |*bw| {
            for (bw.items) |*t| {
                t.deinit();
            }
            bw.deinit();
        }
    }

    /// Save the network to a file
    /// Returns the number of bytes written
    pub fn save(self: *Network, path: []const u8) !usize {
        const serialization = @import("serialization.zig");
        return try serialization.saveModel(self.layers.items, path);
    }

    /// Static method to load a network from a file
    /// Caller owns the returned Network and must call deinit()
    pub fn load(allocator: std.mem.Allocator, path: []const u8, backend_inst: backend_module.Backend) !*Network {
        const serialization = @import("serialization.zig");
        const loaded_layers = try serialization.loadModel(allocator, path, backend_inst);

        const self = try allocator.create(Network);
        errdefer allocator.destroy(self);

        self.allocator = allocator;
        self.backend = backend_inst;
        self.work_buffer = null;
        self.output_buffer = null;
        self.max_layer_size = 0;
        self.grad_work_1 = null;
        self.grad_work_2 = null;

        // Convert ArrayList to Managed
        self.layers = std.array_list.Managed(layer.Layer).init(allocator);
        try self.layers.appendSlice(loaded_layers.items);
        loaded_layers.deinit();

        self.caches = std.array_list.Managed(?LayerCache).init(allocator);
        try self.caches.resize(self.layers.items.len);
        for (self.caches.items) |*cache| {
            cache.* = null;
        }

        self.optimizers = std.array_list.Managed(?optimizer.Optimizer).init(allocator);
        try self.optimizers.resize(self.layers.items.len);
        for (self.optimizers.items) |*opt| {
            opt.* = null;
        }

        // Calculate max layer size
        for (self.layers.items) |l| {
            const size = l.inputSize();
            const out_size = l.outputSize();
            const max_size = @max(size, out_size);
            if (max_size > self.max_layer_size) {
                self.max_layer_size = max_size;
            }
        }

        // Pre-allocate work buffers
        try self.reallocateWorkBuffers(self.max_layer_size);

        return self;
    }
};

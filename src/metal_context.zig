/// Metal Context for managing persistent resources
/// Optimized for Apple Silicon GPU performance
const std = @import("std");
const metal = @import("metal.zig");

pub const MetalContext = struct {
    allocator: std.mem.Allocator,
    device: metal.MTLDevice,
    command_queue: metal.MTLCommandQueue,
    pipelines: std.StringHashMap(metal.MTLComputePipelineState),
    library: *anyopaque,
    active_command_buffer: ?metal.MTLCommandBuffer = null,
    temp_resources: std.ArrayListUnmanaged(metal.MTLBuffer),
    buffer_pool: std.AutoHashMap(usize, std.ArrayListUnmanaged(metal.MTLBuffer)),

    pub fn init(allocator: std.mem.Allocator) !*MetalContext {
        const self = try allocator.create(MetalContext);
        errdefer allocator.destroy(self);

        self.allocator = allocator;
        self.device = try metal.MTLDevice.create();
        errdefer self.device.release();

        self.command_queue = try self.device.newCommandQueue();
        errdefer self.command_queue.release();

        self.temp_resources = .{};
        errdefer self.temp_resources.deinit(self.allocator);

        self.buffer_pool = std.AutoHashMap(usize, std.ArrayListUnmanaged(metal.MTLBuffer)).init(allocator);
        errdefer {
            var it = self.buffer_pool.valueIterator();
            while (it.next()) |list| {
                for (list.items) |buf| buf.release();
                list.deinit(self.allocator);
            }
            self.buffer_pool.deinit();
        }

        // Load Metal shaders from source (runtime compilation)
        const shader_paths = [_][]const u8{
            "shaders/metal/matmul.metal",
            "shaders/metal/activation.metal",
            "shaders/metal/loss.metal",
            "shaders/metal/optimizer.metal",
            "shaders/metal/recurrent.metal",
            "shaders/metal/normalization.metal",
            "shaders/metal/convolution.metal",
            "shaders/metal/attention.metal",
            "shaders/metal/auxiliary.metal",
        };

        var sources = [_][]const u8{""} ** shader_paths.len;
        defer {
            for (sources) |s| {
                if (s.len > 0) allocator.free(s);
            }
        }

        for (shader_paths, 0..) |path, i| {
            const io = std.Io.Threaded.global_single_threaded.io();
            sources[i] = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, std.Io.Limit.limited(1024 * 1024));
        }

        const shader_source = try std.mem.concat(allocator, u8, &sources);
        // SECURITY FIX: Do NOT use defer here - newLibraryWithSource may retain the pointer internally
        // for asynchronous compilation. Free only after successful library creation.

        self.library = try self.device.newLibraryWithSource(shader_source);
        allocator.free(shader_source); // Free after confirmed use
        self.active_command_buffer = null;

        self.pipelines = std.StringHashMap(metal.MTLComputePipelineState).init(allocator);
        errdefer {
            var it = self.pipelines.valueIterator();
            while (it.next()) |p| p.release();
            self.pipelines.deinit();
        }

        // Pre-compile core kernels
        try self.registerPipeline("matmul");
        try self.registerPipeline("test_write");
        try self.registerPipeline("matmul_batch");
        try self.registerPipeline("matmul_tiled");
        try self.registerPipeline("matmul_transpose_a");
        try self.registerPipeline("matmul_transpose_b");
        try self.registerPipeline("matmul_batch_transpose_b");
        try self.registerPipeline("add_bias");

        // Activation kernels
        try self.registerPipeline("relu_forward");
        try self.registerPipeline("relu_backward");
        try self.registerPipeline("sigmoid_forward");
        try self.registerPipeline("sigmoid_backward");
        try self.registerPipeline("tanh_forward");
        try self.registerPipeline("tanh_backward");
        try self.registerPipeline("softmax_forward");
        try self.registerPipeline("softmax_backward");
        try self.registerPipeline("linear_forward");
        try self.registerPipeline("linear_backward");

        // Loss kernels
        try self.registerPipeline("mse_backward");
        try self.registerPipeline("cross_entropy_backward");
        try self.registerPipeline("binary_cross_entropy_backward");
        try self.registerPipeline("kl_divergence_backward");

        // Optimizer kernels
        try self.registerPipeline("sgd_update");
        try self.registerPipeline("sgd_update_bias");
        try self.registerPipeline("accumulate_bias");
        try self.registerPipeline("adam_update");
        try self.registerPipeline("rmsprop_update");

        // Normalization kernels
        try self.registerPipeline("layernorm_forward_optimized");
        try self.registerPipeline("layernorm_backward");
        try self.registerPipeline("batchnorm_forward_training");
        try self.registerPipeline("batchnorm_forward_inference");
        try self.registerPipeline("batchnorm_backward");

        // Convolution kernels
        try self.registerPipeline("conv1d_forward");
        try self.registerPipeline("conv1d_backward");
        try self.registerPipeline("conv2d_forward");
        try self.registerPipeline("conv2d_backward");
        try self.registerPipeline("conv2d_forward");
        try self.registerPipeline("conv2d_backward");

        // Attention kernels
        try self.registerPipeline("attention_forward");

        // Auxiliary kernels
        try self.registerPipeline("dropout_forward");
        try self.registerPipeline("vae_sampling_forward");
        try self.registerPipeline("vae_sampling_backward");
        try self.registerPipeline("fill_constant");
        try self.registerPipeline("scale_buffer");
        try self.registerPipeline("reverse_sequence");
        try self.registerPipeline("concat_buffers");
        try self.registerPipeline("split_buffer");

        // Recurrent kernels
        try self.registerPipeline("lstm_forward_step");
        try self.registerPipeline("gru_forward_step");
        try self.registerPipeline("lstm_backward_step");
        try self.registerPipeline("gru_backward_step");
        try self.registerPipeline("rnn_forward_step");
        try self.registerPipeline("rnn_backward_step");

        // Map kernels
        try self.registerPipeline("map_exp");
        try self.registerPipeline("map_log");
        try self.registerPipeline("map_sqrt");
        try self.registerPipeline("map_abs");
        try self.registerPipeline("map_square");
        try self.registerPipeline("map_inv");

        // Element-wise kernels
        try self.registerPipeline("ew_add");
        try self.registerPipeline("ew_sub");
        try self.registerPipeline("ew_mul");
        try self.registerPipeline("ew_div");

        // Random kernels
        try self.registerPipeline("fill_random_normal");

        return self;
    }

    pub fn deinit(self: *MetalContext) void {
        for (self.temp_resources.items) |res| res.release();
        self.temp_resources.deinit(self.allocator);

        var pool_it = self.buffer_pool.valueIterator();
        while (pool_it.next()) |list| {
            for (list.items) |buf| buf.release();
            list.deinit(self.allocator);
        }
        self.buffer_pool.deinit();

        var it = self.pipelines.valueIterator();
        while (it.next()) |p| p.release();
        self.pipelines.deinit();

        metal.objc.release(self.library);
        self.command_queue.release();
        self.device.release();
        self.allocator.destroy(self);
    }

    fn registerPipeline(self: *MetalContext, name: []const u8) !void {
        const name_z = try self.allocator.dupeZ(u8, name);
        defer self.allocator.free(name_z);

        const function = try metal.getFunctionFromLibrary(self.library, name_z);
        const pipeline = try self.device.newComputePipelineStateWithFunction(function);
        try self.pipelines.put(name, pipeline);
    }

    pub fn getPipeline(self: *const MetalContext, name: []const u8) ?*const metal.MTLComputePipelineState {
        return self.pipelines.getPtr(name);
    }

    pub fn getPipelineConfig(self: *const MetalContext, name: []const u8) !struct { threadsPerThreadgroup: metal.MTLSize, executionWidth: usize } {
        const pipeline = self.getPipeline(name) orelse return error.PipelineNotFound;
        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const width = pipeline.threadExecutionWidth();
        return .{
            .threadsPerThreadgroup = metal.MTLSize.make(max_threads, 1, 1),
            .executionWidth = width,
        };
    }

    pub fn getBuffer(self: *MetalContext, length: usize) !metal.MTLBuffer {
        // Use bucket-based pooling (power of two) to increase reuse
        const pooled_length = if (length == 0) 4 else std.math.ceilPowerOfTwo(usize, length) catch length;

        if (self.buffer_pool.getPtr(pooled_length)) |list| {
            if (list.items.len > 0) {
                return list.pop().?;
            }
        }

        // If not in pool, create a new one
        return try self.device.newBufferWithLength(
            pooled_length,
            .StorageModeShared
        );
    }

    pub fn returnBuffer(self: *MetalContext, buffer: metal.MTLBuffer) void {
        const length = buffer.length();
        const res = self.buffer_pool.getOrPut(length) catch {
            buffer.release();
            return;
        };
        if (!res.found_existing) {
            res.value_ptr.* = std.ArrayListUnmanaged(metal.MTLBuffer){};
        }
        res.value_ptr.append(self.allocator, buffer) catch {
            buffer.release();
        };
    }

    pub fn registerTempResource(self: *MetalContext, resource: metal.MTLBuffer) !void {
        try self.temp_resources.append(self.allocator, resource);
    }

    pub fn clearTempResources(self: *MetalContext) void {
        for (self.temp_resources.items) |res| {
            self.returnBuffer(res);
        }
        self.temp_resources.clearRetainingCapacity();
    }

    pub fn allocBuffer(self: *MetalContext, length: usize, options: metal.MTLResourceOptions) !metal.MTLBuffer {
        // Only pool shared buffers for now (common case)
        if (options.storage_mode == 0) {
            return self.getBuffer(length);
        }
        return try self.device.newBufferWithLength(length, options);
    }

    pub fn freeBuffer(self: *MetalContext, buffer: metal.MTLBuffer) void {
        self.returnBuffer(buffer);
    }
};

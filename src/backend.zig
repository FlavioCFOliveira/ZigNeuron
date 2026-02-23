/// Backend for neural network computation
/// Supports GPU (Metal/Vulkan) and CPU execution
/// GPU is the PRIORITY for all training and inference operations
const std = @import("std");
const activation = @import("activation.zig");
const loss = @import("loss.zig");
const vulkan_module = @import("vulkan.zig");
const metal = @import("metal.zig");
const metal_context = @import("metal_context.zig");

/// Available GPU backends in priority order
pub const GpuBackend = enum {
    /// Apple Silicon - Metal compute shaders
    metal,
    /// Cross-platform - Vulkan compute shaders
    vulkan,
};

/// Backend selection - GPU preferred, CPU fallback
pub const Backend = struct {
    type: BackendType,
    metal_ctx: ?*metal_context.MetalContext = null,

    pub const BackendType = union(enum) {
        gpu: GpuBackend,
        cpu,
    };

    pub fn init(allocator: std.mem.Allocator) !Backend {
        const detected = detect();
        var self = Backend{
            .type = detected,
            .metal_ctx = null,
        };

        if (detected == .gpu and detected.gpu == .metal) {
            self.metal_ctx = try metal_context.MetalContext.init(allocator);
        }

        return self;
    }

    pub fn deinit(self: *Backend) void {
        if (self.metal_ctx) |ctx| {
            ctx.deinit();
            self.metal_ctx = null;
        }
    }

    /// Returns the default backend type based on available hardware
    /// Priority: Metal (Apple Silicon) > Vulkan > CPU
    pub fn detect() BackendType {
        const os_tag = @import("builtin").os.tag;
        if (os_tag == .macos) {
            return .{ .gpu = .metal };
        }
        return .{ .cpu = {} };
    }

    /// Check if we're on macOS (helper)
    fn isMacos() bool {
        const os_tag = @import("builtin").os.tag;
        return os_tag == .macos;
    }

    /// Begin a batch of commands to be executed together (Metal only)
    pub fn beginCommandBatch(self: Backend) !void {
        if (self.metal_ctx) |ctx| {
            if (ctx.active_command_buffer == null) {
                ctx.active_command_buffer = try ctx.command_queue.commandBuffer();
            }
        }
    }

    /// End a batch of commands and execute them (Metal only)
    pub fn endCommandBatch(self: Backend) !void {
        if (self.metal_ctx) |ctx| {
            if (ctx.active_command_buffer) |*cb| {
                cb.commit();
                cb.waitUntilCompleted();
                ctx.active_command_buffer = null;
                ctx.clearTempResources();
            }
        }
    }

    /// Copy data between buffers (GPU or CPU)
    pub fn copyData(self: Backend,
        src: []const f32, src_buf: ?*const metal.MTLBuffer,
        dst: []f32, dst_buf: ?*const metal.MTLBuffer
    ) !void {
        if (src.len != dst.len) return error.BufferTooSmall;

        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalCopyData(src, src_buf, dst, dst_buf),
                .vulkan => if (dst.ptr != src.ptr) @memcpy(dst, src),
            },
            .cpu => if (dst.ptr != src.ptr) @memcpy(dst, src),
        }
    }

    fn metalCopyData(self: Backend,
        src: []const f32, src_buf: ?*const metal.MTLBuffer,
        dst: []f32, dst_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;

        if (src_buf != null and dst_buf != null) {
            // GPU to GPU copy
            var command_buffer = try self.getCommandBuffer();

            const buffer_src = try self.getBuffer(src, src_buf);
            defer if (src_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_src.buffer);
            const buffer_dst = try self.getBuffer(dst, dst_buf);
            defer if (dst_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_dst.buffer);

            // Use blit encoder for GPU to GPU copy
            var encoder = try command_buffer.blitCommandEncoder();
            encoder.copyBuffer(buffer_src.buffer, buffer_src.offset, buffer_dst.buffer, buffer_dst.offset, src.len * @sizeOf(f32));
            encoder.endEncoding();

            if (ctx.active_command_buffer == null) {
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
            }
        } else {
            // Fallback to CPU copy (works for Unified Memory if not in a batch)
            if (dst.ptr != src.ptr) {
                @memcpy(dst, src);
            }
        }
    }

    /// Execute matrix multiplication: C = A * B
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn matMul(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        m: usize, n: usize, k: usize
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMul(a, a_buf, b, b_buf, c, c_buf, m, n, k, false),
                .vulkan => try self.vulkanMatMulBatch(a, b, c, 1, n, k),
            },
            .cpu => cpuMatMul(a, b, c, m, n, k),
        }
    }

    /// Execute batched matrix multiplication
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn matMulBatch(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        batch_size: usize, n: usize, k: usize
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMulBatchGPU(a, a_buf, b, b_buf, c, c_buf, batch_size, n, k),
                .vulkan => try self.vulkanMatMulBatch(a, b, c, batch_size, n, k),
            },
            .cpu => cpuMatMulBatch(a, b, c, batch_size, n, k),
        }
    }

    /// Execute matrix multiplication with transposed A: C = A^T * B
    pub fn matMulTransposeA(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        m: usize, n: usize, k: usize
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMulTransposeA(a, a_buf, b, b_buf, c, c_buf, m, n, k),
                .vulkan => cpuMatMulTransposeA(a, b, c, m, n, k),
            },
            .cpu => cpuMatMulTransposeA(a, b, c, m, n, k),
        }
    }

    /// Execute matrix multiplication with transposed B: C = A * B^T
    pub fn matMulTransposeB(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        m: usize, n: usize, k: usize
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMul(a, a_buf, b, b_buf, c, c_buf, m, n, k, true),
                .vulkan => cpuMatMulTransposeB(a, b, c, m, n, k),
            },
            .cpu => cpuMatMulTransposeB(a, b, c, m, n, k),
        }
    }

    /// Execute activation function on the selected backend
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    /// Execute activation function on the selected backend
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn activationForward(self: Backend,
        act: activation.Activation,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        const batch_size = if (act == .softmax) 1 else input.len / 1; // Default to 1 for now or handle in caller
        _ = batch_size;
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalActivationForward(act, input, input_buf, output, output_buf),
                .vulkan => try self.vulkanActivationForward(act, input, output),
            },
            .cpu => {
                if (act == .softmax) {
                    try act.softmaxForward(input, output);
                } else {
                    for (input, 0..) |x, i| {
                        output[i] = act.forward(x);
                    }
                }
            },
        }
    }

    /// Execute activation backward pass
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn activationBackward(self: Backend,
        act: activation.Activation,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalActivationBackward(act, input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf),
                .vulkan => cpuActivationBackward(act, input, grad_output, grad_input),
            },
            .cpu => cpuActivationBackward(act, input, grad_output, grad_input),
        }
    }

    /// Execute loss function gradient on the selected backend
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn lossBackward(self: Backend,
        loss_fn: loss.Loss,
        output: []const f32, output_buf: ?*const metal.MTLBuffer,
        target: []const f32, target_buf: ?*const metal.MTLBuffer,
        grad_output: []f32, grad_output_buf: ?*const metal.MTLBuffer
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalLossBackward(loss_fn, output, output_buf, target, target_buf, grad_output, grad_output_buf),
                .vulkan => try self.cpuLossBackwardGeneric(loss_fn, output, target, grad_output),
            },
            .cpu => try self.cpuLossBackwardGeneric(loss_fn, output, target, grad_output),
        }
    }

    fn cpuLossBackwardGeneric(self: Backend, loss_fn: loss.Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        _ = self;
        try loss_fn.backward(output, target, grad_output);
    }

    /// Update weights using SGD
    pub fn sgdUpdate(self: Backend,
        weights: []f32, weights_buf: ?*const metal.MTLBuffer,
        gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer,
        learning_rate: f32, weight_decay: f32
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalSgdUpdate(weights, weights_buf, gradients, gradients_buf, learning_rate, weight_decay),
                .vulkan => cpuSgdUpdate(weights, gradients, learning_rate, weight_decay),
            },
            .cpu => cpuSgdUpdate(weights, gradients, learning_rate, weight_decay),
        }
    }

    /// Update bias using SGD
    pub fn sgdUpdateBias(self: Backend,
        bias: []f32, bias_buf: ?*const metal.MTLBuffer,
        gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer,
        learning_rate: f32
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalSgdUpdateBias(bias, bias_buf, gradients, gradients_buf, learning_rate),
                .vulkan => cpuSgdUpdateBias(bias, gradients, learning_rate),
            },
            .cpu => cpuSgdUpdateBias(bias, gradients, learning_rate),
        }
    }

    /// Accumulate bias gradients
    pub fn accumulateBias(self: Backend,
        grad_bias: []f32, grad_bias_buf: ?*const metal.MTLBuffer,
        grad_after_act: []const f32, grad_after_act_buf: ?*const metal.MTLBuffer
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalAccumulateBias(grad_bias, grad_bias_buf, grad_after_act, grad_after_act_buf),
                .vulkan => cpuAccumulateBias(grad_bias, grad_after_act),
            },
            .cpu => cpuAccumulateBias(grad_bias, grad_after_act),
        }
    }

    /// Add bias to output (broadcasted over batch)
    pub fn addBias(self: Backend,
        output: []f32, output_buf: ?*const metal.MTLBuffer,
        bias: []const f32, bias_buf: ?*const metal.MTLBuffer
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalAddBias(output, output_buf, bias, bias_buf),
                .vulkan => {
                    const batch_size = output.len / bias.len;
                    for (0..batch_size) |b| {
                        for (0..bias.len) |i| {
                            output[b * bias.len + i] += bias[i];
                        }
                    }
                },
            },
            .cpu => {
                const batch_size = output.len / bias.len;
                for (0..batch_size) |b| {
                    for (0..bias.len) |i| {
                        output[b * bias.len + i] += bias[i];
                    }
                }
            },
        }
    }

    /// Map a function over a buffer (exp, log, etc)
    pub fn map(self: Backend,
        func: enum { exp, log, sqrt, abs, square, inv },
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMap(func, input, input_buf, output, output_buf),
                .vulkan => try self.cpuMap(func, input, output),
            },
            .cpu => try self.cpuMap(func, input, output),
        }
    }

    fn cpuMap(self: Backend,
        func: enum { exp, log, sqrt, abs, square, inv },
        input: []const f32, output: []f32
    ) !void {
        _ = self;
        switch (func) {
            .exp => for (input, output) |in, *out| { out.* = std.math.exp(in); },
            .log => for (input, output) |in, *out| { out.* = std.math.log(f32, std.math.e, @max(in, 1e-10)); },
            .sqrt => for (input, output) |in, *out| { out.* = @sqrt(in); },
            .abs => for (input, output) |in, *out| { out.* = @abs(in); },
            .square => for (input, output) |in, *out| { out.* = in * in; },
            .inv => for (input, output) |in, *out| { out.* = 1.0 / in; },
        }
    }

    fn metalMap(self: Backend,
        func: enum { exp, log, sqrt, abs, square, inv },
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_input = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_input.buffer);
        var buffer_output = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_output.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline_name = switch (func) {
            .exp => "map_exp",
            .log => "map_log",
            .sqrt => "map_sqrt",
            .abs => "map_abs",
            .square => "map_square",
            .inv => "map_inv",
        };
        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_input.buffer, buffer_input.offset, 0);
        encoder.setBuffer(&buffer_output.buffer, buffer_output.offset, 1);

        const size = @as(u32, @intCast(input.len));
        encoder.setBytes(std.mem.asBytes(&size), 2);

        // Vectorized dispatch (4 elements per thread)
        const num_threads = (size + 3) / 4;
        const tg_size = @min(num_threads, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(num_threads, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    /// Element-wise operations: C = op(A, B)
    pub fn elementWise(self: Backend,
        op: enum { add, sub, mul, div },
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalElementWise(op, a, a_buf, b, b_buf, c, c_buf),
                .vulkan => try self.cpuElementWise(op, a, b, c),
            },
            .cpu => try self.cpuElementWise(op, a, b, c),
        }
    }

    fn cpuElementWise(self: Backend,
        op: enum { add, sub, mul, div },
        a: []const f32, b: []const f32, c: []f32
    ) !void {
        _ = self;
        switch (op) {
            .add => for (a, b, c) |av, bv, *cv| { cv.* = av + bv; },
            .sub => for (a, b, c) |av, bv, *cv| { cv.* = av - bv; },
            .mul => for (a, b, c) |av, bv, *cv| { cv.* = av * bv; },
            .div => for (a, b, c) |av, bv, *cv| { cv.* = av / bv; },
        }
    }

    fn metalElementWise(self: Backend,
        op: enum { add, sub, mul, div },
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_a = try self.getBuffer(a, a_buf);
        defer if (a_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_a.buffer);
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_b.buffer);
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_c.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline_name = switch (op) {
            .add => "ew_add",
            .sub => "ew_sub",
            .mul => "ew_mul",
            .div => "ew_div",
        };
        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a.buffer, buffer_a.offset, 0);
        encoder.setBuffer(&buffer_b.buffer, buffer_b.offset, 1);
        encoder.setBuffer(&buffer_c.buffer, buffer_c.offset, 2);

        const size = @as(u32, @intCast(a.len));
        encoder.setBytes(std.mem.asBytes(&size), 3);

        // Vectorized dispatch (4 elements per thread)
        const num_threads = (size + 3) / 4;
        const tg_size = @min(num_threads, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(num_threads, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    /// Fill buffer with random normal values
    pub fn fillRandomNormal(self: Backend,
        data: []f32, data_buf: ?*const metal.MTLBuffer,
        mean: f32, std_dev: f32, seed: u64
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalFillRandomNormal(data, data_buf, mean, std_dev, seed),
                .vulkan => self.cpuFillRandomNormal(data, mean, std_dev, seed),
            },
            .cpu => self.cpuFillRandomNormal(data, mean, std_dev, seed),
        }
    }

    fn cpuFillRandomNormal(self: Backend, data: []f32, mean: f32, std_dev: f32, seed: u64) void {
        _ = self;
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (data) |*v| {
            v.* = random.floatNorm(f32) * std_dev + mean;
        }
    }

    fn metalFillRandomNormal(self: Backend,
        data: []f32, data_buf: ?*const metal.MTLBuffer,
        mean: f32, std_dev: f32, seed: u64
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_data = try self.getBuffer(data, data_buf);
        defer if (data_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_data.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("fill_random_normal") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_data.buffer, buffer_data.offset, 0);

        encoder.setBytes(std.mem.asBytes(&mean), 1);
        encoder.setBytes(std.mem.asBytes(&std_dev), 2);
        encoder.setBytes(std.mem.asBytes(&seed), 3);
        const size = @as(u32, @intCast(data.len));
        encoder.setBytes(std.mem.asBytes(&size), 4);

        const tg_size = @min(data.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(data.len, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalAddBias(self: Backend,
        output: []f32, output_buf: ?*const metal.MTLBuffer,
        bias: []const f32, bias_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_output = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_output.buffer);
        var buffer_bias = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_bias.buffer);

        const bias_size = @as(u32, @intCast(bias.len));
        const batch_size = @as(u32, @intCast(output.len / bias.len));

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("add_bias") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_output.buffer, buffer_output.offset, 0);
        encoder.setBuffer(&buffer_bias.buffer, buffer_bias.offset, 1);
        encoder.setBytes(std.mem.asBytes(&batch_size), 2);
        encoder.setBytes(std.mem.asBytes(&bias_size), 3);

        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const width = pipeline.threadExecutionWidth();
        const tg_x = @min(@as(usize, bias_size), width);
        const tg_y = @min(@as(usize, batch_size), max_threads / tg_x);
        encoder.dispatchThreads(metal.MTLSize.make(bias_size, batch_size, 1), metal.MTLSize.make(tg_x, tg_y, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    // ================== Metal implementations ==================

    fn metalMatMul(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        m: usize, n: usize, k: usize,
        transpose_b: bool
    ) !void {
        if (!isMacos()) return error.NotAvailable;
        try self.metalMatMulGPU(a, a_buf, b, b_buf, c, c_buf, m, n, k, transpose_b);
    }

    fn metalMatMulBatch(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        batch_size: usize, n: usize, k: usize
    ) !void {
        if (!isMacos()) return error.NotAvailable;
        try self.metalMatMulBatchGPU(a, a_buf, b, b_buf, c, c_buf, batch_size, n, k);
    }

    fn metalMatMulTransposeA(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        m: usize, n: usize, k: usize
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_a = try self.getBuffer(a, a_buf);
        defer if (a_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_a.buffer);
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_b.buffer);
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_c.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("matmul_transpose_a") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a.buffer, buffer_a.offset, 0);
        encoder.setBuffer(&buffer_b.buffer, buffer_b.offset, 1);
        encoder.setBuffer(&buffer_c.buffer, buffer_c.offset, 2);

        const m_u32 = @as(u32, @intCast(m));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        encoder.setBytes(std.mem.asBytes(&m_u32), 3);
        encoder.setBytes(std.mem.asBytes(&n_u32), 4);
        encoder.setBytes(std.mem.asBytes(&k_u32), 5);

        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const width = pipeline.threadExecutionWidth();
        const tg_x = @min(n, width);
        const tg_y = @min(m, max_threads / tg_x);
        encoder.dispatchThreads(metal.MTLSize.make(n, m, 1), metal.MTLSize.make(tg_x, tg_y, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    const BufferWithOffset = struct {
        buffer: metal.MTLBuffer,
        offset: usize,
    };

    // Helper to get buffer, either from provided or create temporary
    fn getBuffer(self: Backend, slice: anytype, provided_buf: ?*const metal.MTLBuffer) !BufferWithOffset {
        if (provided_buf) |buf| {
            const slice_ptr = @intFromPtr(slice.ptr);
            const buf_ptr = @intFromPtr(buf.contents());
            const offset = if (slice_ptr >= buf_ptr) slice_ptr - buf_ptr else 0;
            return .{ .buffer = buf.*, .offset = offset };
        }
        const ctx = self.metal_ctx.?;
        const bytes = std.mem.sliceAsBytes(slice);
        const buffer = try ctx.allocBuffer(bytes.len, .StorageModeShared);
        @memcpy(buffer.contents()[0..bytes.len], bytes);
        if (ctx.active_command_buffer != null) {
            try ctx.registerTempResource(buffer);
        }
        return .{ .buffer = buffer, .offset = 0 };
    }

    fn releaseBuffer(self: Backend, buffer: metal.MTLBuffer) void {
        if (self.metal_ctx) |ctx| {
            ctx.freeBuffer(buffer);
        } else {
            buffer.release();
        }
    }

    fn getCommandBuffer(self: Backend) !metal.MTLCommandBuffer {
        const ctx = self.metal_ctx.?;
        if (ctx.active_command_buffer) |cb| return cb;
        return try ctx.command_queue.commandBuffer();
    }

    fn metalMatMulGPU(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        m: usize, n: usize, k: usize,
        transpose_b: bool
    ) !void {
        const ctx = self.metal_ctx.?;
        var buffer_a = try self.getBuffer(a, a_buf);
        defer if (a_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_a.buffer);
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_b.buffer);
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_c.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        var pipeline_name: []const u8 = "matmul";
        if (transpose_b) {
            pipeline_name = "matmul_transpose_b";
        } else if (m % 16 == 0 and n % 16 == 0 and k % 16 == 0) {
            pipeline_name = "matmul_tiled";
        }

        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a.buffer, buffer_a.offset, 0);
        encoder.setBuffer(&buffer_b.buffer, buffer_b.offset, 1);
        encoder.setBuffer(&buffer_c.buffer, buffer_c.offset, 2);

        const m_u32 = @as(u32, @intCast(m));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        encoder.setBytes(std.mem.asBytes(&m_u32), 3);
        encoder.setBytes(std.mem.asBytes(&n_u32), 4);
        encoder.setBytes(std.mem.asBytes(&k_u32), 5);

        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const width = pipeline.threadExecutionWidth();
        const tg_x = @min(n, width);
        const tg_y = @min(m, max_threads / tg_x);
        encoder.dispatchThreads(metal.MTLSize.make(n, m, 1), metal.MTLSize.make(tg_x, tg_y, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalMatMulBatchGPU(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        batch_size: usize, n: usize, k: usize
    ) !void {
        const ctx = self.metal_ctx.?;
        var buffer_a = try self.getBuffer(a, a_buf);
        defer if (a_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_a.buffer);
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_b.buffer);
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_c.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        const pipeline = ctx.getPipeline("matmul_batch") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a.buffer, buffer_a.offset, 0);
        encoder.setBuffer(&buffer_b.buffer, buffer_b.offset, 1);
        encoder.setBuffer(&buffer_c.buffer, buffer_c.offset, 2);

        const batch_size_u32 = @as(u32, @intCast(batch_size));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        encoder.setBytes(std.mem.asBytes(&batch_size_u32), 3);
        encoder.setBytes(std.mem.asBytes(&n_u32), 4);
        encoder.setBytes(std.mem.asBytes(&k_u32), 5);

        encoder.dispatchThreads(metal.MTLSize.make(n, 1, batch_size), metal.MTLSize.make(16, 1, 16));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalActivationForward(self: Backend,
        act: activation.Activation,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx.?;
        var buffer_input = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_input.buffer);
        var buffer_output = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_output.buffer);

        const pipeline_name = switch (act) {
            .relu => "relu_forward",
            .sigmoid => "sigmoid_forward",
            .tanh => "tanh_forward",
            .softmax => "softmax_forward",
            .linear => "linear_forward",
        };
        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_input.buffer, buffer_input.offset, 0);
        encoder.setBuffer(&buffer_output.buffer, buffer_output.offset, 1);

        const size = @as(u32, @intCast(input.len));
        if (act == .softmax) {
            const num_classes = size;
            const batch_size: u32 = 1;
            encoder.setBytes(std.mem.asBytes(&batch_size), 2);
            encoder.setBytes(std.mem.asBytes(&num_classes), 3);
            const tg_config = try ctx.getPipelineConfig(pipeline_name);
            const width = tg_config.executionWidth;
            // One threadgroup per sample
            const optimized_tg_size = (num_classes + width - 1) / width * width;
            const final_tg_size = @min(optimized_tg_size, tg_config.threadsPerThreadgroup.width);
            encoder.dispatchThreads(metal.MTLSize.make(final_tg_size, 1, 1), metal.MTLSize.make(final_tg_size, 1, 1));
        } else {
            encoder.setBytes(std.mem.asBytes(&size), 2);
            // Vectorized dispatch (4 elements per thread)
            const num_threads = (size + 3) / 4;
            const tg_size = @min(num_threads, pipeline.maxTotalThreadsPerThreadgroup());
            encoder.dispatchThreads(metal.MTLSize.make(num_threads, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        }
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalActivationBackward(self: Backend,
        act: activation.Activation,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx.?;
        var buffer_input = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_input.buffer);
        var buffer_grad_output = try self.getBuffer(grad_output, grad_output_buf);
        defer if (grad_output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_grad_output.buffer);
        var buffer_grad_input = try self.getBuffer(grad_input, grad_input_buf);
        defer if (grad_input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_grad_input.buffer);

        const pipeline_name = switch (act) {
            .relu => "relu_backward",
            .sigmoid => "sigmoid_backward",
            .tanh => "tanh_backward",
            .softmax => "softmax_backward",
            .linear => "linear_backward",
        };
        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_input.buffer, buffer_input.offset, 0);
        encoder.setBuffer(&buffer_grad_output.buffer, buffer_grad_output.offset, 1);
        encoder.setBuffer(&buffer_grad_input.buffer, buffer_grad_input.offset, 2);

        const size = @as(u32, @intCast(input.len));
        if (act == .softmax) {
            const num_classes = size;
            const batch_size: u32 = 1;
            encoder.setBytes(std.mem.asBytes(&batch_size), 3);
            encoder.setBytes(std.mem.asBytes(&num_classes), 4);
            const tg_config = try ctx.getPipelineConfig(pipeline_name);
            const width = tg_config.executionWidth;
            // One threadgroup per sample
            const optimized_tg_size = (num_classes + width - 1) / width * width;
            const final_tg_size = @min(optimized_tg_size, tg_config.threadsPerThreadgroup.width);
            encoder.dispatchThreads(metal.MTLSize.make(final_tg_size, 1, 1), metal.MTLSize.make(final_tg_size, 1, 1));
        } else {
            encoder.setBytes(std.mem.asBytes(&size), 3);
            // Vectorized dispatch (4 elements per thread)
            const num_threads = (size + 3) / 4;
            const tg_size = @min(num_threads, pipeline.maxTotalThreadsPerThreadgroup());
            encoder.dispatchThreads(metal.MTLSize.make(num_threads, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        }
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalLossBackward(self: Backend,
        loss_fn: loss.Loss,
        output: []const f32, output_buf: ?*const metal.MTLBuffer,
        target: []const f32, target_buf: ?*const metal.MTLBuffer,
        grad_output: []f32, grad_output_buf: ?*const metal.MTLBuffer
    ) !void {
        if (!isMacos()) return error.NotAvailable;
        try self.metalLossBackwardGPU(loss_fn, output, output_buf, target, target_buf, grad_output, grad_output_buf);
    }

    fn metalLossBackwardGPU(self: Backend,
        loss_fn: loss.Loss,
        output: []const f32, output_buf: ?*const metal.MTLBuffer,
        target: []const f32, target_buf: ?*const metal.MTLBuffer,
        grad_output: []f32, grad_output_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx.?;
        var buffer_output = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_output.buffer);

        var buffer_target_opt: ?BufferWithOffset = null;
        if (loss_fn != .kl_divergence) {
            buffer_target_opt = try self.getBuffer(target, target_buf);
        }
        defer if (buffer_target_opt) |bt| {
            if (target_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(bt.buffer);
        };

        var buffer_grad_output = try self.getBuffer(grad_output, grad_output_buf);
        defer if (grad_output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_grad_output.buffer);

        const pipeline_name = switch (loss_fn) {
            .mse => "mse_backward",
            .cross_entropy => "cross_entropy_backward",
            .cross_entropy_logits => "cross_entropy_backward",
            .binary_cross_entropy => "binary_cross_entropy_backward",
            .kl_divergence => "kl_divergence_backward",
        };
        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_output.buffer, buffer_output.offset, 0);
        if (buffer_target_opt) |bt| {
            encoder.setBuffer(&bt.buffer, bt.offset, 1);
        }
        encoder.setBuffer(&buffer_grad_output.buffer, buffer_grad_output.offset, 2);

        const size = @as(u32, @intCast(output.len));
        if (loss_fn == .cross_entropy or loss_fn == .cross_entropy_logits) {
            const num_samples: u32 = 1;
            const num_classes = size;
            encoder.setBytes(std.mem.asBytes(&num_samples), 3);
            encoder.setBytes(std.mem.asBytes(&num_classes), 4);
            const tg_config = try ctx.getPipelineConfig(pipeline_name);
            const width = tg_config.executionWidth;
            // One threadgroup per sample
            const optimized_tg_size = (num_classes + width - 1) / width * width;
            const final_tg_size = @min(optimized_tg_size, tg_config.threadsPerThreadgroup.width);
            encoder.dispatchThreads(metal.MTLSize.make(final_tg_size, 1, 1), metal.MTLSize.make(final_tg_size, 1, 1));
        } else if (loss_fn == .kl_divergence) {
            const n: u32 = size / 2;
            encoder.setBytes(std.mem.asBytes(&n), 3);
            const tg_size = @min(output.len, pipeline.maxTotalThreadsPerThreadgroup());
            encoder.dispatchThreads(metal.MTLSize.make(output.len, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        } else {
            const n: u32 = size;
            encoder.setBytes(std.mem.asBytes(&size), 3);
            encoder.setBytes(std.mem.asBytes(&n), 4);
            // Vectorized dispatch (4 elements per thread)
            const num_threads = (size + 3) / 4;
            const tg_size = @min(num_threads, pipeline.maxTotalThreadsPerThreadgroup());
            encoder.dispatchThreads(metal.MTLSize.make(num_threads, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        }
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalSgdUpdate(self: Backend,
        weights: []f32, weights_buf: ?*const metal.MTLBuffer,
        gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer,
        learning_rate: f32, weight_decay: f32
    ) !void {
        const ctx = self.metal_ctx.?;
        var buffer_weights = try self.getBuffer(weights, weights_buf);
        defer if (weights_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_weights.buffer);
        var buffer_gradients = try self.getBuffer(gradients, gradients_buf);
        defer if (gradients_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gradients.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("sgd_update") orelse return error.PipelineNotFound;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_weights.buffer, buffer_weights.offset, 0);
        encoder.setBuffer(&buffer_gradients.buffer, buffer_gradients.offset, 1);
        encoder.setBytes(std.mem.asBytes(&learning_rate), 2);
        encoder.setBytes(std.mem.asBytes(&weight_decay), 3);
        const size = @as(u32, @intCast(weights.len));
        encoder.setBytes(std.mem.asBytes(&size), 4);

        const tg_size = @min(weights.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(weights.len, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalSgdUpdateBias(self: Backend,
        bias: []f32, bias_buf: ?*const metal.MTLBuffer,
        gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer,
        learning_rate: f32
    ) !void {
        const ctx = self.metal_ctx.?;
        var buffer_bias = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_bias.buffer);
        var buffer_gradients = try self.getBuffer(gradients, gradients_buf);
        defer if (gradients_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gradients.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("sgd_update_bias") orelse return error.PipelineNotFound;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_bias.buffer, buffer_bias.offset, 0);
        encoder.setBuffer(&buffer_gradients.buffer, buffer_gradients.offset, 1);
        encoder.setBytes(std.mem.asBytes(&learning_rate), 2);
        const size = @as(u32, @intCast(bias.len));
        encoder.setBytes(std.mem.asBytes(&size), 3);

        const tg_size = @min(bias.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(bias.len, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalAccumulateBias(self: Backend,
        grad_bias: []f32, grad_bias_buf: ?*const metal.MTLBuffer,
        grad_after_act: []const f32, grad_after_act_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx.?;
        var buffer_grad_bias = try self.getBuffer(grad_bias, grad_bias_buf);
        defer if (grad_bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_grad_bias.buffer);
        var buffer_grad_after_act = try self.getBuffer(grad_after_act, grad_after_act_buf);
        defer if (grad_after_act_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_grad_after_act.buffer);

        const bias_size = @as(u32, @intCast(grad_bias.len));
        const batch_size = @as(u32, @intCast(grad_after_act.len / grad_bias.len));

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("accumulate_bias") orelse return error.PipelineNotFound;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_grad_bias.buffer, buffer_grad_bias.offset, 0);
        encoder.setBuffer(&buffer_grad_after_act.buffer, buffer_grad_after_act.offset, 1);
        encoder.setBytes(std.mem.asBytes(&batch_size), 2);
        encoder.setBytes(std.mem.asBytes(&bias_size), 3);

        const tg_size = @min(grad_bias.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(grad_bias.len, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    // ================== Vulkan implementations ==================

    fn vulkanMatMulBatch(self: Backend, a: []const f32, b: []const f32, c: []f32, batch_size: usize, n: usize, k: usize) !void {
        _ = self;
        cpuMatMulBatch(a, b, c, batch_size, n, k);
    }

    fn vulkanActivationForward(self: Backend, act: activation.Activation, input: []const f32, output: []f32) !void {
        _ = self;
        if (act == .softmax) {
            try act.softmaxForward(input, output);
        } else {
            for (input, 0..) |x, i| {
                output[i] = act.forward(x);
            }
        }
    }

    // ================== CPU implementations ==================

    fn cpuMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
        @memset(c, 0);
        const block_size = 32;
        if (m >= block_size and n >= block_size and k >= block_size) {
            var ii: usize = 0;
            while (ii < m) : (ii += block_size) {
                var kk: usize = 0;
                while (kk < k) : (kk += block_size) {
                    var jj: usize = 0;
                    while (jj < n) : (jj += block_size) {
                        const i_end = @min(ii + block_size, m);
                        const k_end = @min(kk + block_size, k);
                        const j_end = @min(jj + block_size, n);

                        for (ii..i_end) |i| {
                            for (kk..k_end) |p| {
                                const a_val = a[i * k + p];
                                const b_row = b[p * n ..];
                                const c_row = c[i * n ..];
                                for (jj..j_end) |j| {
                                    c_row[j] += a_val * b_row[j];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for (0..m) |i| {
                for (0..k) |p| {
                    const a_val = a[i * k + p];
                    for (0..n) |j| {
                        c[i * n + j] += a_val * b[p * n + j];
                    }
                }
            }
        }
    }

    fn cpuMatMulBatch(a: []const f32, b: []const f32, c: []f32, batch_size: usize, n: usize, k: usize) void {
        for (0..batch_size) |i| {
            cpuMatMul(a[i * k ..], b, c[i * n ..], 1, n, k);
        }
    }

    fn cpuMatMulTransposeA(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
        @memset(c, 0);
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0.0;
                for (0..k) |p| {
                    sum += a[p * m + i] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    fn cpuMatMulTransposeB(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
        @memset(c, 0);
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0.0;
                for (0..k) |p| {
                    sum += a[i * k + p] * b[j * k + p];
                }
                c[i * n + j] = sum;
            }
        }
    }

    fn cpuActivationBackward(act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) void {
        if (act == .softmax) {
            // Softmax backward requires per-sample processing
            // We need to know the number of classes.
            // Since we don't pass it, we assume the whole input is one sample
            // UNLESS we update the interface.
            // For now, let's assume we need to loop if input.len > some_heuristic or we just need the size.
            // Actually, for softmax, input is the ACTIVATED output (probabilities).
            // We can't know num_classes without it being passed.
            // Let's assume it's one sample for now, as it was before,
            // but the Network should really pass the sample size.
            act.softmaxBackward(input, grad_output, grad_input) catch @panic("Softmax backward failed");
        } else {
            for (input, 0..) |y, i| {
                grad_input[i] = act.backward(y, grad_output[i]);
            }
        }
    }

    fn cpuLossBackward(loss_fn: loss.Loss, output: []const f32, target: []const f32, grad_output: []f32) void {
        const n: f32 = @floatFromInt(output.len);
        switch (loss_fn) {
            .mse => {
                for (output, target, grad_output) |p, t, *go| {
                    go.* = 2.0 * (p - t) / n;
                }
            },
            .binary_cross_entropy => {
                for (output, target, grad_output) |p, t, *go| {
                    go.* = (p - t) / n;
                }
            },
            .cross_entropy, .cross_entropy_logits => {
                for (output, target, grad_output) |p, t, *go| {
                    go.* = p - t;
                }
            },
        }
    }

    fn cpuSgdUpdate(weights: []f32, gradients: []const f32, learning_rate: f32, weight_decay: f32) void {
        const max_grad: f32 = 5.0;
        const min_grad: f32 = -5.0;
        const max_weight: f32 = 100.0;
        const min_weight: f32 = -100.0;

        for (weights, gradients) |*w, g_raw| {
            var g = g_raw;
            if (std.math.isNan(g)) {
                g = 0.0;
            } else {
                g = @min(max_grad, @max(min_grad, g));
            }

            w.* -= learning_rate * (g + weight_decay * w.*);
            w.* = @min(max_weight, @max(min_weight, w.*));
        }
    }

    fn cpuSgdUpdateBias(bias: []f32, gradients: []const f32, learning_rate: f32) void {
        const max_grad: f32 = 5.0;
        const min_grad: f32 = -5.0;
        const max_bias: f32 = 50.0;
        const min_bias: f32 = -50.0;

        for (bias, gradients) |*b, g_raw| {
            var g = g_raw;
            if (std.math.isNan(g)) {
                g = 0.0;
            } else {
                g = @min(max_grad, @max(min_grad, g));
            }

            b.* -= learning_rate * g;
            b.* = @min(max_bias, @max(min_bias, b.*));
        }
    }

    fn cpuAccumulateBias(grad_bias: []f32, grad_after_act: []const f32) void {
        const batch_size = grad_after_act.len / grad_bias.len;
        for (0..batch_size) |b| {
            const gaa_slice = grad_after_act[b * grad_bias.len .. (b + 1) * grad_bias.len];
            for (grad_bias, gaa_slice) |*gb, gaa| {
                gb.* += gaa;
            }
        }
    }

    /// GRU forward step
    pub fn gruForwardStep(self: Backend,
        gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer,
        gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer,
        bias: []const f32, bias_buf: ?*const metal.MTLBuffer,
        h_prev: []const f32, h_prev_buf: ?*const metal.MTLBuffer,
        h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer,
        gate_acts: []f32, gate_acts_buf: ?*const metal.MTLBuffer,
        n_hh_out: []f32, n_hh_out_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalGruForwardStep(gates_ih, gates_ih_buf, gates_hh, gates_hh_buf, bias, bias_buf, h_prev, h_prev_buf, h_curr, h_curr_buf, gate_acts, gate_acts_buf, n_hh_out, n_hh_out_buf, hidden_size),
                .vulkan => return error.NotAvailable,
            },
            .cpu => {
                // CPU implementation
                for (0..hidden_size) |j| {
                    const z = 1.0 / (1.0 + std.math.exp(-(gates_ih[j] + gates_hh[j] + bias[j])));
                    const r = 1.0 / (1.0 + std.math.exp(-(gates_ih[hidden_size + j] + gates_hh[hidden_size + j] + bias[hidden_size + j])));
                    const n_hh = gates_hh[2 * hidden_size + j];
                    const n = std.math.tanh(gates_ih[2 * hidden_size + j] + bias[2 * hidden_size + j] + r * n_hh);

                    h_curr[j] = (1.0 - z) * n + z * h_prev[j];

                    // Store for backward
                    gate_acts[j] = z;
                    gate_acts[hidden_size + j] = r;
                    gate_acts[2 * hidden_size + j] = n;
                    n_hh_out[j] = n_hh;
                }
            },
        }
    }

    /// Vanilla RNN forward step
    pub fn rnnForwardStep(self: Backend,
        gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer,
        gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer,
        bias: []const f32, bias_buf: ?*const metal.MTLBuffer,
        h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalRnnForwardStep(gates_ih, gates_ih_buf, gates_hh, gates_hh_buf, bias, bias_buf, h_curr, h_curr_buf, hidden_size),
                .vulkan => return error.NotAvailable,
            },
            .cpu => {
                for (0..hidden_size) |j| {
                    h_curr[j] = std.math.tanh(gates_ih[j] + gates_hh[j] + bias[j]);
                }
            },
        }
    }

    /// Vanilla RNN backward step
    pub fn rnnBackwardStep(self: Backend,
        grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer,
        grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer,
        h_curr: []const f32, h_curr_buf: ?*const metal.MTLBuffer,
        grad_after_act: []f32, grad_after_act_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalRnnBackwardStep(grad_h_curr, grad_h_curr_buf, grad_h_next, grad_h_next_buf, h_curr, h_curr_buf, grad_after_act, grad_after_act_buf, hidden_size),
                .vulkan => return error.NotAvailable,
            },
            .cpu => {
                for (0..hidden_size) |j| {
                    const gh = grad_h_curr[j] + grad_h_next[j];
                    const h = h_curr[j];
                    grad_after_act[j] = gh * (1.0 - h * h);
                }
            },
        }
    }

    /// LSTM forward step
    pub fn lstmForwardStep(self: Backend,
        gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer,
        gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer,
        bias: []const f32, bias_buf: ?*const metal.MTLBuffer,
        c_prev: []const f32, c_prev_buf: ?*const metal.MTLBuffer,
        c_curr: []f32, c_curr_buf: ?*const metal.MTLBuffer,
        h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer,
        gate_acts: []f32, gate_acts_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalLstmForwardStep(gates_ih, gates_ih_buf, gates_hh, gates_hh_buf, bias, bias_buf, c_prev, c_prev_buf, c_curr, c_curr_buf, h_curr, h_curr_buf, gate_acts, gate_acts_buf, hidden_size),
                .vulkan => return error.NotAvailable,
            },
            .cpu => {
                for (0..hidden_size) |j| {
                    const i = 1.0 / (1.0 + std.math.exp(-(gates_ih[j] + gates_hh[j] + bias[j])));
                    const f = 1.0 / (1.0 + std.math.exp(-(gates_ih[hidden_size + j] + gates_hh[hidden_size + j] + bias[hidden_size + j])));
                    const g = std.math.tanh(gates_ih[2 * hidden_size + j] + gates_hh[2 * hidden_size + j] + bias[2 * hidden_size + j]);
                    const o = 1.0 / (1.0 + std.math.exp(-(gates_ih[3 * hidden_size + j] + gates_hh[3 * hidden_size + j] + bias[3 * hidden_size + j])));

                    const cc = f * c_prev[j] + i * g;
                    c_curr[j] = cc;
                    h_curr[j] = o * std.math.tanh(cc);

                    gate_acts[j] = i;
                    gate_acts[hidden_size + j] = f;
                    gate_acts[2 * hidden_size + j] = g;
                    gate_acts[3 * hidden_size + j] = o;
                }
            },
        }
    }

    fn metalLstmForwardStep(self: Backend,
        gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer,
        gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer,
        bias: []const f32, bias_buf: ?*const metal.MTLBuffer,
        c_prev: []const f32, c_prev_buf: ?*const metal.MTLBuffer,
        c_curr: []f32, c_curr_buf: ?*const metal.MTLBuffer,
        h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer,
        gate_acts: []f32, gate_acts_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_ih = try self.getBuffer(gates_ih, gates_ih_buf);
        defer if (gates_ih_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ih.buffer);
        var buffer_hh = try self.getBuffer(gates_hh, gates_hh_buf);
        defer if (gates_hh_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_hh.buffer);
        var buffer_bias = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_bias.buffer);
        var buffer_cp = try self.getBuffer(c_prev, c_prev_buf);
        defer if (c_prev_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_cp.buffer);
        var buffer_cc = try self.getBuffer(c_curr, c_curr_buf);
        defer if (c_curr_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_cc.buffer);
        var buffer_hc = try self.getBuffer(h_curr, h_curr_buf);
        defer if (h_curr_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_hc.buffer);
        var buffer_ga = try self.getBuffer(gate_acts, gate_acts_buf);
        defer if (gate_acts_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ga.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("lstm_forward_step") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_ih.buffer, buffer_ih.offset, 0);
        encoder.setBuffer(&buffer_hh.buffer, buffer_hh.offset, 1);
        encoder.setBuffer(&buffer_bias.buffer, buffer_bias.offset, 2);
        encoder.setBuffer(&buffer_cp.buffer, buffer_cp.offset, 3);
        encoder.setBuffer(&buffer_cc.buffer, buffer_cc.offset, 4);
        encoder.setBuffer(&buffer_hc.buffer, buffer_hc.offset, 5);
        encoder.setBuffer(&buffer_ga.buffer, buffer_ga.offset, 6);

        const h_u32 = @as(u32, @intCast(hidden_size));
        encoder.setBytes(std.mem.asBytes(&h_u32), 7);

        const tg_size = @min(hidden_size, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(hidden_size, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalRnnForwardStep(self: Backend,
        gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer,
        gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer,
        bias: []const f32, bias_buf: ?*const metal.MTLBuffer,
        h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_ih = try self.getBuffer(gates_ih, gates_ih_buf);
        defer if (gates_ih_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ih.buffer);
        var buffer_hh = try self.getBuffer(gates_hh, gates_hh_buf);
        defer if (gates_hh_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_hh.buffer);
        var buffer_bias = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_bias.buffer);
        var buffer_hc = try self.getBuffer(h_curr, h_curr_buf);
        defer if (h_curr_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_hc.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("rnn_forward_step") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_ih.buffer, buffer_ih.offset, 0);
        encoder.setBuffer(&buffer_hh.buffer, buffer_hh.offset, 1);
        encoder.setBuffer(&buffer_bias.buffer, buffer_bias.offset, 2);
        encoder.setBuffer(&buffer_hc.buffer, buffer_hc.offset, 3);

        const h_u32 = @as(u32, @intCast(hidden_size));
        encoder.setBytes(std.mem.asBytes(&h_u32), 4);

        const tg_size = @min(hidden_size, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(hidden_size, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalRnnBackwardStep(self: Backend,
        grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer,
        grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer,
        h_curr: []const f32, h_curr_buf: ?*const metal.MTLBuffer,
        grad_after_act: []f32, grad_after_act_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_ghc = try self.getBuffer(grad_h_curr, grad_h_curr_buf);
        defer if (grad_h_curr_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ghc.buffer);
        var buffer_ghn = try self.getBuffer(grad_h_next, grad_h_next_buf);
        defer if (grad_h_next_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ghn.buffer);
        var buffer_hc = try self.getBuffer(h_curr, h_curr_buf);
        defer if (h_curr_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_hc.buffer);
        var buffer_gaa = try self.getBuffer(grad_after_act, grad_after_act_buf);
        defer if (grad_after_act_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gaa.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("rnn_backward_step") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_ghc.buffer, buffer_ghc.offset, 0);
        encoder.setBuffer(&buffer_ghn.buffer, buffer_ghn.offset, 1);
        encoder.setBuffer(&buffer_hc.buffer, buffer_hc.offset, 2);
        encoder.setBuffer(&buffer_gaa.buffer, buffer_gaa.offset, 3);

        const h_u32 = @as(u32, @intCast(hidden_size));
        encoder.setBytes(std.mem.asBytes(&h_u32), 4);

        const tg_size = @min(hidden_size, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(hidden_size, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalGruForwardStep(self: Backend,
        gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer,
        gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer,
        bias: []const f32, bias_buf: ?*const metal.MTLBuffer,
        h_prev: []const f32, h_prev_buf: ?*const metal.MTLBuffer,
        h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer,
        gate_acts: []f32, gate_acts_buf: ?*const metal.MTLBuffer,
        n_hh_out: []f32, n_hh_out_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_ih = try self.getBuffer(gates_ih, gates_ih_buf);
        defer if (gates_ih_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ih.buffer);
        var buffer_hh = try self.getBuffer(gates_hh, gates_hh_buf);
        defer if (gates_hh_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_hh.buffer);
        var buffer_bias = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_bias.buffer);
        var buffer_hp = try self.getBuffer(h_prev, h_prev_buf);
        defer if (h_prev_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_hp.buffer);
        var buffer_hc = try self.getBuffer(h_curr, h_curr_buf);
        defer if (h_curr_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_hc.buffer);
        var buffer_ga = try self.getBuffer(gate_acts, gate_acts_buf);
        defer if (gate_acts_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ga.buffer);
        var buffer_n_hh = try self.getBuffer(n_hh_out, n_hh_out_buf);
        defer if (n_hh_out_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_n_hh.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("gru_forward_step") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_ih.buffer, buffer_ih.offset, 0);
        encoder.setBuffer(&buffer_hh.buffer, buffer_hh.offset, 1);
        encoder.setBuffer(&buffer_bias.buffer, buffer_bias.offset, 2);
        encoder.setBuffer(&buffer_hp.buffer, buffer_hp.offset, 3);
        encoder.setBuffer(&buffer_hc.buffer, buffer_hc.offset, 4);
        encoder.setBuffer(&buffer_ga.buffer, buffer_ga.offset, 5);
        encoder.setBuffer(&buffer_n_hh.buffer, buffer_n_hh.offset, 6);

        const h_u32 = @as(u32, @intCast(hidden_size));
        encoder.setBytes(std.mem.asBytes(&h_u32), 7);

        const tg_size = @min(hidden_size, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(hidden_size, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    /// LSTM backward step
    pub fn lstmBackwardStep(self: Backend,
        grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer,
        grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer,
        grad_c_next: []const f32, grad_c_next_buf: ?*const metal.MTLBuffer,
        gate_acts: []const f32, gate_acts_buf: ?*const metal.MTLBuffer,
        c_curr: []const f32, c_curr_buf: ?*const metal.MTLBuffer,
        c_prev: []const f32, c_prev_buf: ?*const metal.MTLBuffer,
        grad_gates: []f32, grad_gates_buf: ?*const metal.MTLBuffer,
        grad_c_prev: []f32, grad_c_prev_buf: ?*const metal.MTLBuffer,
        grad_h_prev_part: []f32, grad_h_prev_part_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalLstmBackwardStep(grad_h_curr, grad_h_curr_buf, grad_h_next, grad_h_next_buf, grad_c_next, grad_c_next_buf, gate_acts, gate_acts_buf, c_curr, c_curr_buf, c_prev, c_prev_buf, grad_gates, grad_gates_buf, grad_c_prev, grad_c_prev_buf, grad_h_prev_part, grad_h_prev_part_buf, hidden_size),
                .vulkan => return error.NotAvailable,
            },
            .cpu => {
                for (0..hidden_size) |j| {
                    const i = gate_acts[j];
                    const f = gate_acts[hidden_size + j];
                    const g = gate_acts[2 * hidden_size + j];
                    const o = gate_acts[3 * hidden_size + j];

                    const gh = grad_h_curr[j] + grad_h_next[j];
                    const tanh_cc = std.math.tanh(c_curr[j]);

                    const d_o = gh * tanh_cc * (o * (1.0 - o));
                    const d_c = gh * o * (1.0 - tanh_cc * tanh_cc) + grad_c_next[j];

                    grad_gates[j] = d_c * g * (i * (1.0 - i)); // d_i
                    grad_gates[hidden_size + j] = d_c * c_prev[j] * (f * (1.0 - f)); // d_f
                    grad_gates[2 * hidden_size + j] = d_c * i * (1.0 - g * g); // d_g
                    grad_gates[3 * hidden_size + j] = d_o;

                    grad_c_prev[j] = d_c * f;
                    grad_h_prev_part[j] = 0; // Not used in CPU path, but consistent with Metal
                }
            },
        }
    }

    /// GRU backward step
    pub fn gruBackwardStep(self: Backend,
        grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer,
        grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer,
        gate_acts: []const f32, gate_acts_buf: ?*const metal.MTLBuffer,
        h_prev: []const f32, h_prev_buf: ?*const metal.MTLBuffer,
        n_hh: []const f32, n_hh_buf: ?*const metal.MTLBuffer,
        grad_gates_ih: []f32, grad_gates_ih_buf: ?*const metal.MTLBuffer,
        grad_gates_hh: []f32, grad_gates_hh_buf: ?*const metal.MTLBuffer,
        grad_h_prev: []f32, grad_h_prev_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalGruBackwardStep(grad_h_curr, grad_h_curr_buf, grad_h_next, grad_h_next_buf, gate_acts, gate_acts_buf, h_prev, h_prev_buf, n_hh, n_hh_buf, grad_gates_ih, grad_gates_ih_buf, grad_gates_hh, grad_gates_hh_buf, grad_h_prev, grad_h_prev_buf, hidden_size),
                .vulkan => return error.NotAvailable,
            },
            .cpu => {
                for (0..hidden_size) |j| {
                    const z = gate_acts[j];
                    const r = gate_acts[hidden_size + j];
                    const n = gate_acts[2 * hidden_size + j];

                    const gh = grad_h_curr[j] + grad_h_next[j];
                    const hp = h_prev[j];

                    const d_nt = gh * (1.0 - z);
                    const d_zt = gh * (hp - n);

                    const d_n_raw = d_nt * (1.0 - n * n);
                    const d_n_ih = d_n_raw;
                    const d_n_hh = d_n_raw * r;

                    const d_rt = d_n_raw * n_hh[j] * (r * (1.0 - r));
                    const d_z_raw = d_zt * z * (1.0 - z);

                    grad_gates_ih[j] = d_z_raw;
                    grad_gates_ih[hidden_size + j] = d_rt;
                    grad_gates_ih[2 * hidden_size + j] = d_n_ih;

                    grad_gates_hh[j] = d_z_raw;
                    grad_gates_hh[hidden_size + j] = d_rt;
                    grad_gates_hh[2 * hidden_size + j] = d_n_hh;

                    grad_h_prev[j] = gh * z;
                }
            },
        }
    }

    fn metalLstmBackwardStep(self: Backend,
        grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer,
        grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer,
        grad_c_next: []const f32, grad_c_next_buf: ?*const metal.MTLBuffer,
        gate_acts: []const f32, gate_acts_buf: ?*const metal.MTLBuffer,
        c_curr: []const f32, c_curr_buf: ?*const metal.MTLBuffer,
        c_prev: []const f32, c_prev_buf: ?*const metal.MTLBuffer,
        grad_gates: []f32, grad_gates_buf: ?*const metal.MTLBuffer,
        grad_c_prev: []f32, grad_c_prev_buf: ?*const metal.MTLBuffer,
        grad_h_prev_part: []f32, grad_h_prev_part_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_gh = try self.getBuffer(grad_h_curr, grad_h_curr_buf);
        defer if (grad_h_curr_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gh.buffer);
        var buffer_gn = try self.getBuffer(grad_h_next, grad_h_next_buf);
        defer if (grad_h_next_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gn.buffer);
        var buffer_gcn = try self.getBuffer(grad_c_next, grad_c_next_buf);
        defer if (grad_c_next_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gcn.buffer);
        var buffer_ga = try self.getBuffer(gate_acts, gate_acts_buf);
        defer if (gate_acts_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ga.buffer);
        var buffer_cc = try self.getBuffer(c_curr, c_curr_buf);
        defer if (c_curr_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_cc.buffer);
        var buffer_cp = try self.getBuffer(c_prev, c_prev_buf);
        defer if (c_prev_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_cp.buffer);
        var buffer_gg = try self.getBuffer(grad_gates, grad_gates_buf);
        defer if (grad_gates_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gg.buffer);
        var buffer_gcp = try self.getBuffer(grad_c_prev, grad_c_prev_buf);
        defer if (grad_c_prev_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gcp.buffer);
        var buffer_ghp = try self.getBuffer(grad_h_prev_part, grad_h_prev_part_buf);
        defer if (grad_h_prev_part_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ghp.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("lstm_backward_step") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_gh.buffer, buffer_gh.offset, 0);
        encoder.setBuffer(&buffer_gn.buffer, buffer_gn.offset, 1);
        encoder.setBuffer(&buffer_gcn.buffer, buffer_gcn.offset, 2);
        encoder.setBuffer(&buffer_ga.buffer, buffer_ga.offset, 3);
        encoder.setBuffer(&buffer_cc.buffer, buffer_cc.offset, 4);
        encoder.setBuffer(&buffer_cp.buffer, buffer_cp.offset, 5);
        encoder.setBuffer(&buffer_gg.buffer, buffer_gg.offset, 6);
        encoder.setBuffer(&buffer_gcp.buffer, buffer_gcp.offset, 7);
        encoder.setBuffer(&buffer_ghp.buffer, buffer_ghp.offset, 8);

        const h_u32 = @as(u32, @intCast(hidden_size));
        encoder.setBytes(std.mem.asBytes(&h_u32), 9);

        const tg_size = @min(hidden_size, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(hidden_size, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalGruBackwardStep(self: Backend,
        grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer,
        grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer,
        gate_acts: []const f32, gate_acts_buf: ?*const metal.MTLBuffer,
        h_prev: []const f32, h_prev_buf: ?*const metal.MTLBuffer,
        n_hh: []const f32, n_hh_buf: ?*const metal.MTLBuffer,
        grad_gates_ih: []f32, grad_gates_ih_buf: ?*const metal.MTLBuffer,
        grad_gates_hh: []f32, grad_gates_hh_buf: ?*const metal.MTLBuffer,
        grad_h_prev: []f32, grad_h_prev_buf: ?*const metal.MTLBuffer,
        hidden_size: usize
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_gh = try self.getBuffer(grad_h_curr, grad_h_curr_buf);
        defer if (grad_h_curr_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gh.buffer);
        var buffer_gn = try self.getBuffer(grad_h_next, grad_h_next_buf);
        defer if (grad_h_next_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gn.buffer);
        var buffer_ga = try self.getBuffer(gate_acts, gate_acts_buf);
        defer if (gate_acts_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ga.buffer);
        var buffer_hp = try self.getBuffer(h_prev, h_prev_buf);
        defer if (h_prev_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_hp.buffer);
        var buffer_nhh = try self.getBuffer(n_hh, n_hh_buf);
        defer if (n_hh_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_nhh.buffer);
        var buffer_gih = try self.getBuffer(grad_gates_ih, grad_gates_ih_buf);
        defer if (grad_gates_ih_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gih.buffer);
        var buffer_ghh = try self.getBuffer(grad_gates_hh, grad_gates_hh_buf);
        defer if (grad_gates_hh_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ghh.buffer);
        var buffer_ghp = try self.getBuffer(grad_h_prev, grad_h_prev_buf);
        defer if (grad_h_prev_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ghp.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("gru_backward_step") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_gh.buffer, buffer_gh.offset, 0);
        encoder.setBuffer(&buffer_gn.buffer, buffer_gn.offset, 1);
        encoder.setBuffer(&buffer_ga.buffer, buffer_ga.offset, 2);
        encoder.setBuffer(&buffer_hp.buffer, buffer_hp.offset, 3);
        encoder.setBuffer(&buffer_nhh.buffer, buffer_nhh.offset, 4);
        encoder.setBuffer(&buffer_gih.buffer, buffer_gih.offset, 5);
        encoder.setBuffer(&buffer_ghh.buffer, buffer_ghh.offset, 6);
        encoder.setBuffer(&buffer_ghp.buffer, buffer_ghp.offset, 7);

        const h_u32 = @as(u32, @intCast(hidden_size));
        encoder.setBytes(std.mem.asBytes(&h_u32), 8);

        const tg_size = @min(hidden_size, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(hidden_size, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }
};

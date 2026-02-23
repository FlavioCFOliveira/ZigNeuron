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
                .vulkan => @memcpy(dst, src),
            },
            .cpu => @memcpy(dst, src),
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

            // Use blit encoder for GPU to GPU copy
            var encoder = try command_buffer.blitCommandEncoder();
            encoder.copyBuffer(src_buf.?.*, 0, dst_buf.?.*, 0, src.len * @sizeOf(f32));
            encoder.endEncoding();

            if (ctx.active_command_buffer == null) {
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
            }
        } else {
            // Fallback to CPU copy (works for Unified Memory if not in a batch)
            @memcpy(dst, src);
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
    pub fn activationForward(self: Backend,
        act: activation.Activation,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
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
                .vulkan => cpuLossBackward(loss_fn, output, target, grad_output),
            },
            .cpu => cpuLossBackward(loss_fn, output, target, grad_output),
        }
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

    /// Add bias to output
    pub fn addBias(self: Backend,
        output: []f32, output_buf: ?*const metal.MTLBuffer,
        bias: []const f32, bias_buf: ?*const metal.MTLBuffer
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalAddBias(output, output_buf, bias, bias_buf),
                .vulkan => {
                    for (0..output.len) |i| {
                        output[i] += bias[i];
                    }
                },
            },
            .cpu => {
                for (0..output.len) |i| {
                    output[i] += bias[i];
                }
            },
        }
    }

    fn metalAddBias(self: Backend,
        output: []f32, output_buf: ?*const metal.MTLBuffer,
        bias: []const f32, bias_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_output = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) buffer_output.release();
        var buffer_bias = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) buffer_bias.release();

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("add_bias") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_output, 0, 0);
        encoder.setBuffer(&buffer_bias, 0, 1);

        const size = @as(u32, @intCast(output.len));
        encoder.setBytes(std.mem.asBytes(&size), 2);

        encoder.dispatchThreads(metal.MTLSize.make(output.len, 1, 1), metal.MTLSize.make(1, 1, 1));
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
        defer if (a_buf == null and ctx.active_command_buffer == null) buffer_a.release();
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) buffer_b.release();
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) buffer_c.release();

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("matmul_transpose_a") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a, 0, 0);
        encoder.setBuffer(&buffer_b, 0, 1);
        encoder.setBuffer(&buffer_c, 0, 2);

        const m_u32 = @as(u32, @intCast(m));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        encoder.setBytes(std.mem.asBytes(&m_u32), 3);
        encoder.setBytes(std.mem.asBytes(&n_u32), 4);
        encoder.setBytes(std.mem.asBytes(&k_u32), 5);

        encoder.dispatchThreads(metal.MTLSize.make(n, m, 1), metal.MTLSize.make(1, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    // Helper to get buffer, either from provided or create temporary
    fn getBuffer(self: Backend, slice: anytype, provided_buf: ?*const metal.MTLBuffer) !metal.MTLBuffer {
        if (provided_buf) |buf| return buf.*;
        const ctx = self.metal_ctx.?;
        const buffer = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(slice), .StorageModeShared);
        if (ctx.active_command_buffer != null) {
            try ctx.registerTempResource(buffer);
        }
        return buffer;
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
        defer if (a_buf == null and ctx.active_command_buffer == null) buffer_a.release();
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) buffer_b.release();
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) buffer_c.release();

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
        encoder.setBuffer(&buffer_a, 0, 0);
        encoder.setBuffer(&buffer_b, 0, 1);
        encoder.setBuffer(&buffer_c, 0, 2);

        const m_u32 = @as(u32, @intCast(m));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        encoder.setBytes(std.mem.asBytes(&m_u32), 3);
        encoder.setBytes(std.mem.asBytes(&n_u32), 4);
        encoder.setBytes(std.mem.asBytes(&k_u32), 5);

        encoder.dispatchThreads(metal.MTLSize.make(n, m, 1), metal.MTLSize.make(1, 1, 1));
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
        defer if (a_buf == null and ctx.active_command_buffer == null) buffer_a.release();
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) buffer_b.release();
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) buffer_c.release();

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        const pipeline = ctx.getPipeline("matmul_batch") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a, 0, 0);
        encoder.setBuffer(&buffer_b, 0, 1);
        encoder.setBuffer(&buffer_c, 0, 2);

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
        defer if (input_buf == null and ctx.active_command_buffer == null) buffer_input.release();
        var buffer_output = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) buffer_output.release();

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
        encoder.setBuffer(&buffer_input, 0, 0);
        encoder.setBuffer(&buffer_output, 0, 1);

        const size = @as(u32, @intCast(input.len));
        if (act == .softmax) {
            const num_classes = size;
            const batch_size: u32 = 1;
            encoder.setBytes(std.mem.asBytes(&batch_size), 2);
            encoder.setBytes(std.mem.asBytes(&num_classes), 3);
            encoder.dispatchThreads(metal.MTLSize.make(num_classes, 1, 1), metal.MTLSize.make(256, 1, 1));
        } else {
            encoder.setBytes(std.mem.asBytes(&size), 2);
            encoder.dispatchThreads(metal.MTLSize.make(size, 1, 1), metal.MTLSize.make(256, 1, 1));
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
        defer if (input_buf == null and ctx.active_command_buffer == null) buffer_input.release();
        var buffer_grad_output = try self.getBuffer(grad_output, grad_output_buf);
        defer if (grad_output_buf == null and ctx.active_command_buffer == null) buffer_grad_output.release();
        var buffer_grad_input = try self.getBuffer(grad_input, grad_input_buf);
        defer if (grad_input_buf == null and ctx.active_command_buffer == null) buffer_grad_input.release();

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
        encoder.setBuffer(&buffer_input, 0, 0);
        encoder.setBuffer(&buffer_grad_output, 0, 1);
        encoder.setBuffer(&buffer_grad_input, 0, 2);

        const size = @as(u32, @intCast(input.len));
        if (act == .softmax) {
            const num_classes = size;
            const batch_size: u32 = 1;
            encoder.setBytes(std.mem.asBytes(&batch_size), 3);
            encoder.setBytes(std.mem.asBytes(&num_classes), 4);
            encoder.dispatchThreads(metal.MTLSize.make(num_classes, 1, 1), metal.MTLSize.make(256, 1, 1));
        } else {
            encoder.setBytes(std.mem.asBytes(&size), 3);
            encoder.dispatchThreads(metal.MTLSize.make(size, 1, 1), metal.MTLSize.make(256, 1, 1));
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
        defer if (output_buf == null and ctx.active_command_buffer == null) buffer_output.release();
        var buffer_target = try self.getBuffer(target, target_buf);
        defer if (target_buf == null and ctx.active_command_buffer == null) buffer_target.release();
        var buffer_grad_output = try self.getBuffer(grad_output, grad_output_buf);
        defer if (grad_output_buf == null and ctx.active_command_buffer == null) buffer_grad_output.release();

        const pipeline_name = switch (loss_fn) {
            .mse => "mse_backward",
            .cross_entropy => "cross_entropy_backward",
            .cross_entropy_logits => "cross_entropy_logits_backward",
            .binary_cross_entropy => "binary_cross_entropy_backward",
        };
        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_output, 0, 0);
        encoder.setBuffer(&buffer_target, 0, 1);
        encoder.setBuffer(&buffer_grad_output, 0, 2);

        const size = @as(u32, @intCast(output.len));
        if (loss_fn == .cross_entropy or loss_fn == .cross_entropy_logits) {
            const num_samples: u32 = 1;
            const num_classes = size;
            encoder.setBytes(std.mem.asBytes(&num_samples), 3);
            encoder.setBytes(std.mem.asBytes(&num_classes), 4);
        } else {
            const n: u32 = size;
            encoder.setBytes(std.mem.asBytes(&size), 3);
            encoder.setBytes(std.mem.asBytes(&n), 4);
        }

        encoder.dispatchThreads(metal.MTLSize.make(output.len, 1, 1), metal.MTLSize.make(1, 1, 1));
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
        defer if (weights_buf == null and ctx.active_command_buffer == null) buffer_weights.release();
        var buffer_gradients = try self.getBuffer(gradients, gradients_buf);
        defer if (gradients_buf == null and ctx.active_command_buffer == null) buffer_gradients.release();

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("sgd_update") orelse return error.PipelineNotFound;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_weights, 0, 0);
        encoder.setBuffer(&buffer_gradients, 0, 1);
        encoder.setBytes(std.mem.asBytes(&learning_rate), 2);
        encoder.setBytes(std.mem.asBytes(&weight_decay), 3);
        const size = @as(u32, @intCast(weights.len));
        encoder.setBytes(std.mem.asBytes(&size), 4);

        encoder.dispatchThreads(metal.MTLSize.make(weights.len, 1, 1), metal.MTLSize.make(256, 1, 1));
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
        defer if (bias_buf == null and ctx.active_command_buffer == null) buffer_bias.release();
        var buffer_gradients = try self.getBuffer(gradients, gradients_buf);
        defer if (gradients_buf == null and ctx.active_command_buffer == null) buffer_gradients.release();

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("sgd_update_bias") orelse return error.PipelineNotFound;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_bias, 0, 0);
        encoder.setBuffer(&buffer_gradients, 0, 1);
        encoder.setBytes(std.mem.asBytes(&learning_rate), 2);
        const size = @as(u32, @intCast(bias.len));
        encoder.setBytes(std.mem.asBytes(&size), 3);

        encoder.dispatchThreads(metal.MTLSize.make(bias.len, 1, 1), metal.MTLSize.make(256, 1, 1));
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
        defer if (grad_bias_buf == null and ctx.active_command_buffer == null) buffer_grad_bias.release();
        var buffer_grad_after_act = try self.getBuffer(grad_after_act, grad_after_act_buf);
        defer if (grad_after_act_buf == null and ctx.active_command_buffer == null) buffer_grad_after_act.release();

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("accumulate_bias") orelse return error.PipelineNotFound;

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_grad_bias, 0, 0);
        encoder.setBuffer(&buffer_grad_after_act, 0, 1);
        const size = @as(u32, @intCast(grad_bias.len));
        encoder.setBytes(std.mem.asBytes(&size), 2);

        encoder.dispatchThreads(metal.MTLSize.make(grad_bias.len, 1, 1), metal.MTLSize.make(256, 1, 1));
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
                var jj: usize = 0;
                while (jj < n) : (jj += block_size) {
                    var kk: usize = 0;
                    while (kk < k) : (kk += block_size) {
                        const i_end = @min(ii + block_size, m);
                        const j_end = @min(jj + block_size, n);
                        const k_end = @min(kk + block_size, k);
                        for (ii..i_end) |i| {
                            for (jj..j_end) |j| {
                                var sum: f32 = 0.0;
                                for (kk..k_end) |p| {
                                    sum += a[i * k + p] * b[p * n + j];
                                }
                                c[i * n + j] += sum;
                            }
                        }
                    }
                }
            }
        } else {
            for (0..m) |i| {
                for (0..n) |j| {
                    var sum: f32 = 0.0;
                    for (0..k) |p| {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    c[i * n + j] = sum;
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
                for (output, 0..) |p, i| {
                    grad_output[i] = 2.0 * (p - target[i]) / n;
                }
            },
            .binary_cross_entropy => {
                for (output, 0..) |p, i| {
                    grad_output[i] = (p - target[i]) / n;
                }
            },
            .cross_entropy => {
                for (output, 0..) |p, i| {
                    grad_output[i] = p - target[i];
                }
            },
            .cross_entropy_logits => {
                for (output, 0..) |p, i| {
                    grad_output[i] = p - target[i];
                }
            },
        }
    }

    fn cpuSgdUpdate(weights: []f32, gradients: []const f32, learning_rate: f32, weight_decay: f32) void {
        const max_grad: f32 = 5.0;
        for (0..weights.len) |i| {
            var g = gradients[i];
            if (std.math.isNan(g)) {
                g = 0.0;
            } else if (g > max_grad) {
                g = max_grad;
            } else if (g < -max_grad) {
                g = -max_grad;
            }

            weights[i] -= learning_rate * (g + weight_decay * weights[i]);
            if (weights[i] > 100.0) weights[i] = 100.0;
            if (weights[i] < -100.0) weights[i] = -100.0;
        }
    }

    fn cpuSgdUpdateBias(bias: []f32, gradients: []const f32, learning_rate: f32) void {
        const max_grad: f32 = 5.0;
        for (0..bias.len) |i| {
            var g = gradients[i];
            if (std.math.isNan(g)) {
                g = 0.0;
            } else if (g > max_grad) {
                g = max_grad;
            } else if (g < -max_grad) {
                g = -max_grad;
            }

            bias[i] -= learning_rate * g;
            if (bias[i] > 50.0) bias[i] = 50.0;
            if (bias[i] < -50.0) bias[i] = -50.0;
        }
    }

    fn cpuAccumulateBias(grad_bias: []f32, grad_after_act: []const f32) void {
        for (0..grad_bias.len) |i| {
            grad_bias[i] += grad_after_act[i];
        }
    }
};

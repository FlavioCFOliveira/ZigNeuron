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
        // On macOS, try Metal first
        const os_tag = @import("builtin").os.tag;

        // Try Metal first (Apple Silicon)
        if (os_tag == .macos) {
            if (metalSupported()) {
                return BackendType{ .gpu = .metal };
            }
        }

        // Try Vulkan next (cross-platform)
        if (vulkanSupported()) {
            return BackendType{ .gpu = .vulkan };
        }

        // Fall back to CPU
        return BackendType{ .cpu = {} };
    }

    /// Check if Metal is available on this system
    fn metalSupported() bool {
        // Return true on macOS - Metal is available on all modern macOS systems
        const os_tag = @import("builtin").os.tag;
        return os_tag == .macos;
    }

    /// Check if Vulkan is available on this system
    fn vulkanSupported() bool {
        // For now, always return true as a placeholder
        // Full implementation would check for Vulkan library at runtime
        return true;
    }

    /// Check if we're on macOS (helper)
    fn isMacos() bool {
        const os_tag = @import("builtin").os.tag;
        return os_tag == .macos;
    }

    /// Execute matrix multiplication on the selected backend
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
                .vulkan => try self.vulkanMatMul(a, b, c, m, n, k),
            },
            .cpu => cpuMatMul(a, b, c, m, n, k),
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
                .vulkan => {
                    // TODO: Vulkan transpose matmul
                    cpuMatMulTransposeB(a, b, c, m, n, k);
                },
            },
            .cpu => cpuMatMulTransposeB(a, b, c, m, n, k),
        }
    }

    /// Execute batched matrix multiplication on the selected backend
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    /// This is more efficient for batch processing
    pub fn matMulBatch(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        batch_size: usize, n: usize, k: usize
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMulBatch(a, a_buf, b, b_buf, c, c_buf, batch_size, n, k),
                .vulkan => try self.vulkanMatMulBatch(a, b, c, batch_size, n, k),
            },
            .cpu => cpuMatMulBatch(a, b, c, batch_size, n, k),
        }
    }

    /// Execute activation function on array
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn activationForward(self: Backend,
        act: activation.Activation,
        input: []f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalActivationForward(act, input, input_buf, output, output_buf),
                .vulkan => try self.vulkanActivationForward(act, input, output),
            },
            .cpu => cpuActivationForward(act, input, output),
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
                .vulkan => try self.vulkanActivationBackward(act, input, grad_output, grad_input),
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
                .vulkan => try self.vulkanLossBackward(loss_fn, output, target, grad_output),
            },
            .cpu => cpuLossBackward(loss_fn, output, target, grad_output),
        }
    }

    // ================== Metal implementations ==================
    // ( Apple Silicon GPU )

    fn metalMatMul(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        m: usize, n: usize, k: usize,
        transpose_b: bool
    ) !void {
        // Check if we're on macOS
        if (!isMacos()) {
            return error.NotAvailable;
        }

        // Validate inputs
        if (a.len < m * k) return error.BufferTooSmall;
        if (b.len < k * n) return error.BufferTooSmall;
        if (c.len < m * n) return error.BufferTooSmall;

        // Use GPU implementation
        try self.metalMatMulGPU(a, a_buf, b, b_buf, c, c_buf, m, n, k, transpose_b);
    }

    fn metalMatMulBatch(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        batch_size: usize, n: usize, k: usize
    ) !void {
        // Check if we're on macOS
        if (!isMacos()) {
            return error.NotAvailable;
        }

        // Batch operations are perfect for GPU
        const total_size = @as(usize, batch_size) * n * k;
        if (total_size < 64 and a_buf == null) {
            cpuMatMulBatch(a, b, c, batch_size, n, k);
            return;
        }

        // Validate inputs
        if (a.len < batch_size * k) return error.BufferTooSmall;
        if (b.len < k * n) return error.BufferTooSmall;
        if (c.len < batch_size * n) return error.BufferTooSmall;

        // Use GPU implementation
        try self.metalMatMulBatchGPU(a, a_buf, b, b_buf, c, c_buf, batch_size, n, k);
    }

    fn metalActivationForward(self: Backend,
        act: activation.Activation,
        input: []f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        if (input.len < 32 and input_buf == null) {
            cpuActivationForward(act, input, output);
            return;
        }

        if (input.len != output.len) return error.ShapeMismatch;

        try self.metalActivationForwardGPU(act, input, input_buf, output, output_buf);
    }

    fn metalActivationBackward(self: Backend,
        act: activation.Activation,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer
    ) !void {
        if (input.len < 32 and input_buf == null) {
            cpuActivationBackward(act, input, grad_output, grad_input);
            return;
        }

        if (input.len != grad_output.len or input.len != grad_input.len) {
            return error.ShapeMismatch;
        }

        try self.metalActivationBackwardGPU(act, input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf);
    }

    fn metalLossBackward(self: Backend,
        loss_fn: loss.Loss,
        output: []const f32, output_buf: ?*const metal.MTLBuffer,
        target: []const f32, target_buf: ?*const metal.MTLBuffer,
        grad_output: []f32, grad_output_buf: ?*const metal.MTLBuffer
    ) !void {
        if (output.len < 32 and output_buf == null) {
            cpuLossBackward(loss_fn, output, target, grad_output);
            return;
        }

        if (output.len != target.len or output.len != grad_output.len) {
            return error.ShapeMismatch;
        }

        try self.metalLossBackwardGPU(loss_fn, output, output_buf, target, target_buf, grad_output, grad_output_buf);
    }

    /// GPU implementation of batched matrix multiplication using Metal
    fn metalMatMulBatchGPU(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        batch_size: usize, n: usize, k: usize
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;

        // Use provided buffers or create temporary ones
        var buffer_a: metal.MTLBuffer = undefined;
        var own_a = false;
        if (a_buf) |buf| {
            buffer_a = buf.*;
        } else {
            buffer_a = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(a), .StorageModeShared);
            own_a = true;
        }
        defer if (own_a) buffer_a.release();

        var buffer_b: metal.MTLBuffer = undefined;
        var own_b = false;
        if (b_buf) |buf| {
            buffer_b = buf.*;
        } else {
            buffer_b = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(b), .StorageModeShared);
            own_b = true;
        }
        defer if (own_b) buffer_b.release();

        var buffer_c: metal.MTLBuffer = undefined;
        var own_c = false;
        if (c_buf) |buf| {
            buffer_c = buf.*;
        } else {
            buffer_c = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(c), .StorageModeShared);
            own_c = true;
        }
        defer if (own_c) buffer_c.release();

        // Create command buffer and encoder
        var command_buffer = try ctx.command_queue.commandBuffer();
        defer command_buffer.release();

        var encoder = try command_buffer.computeCommandEncoder();
        defer encoder.release();

        // Set pipeline and buffers
        const pipeline = ctx.getPipeline("matmul_batch") orelse {
            encoder.endEncoding();
            return error.PipelineNotFound;
        };
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a, 0, 0);
        encoder.setBuffer(&buffer_b, 0, 1);
        encoder.setBuffer(&buffer_c, 0, 2);

        // Set matrix dimensions
        const batch_size_u32 = @as(u32, @intCast(batch_size));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        encoder.setBytes(std.mem.asBytes(&batch_size_u32), 3);
        encoder.setBytes(std.mem.asBytes(&n_u32), 4);
        encoder.setBytes(std.mem.asBytes(&k_u32), 5);

        // Dispatch threads
        const threads_per_threadgroup = metal.MTLSize.make(16, 16, 1);
        const grid_size = metal.MTLSize.make(n, 1, batch_size);

        encoder.dispatchThreads(grid_size, threads_per_threadgroup);
        encoder.endEncoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Synchronize results if needed
        if (own_c) {
            const result_ptr = buffer_c.contents();
            if (@as(usize, @intFromPtr(result_ptr)) != @as(usize, @intFromPtr(c.ptr))) {
                @memcpy(c, std.mem.bytesAsSlice(f32, result_ptr[0..c.len * 4]));
            }
        }
    }

    /// GPU implementation of matrix multiplication using Metal
    fn metalMatMulGPU(self: Backend,
        a: []const f32, a_buf: ?*const metal.MTLBuffer,
        b: []const f32, b_buf: ?*const metal.MTLBuffer,
        c: []f32, c_buf: ?*const metal.MTLBuffer,
        m: usize, n: usize, k: usize,
        transpose_b: bool
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;

        // Use provided buffers or create temporary ones
        var buffer_a: metal.MTLBuffer = undefined;
        var own_a = false;
        if (a_buf) |buf| {
            buffer_a = buf.*;
        } else {
            buffer_a = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(a), .StorageModeShared);
            own_a = true;
        }
        defer if (own_a) buffer_a.release();

        var buffer_b: metal.MTLBuffer = undefined;
        var own_b = false;
        if (b_buf) |buf| {
            buffer_b = buf.*;
        } else {
            buffer_b = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(b), .StorageModeShared);
            own_b = true;
        }
        defer if (own_b) buffer_b.release();

        var buffer_c: metal.MTLBuffer = undefined;
        var own_c = false;
        if (c_buf) |buf| {
            buffer_c = buf.*;
        } else {
            buffer_c = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(c), .StorageModeShared);
            own_c = true;
        }
        defer if (own_c) buffer_c.release();

        // Create command buffer and encoder
        var command_buffer = try ctx.command_queue.commandBuffer();
        defer command_buffer.release();

        var encoder = try command_buffer.computeCommandEncoder();
        defer encoder.release();

        // Select pipeline
        var pipeline_name: []const u8 = "matmul";
        if (transpose_b) {
            pipeline_name = "matmul_transpose_b";
        } else if (m % 16 == 0 and n % 16 == 0 and k % 16 == 0) {
            pipeline_name = "matmul_tiled";
        }

        const pipeline = ctx.getPipeline(pipeline_name) orelse {
            encoder.endEncoding();
            return error.PipelineNotFound;
        };

        // Set pipeline and buffers
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a, 0, 0);
        encoder.setBuffer(&buffer_b, 0, 1);
        encoder.setBuffer(&buffer_c, 0, 2);

        // Set matrix dimensions
        const m_u32 = @as(u32, @intCast(m));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        encoder.setBytes(std.mem.asBytes(&m_u32), 3);
        encoder.setBytes(std.mem.asBytes(&n_u32), 4);
        encoder.setBytes(std.mem.asBytes(&k_u32), 5);

        // Dispatch threads
        const threads_per_threadgroup = metal.MTLSize.make(16, 16, 1);
        const grid_size = metal.MTLSize.make(n, m, 1);

        encoder.dispatchThreads(grid_size, threads_per_threadgroup);
        encoder.endEncoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Copy results back (ensure CPU sees updates even on Unified Memory)
        const result_ptr = buffer_c.contents();
        if (@as(usize, @intFromPtr(result_ptr)) != @as(usize, @intFromPtr(c.ptr))) {
            @memcpy(c, std.mem.bytesAsSlice(f32, result_ptr[0..c.len * 4]));
        } else {
            // If they are the same pointer, we might still need to invalidate CPU cache on some systems,
            // but on Apple Silicon Shared memory, waitUntilCompleted should be enough.
            // We'll do nothing here as @memcpy to self is invalid.
        }
    }

    /// GPU implementation of activation forward using Metal
    fn metalActivationForwardGPU(self: Backend,
        act: activation.Activation,
        input: []f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;

        // Use provided buffers or create temporary ones
        var buffer_input: metal.MTLBuffer = undefined;
        var own_input = false;
        if (input_buf) |buf| {
            buffer_input = buf.*;
        } else {
            buffer_input = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(input), .StorageModeShared);
            own_input = true;
        }
        defer if (own_input) buffer_input.release();

        var buffer_output: metal.MTLBuffer = undefined;
        var own_output = false;
        if (output_buf) |buf| {
            buffer_output = buf.*;
        } else {
            buffer_output = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(output), .StorageModeShared);
            own_output = true;
        }
        defer if (own_output) buffer_output.release();

        // Get activation function name based on activation type
        const pipeline_name = switch (act) {
            .relu => "relu_forward",
            .sigmoid => "sigmoid_forward",
            .tanh => "tanh_forward",
            .softmax => "softmax_forward",
            .linear => "linear_forward",
        };

        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;

        // Create command buffer and encoder
        var command_buffer = try ctx.command_queue.commandBuffer();
        defer command_buffer.release();

        var encoder = try command_buffer.computeCommandEncoder();
        defer encoder.release();

        // Set pipeline and buffers
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_input, 0, 0);
        encoder.setBuffer(&buffer_output, 0, 1);

        // Special handling for softmax (needs num_classes)
        if (act == .softmax) {
            const size: u32 = 1; // For now, activationForward is called per-sample
            const num_classes = @as(u32, @intCast(input.len));
            encoder.setBytes(std.mem.asBytes(&size), 2);
            encoder.setBytes(std.mem.asBytes(&num_classes), 3);
        } else {
            // Set array size
            const size = @as(u32, @intCast(input.len));
            encoder.setBytes(std.mem.asBytes(&size), 2);
        }

        // Dispatch threads
        const threads_per_threadgroup = metal.MTLSize.make(256, 1, 1);
        const grid_size = if (act == .softmax)
            metal.MTLSize.make(input.len, 1, 1)
        else
            metal.MTLSize.make(input.len, 1, 1);

        encoder.dispatchThreads(grid_size, threads_per_threadgroup);
        encoder.endEncoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Copy results back if pointers differ
        const result_ptr = buffer_output.contents();
        if (@as(usize, @intFromPtr(result_ptr)) != @as(usize, @intFromPtr(output.ptr))) {
            @memcpy(output, std.mem.bytesAsSlice(f32, result_ptr[0..output.len * 4]));
        }
    }

    /// GPU implementation of activation backward using Metal
    fn metalActivationBackwardGPU(self: Backend,
        act: activation.Activation,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;

        // Use provided buffers or create temporary ones
        var buffer_input: metal.MTLBuffer = undefined;
        var own_input = false;
        if (input_buf) |buf| {
            buffer_input = buf.*;
        } else {
            buffer_input = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(input), .StorageModeShared);
            own_input = true;
        }
        defer if (own_input) buffer_input.release();

        var buffer_grad_output: metal.MTLBuffer = undefined;
        var own_grad_output = false;
        if (grad_output_buf) |buf| {
            buffer_grad_output = buf.*;
        } else {
            buffer_grad_output = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(grad_output), .StorageModeShared);
            own_grad_output = true;
        }
        defer if (own_grad_output) buffer_grad_output.release();

        var buffer_grad_input: metal.MTLBuffer = undefined;
        var own_grad_input = false;
        if (grad_input_buf) |buf| {
            buffer_grad_input = buf.*;
        } else {
            buffer_grad_input = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(grad_input), .StorageModeShared);
            own_grad_input = true;
        }
        defer if (own_grad_input) buffer_grad_input.release();

        // Get activation function name based on activation type
        const pipeline_name = switch (act) {
            .relu => "relu_backward",
            .sigmoid => "sigmoid_backward",
            .tanh => "tanh_backward",
            .softmax => "softmax_backward",
            .linear => "linear_backward",
        };

        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;

        // Create command buffer and encoder
        var command_buffer = try ctx.command_queue.commandBuffer();
        defer command_buffer.release();

        var encoder = try command_buffer.computeCommandEncoder();
        defer encoder.release();

        // Set pipeline and buffers
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_input, 0, 0);
        encoder.setBuffer(&buffer_grad_output, 0, 1);
        encoder.setBuffer(&buffer_grad_input, 0, 2);

        // Special handling for softmax (needs num_classes)
        if (act == .softmax) {
            const size: u32 = 1; // Per-sample
            const num_classes = @as(u32, @intCast(input.len));
            encoder.setBytes(std.mem.asBytes(&size), 3);
            encoder.setBytes(std.mem.asBytes(&num_classes), 4); // num_classes
        } else {
            // Set array size
            const size = @as(u32, @intCast(input.len));
            encoder.setBytes(std.mem.asBytes(&size), 3);
        }

        // Dispatch threads
        const threads_per_threadgroup = metal.MTLSize.make(256, 1, 1);
        const grid_size = metal.MTLSize.make(input.len, 1, 1);

        encoder.dispatchThreads(grid_size, threads_per_threadgroup);
        encoder.endEncoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Copy results back if pointers differ
        const result_ptr = buffer_grad_input.contents();
        if (@as(usize, @intFromPtr(result_ptr)) != @as(usize, @intFromPtr(grad_input.ptr))) {
            @memcpy(grad_input, std.mem.bytesAsSlice(f32, result_ptr[0..grad_input.len * 4]));
        }
    }

    /// GPU implementation of loss backward using Metal
    fn metalLossBackwardGPU(self: Backend,
        loss_fn: loss.Loss,
        output: []const f32, output_buf: ?*const metal.MTLBuffer,
        target: []const f32, target_buf: ?*const metal.MTLBuffer,
        grad_output: []f32, grad_output_buf: ?*const metal.MTLBuffer
    ) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;

        // Use provided buffers or create temporary ones
        var buffer_output: metal.MTLBuffer = undefined;
        var own_output = false;
        if (output_buf) |buf| {
            buffer_output = buf.*;
        } else {
            buffer_output = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(output), .StorageModeShared);
            own_output = true;
        }
        defer if (own_output) buffer_output.release();

        var buffer_target: metal.MTLBuffer = undefined;
        var own_target = false;
        if (target_buf) |buf| {
            buffer_target = buf.*;
        } else {
            buffer_target = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(target), .StorageModeShared);
            own_target = true;
        }
        defer if (own_target) buffer_target.release();

        var buffer_grad_output: metal.MTLBuffer = undefined;
        var own_grad_output = false;
        if (grad_output_buf) |buf| {
            buffer_grad_output = buf.*;
        } else {
            buffer_grad_output = try ctx.device.newBufferWithBytes(std.mem.sliceAsBytes(grad_output), .StorageModeShared);
            own_grad_output = true;
        }
        defer if (own_grad_output) buffer_grad_output.release();

        // Get loss function name based on loss type
        const pipeline_name = switch (loss_fn) {
            .mse => "mse_backward",
            .cross_entropy => "cross_entropy_backward",
            .cross_entropy_logits => "cross_entropy_logits_backward",
            .binary_cross_entropy => "binary_cross_entropy_backward",
        };

        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;

        // Create command buffer and encoder
        var command_buffer = try ctx.command_queue.commandBuffer();
        defer command_buffer.release();

        var encoder = try command_buffer.computeCommandEncoder();
        defer encoder.release();

        // Set pipeline and buffers
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_output, 0, 0);
        encoder.setBuffer(&buffer_target, 0, 1);
        encoder.setBuffer(&buffer_grad_output, 0, 2);

        // Special handling for cross_entropy (needs num_classes)
        if (loss_fn == .cross_entropy or loss_fn == .cross_entropy_logits) {
            const num_samples: u32 = 1; // Per-sample
            const num_classes = @as(u32, @intCast(output.len));
            encoder.setBytes(std.mem.asBytes(&num_samples), 3);
            encoder.setBytes(std.mem.asBytes(&num_classes), 4);
        } else if (loss_fn == .mse or loss_fn == .binary_cross_entropy) {
            const size = @as(u32, @intCast(output.len));
            const n = @as(u32, @intCast(output.len)); // Normalization factor
            encoder.setBytes(std.mem.asBytes(&size), 3);
            encoder.setBytes(std.mem.asBytes(&n), 4);
        }

        // Dispatch threads
        const threads_per_threadgroup = metal.MTLSize.make(256, 1, 1);
        const grid_size = metal.MTLSize.make(output.len, 1, 1);

        encoder.dispatchThreads(grid_size, threads_per_threadgroup);
        encoder.endEncoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Copy results back if pointers differ
        const result_ptr = buffer_grad_output.contents();
        if (@as(usize, @intFromPtr(result_ptr)) != @as(usize, @intFromPtr(grad_output.ptr))) {
            @memcpy(grad_output, std.mem.bytesAsSlice(f32, result_ptr[0..grad_output.len * 4]));
        }
    }

    // ================== Vulkan implementations ==================
    // ( cross-platform GPU )

    /// Execute matrix multiplication using Vulkan compute shaders
    fn vulkanMatMulBatch(self: Backend, a: []const f32, b: []const f32, c: []f32, batch_size: usize, n: usize, k: usize) !void {
        _ = self;
        // Validate inputs
        if (a.len < batch_size * k) return error.BufferTooSmall;
        if (b.len < k * n) return error.BufferTooSmall;
        if (c.len < batch_size * n) return error.BufferTooSmall;

        // For small batches, CPU is often faster due to overhead
        const total_size = @as(usize, batch_size) * n * k;
        if (total_size < 2048) {
            cpuMatMulBatch(a, b, c, batch_size, n, k);
            return;
        }

        // Try to create Vulkan device
        const device = vulkan_module.DeviceWrapper.init() catch {
            // Vulkan not available, fall back to CPU
            cpuMatMulBatch(a, b, c, batch_size, n, k);
            return;
        };
        defer device.deinit();

        // Use GPU implementation
        try vulkan_module.vulkanMatMulBatch(&device, a, b, c, batch_size, n, k);
    }

    fn vulkanMatMul(self: Backend, a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) !void {
        _ = self;
        // Validate inputs
        if (a.len < m * k) return error.BufferTooSmall;
        if (b.len < k * n) return error.BufferTooSmall;
        if (c.len < m * n) return error.BufferTooSmall;

        // For small matrices, CPU is often faster due to overhead
        const total_size = @as(usize, m) * n * k;
        if (total_size < 4096) {
            cpuMatMul(a, b, c, m, n, k);
            return;
        }

        // Try to create Vulkan device
        const device = vulkan_module.DeviceWrapper.init() catch {
            // Vulkan not available, fall back to CPU
            cpuMatMul(a, b, c, m, n, k);
            return;
        };
        defer device.deinit();

        // Use GPU implementation
        try vulkan_module.vulkanMatMul(&device, a, b, c, m, n, k);
    }

    fn vulkanActivationForward(self: Backend, act: activation.Activation, input: []f32, output: []f32) !void {
        _ = self;
        // For small arrays, CPU is faster
        if (input.len < 256) {
            cpuActivationForward(act, input, output);
            return;
        }

        if (input.len != output.len) return error.ShapeMismatch;

        // Try to create Vulkan device
        const device = vulkan_module.DeviceWrapper.init() catch {
            // Vulkan not available, fall back to CPU
            cpuActivationForward(act, input, output);
            return;
        };
        defer device.deinit();

        try vulkan_module.vulkanActivationForward(&device, act, input, output);
    }

    fn vulkanActivationBackward(self: Backend, act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        _ = self;
        if (input.len < 256) {
            cpuActivationBackward(act, input, grad_output, grad_input);
            return;
        }

        if (input.len != grad_output.len or input.len != grad_input.len) {
            return error.ShapeMismatch;
        }

        // Try to create Vulkan device
        const device = vulkan_module.DeviceWrapper.init() catch {
            // Vulkan not available, fall back to CPU
            cpuActivationBackward(act, input, grad_output, grad_input);
            return;
        };
        defer device.deinit();

        try vulkan_module.vulkanActivationBackward(&device, act, input, grad_output, grad_input);
    }

    fn vulkanLossBackward(self: Backend, loss_fn: loss.Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        _ = self;
        if (output.len < 256) {
            cpuLossBackward(loss_fn, output, target, grad_output);
            return;
        }

        if (output.len != target.len or output.len != grad_output.len) {
            return error.ShapeMismatch;
        }

        // Vulkan backend - for now fall back to CPU
        // Full implementation would use Vulkan compute shaders
        cpuLossBackward(loss_fn, output, target, grad_output);
    }

    // ================== CPU implementations ==================
    // ( fallbacks when no GPU available )

    fn cpuMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
        // Optimized matrix multiplication with cache-friendly access pattern
        // C = A * B where A is m×k, B is k×n, C is m×n

        // Initialize output to zero
        @memset(c, 0);

        // Use tiling/blocking for better cache utilization on larger matrices
        const block_size: usize = 64; // Increased from 32 for better cache utilization

        if (m >= block_size and n >= block_size and k >= block_size) {
            // Blocked matrix multiplication for large matrices
            var ii: usize = 0;
            while (ii < m) : (ii += block_size) {
                var jj: usize = 0;
                while (jj < n) : (jj += block_size) {
                    var kk: usize = 0;
                    while (kk < k) : (kk += block_size) {
                        // Process block
                        const i_end = @min(ii + block_size, m);
                        const j_end = @min(jj + block_size, n);
                        const k_end = @min(kk + block_size, k);

                        // Loop reordering for better cache locality
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
            // Optimized simple version for small matrices
            // Loop reordering for better cache locality
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

    fn cpuMatMulTransposeB(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
        // C = A * B^T where A is m×k, B is n×k, C is m×n
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

    fn cpuMatMulBatch(a: []const f32, b: []const f32, c: []f32, batch_size: usize, n: usize, k: usize) void {
        // Batched matrix multiplication: A[batch_size, k] * B[k, n] = C[batch_size, n]
        // Optimized for batch processing

        // Initialize output to zero
        @memset(c, 0);

        // Use tiling/blocking for better cache utilization
        const block_size: usize = 32;

        if (batch_size >= block_size and n >= block_size and k >= block_size) {
            // Blocked matrix multiplication for large batches
            var ii: usize = 0;
            while (ii < batch_size) : (ii += block_size) {
                var jj: usize = 0;
                while (jj < n) : (jj += block_size) {
                    var kk: usize = 0;
                    while (kk < k) : (kk += block_size) {
                        // Process block
                        const i_end = @min(ii + block_size, batch_size);
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
            // Optimized simple version for small batches
            for (0..batch_size) |i| {
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

    fn cpuActivationForward(act: activation.Activation, input: []f32, output: []f32) void {
        // Handle softmax specially - needs the whole vector
        if (act == .softmax) {
            act.softmaxForward(input, output) catch unreachable;
            return;
        }

        // Use SIMD for larger arrays
        if (input.len >= 16) {
            activationForwardSIMD(act, input, output);
        } else {
            // Simple loop for small arrays
            for (0..input.len) |i| {
                output[i] = act.forward(input[i]);
            }
        }
    }

    fn cpuActivationBackward(act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) void {
        // Handle softmax specially - needs the whole vector
        if (act == .softmax) {
            act.softmaxBackward(input, grad_output, grad_input) catch unreachable;
            return;
        }

        // Use SIMD for larger arrays
        if (input.len >= 16) {
            activationBackwardSIMD(act, input, grad_output, grad_input);
        } else {
            // Simple loop for small arrays
            for (0..input.len) |i| {
                grad_input[i] = act.backward(input[i], grad_output[i]);
            }
        }
    }

    // SIMD-optimized activation forward pass
    fn activationForwardSIMD(act: activation.Activation, input: []f32, output: []f32) void {
        // For now, use simple loop - in production, implement platform-specific SIMD
        // Apple Silicon: NEON instructions
        // x86: SSE/AVX instructions

        // This is a placeholder for SIMD implementation
        // Real implementation would use inline assembly or compiler intrinsics
        for (0..input.len) |i| {
            output[i] = act.forward(input[i]);
        }
    }

    // SIMD-optimized activation backward pass
    fn activationBackwardSIMD(act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) void {
        // For now, use simple loop - in production, implement platform-specific SIMD
        for (0..input.len) |i| {
            grad_input[i] = act.backward(input[i], grad_output[i]);
        }
    }

    fn cpuLossBackward(loss_fn: loss.Loss, output: []const f32, target: []const f32, grad_output: []f32) void {
        const n = @as(f32, @floatFromInt(output.len));
        switch (loss_fn) {
            .mse => {
                // MSE gradient: dL/dy = 2(y - t) / n
                for (0..output.len) |i| {
                    grad_output[i] = 2 * (output[i] - target[i]) / n;
                }
            },
            .cross_entropy => {
                // Cross-entropy gradient with log-softmax: (p - t)
                // This assumes the output is logits, and we're using log-softmax
                // The gradient simplifies to prediction - target
                for (0..output.len) |i| {
                    grad_output[i] = output[i] - target[i];
                }
            },
            .cross_entropy_logits => {
                // Gradient: softmax(logits) - target
                // Compute softmax first
                var max_logit: f32 = output[0];
                for (output[1..]) |o| {
                    if (o > max_logit) max_logit = o;
                }

                var sum_exp: f32 = 0;
                for (output) |o| {
                    sum_exp += std.math.exp(o - max_logit);
                }

                for (0..output.len) |i| {
                    const prob = std.math.exp(output[i] - max_logit) / sum_exp;
                    grad_output[i] = prob - target[i];
                }
            },
            .binary_cross_entropy => {
                // BCE gradient with sigmoid: (p - t) / n
                // The gradient simplifies to prediction - target
                // This is the correct form when output is passed through sigmoid
                for (0..output.len) |i| {
                    grad_output[i] = (output[i] - target[i]) / n;
                }
            },
        }
    }
};

test "backend default detection" {
    var backend = try Backend.init(std.testing.allocator);
    defer backend.deinit();
}

test "backend matmul cpu fallback" {
    const allocator = std.testing.allocator;

    const m: usize = 4;
    const n: usize = 3;
    const k: usize = 2;

    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    // Initialize test data
    for (a, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 10)) / 10.0;
    }
    for (b, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt((i + 1) % 10)) / 10.0;
    }

    var backend = Backend{ .type = .cpu, .metal_ctx = null };
    defer backend.deinit();
    try backend.matMul(a, b, c, m, n, k);

    // Verify result - C[0,0] = sum of a[0,:] * b[:,0]
    var expected: f32 = 0;
    for (0..k) |p| {
        expected += a[0 * k + p] * b[p * n + 0];
    }

    const tolerance: f32 = 0.0001;
    const diff = if (c[0] > expected) c[0] - expected else expected - c[0];
    try std.testing.expect(diff < tolerance);
}

test "backend activation forward" {
    var backend = Backend{ .type = .cpu, .metal_ctx = null };
    defer backend.deinit();
    const act = activation.Activation{ .relu = {} };

    const allocator = std.testing.allocator;
    const input = try allocator.alloc(f32, 10);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, 10);
    defer allocator.free(output);

    for (input, 0..) |*v, i| {
        const idx = @as(f32, @floatFromInt(i));
        v.* = (idx - 5.0) / 2.0;
    }

    try backend.activationForward(act, input, output);

    // Verify ReLU: negative values become 0
    for (input, output) |in, out| {
        const expected = if (in > 0) in else 0;
        const diff = if (out > expected) out - expected else expected - out;
        try std.testing.expect(diff < 0.0001);
    }
}

test "backend activation backward" {
    var backend = Backend{ .type = .cpu, .metal_ctx = null };
    defer backend.deinit();
    const act = activation.Activation{ .sigmoid = {} };

    const allocator = std.testing.allocator;
    const input = try allocator.alloc(f32, 5);
    defer allocator.free(input);
    const grad_output = try allocator.alloc(f32, 5);
    defer allocator.free(grad_output);
    const grad_input = try allocator.alloc(f32, 5);
    defer allocator.free(grad_input);

    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) / 2.0 - 1.0;
    }
    @memset(grad_output, 1.0);

    try backend.activationBackward(act, input, grad_output, grad_input);

    // Verify sigmoid derivative: s'(x) = s(x) * (1 - s(x))
    for (input, grad_input) |in, gi| {
        const s = act.forward(in);
        const expected = s * (1 - s);
        const diff = if (gi > expected) gi - expected else expected - gi;
        try std.testing.expect(diff < 0.0001);
    }
}

test "backend loss backward mse" {
    var backend = Backend{ .type = .cpu, .metal_ctx = null };
    defer backend.deinit();
    const loss_fn = loss.Loss{ .mse = {} };

    const allocator = std.testing.allocator;
    const output = try allocator.alloc(f32, 5);
    defer allocator.free(output);
    const target = try allocator.alloc(f32, 5);
    defer allocator.free(target);
    const grad_output = try allocator.alloc(f32, 5);
    defer allocator.free(grad_output);

    for (output, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) / 2.0;
    }
    for (target, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 5)) / 2.0;
    }

    try backend.lossBackward(loss_fn, output, target, grad_output);

    // Verify MSE gradient: dL/dy = 2(y - t)
    for (output, target, grad_output) |o, t, g| {
        const expected = 2 * (o - t);
        const diff = if (g > expected) g - expected else expected - g;
        try std.testing.expect(diff < 0.0001);
    }
}

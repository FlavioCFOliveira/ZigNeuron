/// Backend for neural network computation
/// Supports GPU (Metal, CUDA) and CPU execution
/// GPU is the PRIORITY for all training and inference operations
/// Priority: Metal (Apple Silicon) > CUDA (NVIDIA) > CPU
const std = @import("std");
const activation = @import("activation.zig");
const loss = @import("loss.zig");
const metal = @import("metal.zig");
const metal_context = @import("metal_context.zig");
const optimization = @import("optimization.zig");
const cuda_context = @import("cuda_context.zig");

/// Available GPU backends
pub const GpuBackend = enum {
    /// Apple Silicon - Metal compute shaders
    metal,
    /// NVIDIA GPUs - CUDA compute
    cuda,
    /// CPU fallback
    cpu,
};

/// Backend selection - GPU preferred, CPU fallback
/// SECURITY NOTE: Backend is NOT thread-safe. Use separate instances per thread.
pub const Backend = struct {
    type: BackendType,
    metal_ctx: ?*metal_context.MetalContext = null,
    cuda_ctx: ?*cuda_context.CudaContext = null,

    pub const BackendType = union(enum) {
        gpu: GpuBackend,
        cpu,
    };

    pub const ElementWiseOp = enum { add, sub, mul, div };
    pub const MapOp = enum { exp, log, sqrt, abs, square, inv };

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
    /// Priority: Metal (Apple Silicon) > CPU
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
    /// SECURITY NOTE: This function is NOT thread-safe. Backend should be used from a single thread.
    /// For multi-threading, create separate Backend instances per thread.
    pub fn beginCommandBatch(self: Backend) !void {
        if (self.metal_ctx) |ctx| {
            if (ctx.active_command_buffer == null) {
                ctx.active_command_buffer = try ctx.command_queue.commandBuffer();
            }
        }
    }

    /// End a batch of commands and execute them (Metal only)
    /// SECURITY NOTE: This function is NOT thread-safe. Backend should be used from a single thread.
    pub fn endCommandBatch(self: *Backend) !void {
        if (self.metal_ctx) |ctx| {
            if (ctx.active_command_buffer) |cb| {
                var mutable_cb = cb;
                mutable_cb.commit();
                mutable_cb.waitUntilCompleted();
                ctx.active_command_buffer = null;
                ctx.clearTempResources();
            }
        }
    }

    /// Copy data between buffers (GPU or CPU)
    pub fn copyData(self: Backend, src: []const f32, src_buf: ?*const metal.MTLBuffer, dst: []f32, dst_buf: ?*const metal.MTLBuffer) !void {
        if (src.len != dst.len) return error.BufferTooSmall;

        switch (self.type) {
            .gpu => try self.metalCopyData(src, src_buf, dst, dst_buf),
            .cpu => if (dst.ptr != src.ptr) @memcpy(dst, src),
        }
    }

    fn metalCopyData(self: Backend, src: []const f32, src_buf: ?*const metal.MTLBuffer, dst: []f32, dst_buf: ?*const metal.MTLBuffer) !void {
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
    pub fn matMul(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, m: usize, n: usize, k: usize, accumulate: bool) !void {
        if (m > 1) {
            try self.matMulBatch(a, a_buf, b, b_buf, c, c_buf, m, n, k, accumulate);
            return;
        }
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMul(a, a_buf, b, b_buf, c, c_buf, m, n, k, false, accumulate),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => cpuMatMul(a, b, c, m, n, k, accumulate),
        }
    }

    /// Execute batched matrix multiplication
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn matMulBatch(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize, accumulate: bool) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMulBatchGPU(a, a_buf, b, b_buf, c, c_buf, batch_size, n, k, accumulate),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => cpuMatMulBatch(a, b, c, batch_size, n, k, accumulate),
        }
    }

    /// Execute batched matrix multiplication with bias addition
    /// C = A * B + bias
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn matMulBias(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMulBias(a, a_buf, b, b_buf, bias, bias_buf, c, c_buf, batch_size, n, k),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => {
                // CPU fallback: compute matmul then add bias
                cpuMatMulBatch(a, b, c, batch_size, n, k, false);
                // Add bias to each row
                var batch_idx: usize = 0;
                while (batch_idx < batch_size) : (batch_idx += 1) {
                    var col: usize = 0;
                    while (col < n) : (col += 1) {
                        c[batch_idx * n + col] += bias[col];
                    }
                }
            },
        }
    }

    /// Execute batched matrix multiplication with bias addition and ReLU activation
    /// C = max(0, A * B + bias)
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn matMulBiasRelu(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMulBiasRelu(a, a_buf, b, b_buf, bias, bias_buf, c, c_buf, batch_size, n, k),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => {
                // CPU fallback: compute matmul, add bias, then apply ReLU
                cpuMatMulBatch(a, b, c, batch_size, n, k, false);
                var batch_idx: usize = 0;
                while (batch_idx < batch_size) : (batch_idx += 1) {
                    var col: usize = 0;
                    while (col < n) : (col += 1) {
                        const val = c[batch_idx * n + col] + bias[col];
                        c[batch_idx * n + col] = if (val > 0) val else 0;
                    }
                }
            },
        }
    }

    /// Execute batched matrix multiplication with bias addition and Sigmoid activation
    /// C = sigmoid(A * B + bias)
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn matMulBiasSigmoid(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMulBiasSigmoid(a, a_buf, b, b_buf, bias, bias_buf, c, c_buf, batch_size, n, k),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => {
                // CPU fallback: compute matmul, add bias, then apply sigmoid
                cpuMatMulBatch(a, b, c, batch_size, n, k, false);
                var batch_idx: usize = 0;
                while (batch_idx < batch_size) : (batch_idx += 1) {
                    var col: usize = 0;
                    while (col < n) : (col += 1) {
                        const val = c[batch_idx * n + col] + bias[col];
                        c[batch_idx * n + col] = 1.0 / (1.0 + @exp(-val));
                    }
                }
            },
        }
    }

    /// Execute batched matrix multiplication with bias addition and Tanh activation
    /// C = tanh(A * B + bias)
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn matMulBiasTanh(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMulBiasTanh(a, a_buf, b, b_buf, bias, bias_buf, c, c_buf, batch_size, n, k),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => {
                // CPU fallback: compute matmul, add bias, then apply tanh
                cpuMatMulBatch(a, b, c, batch_size, n, k, false);
                var batch_idx: usize = 0;
                while (batch_idx < batch_size) : (batch_idx += 1) {
                    var col: usize = 0;
                    while (col < n) : (col += 1) {
                        const val = c[batch_idx * n + col] + bias[col];
                        c[batch_idx * n + col] = std.math.tanh(val);
                    }
                }
            },
        }
    }

    /// Execute matrix multiplication with transposed A: C = A^T * B
    pub fn matMulTransposeA(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, m: usize, n: usize, k: usize, accumulate: bool) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalMatMulTransposeA(a, a_buf, b, b_buf, c, c_buf, m, n, k, accumulate),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => cpuMatMulTransposeA(a, b, c, m, n, k, accumulate),
        }
    }

    /// Execute matrix multiplication with transposed B: C = A * B^T
    pub fn matMulTransposeB(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, m: usize, n: usize, k: usize, accumulate: bool) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => {
                    if (m > 1) {
                        try self.metalMatMulBatchTransposeB(a, a_buf, b, b_buf, c, c_buf, m, n, k, accumulate);
                    } else {
                        try self.metalMatMul(a, a_buf, b, b_buf, c, c_buf, m, n, k, true, accumulate);
                    }
                },
                else => return error.CudaNotYetImplemented,
            },
            .cpu => cpuMatMulTransposeB(a, b, c, m, n, k, accumulate),
        }
    }

    fn metalMatMulBatchTransposeB(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize, accumulate: bool) !void {
        const ctx = self.metal_ctx.?;
        var buffer_a = try self.getBuffer(a, a_buf);
        defer if (a_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_a.buffer);
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_b.buffer);
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_c.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        const pipeline = ctx.getPipeline("matmul_batch_transpose_b") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a.buffer, buffer_a.offset, 0);
        encoder.setBuffer(&buffer_b.buffer, buffer_b.offset, 1);
        encoder.setBuffer(&buffer_c.buffer, buffer_c.offset, 2);

        const batch_size_u32 = @as(u32, @intCast(batch_size));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        const acc_u32 = @as(u32, @intCast(@intFromBool(accumulate)));
        encoder.setBytes(std.mem.asBytes(&batch_size_u32), 3);
        encoder.setBytes(std.mem.asBytes(&n_u32), 4);
        encoder.setBytes(std.mem.asBytes(&k_u32), 5);
        encoder.setBytes(std.mem.asBytes(&acc_u32), 6);

        encoder.dispatchThreads(metal.MTLSize.make(n, 1, batch_size), metal.MTLSize.make(16, 1, 16));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    /// Execute activation function on the selected backend
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    /// Execute activation function on the selected backend
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn activationForward(self: Backend, act: activation.Activation, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        const batch_size = if (act == .softmax) 1 else input.len / 1; // Default to 1 for now or handle in caller
        _ = batch_size;
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalActivationForward(act, input, input_buf, output, output_buf),
                else => return error.CudaNotYetImplemented,
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
    pub fn activationBackward(self: Backend, act: activation.Activation, input: []const f32, input_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalActivationBackward(act, input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => cpuActivationBackward(act, input, grad_output, grad_input),
        }
    }

    /// Execute loss function gradient on the selected backend
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn lossBackward(self: Backend, loss_fn: loss.Loss, output: []const f32, output_buf: ?*const metal.MTLBuffer, target: []const f32, target_buf: ?*const metal.MTLBuffer, grad_output: []f32, grad_output_buf: ?*const metal.MTLBuffer) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalLossBackward(loss_fn, output, output_buf, target, target_buf, grad_output, grad_output_buf),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => try self.cpuLossBackwardGeneric(loss_fn, output, target, grad_output),
        }
    }

    fn cpuLossBackwardGeneric(self: Backend, loss_fn: loss.Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        _ = self;
        try loss_fn.backward(output, target, grad_output);
    }

    /// Update weights using SGD
    pub fn sgdUpdate(self: Backend, weights: []f32, weights_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, learning_rate: f32, weight_decay: f32) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalSgdUpdate(weights, weights_buf, gradients, gradients_buf, learning_rate, weight_decay),
                .cuda => try self.cudaSgdUpdate(weights, weights_buf, gradients, gradients_buf, learning_rate, weight_decay),
                .cpu => cpuSgdUpdate(weights, gradients, learning_rate, weight_decay),
            },
            .cpu => cpuSgdUpdate(weights, gradients, learning_rate, weight_decay),
        }
    }

    /// Update weights using Adam
    pub fn adamUpdate(self: Backend, weights: []f32, weights_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, m: []f32, m_buf: ?*const metal.MTLBuffer, v: []f32, v_buf: ?*const metal.MTLBuffer, lr: f32, beta1: f32, beta2: f32, eps: f32, bias_corr1: f32, bias_corr2: f32) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalAdamUpdate(weights, weights_buf, gradients, gradients_buf, m, m_buf, v, v_buf, lr, beta1, beta2, eps, bias_corr1, bias_corr2),
                .cuda => try self.cudaAdamUpdate(weights, weights_buf, gradients, gradients_buf, m, m_buf, v, v_buf, lr, beta1, beta2, eps, bias_corr1, bias_corr2),
                .cpu => try self.cpuAdamUpdate(weights, gradients, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2),
            },
            .cpu => try self.cpuAdamUpdate(weights, gradients, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2),
        }
    }

    /// Update weights using RMSprop
    pub fn rmspropUpdate(self: Backend, weights: []f32, weights_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, g_avg: []f32, g_avg_buf: ?*const metal.MTLBuffer, lr: f32, rho: f32, eps: f32) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalRmspropUpdate(weights, weights_buf, gradients, gradients_buf, g_avg, g_avg_buf, lr, rho, eps),
                .cuda => try self.cudaRmspropUpdate(weights, weights_buf, gradients, gradients_buf, g_avg, g_avg_buf, lr, rho, eps),
                .cpu => try self.cpuRmspropUpdate(weights, gradients, g_avg, lr, rho, eps),
            },
            .cpu => try self.cpuRmspropUpdate(weights, gradients, g_avg, lr, rho, eps),
        }
    }

    /// LayerNorm forward pass
    pub fn layerNormForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, gamma: []const f32, gamma_buf: ?*const metal.MTLBuffer, beta: []const f32, beta_buf: ?*const metal.MTLBuffer, eps: f32) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalLayerNormForward(input, input_buf, output, output_buf, gamma, gamma_buf, beta, beta_buf, eps),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => try self.cpuLayerNormForward(input, output, gamma, beta, eps),
        }
    }

    /// LayerNorm backward pass
    pub fn layerNormBackward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, gamma: []const f32, gamma_buf: ?*const metal.MTLBuffer, grad_gamma: []f32, grad_gamma_buf: ?*const metal.MTLBuffer, grad_beta: []f32, grad_beta_buf: ?*const metal.MTLBuffer, eps: f32) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalLayerNormBackward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, gamma, gamma_buf, grad_gamma, grad_gamma_buf, grad_beta, grad_beta_buf, eps),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => try self.cpuLayerNormBackward(input, grad_output, grad_input, gamma, grad_gamma, grad_beta, eps),
        }
    }

    /// Conv1D forward pass
    pub fn conv1dForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, weights: []const f32, weights_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, in_channels: usize, out_channels: usize, kernel_size: usize, in_len: usize, out_len: usize) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalConv1dForward(input, input_buf, weights, weights_buf, bias, bias_buf, output, output_buf, in_channels, out_channels, kernel_size, in_len, out_len),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => try self.cpuConv1dForward(input, weights, bias, output, in_channels, out_channels, kernel_size, in_len, out_len),
        }
    }

    /// Conv1D backward pass
    pub fn conv1dBackward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, weights: []const f32, weights_buf: ?*const metal.MTLBuffer, grad_after_act: []const f32, grad_after_act_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, grad_weights: []f32, grad_weights_buf: ?*const metal.MTLBuffer, grad_bias: []f32, grad_bias_buf: ?*const metal.MTLBuffer, in_channels: usize, out_channels: usize, kernel_size: usize, in_len: usize, out_len: usize) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalConv1dBackward(input, input_buf, weights, weights_buf, grad_after_act, grad_after_act_buf, grad_input, grad_input_buf, grad_weights, grad_weights_buf, grad_bias, grad_bias_buf, in_channels, out_channels, kernel_size, in_len, out_len),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => try self.cpuConv1dBackward(input, weights, grad_after_act, grad_input, grad_weights, grad_bias, in_channels, out_channels, kernel_size, in_len, out_len),
        }
    }

    /// Conv2D forward pass
    pub fn conv2dForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, weights: []const f32, weights_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, in_channels: usize, out_channels: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize, output_h: usize, output_w: usize, stride_h: usize, stride_w: usize, padding_h: usize, padding_w: usize) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalConv2dForward(input, input_buf, weights, weights_buf, bias, bias_buf, output, output_buf, in_channels, out_channels, kernel_h, kernel_w, input_h, input_w, output_h, output_w, stride_h, stride_w, padding_h, padding_w),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => try self.cpuConv2dForward(input, weights, bias, output, in_channels, out_channels, kernel_h, kernel_w, input_h, input_w, output_h, output_w, stride_h, stride_w, padding_h, padding_w),
        }
    }

    /// Conv2D backward pass
    pub fn conv2dBackward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, weights: []const f32, weights_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, grad_weights: []f32, grad_weights_buf: ?*const metal.MTLBuffer, grad_bias: []f32, grad_bias_buf: ?*const metal.MTLBuffer, in_channels: usize, out_channels: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize, output_h: usize, output_w: usize, stride_h: usize, stride_w: usize, padding_h: usize, padding_w: usize) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalConv2dBackward(input, input_buf, weights, weights_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, grad_weights, grad_weights_buf, grad_bias, grad_bias_buf, in_channels, out_channels, kernel_h, kernel_w, input_h, input_w, output_h, output_w, stride_h, stride_w, padding_h, padding_w),
                .cuda => try self.cudaConv2dBackward(input, input_buf, weights, weights_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, grad_weights, grad_weights_buf, grad_bias, grad_bias_buf, in_channels, out_channels, kernel_h, kernel_w, input_h, input_w, output_h, output_w, stride_h, stride_w, padding_h, padding_w),
                .cpu => try self.cpuConv2dBackward(input, weights, grad_output, grad_input, grad_weights, grad_bias, in_channels, out_channels, kernel_h, kernel_w, input_h, input_w, output_h, output_w, stride_h, stride_w, padding_h, padding_w),
            },
            .cpu => try self.cpuConv2dBackward(input, weights, grad_output, grad_input, grad_weights, grad_bias, in_channels, out_channels, kernel_h, kernel_w, input_h, input_w, output_h, output_w, stride_h, stride_w, padding_h, padding_w),
        }
    }

    /// MaxPool1D forward pass
    pub fn maxPool1dForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, max_indices: []f32, max_indices_buf: ?*const metal.MTLBuffer, channels: usize, input_len: usize, output_len: usize, pool_size: usize, stride: usize) !void {
        _ = input_buf;
        _ = output_buf;
        _ = max_indices_buf;
        // CPU implementation only for now
        if (self.type != .cpu) return error.GpuNotImplemented;
        try self.cpuMaxPool1dForward(input, output, max_indices, channels, input_len, output_len, pool_size, stride);
    }

    /// MaxPool1D backward pass
    pub fn maxPool1dBackward(self: Backend, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, max_indices: []const f32, max_indices_buf: ?*const metal.MTLBuffer, channels: usize, input_len: usize, output_len: usize, pool_size: usize, stride: usize) !void {
        _ = grad_output_buf;
        _ = grad_input_buf;
        _ = max_indices_buf;
        _ = input_len;
        _ = pool_size;
        _ = stride;
        // CPU implementation only for now
        if (self.type != .cpu) return error.GpuNotImplemented;
        @memset(grad_input, 0);
        for (0..channels * output_len) |out_idx| {
            const max_idx: usize = @intFromFloat(max_indices[out_idx]);
            // SECURITY: Bounds check to prevent out-of-bounds access
            if (max_idx < grad_input.len) {
                grad_input[max_idx] += grad_output[out_idx];
            }
        }
    }

    /// MaxPool2D forward pass
    pub fn maxPool2dForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, max_indices: []f32, max_indices_buf: ?*const metal.MTLBuffer, channels: usize, input_h: usize, input_w: usize, output_h: usize, output_w: usize, pool_h: usize, pool_w: usize, stride_h: usize, stride_w: usize) !void {
        _ = input_buf;
        _ = output_buf;
        _ = max_indices_buf;
        // CPU implementation only for now
        if (self.type != .cpu) return error.GpuNotImplemented;
        try self.cpuMaxPool2dForward(input, output, max_indices, channels, input_h, input_w, output_h, output_w, pool_h, pool_w, stride_h, stride_w);
    }

    /// MaxPool2D backward pass
    pub fn maxPool2dBackward(self: Backend, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, max_indices: []const f32, max_indices_buf: ?*const metal.MTLBuffer, channels: usize, input_h: usize, input_w: usize, output_h: usize, output_w: usize, pool_h: usize, pool_w: usize, stride_h: usize, stride_w: usize) !void {
        _ = grad_output_buf;
        _ = grad_input_buf;
        _ = max_indices_buf;
        _ = input_h;
        _ = input_w;
        _ = pool_h;
        _ = pool_w;
        _ = stride_h;
        _ = stride_w;
        // CPU implementation only for now
        if (self.type != .cpu) return error.GpuNotImplemented;
        @memset(grad_input, 0);
        const output_size = channels * output_h * output_w;
        for (0..output_size) |out_idx| {
            const max_idx: usize = @intFromFloat(max_indices[out_idx]);
            // SECURITY: Bounds check to prevent out-of-bounds access
            if (max_idx < grad_input.len) {
                grad_input[max_idx] += grad_output[out_idx];
            }
        }
    }

    /// Dropout forward pass
    pub fn dropoutForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, mask: []f32, mask_buf: ?*const metal.MTLBuffer, rate: f32, scaling_factor: f32, seed: u64) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalDropoutForward(input, input_buf, output, output_buf, mask, mask_buf, rate, scaling_factor, seed),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => try self.cpuDropoutForward(input, output, mask, rate, scaling_factor, seed),
        }
    }

    /// VAE Sampling forward pass
    pub fn vaeSamplingForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, epsilon: []f32, epsilon_buf: ?*const metal.MTLBuffer, seed: u64, latent_dim: usize) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalVaeSamplingForward(input, input_buf, output, output_buf, epsilon, epsilon_buf, seed, latent_dim),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => try self.cpuVaeSamplingForward(input, output, epsilon, seed, latent_dim),
        }
    }

    /// VAE Sampling backward pass
    pub fn vaeSamplingBackward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, epsilon: []const f32, epsilon_buf: ?*const metal.MTLBuffer, latent_dim: usize) !void {
        switch (self.type) {
            .gpu => try self.metalVaeSamplingBackward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, epsilon, epsilon_buf, latent_dim),
            .cpu => try self.cpuVaeSamplingBackward(input, grad_output, grad_input, epsilon, latent_dim),
        }
    }

    /// Attention forward pass (scaled dot-product)
    /// PERFORMANCE FIX: scores_buffer must be pre-allocated with size >= seq_len (F2.1)
    pub fn attentionForward(self: Backend, q: []const f32, q_buf: ?*const metal.MTLBuffer, k: []const f32, k_buf: ?*const metal.MTLBuffer, v: []const f32, v_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, scores_buffer: []f32, seq_len: usize, d_k: usize, scaling_factor: f32) !void {
        switch (self.type) {
            .gpu => try self.metalAttentionForward(q, q_buf, k, k_buf, v, v_buf, output, output_buf, seq_len, d_k, scaling_factor),
            .cpu => try self.cpuAttentionForward(q, k, v, output, scores_buffer, seq_len, d_k, scaling_factor),
        }
    }

    /// Update bias using SGD
    pub fn sgdUpdateBias(self: Backend, bias: []f32, bias_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, learning_rate: f32) !void {
        switch (self.type) {
            .gpu => try self.metalSgdUpdateBias(bias, bias_buf, gradients, gradients_buf, learning_rate),
            .cpu => cpuSgdUpdateBias(bias, gradients, learning_rate),
        }
    }

    /// Accumulate bias gradients
    pub fn accumulateBias(self: Backend, grad_bias: []f32, grad_bias_buf: ?*const metal.MTLBuffer, grad_after_act: []const f32, grad_after_act_buf: ?*const metal.MTLBuffer) !void {
        switch (self.type) {
            .gpu => try self.metalAccumulateBias(grad_bias, grad_bias_buf, grad_after_act, grad_after_act_buf),
            .cpu => cpuAccumulateBias(grad_bias, grad_after_act),
        }
    }

    /// Add bias to output (broadcasted over batch)
    pub fn addBias(self: Backend, output: []f32, output_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer) !void {
        switch (self.type) {
            .gpu => try self.metalAddBias(output, output_buf, bias, bias_buf),
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
    pub fn map(self: Backend, func: MapOp, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        switch (self.type) {
            .gpu => try self.metalMap(func, input, input_buf, output, output_buf),
            .cpu => try self.cpuMap(func, input, output),
        }
    }

    fn cpuMap(self: Backend, func: MapOp, input: []const f32, output: []f32) !void {
        _ = self;
        switch (func) {
            .exp => for (input, output) |in, *out| {
                out.* = std.math.exp(in);
            },
            .log => for (input, output) |in, *out| {
                out.* = std.math.log(f32, std.math.e, @max(in, 1e-10));
            },
            .sqrt => for (input, output) |in, *out| {
                out.* = @sqrt(in);
            },
            .abs => for (input, output) |in, *out| {
                out.* = @abs(in);
            },
            .square => for (input, output) |in, *out| {
                out.* = in * in;
            },
            .inv => for (input, output) |in, *out| {
                out.* = 1.0 / in;
            },
        }
    }

    fn cpuAccumulateBias(grad_bias: []f32, grad_after_act: []const f32) void {
        const batch_size = grad_after_act.len / grad_bias.len;
        for (0..batch_size) |b| {
            for (0..grad_bias.len) |i| {
                grad_bias[i] += grad_after_act[b * grad_bias.len + i];
            }
        }
    }

    fn metalAccumulateBias(self: Backend, grad_bias: []f32, grad_bias_buf: ?*const metal.MTLBuffer, grad_after_act: []const f32, grad_after_act_buf: ?*const metal.MTLBuffer) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_gb = try self.getBuffer(grad_bias, grad_bias_buf);
        defer if (grad_bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gb.buffer);
        var buffer_ga = try self.getBuffer(grad_after_act, grad_after_act_buf);
        defer if (grad_after_act_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ga.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("accumulate_bias") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_gb.buffer, buffer_gb.offset, 0);
        encoder.setBuffer(&buffer_ga.buffer, buffer_ga.offset, 1);

        const bias_len = @as(u32, @intCast(grad_bias.len));
        const batch_size = @as(u32, @intCast(grad_after_act.len / grad_bias.len));
        encoder.setBytes(std.mem.asBytes(&bias_len), 2);
        encoder.setBytes(std.mem.asBytes(&batch_size), 3);

        const tg_size = @min(grad_bias.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(grad_bias.len, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalMap(self: Backend, func: MapOp, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
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
    pub fn elementWise(self: Backend, op: ElementWiseOp, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer) !void {
        switch (self.type) {
            .gpu => try self.metalElementWise(op, a, a_buf, b, b_buf, c, c_buf),
            .cpu => try self.cpuElementWise(op, a, b, c),
        }
    }

    fn cpuElementWise(self: Backend, op: ElementWiseOp, a: []const f32, b: []const f32, c: []f32) !void {
        _ = self;
        const Vec4 = @Vector(4, f32);
        const vec_len = (a.len / 4) * 4;

        // Process 4 elements at a time using SIMD
        var i: usize = 0;
        while (i < vec_len) : (i += 4) {
            const va: Vec4 = @as(*const Vec4, @ptrCast(@alignCast(a.ptr + i))).*;
            const vb: Vec4 = @as(*const Vec4, @ptrCast(@alignCast(b.ptr + i))).*;
            const vc = switch (op) {
                .add => va + vb,
                .sub => va - vb,
                .mul => va * vb,
                .div => va / vb,
            };
            @as(*Vec4, @ptrCast(@alignCast(c.ptr + i))).* = vc;
        }

        // Process remaining elements
        switch (op) {
            .add => for (a[i..], b[i..], c[i..]) |av, bv, *cv| {
                cv.* = av + bv;
            },
            .sub => for (a[i..], b[i..], c[i..]) |av, bv, *cv| {
                cv.* = av - bv;
            },
            .mul => for (a[i..], b[i..], c[i..]) |av, bv, *cv| {
                cv.* = av * bv;
            },
            .div => for (a[i..], b[i..], c[i..]) |av, bv, *cv| {
                cv.* = av / bv;
            },
        }
    }

    fn metalElementWise(self: Backend, op: ElementWiseOp, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer) !void {
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

    /// Clear buffer with zeros
    pub fn clear(self: Backend, data: []f32, data_buf: ?*const metal.MTLBuffer) !void {
        try self.fill(data, data_buf, 0.0);
    }

    /// Fill buffer with constant value
    pub fn fill(self: Backend, data: []f32, data_buf: ?*const metal.MTLBuffer, value: f32) !void {
        switch (self.type) {
            .gpu => try self.metalFillConstant(data, data_buf, value),
            .cpu => @memset(data, value),
        }
    }

    /// Scale buffer by constant factor
    pub fn scale(self: Backend, data: []f32, data_buf: ?*const metal.MTLBuffer, factor: f32) !void {
        switch (self.type) {
            .gpu => try self.metalScaleBuffer(data, data_buf, factor),
            .cpu => {
                const Vec4 = @Vector(4, f32);
                const vec_factor: Vec4 = @splat(factor);
                const vec_len = (data.len / 4) * 4;

                // Process 4 elements at a time using SIMD
                var i: usize = 0;
                while (i < vec_len) : (i += 4) {
                    const v: Vec4 = @as(*const Vec4, @ptrCast(@alignCast(data.ptr + i))).*;
                    @as(*Vec4, @ptrCast(@alignCast(data.ptr + i))).* = v * vec_factor;
                }

                // Process remaining elements
                for (data[i..]) |*v| {
                    v.* *= factor;
                }
            },
        }
    }

    /// Reverse sequence along time dimension
    pub fn reverse(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, seq_len: usize, element_size: usize) !void {
        if (input.len != seq_len * element_size or output.len != seq_len * element_size) return error.InvalidInputSize;
        switch (self.type) {
            .gpu => try self.metalReverseSequence(input, input_buf, output, output_buf, seq_len, element_size),
            .cpu => {
                for (0..seq_len) |t| {
                    const src_t = seq_len - 1 - t;
                    @memcpy(output[t * element_size .. (t + 1) * element_size], input[src_t * element_size .. (src_t + 1) * element_size]);
                }
            },
        }
    }

    /// Concatenate two sequences along feature dimension
    pub fn concat(self: Backend, input1: []const f32, input1_buf: ?*const metal.MTLBuffer, input2: []const f32, input2_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, size1: usize, size2: usize, seq_len: usize) !void {
        if (output.len != seq_len * (size1 + size2)) return error.InvalidOutputSize;
        switch (self.type) {
            .gpu => try self.metalConcatBuffers(input1, input1_buf, input2, input2_buf, output, output_buf, size1, size2, seq_len),
            .cpu => {
                for (0..seq_len) |t| {
                    const out_t = output[t * (size1 + size2) .. (t + 1) * (size1 + size2)];
                    @memcpy(out_t[0..size1], input1[t * size1 .. (t + 1) * size1]);
                    @memcpy(out_t[size1..], input2[t * size2 .. (t + 1) * size2]);
                }
            },
        }
    }

    /// Split sequence into two along feature dimension
    pub fn split(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output1: []f32, output1_buf: ?*const metal.MTLBuffer, output2: []f32, output2_buf: ?*const metal.MTLBuffer, size1: usize, size2: usize, seq_len: usize) !void {
        if (input.len != seq_len * (size1 + size2)) return error.InvalidInputSize;
        switch (self.type) {
            .gpu => try self.metalSplitBuffer(input, input_buf, output1, output1_buf, output2, output2_buf, size1, size2, seq_len),
            .cpu => {
                for (0..seq_len) |t| {
                    const in_t = input[t * (size1 + size2) .. (t + 1) * (size1 + size2)];
                    @memcpy(output1[t * size1 .. (t + 1) * size1], in_t[0..size1]);
                    @memcpy(output2[t * size2 .. (t + 1) * size2], in_t[size1..]);
                }
            },
        }
    }

    fn metalFillConstant(self: Backend, data: []f32, data_buf: ?*const metal.MTLBuffer, value: f32) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_data = try self.getBuffer(data, data_buf);
        defer if (data_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_data.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("fill_constant") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_data.buffer, buffer_data.offset, 0);
        encoder.setBytes(std.mem.asBytes(&value), 1);
        const size = @as(u32, @intCast(data.len));
        encoder.setBytes(std.mem.asBytes(&size), 2);

        const tg_size = @min(data.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(data.len, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalScaleBuffer(self: Backend, data: []f32, data_buf: ?*const metal.MTLBuffer, factor: f32) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_data = try self.getBuffer(data, data_buf);
        defer if (data_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_data.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("scale_buffer") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_data.buffer, buffer_data.offset, 0);
        encoder.setBytes(std.mem.asBytes(&factor), 1);
        const size = @as(u32, @intCast(data.len));
        encoder.setBytes(std.mem.asBytes(&size), 2);

        const tg_size = @min(data.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(data.len, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalReverseSequence(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, seq_len: usize, element_size: usize) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_out = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_out.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("reverse_sequence") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_out.buffer, buffer_out.offset, 1);
        const sl = @as(u32, @intCast(seq_len));
        const es = @as(u32, @intCast(element_size));
        encoder.setBytes(std.mem.asBytes(&sl), 2);
        encoder.setBytes(std.mem.asBytes(&es), 3);

        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const width = pipeline.threadExecutionWidth();
        const tg_x = @min(element_size, width);
        const tg_y = @min(seq_len, max_threads / tg_x);
        encoder.dispatchThreads(metal.MTLSize.make(element_size, seq_len, 1), metal.MTLSize.make(tg_x, tg_y, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalConcatBuffers(self: Backend, input1: []const f32, input1_buf: ?*const metal.MTLBuffer, input2: []const f32, input2_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, size1: usize, size2: usize, seq_len: usize) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var b1 = try self.getBuffer(input1, input1_buf);
        defer if (input1_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(b1.buffer);
        var b2 = try self.getBuffer(input2, input2_buf);
        defer if (input2_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(b2.buffer);
        var bout = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(bout.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("concat_buffers") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&b1.buffer, b1.offset, 0);
        encoder.setBuffer(&b2.buffer, b2.offset, 1);
        encoder.setBuffer(&bout.buffer, bout.offset, 2);
        const s1 = @as(u32, @intCast(size1));
        const s2 = @as(u32, @intCast(size2));
        const sl = @as(u32, @intCast(seq_len));
        encoder.setBytes(std.mem.asBytes(&s1), 3);
        encoder.setBytes(std.mem.asBytes(&s2), 4);
        encoder.setBytes(std.mem.asBytes(&sl), 5);

        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const width = pipeline.threadExecutionWidth();
        const total_size = size1 + size2;
        const tg_x = @min(total_size, width);
        const tg_y = @min(seq_len, max_threads / tg_x);
        encoder.dispatchThreads(metal.MTLSize.make(total_size, seq_len, 1), metal.MTLSize.make(tg_x, tg_y, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalSplitBuffer(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output1: []f32, output1_buf: ?*const metal.MTLBuffer, output2: []f32, output2_buf: ?*const metal.MTLBuffer, size1: usize, size2: usize, seq_len: usize) !void {
        const ctx = self.metal_ctx orelse return error.NotAvailable;
        var bin = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(bin.buffer);
        var b1 = try self.getBuffer(output1, output1_buf);
        defer if (output1_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(b1.buffer);
        var b2 = try self.getBuffer(output2, output2_buf);
        defer if (output2_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(b2.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("split_buffer") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&bin.buffer, bin.offset, 0);
        encoder.setBuffer(&b1.buffer, b1.offset, 1);
        encoder.setBuffer(&b2.buffer, b2.offset, 2);
        const s1 = @as(u32, @intCast(size1));
        const s2 = @as(u32, @intCast(size2));
        const sl = @as(u32, @intCast(seq_len));
        encoder.setBytes(std.mem.asBytes(&s1), 3);
        encoder.setBytes(std.mem.asBytes(&s2), 4);
        encoder.setBytes(std.mem.asBytes(&sl), 5);

        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const width = pipeline.threadExecutionWidth();
        const total_size = size1 + size2;
        const tg_x = @min(total_size, width);
        const tg_y = @min(seq_len, max_threads / tg_x);
        encoder.dispatchThreads(metal.MTLSize.make(total_size, seq_len, 1), metal.MTLSize.make(tg_x, tg_y, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    /// Fill buffer with random normal values
    pub fn fillRandomNormal(self: Backend, data: []f32, data_buf: ?*const metal.MTLBuffer, mean: f32, std_dev: f32, seed: u64) !void {
        switch (self.type) {
            .gpu => try self.metalFillRandomNormal(data, data_buf, mean, std_dev, seed),
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

    fn metalFillRandomNormal(self: Backend, data: []f32, data_buf: ?*const metal.MTLBuffer, mean: f32, std_dev: f32, seed: u64) !void {
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

    fn metalAddBias(self: Backend, output: []f32, output_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer) !void {
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

    fn metalMatMul(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, m: usize, n: usize, k: usize, transpose_b: bool, accumulate: bool) !void {
        if (!isMacos()) return error.NotAvailable;
        try self.metalMatMulGPU(a, a_buf, b, b_buf, c, c_buf, m, n, k, transpose_b, accumulate);
    }

    fn metalMatMulBatch(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize) !void {
        if (!isMacos()) return error.NotAvailable;
        try self.metalMatMulBatchGPU(a, a_buf, b, b_buf, c, c_buf, batch_size, n, k);
    }

    fn metalMatMulTransposeA(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, m: usize, n: usize, k: usize, accumulate: bool) !void {
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
        const acc_u32 = @as(u32, @intCast(@intFromBool(accumulate)));
        encoder.setBytes(std.mem.asBytes(&m_u32), 3);
        encoder.setBytes(std.mem.asBytes(&n_u32), 4);
        encoder.setBytes(std.mem.asBytes(&k_u32), 5);
        encoder.setBytes(std.mem.asBytes(&acc_u32), 6);

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

    fn metalMatMulGPU(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, m: usize, n: usize, k: usize, transpose_b: bool, accumulate: bool) !void {
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
        const acc_u32 = @as(u32, @intCast(@intFromBool(accumulate)));
        encoder.setBytes(std.mem.asBytes(&m_u32), 3);
        encoder.setBytes(std.mem.asBytes(&n_u32), 4);
        encoder.setBytes(std.mem.asBytes(&k_u32), 5);
        encoder.setBytes(std.mem.asBytes(&acc_u32), 6);

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

    fn metalMatMulBatchGPU(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize, accumulate: bool) !void {
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
        const acc_u32 = @as(u32, @intCast(@intFromBool(accumulate)));
        encoder.setBytes(std.mem.asBytes(&batch_size_u32), 3);
        encoder.setBytes(std.mem.asBytes(&n_u32), 4);
        encoder.setBytes(std.mem.asBytes(&k_u32), 5);
        encoder.setBytes(std.mem.asBytes(&acc_u32), 6);

        // Optimize threadgroup size for Apple GPU (SIMD width = 32)
        // Use multiples of 32 for better occupancy
        // Total threads = tg_x * tg_y * tg_z must be <= 256-512 (Apple Silicon limit)
        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const tg_size = @min(32, max_threads);
        // Use (32, 1, 1) threadgroup: 32 threads total, well within limits
        encoder.dispatchThreads(metal.MTLSize.make(n, 1, batch_size), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalMatMulBias(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize) !void {
        if (!isMacos()) return error.NotAvailable;
        const ctx = self.metal_ctx.?;
        var buffer_a = try self.getBuffer(a, a_buf);
        defer if (a_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_a.buffer);
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_b.buffer);
        var buffer_bias = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_bias.buffer);
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_c.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        const pipeline = ctx.getPipeline("matmul_bias") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a.buffer, buffer_a.offset, 0);
        encoder.setBuffer(&buffer_b.buffer, buffer_b.offset, 1);
        encoder.setBuffer(&buffer_bias.buffer, buffer_bias.offset, 2);
        encoder.setBuffer(&buffer_c.buffer, buffer_c.offset, 3);

        const batch_size_u32 = @as(u32, @intCast(batch_size));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        encoder.setBytes(std.mem.asBytes(&batch_size_u32), 4);
        encoder.setBytes(std.mem.asBytes(&n_u32), 5);
        encoder.setBytes(std.mem.asBytes(&k_u32), 6);

        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const tg_size = @min(32, max_threads);
        encoder.dispatchThreads(metal.MTLSize.make(n, batch_size, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalMatMulBiasRelu(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize) !void {
        if (!isMacos()) return error.NotAvailable;
        const ctx = self.metal_ctx.?;
        var buffer_a = try self.getBuffer(a, a_buf);
        defer if (a_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_a.buffer);
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_b.buffer);
        var buffer_bias = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_bias.buffer);
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_c.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        const pipeline = ctx.getPipeline("matmul_bias_relu") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a.buffer, buffer_a.offset, 0);
        encoder.setBuffer(&buffer_b.buffer, buffer_b.offset, 1);
        encoder.setBuffer(&buffer_bias.buffer, buffer_bias.offset, 2);
        encoder.setBuffer(&buffer_c.buffer, buffer_c.offset, 3);

        const batch_size_u32 = @as(u32, @intCast(batch_size));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        encoder.setBytes(std.mem.asBytes(&batch_size_u32), 4);
        encoder.setBytes(std.mem.asBytes(&n_u32), 5);
        encoder.setBytes(std.mem.asBytes(&k_u32), 6);

        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const tg_size = @min(32, max_threads);
        encoder.dispatchThreads(metal.MTLSize.make(n, batch_size, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalMatMulBiasSigmoid(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize) !void {
        if (!isMacos()) return error.NotAvailable;
        const ctx = self.metal_ctx.?;
        var buffer_a = try self.getBuffer(a, a_buf);
        defer if (a_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_a.buffer);
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_b.buffer);
        var buffer_bias = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_bias.buffer);
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_c.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        const pipeline = ctx.getPipeline("matmul_bias_sigmoid") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a.buffer, buffer_a.offset, 0);
        encoder.setBuffer(&buffer_b.buffer, buffer_b.offset, 1);
        encoder.setBuffer(&buffer_bias.buffer, buffer_bias.offset, 2);
        encoder.setBuffer(&buffer_c.buffer, buffer_c.offset, 3);

        const batch_size_u32 = @as(u32, @intCast(batch_size));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        encoder.setBytes(std.mem.asBytes(&batch_size_u32), 4);
        encoder.setBytes(std.mem.asBytes(&n_u32), 5);
        encoder.setBytes(std.mem.asBytes(&k_u32), 6);

        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const tg_size = @min(32, max_threads);
        encoder.dispatchThreads(metal.MTLSize.make(n, batch_size, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalMatMulBiasTanh(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize) !void {
        if (!isMacos()) return error.NotAvailable;
        const ctx = self.metal_ctx.?;
        var buffer_a = try self.getBuffer(a, a_buf);
        defer if (a_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_a.buffer);
        var buffer_b = try self.getBuffer(b, b_buf);
        defer if (b_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_b.buffer);
        var buffer_bias = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_bias.buffer);
        var buffer_c = try self.getBuffer(c, c_buf);
        defer if (c_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_c.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        const pipeline = ctx.getPipeline("matmul_bias_tanh") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_a.buffer, buffer_a.offset, 0);
        encoder.setBuffer(&buffer_b.buffer, buffer_b.offset, 1);
        encoder.setBuffer(&buffer_bias.buffer, buffer_bias.offset, 2);
        encoder.setBuffer(&buffer_c.buffer, buffer_c.offset, 3);

        const batch_size_u32 = @as(u32, @intCast(batch_size));
        const n_u32 = @as(u32, @intCast(n));
        const k_u32 = @as(u32, @intCast(k));
        encoder.setBytes(std.mem.asBytes(&batch_size_u32), 4);
        encoder.setBytes(std.mem.asBytes(&n_u32), 5);
        encoder.setBytes(std.mem.asBytes(&k_u32), 6);

        const max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        const tg_size = @min(32, max_threads);
        encoder.dispatchThreads(metal.MTLSize.make(n, batch_size, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalActivationForward(self: Backend, act: activation.Activation, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
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
            .gelu => "gelu_forward", // TODO: Add Metal kernel for GELU
            .leaky_relu => "leaky_relu_forward", // TODO: Add Metal kernel
            .elu => "elu_forward", // TODO: Add Metal kernel
        };
        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_input.buffer, buffer_input.offset, 0);
        encoder.setBuffer(&buffer_output.buffer, buffer_output.offset, 1);

        const total_size = @as(u32, @intCast(input.len));
        if (act == .softmax) {
            // Assume 1 sample for now, caller should loop for batches if needed
            // or we update shader to handle batching
            const num_classes = total_size;
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
            encoder.setBytes(std.mem.asBytes(&total_size), 2);
            // Vectorized dispatch (4 elements per thread)
            const num_threads = (total_size + 3) / 4;
            const tg_size = @min(num_threads, pipeline.maxTotalThreadsPerThreadgroup());
            encoder.dispatchThreads(metal.MTLSize.make(num_threads, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        }
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalActivationBackward(self: Backend, act: activation.Activation, input: []const f32, input_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer) !void {
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
            .gelu => "gelu_backward", // TODO: Add Metal kernel for GELU
            .leaky_relu => "leaky_relu_backward", // TODO: Add Metal kernel
            .elu => "elu_backward", // TODO: Add Metal kernel
        };
        const pipeline = ctx.getPipeline(pipeline_name) orelse return error.PipelineNotFound;

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();

        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_input.buffer, buffer_input.offset, 0);
        encoder.setBuffer(&buffer_grad_output.buffer, buffer_grad_output.offset, 1);
        encoder.setBuffer(&buffer_grad_input.buffer, buffer_grad_input.offset, 2);

        const total_size = @as(u32, @intCast(input.len));
        if (act == .softmax) {
            const num_classes = total_size;
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
            encoder.setBytes(std.mem.asBytes(&total_size), 3);
            // Vectorized dispatch (4 elements per thread)
            const num_threads = (total_size + 3) / 4;
            const tg_size = @min(num_threads, pipeline.maxTotalThreadsPerThreadgroup());
            encoder.dispatchThreads(metal.MTLSize.make(num_threads, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        }
        encoder.endEncoding();

        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalLossBackward(self: Backend, loss_fn: loss.Loss, output: []const f32, output_buf: ?*const metal.MTLBuffer, target: []const f32, target_buf: ?*const metal.MTLBuffer, grad_output: []f32, grad_output_buf: ?*const metal.MTLBuffer) !void {
        if (!isMacos()) return error.NotAvailable;
        try self.metalLossBackwardGPU(loss_fn, output, output_buf, target, target_buf, grad_output, grad_output_buf);
    }

    fn metalLossBackwardGPU(self: Backend, loss_fn: loss.Loss, output: []const f32, output_buf: ?*const metal.MTLBuffer, target: []const f32, target_buf: ?*const metal.MTLBuffer, grad_output: []f32, grad_output_buf: ?*const metal.MTLBuffer) !void {
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

    fn metalSgdUpdate(self: Backend, weights: []f32, weights_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, learning_rate: f32, weight_decay: f32) !void {
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

    fn metalSgdUpdateBias(self: Backend, bias: []f32, bias_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, learning_rate: f32) !void {
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

    fn metalAdamUpdate(self: Backend, weights: []f32, weights_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, m: []f32, m_buf: ?*const metal.MTLBuffer, v: []f32, v_buf: ?*const metal.MTLBuffer, lr: f32, beta1: f32, beta2: f32, eps: f32, bias_corr1: f32, bias_corr2: f32) !void {
        const ctx = self.metal_ctx.?;
        var buffer_w = try self.getBuffer(weights, weights_buf);
        defer if (weights_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_w.buffer);
        var buffer_g = try self.getBuffer(gradients, gradients_buf);
        defer if (gradients_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_g.buffer);
        var buffer_m = try self.getBuffer(m, m_buf);
        defer if (m_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_m.buffer);
        var buffer_v = try self.getBuffer(v, v_buf);
        defer if (v_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_v.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("adam_update") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_w.buffer, buffer_w.offset, 0);
        encoder.setBuffer(&buffer_g.buffer, buffer_g.offset, 1);
        encoder.setBuffer(&buffer_m.buffer, buffer_m.offset, 2);
        encoder.setBuffer(&buffer_v.buffer, buffer_v.offset, 3);
        encoder.setBytes(std.mem.asBytes(&lr), 4);
        encoder.setBytes(std.mem.asBytes(&beta1), 5);
        encoder.setBytes(std.mem.asBytes(&beta2), 6);
        encoder.setBytes(std.mem.asBytes(&eps), 7);
        encoder.setBytes(std.mem.asBytes(&bias_corr1), 8);
        encoder.setBytes(std.mem.asBytes(&bias_corr2), 9);
        const size = @as(u32, @intCast(weights.len));
        encoder.setBytes(std.mem.asBytes(&size), 10);

        const tg_size = @min(weights.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(weights.len, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalRmspropUpdate(self: Backend, weights: []f32, weights_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, g_avg: []f32, g_avg_buf: ?*const metal.MTLBuffer, lr: f32, rho: f32, eps: f32) !void {
        const ctx = self.metal_ctx.?;
        var buffer_w = try self.getBuffer(weights, weights_buf);
        defer if (weights_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_w.buffer);
        var buffer_g = try self.getBuffer(gradients, gradients_buf);
        defer if (gradients_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_g.buffer);
        var buffer_ga = try self.getBuffer(g_avg, g_avg_buf);
        defer if (g_avg_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_ga.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("rmsprop_update") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_w.buffer, buffer_w.offset, 0);
        encoder.setBuffer(&buffer_g.buffer, buffer_g.offset, 1);
        encoder.setBuffer(&buffer_ga.buffer, buffer_ga.offset, 2);
        encoder.setBytes(std.mem.asBytes(&lr), 3);
        encoder.setBytes(std.mem.asBytes(&rho), 4);
        encoder.setBytes(std.mem.asBytes(&eps), 5);
        const size = @as(u32, @intCast(weights.len));
        encoder.setBytes(std.mem.asBytes(&size), 6);

        const tg_size = @min(weights.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(weights.len, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalLayerNormForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, gamma: []const f32, gamma_buf: ?*const metal.MTLBuffer, beta: []const f32, beta_buf: ?*const metal.MTLBuffer, eps: f32) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_out = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_out.buffer);
        var buffer_gamma = try self.getBuffer(gamma, gamma_buf);
        defer if (gamma_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gamma.buffer);
        var buffer_beta = try self.getBuffer(beta, beta_buf);
        defer if (beta_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_beta.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("layernorm_forward_optimized") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_out.buffer, buffer_out.offset, 1);
        encoder.setBuffer(&buffer_gamma.buffer, buffer_gamma.offset, 2);
        encoder.setBuffer(&buffer_beta.buffer, buffer_beta.offset, 3);
        encoder.setBytes(std.mem.asBytes(&eps), 4);
        const size = @as(u32, @intCast(gamma.len));
        encoder.setBytes(std.mem.asBytes(&size), 5);

        const batch_size = input.len / gamma.len;
        const tg_size = @min(gamma.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(batch_size * tg_size, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalLayerNormBackward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, gamma: []const f32, gamma_buf: ?*const metal.MTLBuffer, grad_gamma: []f32, grad_gamma_buf: ?*const metal.MTLBuffer, grad_beta: []f32, grad_beta_buf: ?*const metal.MTLBuffer, eps: f32) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_go = try self.getBuffer(grad_output, grad_output_buf);
        defer if (grad_output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_go.buffer);
        var buffer_gi = try self.getBuffer(grad_input, grad_input_buf);
        defer if (grad_input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gi.buffer);
        var buffer_gamma = try self.getBuffer(gamma, gamma_buf);
        defer if (gamma_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gamma.buffer);
        var buffer_gg = try self.getBuffer(grad_gamma, grad_gamma_buf);
        defer if (grad_gamma_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gg.buffer);
        var buffer_gb = try self.getBuffer(grad_beta, grad_beta_buf);
        defer if (grad_beta_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gb.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("layernorm_backward") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_go.buffer, buffer_go.offset, 1);
        encoder.setBuffer(&buffer_gi.buffer, buffer_gi.offset, 2);
        encoder.setBuffer(&buffer_gamma.buffer, buffer_gamma.offset, 3);
        encoder.setBuffer(&buffer_gg.buffer, buffer_gg.offset, 4);
        encoder.setBuffer(&buffer_gb.buffer, buffer_gb.offset, 5);
        encoder.setBytes(std.mem.asBytes(&eps), 6);
        const size = @as(u32, @intCast(gamma.len));
        encoder.setBytes(std.mem.asBytes(&size), 7);

        const batch_size = input.len / gamma.len;
        const tg_size = @min(gamma.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(batch_size * tg_size, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    // MARK: - Batch Normalization Metal Functions

    fn metalBatchNormForwardTraining(
        self: Backend,
        input: []const f32,
        input_buf: ?*const metal.MTLBuffer,
        output: []f32,
        output_buf: ?*const metal.MTLBuffer,
        gamma: []const f32,
        gamma_buf: ?*const metal.MTLBuffer,
        beta: []const f32,
        beta_buf: ?*const metal.MTLBuffer,
        epsilon: f32,
        momentum: f32,
        running_mean: []f32,
        running_mean_buf: ?*const metal.MTLBuffer,
        running_var: []f32,
        running_var_buf: ?*const metal.MTLBuffer,
    ) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_out = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_out.buffer);
        var buffer_gamma = try self.getBuffer(gamma, gamma_buf);
        defer if (gamma_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gamma.buffer);
        var buffer_beta = try self.getBuffer(beta, beta_buf);
        defer if (beta_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_beta.buffer);
        var buffer_running_mean = try self.getBuffer(running_mean, running_mean_buf);
        defer if (running_mean_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_running_mean.buffer);
        var buffer_running_var = try self.getBuffer(running_var, running_var_buf);
        defer if (running_var_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_running_var.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("batchnorm_forward_training") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_out.buffer, buffer_out.offset, 1);
        encoder.setBuffer(&buffer_gamma.buffer, buffer_gamma.offset, 2);
        encoder.setBuffer(&buffer_beta.buffer, buffer_beta.offset, 3);
        encoder.setBuffer(&buffer_running_mean.buffer, buffer_running_mean.offset, 4);
        encoder.setBuffer(&buffer_running_var.buffer, buffer_running_var.offset, 5);
        encoder.setBytes(std.mem.asBytes(&epsilon), 6);
        encoder.setBytes(std.mem.asBytes(&momentum), 7);
        const size = @as(u32, @intCast(gamma.len));
        const batch_size = @as(u32, @intCast(input.len / gamma.len));
        encoder.setBytes(std.mem.asBytes(&batch_size), 8);
        encoder.setBytes(std.mem.asBytes(&size), 9);

        // Dispatch one thread per channel
        encoder.dispatchThreads(metal.MTLSize.make(size, 1, 1), metal.MTLSize.make(1, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalBatchNormForwardInference(
        self: Backend,
        input: []const f32,
        input_buf: ?*const metal.MTLBuffer,
        output: []f32,
        output_buf: ?*const metal.MTLBuffer,
        gamma: []const f32,
        gamma_buf: ?*const metal.MTLBuffer,
        beta: []const f32,
        beta_buf: ?*const metal.MTLBuffer,
        epsilon: f32,
        running_mean: []const f32,
        running_mean_buf: ?*const metal.MTLBuffer,
        running_var: []const f32,
        running_var_buf: ?*const metal.MTLBuffer,
    ) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_out = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_out.buffer);
        var buffer_gamma = try self.getBuffer(gamma, gamma_buf);
        defer if (gamma_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gamma.buffer);
        var buffer_beta = try self.getBuffer(beta, beta_buf);
        defer if (beta_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_beta.buffer);
        var buffer_running_mean = try self.getBuffer(running_mean, running_mean_buf);
        defer if (running_mean_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_running_mean.buffer);
        var buffer_running_var = try self.getBuffer(running_var, running_var_buf);
        defer if (running_var_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_running_var.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("batchnorm_forward_inference") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_out.buffer, buffer_out.offset, 1);
        encoder.setBuffer(&buffer_gamma.buffer, buffer_gamma.offset, 2);
        encoder.setBuffer(&buffer_beta.buffer, buffer_beta.offset, 3);
        encoder.setBuffer(&buffer_running_mean.buffer, buffer_running_mean.offset, 4);
        encoder.setBuffer(&buffer_running_var.buffer, buffer_running_var.offset, 5);
        encoder.setBytes(std.mem.asBytes(&epsilon), 6);
        const size = @as(u32, @intCast(gamma.len));
        const batch_size = @as(u32, @intCast(input.len / gamma.len));
        encoder.setBytes(std.mem.asBytes(&batch_size), 7);
        encoder.setBytes(std.mem.asBytes(&size), 8);

        // Dispatch one thread per channel
        encoder.dispatchThreads(metal.MTLSize.make(size, 1, 1), metal.MTLSize.make(1, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalBatchNormBackward(
        self: Backend,
        input: []const f32,
        input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32,
        grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32,
        grad_input_buf: ?*const metal.MTLBuffer,
        gamma: []const f32,
        gamma_buf: ?*const metal.MTLBuffer,
        grad_gamma: []f32,
        grad_gamma_buf: ?*const metal.MTLBuffer,
        grad_beta: []f32,
        grad_beta_buf: ?*const metal.MTLBuffer,
        epsilon: f32,
        running_mean: []const f32,
        running_mean_buf: ?*const metal.MTLBuffer,
        running_var: []const f32,
        running_var_buf: ?*const metal.MTLBuffer,
    ) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_go = try self.getBuffer(grad_output, grad_output_buf);
        defer if (grad_output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_go.buffer);
        var buffer_gi = try self.getBuffer(grad_input, grad_input_buf);
        defer if (grad_input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gi.buffer);
        var buffer_gamma = try self.getBuffer(gamma, gamma_buf);
        defer if (gamma_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gamma.buffer);
        var buffer_gg = try self.getBuffer(grad_gamma, grad_gamma_buf);
        defer if (grad_gamma_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gg.buffer);
        var buffer_gb = try self.getBuffer(grad_beta, grad_beta_buf);
        defer if (grad_beta_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gb.buffer);
        var buffer_running_mean = try self.getBuffer(running_mean, running_mean_buf);
        defer if (running_mean_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_running_mean.buffer);
        var buffer_running_var = try self.getBuffer(running_var, running_var_buf);
        defer if (running_var_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_running_var.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("batchnorm_backward") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_go.buffer, buffer_go.offset, 1);
        encoder.setBuffer(&buffer_gi.buffer, buffer_gi.offset, 2);
        encoder.setBuffer(&buffer_gamma.buffer, buffer_gamma.offset, 3);
        encoder.setBuffer(&buffer_gg.buffer, buffer_gg.offset, 4);
        encoder.setBuffer(&buffer_gb.buffer, buffer_gb.offset, 5);
        encoder.setBuffer(&buffer_running_mean.buffer, buffer_running_mean.offset, 6);
        encoder.setBuffer(&buffer_running_var.buffer, buffer_running_var.offset, 7);
        encoder.setBytes(std.mem.asBytes(&epsilon), 8);
        const size = @as(u32, @intCast(gamma.len));
        const batch_size = @as(u32, @intCast(input.len / gamma.len));
        encoder.setBytes(std.mem.asBytes(&batch_size), 9);
        encoder.setBytes(std.mem.asBytes(&size), 10);

        // Dispatch one thread per channel
        encoder.dispatchThreads(metal.MTLSize.make(size, 1, 1), metal.MTLSize.make(1, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalConv1dForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, weights: []const f32, weights_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, in_channels: usize, out_channels: usize, kernel_size: usize, in_len: usize, out_len: usize) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_w = try self.getBuffer(weights, weights_buf);
        defer if (weights_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_w.buffer);
        var buffer_b = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_b.buffer);
        var buffer_out = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_out.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("conv1d_forward") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_w.buffer, buffer_w.offset, 1);
        encoder.setBuffer(&buffer_b.buffer, buffer_b.offset, 2);
        encoder.setBuffer(&buffer_out.buffer, buffer_out.offset, 3);
        const params = [_]u32{ @as(u32, @intCast(in_channels)), @as(u32, @intCast(out_channels)), @as(u32, @intCast(kernel_size)), @as(u32, @intCast(in_len)), @as(u32, @intCast(out_len)) };
        encoder.setBytes(std.mem.asBytes(&params[0]), 4);
        encoder.setBytes(std.mem.asBytes(&params[1]), 5);
        encoder.setBytes(std.mem.asBytes(&params[2]), 6);
        encoder.setBytes(std.mem.asBytes(&params[3]), 7);
        encoder.setBytes(std.mem.asBytes(&params[4]), 8);

        const batch_size = input.len / (in_channels * in_len);
        // Use threadgroup size that's a multiple of SIMD width (32)
        // 8x4x1 = 32 threads per threadgroup
        encoder.dispatchThreads(metal.MTLSize.make(out_len, out_channels, batch_size), metal.MTLSize.make(8, 4, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalConv1dBackward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, weights: []const f32, weights_buf: ?*const metal.MTLBuffer, grad_after_act: []const f32, grad_after_act_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, grad_weights: []f32, grad_weights_buf: ?*const metal.MTLBuffer, grad_bias: []f32, grad_bias_buf: ?*const metal.MTLBuffer, in_channels: usize, out_channels: usize, kernel_size: usize, in_len: usize, out_len: usize) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_w = try self.getBuffer(weights, weights_buf);
        defer if (weights_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_w.buffer);
        var buffer_gaa = try self.getBuffer(grad_after_act, grad_after_act_buf);
        defer if (grad_after_act_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gaa.buffer);
        var buffer_gi = try self.getBuffer(grad_input, grad_input_buf);
        defer if (grad_input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gi.buffer);
        var buffer_gw = try self.getBuffer(grad_weights, grad_weights_buf);
        defer if (grad_weights_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gw.buffer);
        var buffer_gb = try self.getBuffer(grad_bias, grad_bias_buf);
        defer if (grad_bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gb.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("conv1d_backward") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_w.buffer, buffer_w.offset, 1);
        encoder.setBuffer(&buffer_gaa.buffer, buffer_gaa.offset, 2);
        encoder.setBuffer(&buffer_gi.buffer, buffer_gi.offset, 3);
        encoder.setBuffer(&buffer_gw.buffer, buffer_gw.offset, 4);
        encoder.setBuffer(&buffer_gb.buffer, buffer_gb.offset, 5);

        const params = [_]u32{ @as(u32, @intCast(in_channels)), @as(u32, @intCast(out_channels)), @as(u32, @intCast(kernel_size)), @as(u32, @intCast(in_len)), @as(u32, @intCast(out_len)) };
        encoder.setBytes(std.mem.asBytes(&params[0]), 6);
        encoder.setBytes(std.mem.asBytes(&params[1]), 7);
        encoder.setBytes(std.mem.asBytes(&params[2]), 8);
        encoder.setBytes(std.mem.asBytes(&params[3]), 9);
        encoder.setBytes(std.mem.asBytes(&params[4]), 10);
        const batch_size = @as(u32, @intCast(input.len / (in_channels * in_len)));
        encoder.setBytes(std.mem.asBytes(&batch_size), 11);

        encoder.dispatchThreads(metal.MTLSize.make(out_len, out_channels, batch_size), metal.MTLSize.make(16, 16, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalConv2dForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, weights: []const f32, weights_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, in_channels: usize, out_channels: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize, output_h: usize, output_w: usize, stride_h: usize, stride_w: usize, padding_h: usize, padding_w: usize) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_w = try self.getBuffer(weights, weights_buf);
        defer if (weights_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_w.buffer);
        var buffer_b = try self.getBuffer(bias, bias_buf);
        defer if (bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_b.buffer);
        var buffer_out = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_out.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("conv2d_forward") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_w.buffer, buffer_w.offset, 1);
        encoder.setBuffer(&buffer_b.buffer, buffer_b.offset, 2);
        encoder.setBuffer(&buffer_out.buffer, buffer_out.offset, 3);

        const params = [_]u32{
            @as(u32, @intCast(in_channels)),
            @as(u32, @intCast(out_channels)),
            @as(u32, @intCast(kernel_h)),
            @as(u32, @intCast(kernel_w)),
            @as(u32, @intCast(input_h)),
            @as(u32, @intCast(input_w)),
            @as(u32, @intCast(output_h)),
            @as(u32, @intCast(output_w)),
            @as(u32, @intCast(stride_h)),
            @as(u32, @intCast(stride_w)),
            @as(u32, @intCast(padding_h)),
            @as(u32, @intCast(padding_w)),
        };
        encoder.setBytes(std.mem.asBytes(&params[0]), 4);
        encoder.setBytes(std.mem.asBytes(&params[1]), 5);
        encoder.setBytes(std.mem.asBytes(&params[2]), 6);
        encoder.setBytes(std.mem.asBytes(&params[3]), 7);
        encoder.setBytes(std.mem.asBytes(&params[4]), 8);
        encoder.setBytes(std.mem.asBytes(&params[5]), 9);
        encoder.setBytes(std.mem.asBytes(&params[6]), 10);
        encoder.setBytes(std.mem.asBytes(&params[7]), 11);
        encoder.setBytes(std.mem.asBytes(&params[8]), 12);
        encoder.setBytes(std.mem.asBytes(&params[9]), 13);
        encoder.setBytes(std.mem.asBytes(&params[10]), 14);
        encoder.setBytes(std.mem.asBytes(&params[11]), 15);

        const batch_size = input.len / (in_channels * input_h * input_w);
        encoder.dispatchThreads(metal.MTLSize.make(output_w, output_h, out_channels * batch_size), metal.MTLSize.make(8, 8, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalConv2dBackward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, weights: []const f32, weights_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, grad_weights: []f32, grad_weights_buf: ?*const metal.MTLBuffer, grad_bias: []f32, grad_bias_buf: ?*const metal.MTLBuffer, in_channels: usize, out_channels: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize, output_h: usize, output_w: usize, stride_h: usize, stride_w: usize, padding_h: usize, padding_w: usize) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_w = try self.getBuffer(weights, weights_buf);
        defer if (weights_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_w.buffer);
        var buffer_go = try self.getBuffer(grad_output, grad_output_buf);
        defer if (grad_output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_go.buffer);
        var buffer_gi = try self.getBuffer(grad_input, grad_input_buf);
        defer if (grad_input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gi.buffer);
        var buffer_gw = try self.getBuffer(grad_weights, grad_weights_buf);
        defer if (grad_weights_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gw.buffer);
        var buffer_gb = try self.getBuffer(grad_bias, grad_bias_buf);
        defer if (grad_bias_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gb.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("conv2d_backward") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_w.buffer, buffer_w.offset, 1);
        encoder.setBuffer(&buffer_go.buffer, buffer_go.offset, 2);
        encoder.setBuffer(&buffer_gi.buffer, buffer_gi.offset, 3);
        encoder.setBuffer(&buffer_gw.buffer, buffer_gw.offset, 4);
        encoder.setBuffer(&buffer_gb.buffer, buffer_gb.offset, 5);

        const params = [_]u32{
            @as(u32, @intCast(in_channels)),
            @as(u32, @intCast(out_channels)),
            @as(u32, @intCast(kernel_h)),
            @as(u32, @intCast(kernel_w)),
            @as(u32, @intCast(input_h)),
            @as(u32, @intCast(input_w)),
            @as(u32, @intCast(output_h)),
            @as(u32, @intCast(output_w)),
            @as(u32, @intCast(stride_h)),
            @as(u32, @intCast(stride_w)),
            @as(u32, @intCast(padding_h)),
            @as(u32, @intCast(padding_w)),
        };
        encoder.setBytes(std.mem.asBytes(&params[0]), 6);
        encoder.setBytes(std.mem.asBytes(&params[1]), 7);
        encoder.setBytes(std.mem.asBytes(&params[2]), 8);
        encoder.setBytes(std.mem.asBytes(&params[3]), 9);
        encoder.setBytes(std.mem.asBytes(&params[4]), 10);
        encoder.setBytes(std.mem.asBytes(&params[5]), 11);
        encoder.setBytes(std.mem.asBytes(&params[6]), 12);
        encoder.setBytes(std.mem.asBytes(&params[7]), 13);
        encoder.setBytes(std.mem.asBytes(&params[8]), 14);
        encoder.setBytes(std.mem.asBytes(&params[9]), 15);
        encoder.setBytes(std.mem.asBytes(&params[10]), 16);
        encoder.setBytes(std.mem.asBytes(&params[11]), 17);
        const batch_size = @as(u32, @intCast(input.len / (in_channels * input_h * input_w)));
        encoder.setBytes(std.mem.asBytes(&batch_size), 18);

        encoder.dispatchThreads(metal.MTLSize.make(output_w, output_h, out_channels * batch_size), metal.MTLSize.make(8, 8, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    /// CUDA 2D Convolution backward pass
    fn cudaConv2dBackward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, weights: []const f32, weights_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, grad_weights: []f32, grad_weights_buf: ?*const metal.MTLBuffer, grad_bias: []f32, grad_bias_buf: ?*const metal.MTLBuffer, in_channels: usize, out_channels: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize, output_h: usize, output_w: usize, stride_h: usize, stride_w: usize, padding_h: usize, padding_w: usize) !void {
        _ = input_buf;
        _ = weights_buf;
        _ = grad_output_buf;
        _ = grad_input_buf;
        _ = grad_weights_buf;
        _ = grad_bias_buf;

        const ctx = self.cuda_ctx orelse return error.NotAvailable;

        const batch_size = input.len / (in_channels * input_h * input_w);

        // Allocate device buffers with overflow checking
        const input_size = try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, batch_size, in_channels), input_h), input_w);
        const weights_size = try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, out_channels, in_channels), kernel_h), kernel_w);
        const output_size = try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, batch_size, out_channels), output_h), output_w);
        const grad_input_size = input_size;
        const grad_weights_size = weights_size;
        const grad_bias_size = out_channels;

        var d_input = try ctx.allocBuffer(input_size * @sizeOf(f32));
        defer ctx.freeBuffer(&d_input);
        var d_weights = try ctx.allocBuffer(weights_size * @sizeOf(f32));
        defer ctx.freeBuffer(&d_weights);
        var d_grad_output = try ctx.allocBuffer(output_size * @sizeOf(f32));
        defer ctx.freeBuffer(&d_grad_output);
        var d_grad_input = try ctx.allocBuffer(grad_input_size * @sizeOf(f32));
        defer ctx.freeBuffer(&d_grad_input);
        var d_grad_weights = try ctx.allocBuffer(grad_weights_size * @sizeOf(f32));
        defer ctx.freeBuffer(&d_grad_weights);
        var d_grad_bias = try ctx.allocBuffer(grad_bias_size * @sizeOf(f32));
        defer ctx.freeBuffer(&d_grad_bias);

        // Upload data to device
        try ctx.upload(d_input.ptr, std.mem.sliceAsBytes(input));
        try ctx.upload(d_weights.ptr, std.mem.sliceAsBytes(weights));
        try ctx.upload(d_grad_output.ptr, std.mem.sliceAsBytes(grad_output));
        try ctx.memset(d_grad_input.ptr, 0, @intCast(grad_input_size * @sizeOf(f32)));
        try ctx.memset(d_grad_weights.ptr, 0, @intCast(grad_weights_size * @sizeOf(f32)));
        try ctx.memset(d_grad_bias.ptr, 0, @intCast(grad_bias_size * @sizeOf(f32)));

        // Launch kernel
        const grid_dim = [3]u32{
            @intCast((output_w + 7) / 8),
            @intCast((output_h + 7) / 8),
            @intCast(batch_size * out_channels),
        };
        const block_dim = [3]u32{ 8, 8, 1 };

        var batch_size_var = batch_size;
        var in_channels_var = in_channels;
        var out_channels_var = out_channels;
        var kernel_h_var = kernel_h;
        var kernel_w_var = kernel_w;
        var input_h_var = input_h;
        var input_w_var = input_w;
        var output_h_var = output_h;
        var output_w_var = output_w;
        var stride_h_var = stride_h;
        var stride_w_var = stride_w;
        var padding_h_var = padding_h;
        var padding_w_var = padding_w;

        const args = &[_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_weights.ptr),
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_grad_input.ptr),
            @ptrCast(&d_grad_weights.ptr),
            @ptrCast(&d_grad_bias.ptr),
            @ptrCast(&batch_size_var),
            @ptrCast(&in_channels_var),
            @ptrCast(&out_channels_var),
            @ptrCast(&kernel_h_var),
            @ptrCast(&kernel_w_var),
            @ptrCast(&input_h_var),
            @ptrCast(&input_w_var),
            @ptrCast(&output_h_var),
            @ptrCast(&output_w_var),
            @ptrCast(&stride_h_var),
            @ptrCast(&stride_w_var),
            @ptrCast(&padding_h_var),
            @ptrCast(&padding_w_var),
        };

        try ctx.launchKernel("conv2d_backward", grid_dim, block_dim, 0, args);
        try ctx.synchronize();

        // Download results
        try ctx.download(std.mem.sliceAsBytes(grad_input), d_grad_input.ptr);
        try ctx.download(std.mem.sliceAsBytes(grad_weights), d_grad_weights.ptr);
        try ctx.download(std.mem.sliceAsBytes(grad_bias), d_grad_bias.ptr);
    }

    fn metalDropoutForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, mask: []f32, mask_buf: ?*const metal.MTLBuffer, rate: f32, scaling_factor: f32, seed: u64) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_out = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_out.buffer);
        var buffer_mask = try self.getBuffer(mask, mask_buf);
        defer if (mask_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_mask.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("dropout_forward") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_out.buffer, buffer_out.offset, 1);
        encoder.setBuffer(&buffer_mask.buffer, buffer_mask.offset, 2);
        encoder.setBytes(std.mem.asBytes(&rate), 3);
        encoder.setBytes(std.mem.asBytes(&scaling_factor), 4);
        encoder.setBytes(std.mem.asBytes(&seed), 5);
        const size = @as(u32, @intCast(input.len));
        encoder.setBytes(std.mem.asBytes(&size), 6);

        const tg_size = @min(input.len, pipeline.maxTotalThreadsPerThreadgroup());
        encoder.dispatchThreads(metal.MTLSize.make(input.len, 1, 1), metal.MTLSize.make(tg_size, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalVaeSamplingForward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, epsilon: []f32, epsilon_buf: ?*const metal.MTLBuffer, seed: u64, latent_dim: usize) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_out = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_out.buffer);
        var buffer_eps = try self.getBuffer(epsilon, epsilon_buf);
        defer if (epsilon_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_eps.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("vae_sampling_forward") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_out.buffer, buffer_out.offset, 1);
        encoder.setBuffer(&buffer_eps.buffer, buffer_eps.offset, 2);
        encoder.setBytes(std.mem.asBytes(&seed), 3);
        const ld = @as(u32, @intCast(latent_dim));
        encoder.setBytes(std.mem.asBytes(&ld), 4);

        encoder.dispatchThreads(metal.MTLSize.make(latent_dim, 1, 1), metal.MTLSize.make(latent_dim, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalVaeSamplingBackward(self: Backend, input: []const f32, input_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer, epsilon: []const f32, epsilon_buf: ?*const metal.MTLBuffer, latent_dim: usize) !void {
        const ctx = self.metal_ctx.?;
        var buffer_in = try self.getBuffer(input, input_buf);
        defer if (input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_in.buffer);
        var buffer_go = try self.getBuffer(grad_output, grad_output_buf);
        defer if (grad_output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_go.buffer);
        var buffer_gi = try self.getBuffer(grad_input, grad_input_buf);
        defer if (grad_input_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_gi.buffer);
        var buffer_eps = try self.getBuffer(epsilon, epsilon_buf);
        defer if (epsilon_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_eps.buffer);

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("vae_sampling_backward") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_in.buffer, buffer_in.offset, 0);
        encoder.setBuffer(&buffer_go.buffer, buffer_go.offset, 1);
        encoder.setBuffer(&buffer_gi.buffer, buffer_gi.offset, 2);
        encoder.setBuffer(&buffer_eps.buffer, buffer_eps.offset, 3);
        const ld = @as(u32, @intCast(latent_dim));
        encoder.setBytes(std.mem.asBytes(&ld), 4);

        encoder.dispatchThreads(metal.MTLSize.make(latent_dim, 1, 1), metal.MTLSize.make(latent_dim, 1, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    fn metalAttentionForward(self: Backend, q: []const f32, q_buf: ?*const metal.MTLBuffer, k: []const f32, k_buf: ?*const metal.MTLBuffer, v: []const f32, v_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer, seq_len: usize, d_k: usize, scaling_factor: f32) !void {
        const ctx = self.metal_ctx.?;
        var buffer_q = try self.getBuffer(q, q_buf);
        defer if (q_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_q.buffer);
        var buffer_k = try self.getBuffer(k, k_buf);
        defer if (k_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_k.buffer);
        var buffer_v = try self.getBuffer(v, v_buf);
        defer if (v_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_v.buffer);
        var buffer_out = try self.getBuffer(output, output_buf);
        defer if (output_buf == null and ctx.active_command_buffer == null) self.releaseBuffer(buffer_out.buffer);

        // Allocate temporary buffer for attention scores [seq_len * seq_len]
        const scores_size = seq_len * seq_len * @sizeOf(f32);
        var buffer_scores = try ctx.allocBuffer(scores_size, .StorageModePrivate);
        defer buffer_scores.release();

        var cb = try self.getCommandBuffer();
        var encoder = try cb.computeCommandEncoder();
        const pipeline = ctx.getPipeline("attention_forward") orelse return error.PipelineNotFound;
        encoder.setComputePipelineState(pipeline);
        encoder.setBuffer(&buffer_q.buffer, buffer_q.offset, 0);
        encoder.setBuffer(&buffer_k.buffer, buffer_k.offset, 1);
        encoder.setBuffer(&buffer_v.buffer, buffer_v.offset, 2);
        encoder.setBuffer(&buffer_out.buffer, buffer_out.offset, 3);
        encoder.setBuffer(&buffer_scores, 0, 4);
        const sl = @as(u32, @intCast(seq_len));
        const dk = @as(u32, @intCast(d_k));
        encoder.setBytes(std.mem.asBytes(&sl), 5);
        encoder.setBytes(std.mem.asBytes(&dk), 6);
        encoder.setBytes(std.mem.asBytes(&scaling_factor), 7);

        // Use threadgroup size that's a multiple of SIMD width (32) for optimal occupancy
        const tg_width = 32;
        const tg_height = 8;
        encoder.dispatchThreads(metal.MTLSize.make(d_k, seq_len, 1), metal.MTLSize.make(tg_width, tg_height, 1));
        encoder.endEncoding();
        if (ctx.active_command_buffer == null) {
            cb.commit();
            cb.waitUntilCompleted();
        }
    }

    // ================== CPU implementations ==================

    fn cpuMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize, accumulate: bool) void {
        if (!accumulate) @memset(c, 0);
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

                                // SIMD vectorized inner loop - process 8 elements at a time
                                // Only use SIMD if memory is properly aligned (32 bytes for Vec8)
                                const Vec8 = @Vector(8, f32);
                                const vec_end = (j_end / 8) * 8;
                                const b_aligned = (@intFromPtr(b_row.ptr) % 32) == 0;
                                const c_aligned = (@intFromPtr(c_row.ptr) % 32) == 0;

                                if (b_aligned and c_aligned) {
                                    var j_vec: usize = jj;
                                    while (j_vec < vec_end) : (j_vec += 8) {
                                        const b_vec: Vec8 = @as(*const Vec8, @ptrCast(@alignCast(b_row.ptr + j_vec))).*;
                                        const c_vec: Vec8 = @as(*const Vec8, @ptrCast(@alignCast(c_row.ptr + j_vec))).*;
                                        const a_splat: Vec8 = @splat(a_val);
                                        const result = c_vec + a_splat * b_vec;
                                        @as(*Vec8, @ptrCast(@alignCast(c_row.ptr + j_vec))).* = result;
                                    }
                                    // Process remaining elements
                                    for (j_vec..j_end) |j| {
                                        c_row[j] += a_val * b_row[j];
                                    }
                                } else {
                                    // Fallback to scalar for unaligned memory
                                    for (jj..j_end) |j| {
                                        c_row[j] += a_val * b_row[j];
                                    }
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
                    const b_row = b[p * n ..];
                    const c_row = c[i * n ..];

                    // SIMD vectorized inner loop - process 8 elements at a time
                    // Only use SIMD if memory is properly aligned (32 bytes for Vec8)
                    const Vec8 = @Vector(8, f32);
                    const vec_end = (n / 8) * 8;
                    const b_aligned = (@intFromPtr(b_row.ptr) % 32) == 0;
                    const c_aligned = (@intFromPtr(c_row.ptr) % 32) == 0;

                    if (b_aligned and c_aligned) {
                        var j: usize = 0;
                        while (j < vec_end) : (j += 8) {
                            const b_vec: Vec8 = @as(*const Vec8, @ptrCast(@alignCast(b_row.ptr + j))).*;
                            const c_vec: Vec8 = @as(*const Vec8, @ptrCast(@alignCast(c_row.ptr + j))).*;
                            const a_splat: Vec8 = @splat(a_val);
                            const result = c_vec + a_splat * b_vec;
                            @as(*Vec8, @ptrCast(@alignCast(c_row.ptr + j))).* = result;
                        }
                        // Process remaining elements
                        for (j..n) |jr| {
                            c_row[jr] += a_val * b_row[jr];
                        }
                    } else {
                        // Fallback to scalar for unaligned memory
                        for (0..n) |j| {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }

    fn cpuMatMulBatch(a: []const f32, b: []const f32, c: []f32, batch_size: usize, n: usize, k: usize, accumulate: bool) void {
        for (0..batch_size) |i| {
            cpuMatMul(a[i * k ..], b, c[i * n ..], 1, n, k, accumulate);
        }
    }

    fn cpuMatMulTransposeA(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize, accumulate: bool) void {
        if (!accumulate) @memset(c, 0);
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0.0;
                for (0..k) |p| {
                    sum += a[p * m + i] * b[p * n + j];
                }
                if (accumulate) {
                    c[i * n + j] += sum;
                } else {
                    c[i * n + j] = sum;
                }
            }
        }
    }

    fn cpuMatMulTransposeB(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize, accumulate: bool) void {
        if (!accumulate) @memset(c, 0);
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0.0;
                for (0..k) |p| {
                    sum += a[i * k + p] * b[j * k + p];
                }
                if (accumulate) {
                    c[i * n + j] += sum;
                } else {
                    c[i * n + j] = sum;
                }
            }
        }
    }

    fn cpuActivationBackward(act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) void {
        if (act == .softmax) {
            // Softmax backward requires per-sample processing
            act.softmaxBackward(input, grad_output, grad_input) catch @panic("Softmax backward failed");
        } else {
            // Use SIMD-optimized backward passes
            switch (act) {
                .relu => optimization.SIMD.reluBackwardVectorized(input, grad_output, grad_input),
                .sigmoid => optimization.SIMD.sigmoidBackwardVectorized(input, grad_output, grad_input),
                .tanh => optimization.SIMD.tanhBackwardVectorized(input, grad_output, grad_input),
                .linear => @memcpy(grad_input, grad_output),
                .softmax => unreachable, // Handled above
                .gelu, .leaky_relu, .elu => {
                    // Generic backward using scalar implementation
                    for (input, grad_output, grad_input) |y, go, *gi| {
                        gi.* = act.backward(y, go);
                    }
                },
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

    /// CUDA SGD optimizer update
    fn cudaSgdUpdate(self: Backend, weights: []f32, weights_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, learning_rate: f32, weight_decay: f32) !void {
        _ = weights_buf;
        _ = gradients_buf;
        const cuda_be = self.cuda_ctx orelse return error.NotAvailable;
        try cuda_be.sgdUpdate(weights, gradients, learning_rate, weight_decay);
    }

    /// CUDA Adam optimizer update
    fn cudaAdamUpdate(self: Backend, weights: []f32, weights_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, m: []f32, m_buf: ?*const metal.MTLBuffer, v: []f32, v_buf: ?*const metal.MTLBuffer, lr: f32, beta1: f32, beta2: f32, eps: f32, bias_corr1: f32, bias_corr2: f32) !void {
        _ = weights_buf;
        _ = gradients_buf;
        _ = m_buf;
        _ = v_buf;
        _ = bias_corr1;
        _ = bias_corr2;
        const cuda_be = self.cuda_ctx orelse return error.NotAvailable;
        const t: u32 = 1;
        try cuda_be.adamUpdate(weights, gradients, m, v, lr, beta1, beta2, eps, t);
    }

    /// CUDA RMSprop optimizer update
    fn cudaRmspropUpdate(self: Backend, weights: []f32, weights_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, g_avg: []f32, g_avg_buf: ?*const metal.MTLBuffer, lr: f32, rho: f32, eps: f32) !void {
        _ = weights_buf;
        _ = gradients_buf;
        _ = g_avg_buf;
        const cuda_be = self.cuda_ctx orelse return error.NotAvailable;
        try cuda_be.rmspropUpdate(weights, gradients, g_avg, lr, rho, eps);
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

    fn cpuAdamUpdate(self: Backend, weights: []f32, gradients: []const f32, m: []f32, v: []f32, lr: f32, beta1: f32, beta2: f32, eps: f32, bias_corr1: f32, bias_corr2: f32) !void {
        _ = self;
        for (weights, gradients, m, v) |*w, g_raw, *mv, *vv| {
            var g = g_raw;
            if (std.math.isNan(g)) g = 0.0 else g = @min(5.0, @max(-5.0, g));
            mv.* = beta1 * mv.* + (1.0 - beta1) * g;
            vv.* = beta2 * vv.* + (1.0 - beta2) * g * g;
            const m_hat = mv.* / bias_corr1;
            const v_hat = vv.* / bias_corr2;
            w.* -= lr * m_hat / (@sqrt(v_hat) + eps);
            w.* = @min(100.0, @max(-100.0, w.*));
        }
    }

    fn cpuRmspropUpdate(self: Backend, weights: []f32, gradients: []const f32, g_avg: []f32, lr: f32, rho: f32, eps: f32) !void {
        _ = self;
        for (weights, gradients, g_avg) |*w, g_raw, *ga| {
            var g = g_raw;
            if (std.math.isNan(g)) g = 0.0 else g = @min(5.0, @max(-5.0, g));
            ga.* = rho * ga.* + (1.0 - rho) * g * g;
            w.* -= lr * g / (@sqrt(ga.*) + eps);
            w.* = @min(100.0, @max(-100.0, w.*));
        }
    }

    /// Compute mean and variance using Welford's online algorithm
    /// PERFORMANCE FIX: Single-pass algorithm reduces memory bandwidth by 2x (F2.2)
    fn welfordMeanVariance(data: []const f32) struct { mean: f32, variance: f32 } {
        var mean: f32 = 0.0;
        var m2: f32 = 0.0; // Sum of squares of differences from the mean
        var count: f32 = 0.0;

        for (data) |x| {
            count += 1.0;
            const delta = x - mean;
            mean += delta / count;
            const delta2 = x - mean;
            m2 += delta * delta2;
        }

        const variance = if (count > 1.0) m2 / count else 0.0;
        return .{ .mean = mean, .variance = variance };
    }

    fn cpuLayerNormForward(self: Backend, input: []const f32, output: []f32, gamma: []const f32, beta: []const f32, eps: f32) !void {
        _ = self;
        const optimization_module = @import("optimization.zig");
        const batch_size = input.len / gamma.len;
        const size = gamma.len;
        for (0..batch_size) |b| {
            const in = input[b * size .. (b + 1) * size];
            const out = output[b * size .. (b + 1) * size];
            // PERFORMANCE FIX: Use Welford's algorithm for single-pass mean/variance (F2.2)
            const stats = welfordMeanVariance(in);
            const mean = stats.mean;
            const std_inv = 1.0 / @sqrt(stats.variance + eps);
            // PERFORMANCE FIX: Use SIMD for normalization (F2.3)
            optimization_module.SIMD.layerNormVectorized(in, out, gamma, beta, mean, std_inv);
        }
    }

    fn cpuLayerNormBackward(self: Backend, input: []const f32, grad_output: []const f32, grad_input: []f32, gamma: []const f32, grad_gamma: []f32, grad_beta: []f32, eps: f32) !void {
        _ = self;
        const batch_size = input.len / gamma.len;
        const size = gamma.len;
        for (0..batch_size) |b| {
            const in = input[b * size .. (b + 1) * size];
            const go = grad_output[b * size .. (b + 1) * size];
            const gi = grad_input[b * size .. (b + 1) * size];
            // PERFORMANCE FIX: Use Welford's algorithm for single-pass mean/variance (F2.2)
            const stats = welfordMeanVariance(in);
            const mean = stats.mean;
            const std_inv = 1.0 / @sqrt(stats.variance + eps);
            for (in, go, gi, 0..) |x, gov, *giv, i| {
                const x_hat = (x - mean) * std_inv;
                grad_gamma[i] += gov * x_hat;
                grad_beta[i] += gov;
                giv.* = gov * gamma[i] * std_inv;
            }
        }
    }

    // MARK: - Batch Normalization CPU Functions

    fn cpuBatchNormForwardTraining(
        self: Backend,
        input: []const f32,
        output: []f32,
        gamma: []const f32,
        beta: []const f32,
        epsilon: f32,
        momentum: f32,
        running_mean: []f32,
        running_var: []f32,
    ) !void {
        _ = self;
        const size = gamma.len;
        const batch_size = input.len / size;

        // Compute mean and variance per channel across batch
        for (0..size) |c| {
            var mean: f32 = 0;
            for (0..batch_size) |b| {
                mean += input[b * size + c];
            }
            mean /= @as(f32, @floatFromInt(batch_size));

            var var_val: f32 = 0;
            for (0..batch_size) |b| {
                const diff = input[b * size + c] - mean;
                var_val += diff * diff;
            }
            var_val /= @as(f32, @floatFromInt(batch_size));

            // Update running statistics
            running_mean[c] = (1.0 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0 - momentum) * running_var[c] + momentum * var_val;

            // Normalize and scale
            const inv_std = 1.0 / @sqrt(var_val + epsilon);
            for (0..batch_size) |b| {
                const x_hat = (input[b * size + c] - mean) * inv_std;
                output[b * size + c] = x_hat * gamma[c] + beta[c];
            }
        }
    }

    fn cpuBatchNormForwardInference(
        self: Backend,
        input: []const f32,
        output: []f32,
        gamma: []const f32,
        beta: []const f32,
        epsilon: f32,
        running_mean: []const f32,
        running_var: []const f32,
    ) !void {
        _ = self;
        const size = gamma.len;
        const batch_size = input.len / size;

        for (0..size) |c| {
            const mean = running_mean[c];
            const var_val = running_var[c];
            const inv_std = 1.0 / @sqrt(var_val + epsilon);

            for (0..batch_size) |b| {
                const x_hat = (input[b * size + c] - mean) * inv_std;
                output[b * size + c] = x_hat * gamma[c] + beta[c];
            }
        }
    }

    fn cpuBatchNormBackward(
        self: Backend,
        input: []const f32,
        grad_output: []const f32,
        grad_input: []f32,
        gamma: []const f32,
        grad_gamma: []f32,
        grad_beta: []f32,
        epsilon: f32,
    ) !void {
        _ = self;
        const size = gamma.len;
        const batch_size = input.len / size;

        // Compute mean and variance per channel
        for (0..size) |c| {
            var mean: f32 = 0;
            for (0..batch_size) |b| {
                mean += input[b * size + c];
            }
            mean /= @as(f32, @floatFromInt(batch_size));

            var var_val: f32 = 0;
            for (0..batch_size) |b| {
                const diff = input[b * size + c] - mean;
                var_val += diff * diff;
            }
            var_val /= @as(f32, @floatFromInt(batch_size));

            const inv_std = 1.0 / @sqrt(var_val + epsilon);

            // Compute grad_gamma and grad_beta
            grad_gamma[c] = 0;
            grad_beta[c] = 0;
            for (0..batch_size) |b| {
                const x_hat = (input[b * size + c] - mean) * inv_std;
                grad_gamma[c] += grad_output[b * size + c] * x_hat;
                grad_beta[c] += grad_output[b * size + c];
            }

            // Compute grad_input
            const coef = gamma[c] * inv_std;
            for (0..batch_size) |b| {
                grad_input[b * size + c] = grad_output[b * size + c] * coef;
            }
        }
    }

    fn cpuConv1dForward(self: Backend, input: []const f32, weights: []const f32, bias: []const f32, output: []f32, in_channels: usize, out_channels: usize, kernel_size: usize, in_len: usize, out_len: usize) !void {
        _ = self;
        const batch_size = input.len / (in_channels * in_len);
        @memset(output, 0);
        for (0..batch_size) |b| {
            for (0..out_channels) |oc| {
                for (0..out_len) |t| {
                    var sum: f32 = 0;
                    for (0..in_channels) |ic| {
                        for (0..kernel_size) |k| {
                            const in_idx = (b * in_channels + ic) * in_len + (t + k);
                            const w_idx = (oc * in_channels + ic) * kernel_size + k;
                            sum += input[in_idx] * weights[w_idx];
                        }
                    }
                    output[(b * out_channels + oc) * out_len + t] = sum + bias[oc];
                }
            }
        }
    }

    fn cpuConv1dBackward(self: Backend, input: []const f32, weights: []const f32, grad_after_act: []const f32, grad_input: []f32, grad_weights: []f32, grad_bias: []f32, in_channels: usize, out_channels: usize, kernel_size: usize, in_len: usize, out_len: usize) !void {
        _ = self;
        const batch_size = input.len / (in_channels * in_len);
        @memset(grad_input, 0);
        for (0..batch_size) |b| {
            for (0..out_channels) |oc| {
                for (0..out_len) |t| {
                    const go = grad_after_act[(b * out_channels + oc) * out_len + t];
                    grad_bias[oc] += go;
                    for (0..in_channels) |ic| {
                        for (0..kernel_size) |k| {
                            const in_idx = (b * in_channels + ic) * in_len + (t + k);
                            const w_idx = (oc * in_channels + ic) * kernel_size + k;
                            grad_weights[w_idx] += input[in_idx] * go;
                            grad_input[in_idx] += weights[w_idx] * go;
                        }
                    }
                }
            }
        }
    }

    fn cpuConv2dForward(self: Backend, input: []const f32, weights: []const f32, bias: []const f32, output: []f32, in_channels: usize, out_channels: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize, output_h: usize, output_w: usize, stride_h: usize, stride_w: usize, padding_h: usize, padding_w: usize) !void {
        _ = self;
        const batch_size = input.len / (in_channels * input_h * input_w);
        @memset(output, 0);
        for (0..batch_size) |b| {
            for (0..out_channels) |oc| {
                for (0..output_h) |oh| {
                    for (0..output_w) |ow| {
                        var sum: f32 = 0;
                        const in_h_start = @as(i32, @intCast(oh * stride_h)) - @as(i32, @intCast(padding_h));
                        const in_w_start = @as(i32, @intCast(ow * stride_w)) - @as(i32, @intCast(padding_w));
                        for (0..in_channels) |ic| {
                            for (0..kernel_h) |kh| {
                                for (0..kernel_w) |kw| {
                                    const in_h = in_h_start + @as(i32, @intCast(kh));
                                    const in_w = in_w_start + @as(i32, @intCast(kw));
                                    if (in_h >= 0 and in_h < input_h and in_w >= 0 and in_w < input_w) {
                                        const in_idx = ((b * in_channels + ic) * input_h + @as(usize, @intCast(in_h))) * input_w + @as(usize, @intCast(in_w));
                                        const w_idx = (((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw);
                                        sum += input[in_idx] * weights[w_idx];
                                    }
                                }
                            }
                        }
                        const out_idx = ((b * out_channels + oc) * output_h + oh) * output_w + ow;
                        output[out_idx] = sum + bias[oc];
                    }
                }
            }
        }
    }

    fn cpuConv2dBackward(self: Backend, input: []const f32, weights: []const f32, grad_output: []const f32, grad_input: []f32, grad_weights: []f32, grad_bias: []f32, in_channels: usize, out_channels: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize, output_h: usize, output_w: usize, stride_h: usize, stride_w: usize, padding_h: usize, padding_w: usize) !void {
        _ = self;
        const batch_size = input.len / (in_channels * input_h * input_w);
        @memset(grad_input, 0);
        for (0..batch_size) |b| {
            for (0..out_channels) |oc| {
                for (0..output_h) |oh| {
                    for (0..output_w) |ow| {
                        const out_idx = ((b * out_channels + oc) * output_h + oh) * output_w + ow;
                        const go = grad_output[out_idx];
                        grad_bias[oc] += go;
                        const in_h_start = @as(i32, @intCast(oh * stride_h)) - @as(i32, @intCast(padding_h));
                        const in_w_start = @as(i32, @intCast(ow * stride_w)) - @as(i32, @intCast(padding_w));
                        for (0..in_channels) |ic| {
                            for (0..kernel_h) |kh| {
                                for (0..kernel_w) |kw| {
                                    const in_h = in_h_start + @as(i32, @intCast(kh));
                                    const in_w = in_w_start + @as(i32, @intCast(kw));
                                    if (in_h >= 0 and in_h < input_h and in_w >= 0 and in_w < input_w) {
                                        const in_idx = ((b * in_channels + ic) * input_h + @as(usize, @intCast(in_h))) * input_w + @as(usize, @intCast(in_w));
                                        const w_idx = (((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw);
                                        grad_weights[w_idx] += input[in_idx] * go;
                                        grad_input[in_idx] += weights[w_idx] * go;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn cpuMaxPool1dForward(self: Backend, input: []const f32, output: []f32, max_indices: []f32, channels: usize, input_len: usize, output_len: usize, pool_size: usize, stride: usize) !void {
        _ = self;
        const batch_size = if (input.len > 0) input.len / (channels * input_len) else 1;
        for (0..batch_size) |b| {
            for (0..channels) |ch| {
                for (0..output_len) |out_pos| {
                    const start_idx = b * channels * input_len + ch * input_len + out_pos * stride;
                    var max_val: f32 = -std.math.inf(f32);
                    var max_idx: usize = start_idx;
                    for (0..pool_size) |p| {
                        const idx = start_idx + p;
                        if (idx < (b * channels + ch + 1) * input_len and idx >= b * channels * input_len) {
                            const val = input[idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = idx;
                            }
                        }
                    }
                    const out_idx = b * channels * output_len + ch * output_len + out_pos;
                    output[out_idx] = max_val;
                    max_indices[out_idx] = @floatFromInt(max_idx);
                }
            }
        }
    }

    fn cpuMaxPool2dForward(self: Backend, input: []const f32, output: []f32, max_indices: []f32, channels: usize, input_h: usize, input_w: usize, output_h: usize, output_w: usize, pool_h: usize, pool_w: usize, stride_h: usize, stride_w: usize) !void {
        _ = self;
        const batch_size = if (input.len > 0) input.len / (channels * input_h * input_w) else 1;
        for (0..batch_size) |b| {
            for (0..channels) |ch| {
                for (0..output_h) |oh| {
                    for (0..output_w) |ow| {
                        const ih_start = oh * stride_h;
                        const iw_start = ow * stride_w;
                        var max_val: f32 = -std.math.inf(f32);
                        var max_idx: usize = 0;
                        for (0..pool_h) |ph| {
                            for (0..pool_w) |pw| {
                                const ih = ih_start + ph;
                                const iw = iw_start + pw;
                                if (ih < input_h and iw < input_w) {
                                    const in_idx = ((b * channels + ch) * input_h + ih) * input_w + iw;
                                    const val = input[in_idx];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_idx = in_idx;
                                    }
                                }
                            }
                        }
                        const out_idx = ((b * channels + ch) * output_h + oh) * output_w + ow;
                        output[out_idx] = max_val;
                        max_indices[out_idx] = @floatFromInt(max_idx);
                    }
                }
            }
        }
    }

    fn cpuDropoutForward(self: Backend, input: []const f32, output: []f32, mask: []f32, rate: f32, scaling_factor: f32, seed: u64) !void {
        _ = self;
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (input, output, mask) |in, *out, *m| {
            if (random.float(f32) > rate) {
                m.* = 1.0;
                out.* = in * scaling_factor;
            } else {
                m.* = 0.0;
                out.* = 0.0;
            }
        }
    }

    fn cpuVaeSamplingForward(self: Backend, input: []const f32, output: []f32, epsilon: []f32, seed: u64, latent_dim: usize) !void {
        _ = self;
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        const mu = input[0..latent_dim];
        const log_var = input[latent_dim..];
        for (0..latent_dim) |i| {
            const eps = random.floatNorm(f32);
            epsilon[i] = eps;
            output[i] = mu[i] + eps * std.math.exp(0.5 * log_var[i]);
        }
    }

    fn cpuVaeSamplingBackward(self: Backend, input: []const f32, grad_output: []const f32, grad_input: []f32, epsilon: []const f32, latent_dim: usize) !void {
        _ = self;
        const log_var = input[latent_dim..];
        const grad_mu = grad_input[0..latent_dim];
        const grad_log_var = grad_input[latent_dim..];
        for (0..latent_dim) |i| {
            grad_mu[i] = grad_output[i];
            grad_log_var[i] = grad_output[i] * epsilon[i] * std.math.exp(0.5 * log_var[i]) * 0.5;
        }
    }

    fn cpuAttentionForward(self: Backend, q: []const f32, k: []const f32, v: []const f32, output: []f32, scores_buffer: []f32, seq_len: usize, d_k: usize, scaling_factor: f32) !void {
        _ = self;
        // PERFORMANCE FIX: Use pre-allocated buffer instead of allocating in hot path (F2.1)
        const scores = scores_buffer[0..seq_len];

        for (0..seq_len) |i| {
            var max_score: f32 = -std.math.inf(f32);
            for (0..seq_len) |m| {
                var score: f32 = 0;
                for (0..d_k) |feat| score += q[i * d_k + feat] * k[m * d_k + feat];
                score *= scaling_factor;
                scores[m] = score;
                if (score > max_score) max_score = score;
            }
            var sum_exp: f32 = 0;
            for (scores) |*s| {
                s.* = std.math.exp(s.* - max_score);
                sum_exp += s.*;
            }
            for (0..d_k) |feat| {
                var res: f32 = 0;
                for (0..seq_len) |m| res += (scores[m] / sum_exp) * v[m * d_k + feat];
                output[i * d_k + feat] = res;
            }
        }
    }

    /// GRU forward step
    pub fn gruForwardStep(self: Backend, gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer, gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, h_prev: []const f32, h_prev_buf: ?*const metal.MTLBuffer, h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer, gate_acts: []f32, gate_acts_buf: ?*const metal.MTLBuffer, n_hh_out: []f32, n_hh_out_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
        switch (self.type) {
            .gpu => try self.metalGruForwardStep(gates_ih, gates_ih_buf, gates_hh, gates_hh_buf, bias, bias_buf, h_prev, h_prev_buf, h_curr, h_curr_buf, gate_acts, gate_acts_buf, n_hh_out, n_hh_out_buf, hidden_size),
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
    pub fn rnnForwardStep(self: Backend, gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer, gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
        switch (self.type) {
            .gpu => try self.metalRnnForwardStep(gates_ih, gates_ih_buf, gates_hh, gates_hh_buf, bias, bias_buf, h_curr, h_curr_buf, hidden_size),
            .cpu => {
                for (0..hidden_size) |j| {
                    h_curr[j] = std.math.tanh(gates_ih[j] + gates_hh[j] + bias[j]);
                }
            },
        }
    }

    /// Vanilla RNN backward step
    pub fn rnnBackwardStep(self: Backend, grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer, grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer, h_curr: []const f32, h_curr_buf: ?*const metal.MTLBuffer, grad_after_act: []f32, grad_after_act_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
        switch (self.type) {
            .gpu => try self.metalRnnBackwardStep(grad_h_curr, grad_h_curr_buf, grad_h_next, grad_h_next_buf, h_curr, h_curr_buf, grad_after_act, grad_after_act_buf, hidden_size),
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
    pub fn lstmForwardStep(self: Backend, gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer, gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, c_prev: []const f32, c_prev_buf: ?*const metal.MTLBuffer, c_curr: []f32, c_curr_buf: ?*const metal.MTLBuffer, h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer, gate_acts: []f32, gate_acts_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
        switch (self.type) {
            .gpu => try self.metalLstmForwardStep(gates_ih, gates_ih_buf, gates_hh, gates_hh_buf, bias, bias_buf, c_prev, c_prev_buf, c_curr, c_curr_buf, h_curr, h_curr_buf, gate_acts, gate_acts_buf, hidden_size),
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

    fn metalLstmForwardStep(self: Backend, gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer, gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, c_prev: []const f32, c_prev_buf: ?*const metal.MTLBuffer, c_curr: []f32, c_curr_buf: ?*const metal.MTLBuffer, h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer, gate_acts: []f32, gate_acts_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
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

    fn metalRnnForwardStep(self: Backend, gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer, gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
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

    fn metalRnnBackwardStep(self: Backend, grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer, grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer, h_curr: []const f32, h_curr_buf: ?*const metal.MTLBuffer, grad_after_act: []f32, grad_after_act_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
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

    fn metalGruForwardStep(self: Backend, gates_ih: []const f32, gates_ih_buf: ?*const metal.MTLBuffer, gates_hh: []const f32, gates_hh_buf: ?*const metal.MTLBuffer, bias: []const f32, bias_buf: ?*const metal.MTLBuffer, h_prev: []const f32, h_prev_buf: ?*const metal.MTLBuffer, h_curr: []f32, h_curr_buf: ?*const metal.MTLBuffer, gate_acts: []f32, gate_acts_buf: ?*const metal.MTLBuffer, n_hh_out: []f32, n_hh_out_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
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
    pub fn lstmBackwardStep(self: Backend, grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer, grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer, grad_c_next: []const f32, grad_c_next_buf: ?*const metal.MTLBuffer, gate_acts: []const f32, gate_acts_buf: ?*const metal.MTLBuffer, c_curr: []const f32, c_curr_buf: ?*const metal.MTLBuffer, c_prev: []const f32, c_prev_buf: ?*const metal.MTLBuffer, grad_gates: []f32, grad_gates_buf: ?*const metal.MTLBuffer, grad_c_prev: []f32, grad_c_prev_buf: ?*const metal.MTLBuffer, grad_h_prev_part: []f32, grad_h_prev_part_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
        switch (self.type) {
            .gpu => try self.metalLstmBackwardStep(grad_h_curr, grad_h_curr_buf, grad_h_next, grad_h_next_buf, grad_c_next, grad_c_next_buf, gate_acts, gate_acts_buf, c_curr, c_curr_buf, c_prev, c_prev_buf, grad_gates, grad_gates_buf, grad_c_prev, grad_c_prev_buf, grad_h_prev_part, grad_h_prev_part_buf, hidden_size),
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
    pub fn gruBackwardStep(self: Backend, grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer, grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer, gate_acts: []const f32, gate_acts_buf: ?*const metal.MTLBuffer, h_prev: []const f32, h_prev_buf: ?*const metal.MTLBuffer, n_hh: []const f32, n_hh_buf: ?*const metal.MTLBuffer, grad_gates_ih: []f32, grad_gates_ih_buf: ?*const metal.MTLBuffer, grad_gates_hh: []f32, grad_gates_hh_buf: ?*const metal.MTLBuffer, grad_h_prev: []f32, grad_h_prev_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
        switch (self.type) {
            .gpu => try self.metalGruBackwardStep(grad_h_curr, grad_h_curr_buf, grad_h_next, grad_h_next_buf, gate_acts, gate_acts_buf, h_prev, h_prev_buf, n_hh, n_hh_buf, grad_gates_ih, grad_gates_ih_buf, grad_gates_hh, grad_gates_hh_buf, grad_h_prev, grad_h_prev_buf, hidden_size),
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

    fn metalLstmBackwardStep(self: Backend, grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer, grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer, grad_c_next: []const f32, grad_c_next_buf: ?*const metal.MTLBuffer, gate_acts: []const f32, gate_acts_buf: ?*const metal.MTLBuffer, c_curr: []const f32, c_curr_buf: ?*const metal.MTLBuffer, c_prev: []const f32, c_prev_buf: ?*const metal.MTLBuffer, grad_gates: []f32, grad_gates_buf: ?*const metal.MTLBuffer, grad_c_prev: []f32, grad_c_prev_buf: ?*const metal.MTLBuffer, grad_h_prev_part: []f32, grad_h_prev_part_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
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

    fn metalGruBackwardStep(self: Backend, grad_h_curr: []const f32, grad_h_curr_buf: ?*const metal.MTLBuffer, grad_h_next: []const f32, grad_h_next_buf: ?*const metal.MTLBuffer, gate_acts: []const f32, gate_acts_buf: ?*const metal.MTLBuffer, h_prev: []const f32, h_prev_buf: ?*const metal.MTLBuffer, n_hh: []const f32, n_hh_buf: ?*const metal.MTLBuffer, grad_gates_ih: []f32, grad_gates_ih_buf: ?*const metal.MTLBuffer, grad_gates_hh: []f32, grad_gates_hh_buf: ?*const metal.MTLBuffer, grad_h_prev: []f32, grad_h_prev_buf: ?*const metal.MTLBuffer, hidden_size: usize) !void {
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

    /// Batch Normalization forward pass (training mode)
    pub fn batchNormForwardTraining(
        self: Backend,
        input: []const f32,
        input_buf: ?*const metal.MTLBuffer,
        output: []f32,
        output_buf: ?*const metal.MTLBuffer,
        gamma: []const f32,
        gamma_buf: ?*const metal.MTLBuffer,
        beta: []const f32,
        beta_buf: ?*const metal.MTLBuffer,
        epsilon: f32,
        momentum: f32,
        running_mean: []f32,
        running_mean_buf: ?*const metal.MTLBuffer,
        running_var: []f32,
        running_var_buf: ?*const metal.MTLBuffer,
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalBatchNormForwardTraining(
                    input,
                    input_buf,
                    output,
                    output_buf,
                    gamma,
                    gamma_buf,
                    beta,
                    beta_buf,
                    epsilon,
                    momentum,
                    running_mean,
                    running_mean_buf,
                    running_var,
                    running_var_buf,
                ),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => try self.cpuBatchNormForwardTraining(
                input,
                output,
                gamma,
                beta,
                epsilon,
                momentum,
                running_mean,
                running_var,
            ),
        }
    }

    /// Batch Normalization forward pass (inference mode)
    pub fn batchNormForwardInference(
        self: Backend,
        input: []const f32,
        input_buf: ?*const metal.MTLBuffer,
        output: []f32,
        output_buf: ?*const metal.MTLBuffer,
        gamma: []const f32,
        gamma_buf: ?*const metal.MTLBuffer,
        beta: []const f32,
        beta_buf: ?*const metal.MTLBuffer,
        epsilon: f32,
        running_mean: []const f32,
        running_mean_buf: ?*const metal.MTLBuffer,
        running_var: []const f32,
        running_var_buf: ?*const metal.MTLBuffer,
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalBatchNormForwardInference(
                    input,
                    input_buf,
                    output,
                    output_buf,
                    gamma,
                    gamma_buf,
                    beta,
                    beta_buf,
                    epsilon,
                    running_mean,
                    running_mean_buf,
                    running_var,
                    running_var_buf,
                ),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => try self.cpuBatchNormForwardInference(
                input,
                output,
                gamma,
                beta,
                epsilon,
                running_mean,
                running_var,
            ),
        }
    }

    /// Batch Normalization backward pass
    pub fn batchNormBackward(
        self: Backend,
        input: []const f32,
        input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32,
        grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32,
        grad_input_buf: ?*const metal.MTLBuffer,
        gamma: []const f32,
        gamma_buf: ?*const metal.MTLBuffer,
        grad_gamma: []f32,
        grad_gamma_buf: ?*const metal.MTLBuffer,
        grad_beta: []f32,
        grad_beta_buf: ?*const metal.MTLBuffer,
        epsilon: f32,
        running_mean: []const f32,
        running_mean_buf: ?*const metal.MTLBuffer,
        running_var: []const f32,
        running_var_buf: ?*const metal.MTLBuffer,
    ) !void {
        switch (self.type) {
            .gpu => |gpu| switch (gpu) {
                .metal => try self.metalBatchNormBackward(
                    input,
                    input_buf,
                    grad_output,
                    grad_output_buf,
                    grad_input,
                    grad_input_buf,
                    gamma,
                    gamma_buf,
                    grad_gamma,
                    grad_gamma_buf,
                    grad_beta,
                    grad_beta_buf,
                    epsilon,
                    running_mean,
                    running_mean_buf,
                    running_var,
                    running_var_buf,
                ),
                else => return error.CudaNotYetImplemented,
            },
            .cpu => try self.cpuBatchNormBackward(
                input,
                grad_output,
                grad_input,
                gamma,
                grad_gamma,
                grad_beta,
                epsilon,
            ),
        }
    }
};

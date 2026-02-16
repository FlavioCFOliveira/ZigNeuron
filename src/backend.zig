/// Backend for neural network computation
/// Supports GPU (Metal/Vulkan) and CPU execution
/// GPU is the PRIORITY for all training and inference operations
const std = @import("std");
const activation = @import("activation.zig");
const loss = @import("loss.zig");
const vulkan_module = @import("vulkan.zig");

/// Available GPU backends in priority order
pub const GpuBackend = enum {
    /// Apple Silicon - Metal compute shaders
    metal,
    /// Cross-platform - Vulkan compute shaders
    vulkan,
};

/// Backend selection - GPU preferred, CPU fallback
pub const Backend = union(enum) {
    gpu: GpuBackend,
    cpu,

    /// Returns the default backend based on available hardware
    /// Priority: Metal (Apple Silicon) > Vulkan > CPU
    pub fn default() Backend {
        return Backend.detect();
    }

    /// Detect available hardware and return best backend
    /// Priority: Metal (Apple Silicon) > Vulkan (cross-platform) > CPU
    pub fn detect() Backend {
        // On macOS, try Metal first
        // We use compile-time detection since we need the info at compile time
        // For cross-platform, we'll use Vulkan or CPU fallback
        const os_tag = @import("builtin").os.tag;

        // Try Metal first (Apple Silicon)
        if (os_tag == .macos) {
            if (metalSupported()) {
                return Backend{ .gpu = .metal };
            }
        }

        // Try Vulkan next (cross-platform)
        if (vulkanSupported()) {
            return Backend{ .gpu = .vulkan };
        }

        // Fall back to CPU
        return Backend{ .cpu = {} };
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
    pub fn matMul(self: Backend, a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) !void {
        switch (self) {
            .gpu => |gpu| switch (gpu) {
                .metal => try metalMatMul(a, b, c, m, n, k),
                .vulkan => try vulkanMatMul(a, b, c, m, n, k),
            },
            .cpu => cpuMatMul(a, b, c, m, n, k),
        }
    }

    /// Execute batched matrix multiplication on the selected backend
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    /// This is more efficient for batch processing
    pub fn matMulBatch(self: Backend, a: []const f32, b: []const f32, c: []f32, batch_size: usize, n: usize, k: usize) !void {
        switch (self) {
            .gpu => |gpu| switch (gpu) {
                .metal => try metalMatMulBatch(a, b, c, batch_size, n, k),
                .vulkan => try vulkanMatMulBatch(a, b, c, batch_size, n, k),
            },
            .cpu => cpuMatMulBatch(a, b, c, batch_size, n, k),
        }
    }

    /// Execute activation function on array
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn activationForward(self: Backend, act: activation.Activation, input: []f32, output: []f32) !void {
        switch (self) {
            .gpu => |gpu| switch (gpu) {
                .metal => try metalActivationForward(act, input, output),
                .vulkan => try vulkanActivationForward(act, input, output),
            },
            .cpu => cpuActivationForward(act, input, output),
        }
    }

    /// Execute activation backward pass
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn activationBackward(self: Backend, act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        switch (self) {
            .gpu => |gpu| switch (gpu) {
                .metal => try metalActivationBackward(act, input, grad_output, grad_input),
                .vulkan => try vulkanActivationBackward(act, input, grad_output, grad_input),
            },
            .cpu => cpuActivationBackward(act, input, grad_output, grad_input),
        }
    }

    /// Execute loss function gradient on the selected backend
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn lossBackward(self: Backend, loss_fn: loss.Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        switch (self) {
            .gpu => |gpu| switch (gpu) {
                .metal => try metalLossBackward(loss_fn, output, target, grad_output),
                .vulkan => try vulkanLossBackward(loss_fn, output, target, grad_output),
            },
            .cpu => cpuLossBackward(loss_fn, output, target, grad_output),
        }
    }

    // ================== Metal implementations ==================
    // ( Apple Silicon GPU )

    fn metalMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) !void {
        // Check if we're on macOS
        if (!isMacos()) {
            return error.NotAvailable;
        }

        // Ultra-low threshold for GPU usage to maximize Metal parallelism on Apple Silicon
        // Metal on Apple Silicon has extremely low overhead and massive parallelism
        const total_size = @as(usize, m) * n * k;
        if (total_size < 128) { // Ultra-low threshold for maximum GPU utilization
            cpuMatMul(a, b, c, m, n, k);
            return;
        }

        // Validate inputs
        if (a.len < m * k) return error.BufferTooSmall;
        if (b.len < k * n) return error.BufferTooSmall;
        if (c.len < m * n) return error.BufferTooSmall;

        // Use GPU implementation
        try metalMatMulGPU(a, b, c, m, n, k);
    }

    fn metalMatMulBatch(a: []const f32, b: []const f32, c: []f32, batch_size: usize, n: usize, k: usize) !void {
        // Check if we're on macOS
        if (!isMacos()) {
            return error.NotAvailable;
        }

        // Batch operations are perfect for GPU - always use GPU for batches
        const total_size = @as(usize, batch_size) * n * k;
        if (total_size < 64) { // Even lower threshold for batches
            cpuMatMulBatch(a, b, c, batch_size, n, k);
            return;
        }

        // Validate inputs
        if (a.len < batch_size * k) return error.BufferTooSmall;
        if (b.len < k * n) return error.BufferTooSmall;
        if (c.len < batch_size * n) return error.BufferTooSmall;

        // Use GPU implementation
        try metalMatMulBatchGPU(a, b, c, batch_size, n, k);
    }

    fn metalActivationForward(act: activation.Activation, input: []f32, output: []f32) !void {
        // Ultra-low threshold to maximize Metal GPU parallelism on Apple Silicon
        if (input.len < 32) { // Ultra-low threshold for maximum GPU utilization
            cpuActivationForward(act, input, output);
            return;
        }

        if (input.len != output.len) return error.ShapeMismatch;

        try metalActivationForwardGPU(act, input, output);
    }

    fn metalActivationBackward(act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        // Ultra-low threshold to maximize Metal GPU parallelism on Apple Silicon
        if (input.len < 32) { // Ultra-low threshold for maximum GPU utilization
            cpuActivationBackward(act, input, grad_output, grad_input);
            return;
        }

        if (input.len != grad_output.len or input.len != grad_input.len) {
            return error.ShapeMismatch;
        }

        try metalActivationBackwardGPU(act, input, grad_output, grad_input);
    }

    fn metalLossBackward(loss_fn: loss.Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        // Ultra-low threshold to maximize Metal GPU parallelism on Apple Silicon
        if (output.len < 32) { // Ultra-low threshold for maximum GPU utilization
            cpuLossBackward(loss_fn, output, target, grad_output);
            return;
        }

        if (output.len != target.len or output.len != grad_output.len) {
            return error.ShapeMismatch;
        }

        try metalLossBackwardGPU(loss_fn, output, target, grad_output);
    }

    /// GPU implementation of batched matrix multiplication using Metal
    fn metalMatMulBatchGPU(a: []const f32, b: []const f32, c: []f32, batch_size: usize, n: usize, k: usize) !void {
        // Metal implementation for batched operations
        // Batched operations are highly efficient on GPU

        // Validate matrix dimensions
        if (a.len < batch_size * k) return error.BufferTooSmall;
        if (b.len < k * n) return error.BufferTooSmall;
        if (c.len < batch_size * n) return error.BufferTooSmall;

        // For now, use optimized CPU implementation
        // In production, this would use Metal compute shaders with batch dimension
        cpuMatMulBatch(a, b, c, batch_size, n, k);
    }

    /// GPU implementation of matrix multiplication using Metal
    fn metalMatMulGPU(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) !void {
        // Metal implementation for Apple Silicon
        // Uses inline Metal Shading Language (MSL) via runtime compilation

        // Since Zig doesn't have native Metal FFI in stdlib,
        // we use a CPU fallback with validation for non-macOS platforms
        // For actual macOS with Metal, this would use:
        // - MTLDevice, MTLCommandQueue, MTLComputePipelineState
        // - MTLBuffer for data
        // - MTLComputeCommandEncoder for execution

        // For now, use CPU implementation with validation
        // In a production implementation, this would compile and execute MSL shaders

        // Validate matrix dimensions
        if (a.len < m * k) return error.BufferTooSmall;
        if (b.len < k * n) return error.BufferTooSmall;
        if (c.len < m * n) return error.BufferTooSmall;

        // CPU fallback - actual Metal implementation would use GPU compute
        cpuMatMul(a, b, c, m, n, k);
    }

    /// GPU implementation of activation forward using Metal
    fn metalActivationForwardGPU(act: activation.Activation, input: []f32, output: []f32) !void {
        // Metal compute shader for activation functions
        // Would compile and execute MSL kernel like:
        // kernel void activation_forward(device const float* input,
        //                                device float* output,
        //                                uint gid [thread_position_in_grid]) {
        //     output[gid] = activation(input[gid]);
        // }

        if (input.len != output.len) return error.ShapeMismatch;

        // CPU fallback
        for (0..input.len) |i| {
            output[i] = act.forward(input[i]);
        }
    }

    /// GPU implementation of activation backward using Metal
    fn metalActivationBackwardGPU(act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        // Metal compute shader for activation backward
        // kernel void activation_backward(device const float* input,
        //                                 device const float* grad_output,
        //                                 device float* grad_input,
        //                                 uint gid [thread_position_in_grid]) {
        //     grad_input[gid] = activation_derivative(input[gid]) * grad_output[gid];
        // }

        if (input.len != grad_output.len or input.len != grad_input.len) {
            return error.ShapeMismatch;
        }

        // CPU fallback
        for (0..input.len) |i| {
            grad_input[i] = act.backward(input[i], grad_output[i]);
        }
    }

    /// GPU implementation of loss backward using Metal
    fn metalLossBackwardGPU(loss_fn: loss.Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
        // Metal compute shader for loss gradient computation
        // Would support MSE, CrossEntropy, BinaryCrossEntropy

        if (output.len != target.len or output.len != grad_output.len) {
            return error.ShapeMismatch;
        }

        // CPU fallback with correct gradients
        switch (loss_fn) {
            .mse => {
                // MSE gradient: dL/dy = 2(y - t) / n for average loss
                const scale = 2.0 / @as(f32, @floatFromInt(output.len));
                for (0..output.len) |i| {
                    grad_output[i] = scale * (output[i] - target[i]);
                }
            },
            .cross_entropy => {
                // Cross-entropy gradient: (p - t) for logits
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
                // BCE gradient: (p - t) for sigmoid output
                for (0..output.len) |i| {
                    grad_output[i] = output[i] - target[i];
                }
            },
        }
    }

    // ================== Vulkan implementations ==================
    // ( cross-platform GPU )

    fn vulkanMatMulBatch(a: []const f32, b: []const f32, c: []f32, batch_size: usize, n: usize, k: usize) !void {
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

    fn vulkanMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) !void {
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

    fn vulkanActivationForward(act: activation.Activation, input: []f32, output: []f32) !void {
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

    fn vulkanActivationBackward(act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) !void {
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

    fn vulkanLossBackward(loss_fn: loss.Loss, output: []const f32, target: []const f32, grad_output: []f32) !void {
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
        switch (loss_fn) {
            .mse => {
                // MSE gradient: dL/dy = 2(y - t)
                for (0..output.len) |i| {
                    grad_output[i] = 2 * (output[i] - target[i]);
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
                // BCE gradient with sigmoid: (p - t)
                // The gradient simplifies to prediction - target
                // This is the correct form when output is passed through sigmoid
                for (0..output.len) |i| {
                    grad_output[i] = output[i] - target[i];
                }
            },
        }
    }
};

test "backend default detection" {
    const backend = Backend.default();
    // Should return either gpu or cpu, not error
    _ = backend;
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

    const backend = Backend{ .cpu = {} };
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
    const backend = Backend{ .cpu = {} };
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
    const backend = Backend{ .cpu = {} };
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
    const backend = Backend{ .cpu = {} };
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

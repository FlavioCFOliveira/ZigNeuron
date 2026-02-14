/// Backend for neural network computation
/// Supports GPU (Metal/Vulkan) and CPU execution
/// GPU is the PRIORITY for all training and inference operations
const std = @import("std");
const activation = @import("activation.zig");

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
        // Try Metal first (Apple Silicon)
        if (std.zig.system.target.get().os.tag == .macos) {
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
        // For now, return true on macOS
        // Will be expanded to check for actual Metal device
        return std.zig.system.target.get().os.tag == .macos;
    }

    /// Check if Vulkan is available on this system
    fn vulkanSupported() bool {
        // Check for Vulkan support at runtime
        // For now, return false until Vulkan backend is implemented
        return false;
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

    // ================== Metal implementations ==================
    // ( stubs for Apple Silicon GPU )

    fn metalMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) !void {
        _ = a;
        _ = b;
        _ = c;
        _ = m;
        _ = n;
        _ = k;
        // TODO: Implement Metal GPU matmul
        return error.NotImplemented;
    }

    fn metalActivationForward(act: activation.Activation, input: []f32, output: []f32) !void {
        _ = act;
        _ = input;
        _ = output;
        // TODO: Implement Metal GPU activation
        return error.NotImplemented;
    }

    fn metalActivationBackward(act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        _ = act;
        _ = input;
        _ = grad_output;
        _ = grad_input;
        // TODO: Implement Metal GPU activation backward
        return error.NotImplemented;
    }

    // ================== Vulkan implementations ==================
    // ( stubs for cross-platform GPU )

    fn vulkanMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) !void {
        _ = a;
        _ = b;
        _ = c;
        _ = m;
        _ = n;
        _ = k;
        // TODO: Implement Vulkan GPU matmul
        return error.NotImplemented;
    }

    fn vulkanActivationForward(act: activation.Activation, input: []f32, output: []f32) !void {
        _ = act;
        _ = input;
        _ = output;
        // TODO: Implement Vulkan GPU activation
        return error.NotImplemented;
    }

    fn vulkanActivationBackward(act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        _ = act;
        _ = input;
        _ = grad_output;
        _ = grad_input;
        // TODO: Implement Vulkan GPU activation backward
        return error.NotImplemented;
    }

    // ================== CPU implementations ==================
    // ( fallbacks when no GPU available )

    fn cpuMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
        // Standard matrix multiplication: C = A * B
        // A is m×k, B is k×n, C is m×n
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                for (0..k) |p| {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    fn cpuActivationForward(act: activation.Activation, input: []f32, output: []f32) void {
        for (input, output) |in, out| {
            out.* = act.forward(in);
        }
    }

    fn cpuActivationBackward(act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) void {
        for (input, grad_output, grad_input) |in, go, gi| {
            gi.* = act.backward(in, go);
        }
    }
};

/// Backend for neural network computation
/// Supports GPU (Metal) and CPU execution for Apple Silicon
/// GPU is the PRIORITY for all training and inference operations
const std = @import("std");
const activation = @import("activation.zig");
const metal = @import("metal/backend.zig");

pub const Backend = union(enum) {
    gpu: *metal.Backend,
    cpu,

    /// Returns the default backend (GPU if available, CPU otherwise)
    /// GPU is PRIORITY for all operations
    pub fn default() !Backend {
        // Try GPU first - it's the priority
        return Backend.initGPU() orelse Backend{ .cpu = {} };
    }

    /// Initialize GPU backend if available
    pub fn initGPU() !Backend {
        const gpu = try metal.Backend.init();
        return Backend{ .gpu = gpu };
    }

    /// Execute matrix multiplication on the selected backend
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn matMul(self: *Backend, a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) !void {
        switch (self.*) {
            .gpu => |gpu| {
                try gpu.matMul(a, b, c, m, n, k);
            },
            .cpu => {
                self.cpuMatMul(a, b, c, m, n, k);
            },
        }
    }

    /// CPU-based matrix multiplication (fallback)
    fn cpuMatMul(self: *Backend, a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
        _ = self;
        // TODO: Optimize with SIMD/vectorization when GPU unavailable
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

    /// Execute activation function on array
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn activationForward(self: *Backend, act: activation.Activation, input: []f32, output: []f32) !void {
        switch (self.*) {
            .gpu => |gpu| {
                _ = gpu;
                // TODO: Implement Metal GPU activation
                // Fall through to CPU for now
                self.cpuActivationForward(act, input, output);
            },
            .cpu => {
                self.cpuActivationForward(act, input, output);
            },
        }
    }

    fn cpuActivationForward(self: *Backend, act: activation.Activation, input: []f32, output: []f32) void {
        _ = self;
        for (input, output) |in, out| {
            out.* = act.forward(in);
        }
    }

    /// Execute activation backward pass
    /// GPU is PRIORITY - CPU only used if GPU unavailable
    pub fn activationBackward(self: *Backend, act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        switch (self.*) {
            .gpu => |gpu| {
                _ = gpu;
                // TODO: Implement Metal GPU activation backward
                // Fall through to CPU for now
                self.cpuActivationBackward(act, input, grad_output, grad_input);
            },
            .cpu => {
                self.cpuActivationBackward(act, input, grad_output, grad_input);
            },
        }
    }

    fn cpuActivationBackward(self: *Backend, act: activation.Activation, input: []const f32, grad_output: []const f32, grad_input: []f32) void {
        _ = self;
        for (input, grad_output, grad_input) |in, go, gi| {
            gi.* = act.backward(in, go);
        }
    }
};

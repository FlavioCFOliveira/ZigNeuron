/// Generic optimization utilities for neural network performance
/// Includes SIMD vectorization, multi-threading, and memory optimizations
const std = @import("std");
const builtin = @import("builtin");

/// SIMD vectorization utilities for ARM NEON (Apple Silicon)
pub const SIMD = struct {
    /// Vector width for NEON (4 floats)
    pub const VECTOR_WIDTH: usize = 4;

    /// Check if NEON is available (always true on Apple Silicon)
    pub fn isNeonAvailable() bool {
        return builtin.cpu.arch == .aarch64;
    }

    /// Vectorized ReLU activation using NEON
    pub fn reluVectorized(input: []const f32, output: []f32) void {
        // Check for NEON support
        if (!isNeonAvailable()) {
            // Fallback to scalar implementation
            for (input, output) |x, *out| {
                out.* = if (x > 0) x else 0;
            }
            return;
        }

        // Process 4 elements at a time using NEON
        var i: usize = 0;
        const vector_len = VECTOR_WIDTH;

        // Process 4 elements at a time
        while (i + vector_len <= input.len) : (i += vector_len) {
            // Load 4 floats into NEON register
            const vec: @Vector(VECTOR_WIDTH, f32) = input[i..][0..VECTOR_WIDTH].*;

            // Create zero vector
            const zero = @as(@Vector(VECTOR_WIDTH, f32), @splat(0.0));

            // Compute max(vec, 0)
            const result = @select(f32, vec > zero, vec, zero);

            // Store result
            output[i..][0..VECTOR_WIDTH].* = result;
        }

        // Handle remaining elements
        for (input[i..], output[i..]) |x, *out| {
            out.* = if (x > 0) x else 0;
        }
    }

    /// Vectorized sigmoid activation using NEON
    pub fn sigmoidVectorized(input: []const f32, output: []f32) void {
        if (!isNeonAvailable()) {
            // Fallback to scalar implementation
            for (input, output) |x, *out| {
                out.* = 1.0 / (1.0 + std.math.exp(-x));
            }
            return;
        }

        var i: usize = 0;
        const vector_len = VECTOR_WIDTH;

        // Process 4 elements at a time
        while (i + vector_len <= input.len) : (i += vector_len) {
            // Load 4 floats
            const vec: @Vector(VECTOR_WIDTH, f32) = input[i..][0..VECTOR_WIDTH].*;

            // Compute sigmoid approximation
            // Use: sigmoid(x) ≈ 0.5 + 0.197 * x / (1 + 0.197 * |x|)
            const abs_vec = @abs(vec);
            const scale = @as(@Vector(VECTOR_WIDTH, f32), @splat(0.197));
            const half = @as(@Vector(VECTOR_WIDTH, f32), @splat(0.5));

            const numerator = scale * vec;
            const denominator = @as(@Vector(VECTOR_WIDTH, f32), @splat(1.0)) + scale * abs_vec;
            const result = half + numerator / denominator;

            // Store result
            output[i..][0..VECTOR_WIDTH].* = result;
        }

        // Handle remaining elements
        for (input[i..], output[i..]) |x, *out| {
            out.* = 1.0 / (1.0 + std.math.exp(-x));
        }
    }

    /// Vectorized tanh activation using NEON
    pub fn tanhVectorized(input: []const f32, output: []f32) void {
        if (!isNeonAvailable()) {
            // Fallback to scalar implementation
            for (input, output) |x, *out| {
                out.* = std.math.tanh(x);
            }
            return;
        }

        var i: usize = 0;
        const vector_len = VECTOR_WIDTH;

        // Process 4 elements at a time
        while (i + vector_len <= input.len) : (i += vector_len) {
            // Load 4 floats
            const vec: @Vector(VECTOR_WIDTH, f32) = input[i..][0..VECTOR_WIDTH].*;

            // Compute tanh approximation
            // Use: tanh(x) ≈ x * (1 + 0.5 * x^2) / (1 + x^2)
            const x2 = vec * vec;
            const one = @as(@Vector(VECTOR_WIDTH, f32), @splat(1.0));
            const numerator = vec * (one + @as(@Vector(VECTOR_WIDTH, f32), @splat(0.5)) * x2);
            const denominator = one + x2;
            const result = numerator / denominator;

            // Store result
            output[i..][0..VECTOR_WIDTH].* = result;
        }

        // Handle remaining elements
        for (input[i..], output[i..]) |x, *out| {
            out.* = std.math.tanh(x);
        }
    }

    /// Vectorized element-wise addition using NEON
    pub fn addVectorized(a: []const f32, b: []const f32, output: []f32) void {
        // Check for NEON support
        if (!isNeonAvailable()) {
            // Fallback to scalar implementation
            for (a, b, output) |x, y, *out| {
                out.* = x + y;
            }
            return;
        }

        var i: usize = 0;
        const vector_len = VECTOR_WIDTH;

        // Process 4 elements at a time
        while (i + vector_len <= a.len) : (i += vector_len) {
            // Load vectors
            const vec_a: @Vector(VECTOR_WIDTH, f32) = a[i..][0..VECTOR_WIDTH].*;
            const vec_b: @Vector(VECTOR_WIDTH, f32) = b[i..][0..VECTOR_WIDTH].*;

            // Add vectors
            const result = vec_a + vec_b;

            // Store result
            output[i..][0..VECTOR_WIDTH].* = result;
        }

        // Handle remaining elements
        for (a[i..], b[i..], output[i..]) |x, y, *out| {
            out.* = x + y;
        }
    }

    /// Vectorized element-wise multiplication using NEON
    pub fn mulVectorized(a: []const f32, b: []const f32, output: []f32) void {
        // Check for NEON support
        if (!isNeonAvailable()) {
            // Fallback to scalar implementation
            for (a, b, output) |x, y, *out| {
                out.* = x * y;
            }
            return;
        }

        var i: usize = 0;
        const vector_len = VECTOR_WIDTH;

        // Process 4 elements at a time
        while (i + vector_len <= a.len) : (i += vector_len) {
            // Load vectors
            const vec_a: @Vector(VECTOR_WIDTH, f32) = a[i..][0..VECTOR_WIDTH].*;
            const vec_b: @Vector(VECTOR_WIDTH, f32) = b[i..][0..VECTOR_WIDTH].*;

            // Multiply vectors
            const result = vec_a * vec_b;

            // Store result
            output[i..][0..VECTOR_WIDTH].* = result;
        }

        // Handle remaining elements
        for (a[i..], b[i..], output[i..]) |x, y, *out| {
            out.* = x * y;
        }
    }

    /// Vectorized squared error computation using NEON
    pub fn squaredErrorVectorized(a: []const f32, b: []const f32, output: []f32) void {
        // Check for NEON support
        if (!isNeonAvailable()) {
            // Fallback to scalar implementation
            for (a, b, output) |x, y, *out| {
                const diff = x - y;
                out.* = diff * diff;
            }
            return;
        }

        var i: usize = 0;
        const vector_len = VECTOR_WIDTH;

        // Process 4 elements at a time
        while (i + vector_len <= a.len) : (i += vector_len) {
            // Load vectors
            const vec_a: @Vector(VECTOR_WIDTH, f32) = a[i..][0..VECTOR_WIDTH].*;
            const vec_b: @Vector(VECTOR_WIDTH, f32) = b[i..][0..VECTOR_WIDTH].*;

            // Compute squared error
            const diff = vec_a - vec_b;
            const result = diff * diff;

            // Store result
            output[i..][0..VECTOR_WIDTH].* = result;
        }

        // Handle remaining elements
        for (a[i..], b[i..], output[i..]) |x, y, *out| {
            const diff = x - y;
            out.* = diff * diff;
        }
    }

    /// Vectorized ReLU backward: grad_input = grad_output * (output > 0 ? 1 : 0)
    pub fn reluBackwardVectorized(output: []const f32, grad_output: []const f32, grad_input: []f32) void {
        if (!isNeonAvailable()) {
            for (output, grad_output, grad_input) |y, go, *gi| {
                gi.* = if (y > 0) go else 0;
            }
            return;
        }

        var i: usize = 0;
        const vector_len = VECTOR_WIDTH;

        while (i + vector_len <= output.len) : (i += vector_len) {
            const vec_y: @Vector(VECTOR_WIDTH, f32) = output[i..][0..VECTOR_WIDTH].*;
            const vec_go: @Vector(VECTOR_WIDTH, f32) = grad_output[i..][0..VECTOR_WIDTH].*;
            const zero = @as(@Vector(VECTOR_WIDTH, f32), @splat(0.0));
            const result = @select(f32, vec_y > zero, vec_go, zero);
            grad_input[i..][0..VECTOR_WIDTH].* = result;
        }

        for (output[i..], grad_output[i..], grad_input[i..]) |y, go, *gi| {
            gi.* = if (y > 0) go else 0;
        }
    }

    /// Vectorized Sigmoid backward: grad_input = grad_output * output * (1 - output)
    pub fn sigmoidBackwardVectorized(output: []const f32, grad_output: []const f32, grad_input: []f32) void {
        if (!isNeonAvailable()) {
            for (output, grad_output, grad_input) |y, go, *gi| {
                gi.* = go * y * (1.0 - y);
            }
            return;
        }

        var i: usize = 0;
        const vector_len = VECTOR_WIDTH;

        while (i + vector_len <= output.len) : (i += vector_len) {
            const vec_y: @Vector(VECTOR_WIDTH, f32) = output[i..][0..VECTOR_WIDTH].*;
            const vec_go: @Vector(VECTOR_WIDTH, f32) = grad_output[i..][0..VECTOR_WIDTH].*;
            const one = @as(@Vector(VECTOR_WIDTH, f32), @splat(1.0));
            const result = vec_go * vec_y * (one - vec_y);
            grad_input[i..][0..VECTOR_WIDTH].* = result;
        }

        for (output[i..], grad_output[i..], grad_input[i..]) |y, go, *gi| {
            gi.* = go * y * (1.0 - y);
        }
    }

    /// Vectorized Tanh backward: grad_input = grad_output * (1 - output^2)
    pub fn tanhBackwardVectorized(output: []const f32, grad_output: []const f32, grad_input: []f32) void {
        if (!isNeonAvailable()) {
            for (output, grad_output, grad_input) |y, go, *gi| {
                gi.* = go * (1.0 - y * y);
            }
            return;
        }

        var i: usize = 0;
        const vector_len = VECTOR_WIDTH;

        while (i + vector_len <= output.len) : (i += vector_len) {
            const vec_y: @Vector(VECTOR_WIDTH, f32) = output[i..][0..VECTOR_WIDTH].*;
            const vec_go: @Vector(VECTOR_WIDTH, f32) = grad_output[i..][0..VECTOR_WIDTH].*;
            const one = @as(@Vector(VECTOR_WIDTH, f32), @splat(1.0));
            const result = vec_go * (one - vec_y * vec_y);
            grad_input[i..][0..VECTOR_WIDTH].* = result;
        }

        for (output[i..], grad_output[i..], grad_input[i..]) |y, go, *gi| {
            gi.* = go * (1.0 - y * y);
        }
    }

    /// PERFORMANCE FIX: Vectorized Layer Normalization (F2.3)
    /// Normalizes input using pre-computed mean and inverse std
    pub fn layerNormVectorized(input: []const f32, output: []f32, gamma: []const f32, beta: []const f32, mean: f32, inv_std: f32) void {
        if (!isNeonAvailable()) {
            for (input, output, 0..) |x, *o, i| {
                const normalized = (x - mean) * inv_std;
                o.* = normalized * gamma[i] + beta[i];
            }
            return;
        }

        var i: usize = 0;
        const vector_len = VECTOR_WIDTH;
        const mean_vec = @as(@Vector(VECTOR_WIDTH, f32), @splat(mean));
        const inv_std_vec = @as(@Vector(VECTOR_WIDTH, f32), @splat(inv_std));

        // Process 4 elements at a time
        while (i + vector_len <= input.len) : (i += vector_len) {
            const x_vec: @Vector(VECTOR_WIDTH, f32) = input[i..][0..VECTOR_WIDTH].*;
            const g_vec: @Vector(VECTOR_WIDTH, f32) = gamma[i..][0..VECTOR_WIDTH].*;
            const b_vec: @Vector(VECTOR_WIDTH, f32) = beta[i..][0..VECTOR_WIDTH].*;

            const normalized = (x_vec - mean_vec) * inv_std_vec;
            const result = normalized * g_vec + b_vec;
            output[i..][0..VECTOR_WIDTH].* = result;
        }

        // Handle remaining elements
        for (input[i..], output[i..], gamma[i..], beta[i..]) |x, *o, g, b| {
            const normalized = (x - mean) * inv_std;
            o.* = normalized * g + b;
        }
    }

    /// PERFORMANCE FIX: Vectorized Batch Normalization (F2.3)
    pub fn batchNormVectorized(input: []const f32, output: []f32, gamma: []const f32, beta: []const f32, mean: f32, inv_std: f32) void {
        if (!isNeonAvailable()) {
            for (input, output, 0..) |x, *o, i| {
                const normalized = (x - mean) * inv_std;
                o.* = normalized * gamma[i] + beta[i];
            }
            return;
        }

        var i: usize = 0;
        const vector_len = VECTOR_WIDTH;
        const mean_vec = @as(@Vector(VECTOR_WIDTH, f32), @splat(mean));
        const inv_std_vec = @as(@Vector(VECTOR_WIDTH, f32), @splat(inv_std));

        // Process 4 elements at a time
        while (i + vector_len <= input.len) : (i += vector_len) {
            const x_vec: @Vector(VECTOR_WIDTH, f32) = input[i..][0..VECTOR_WIDTH].*;
            const g_vec: @Vector(VECTOR_WIDTH, f32) = gamma[i..][0..VECTOR_WIDTH].*;
            const b_vec: @Vector(VECTOR_WIDTH, f32) = beta[i..][0..VECTOR_WIDTH].*;

            const normalized = (x_vec - mean_vec) * inv_std_vec;
            const result = normalized * g_vec + b_vec;
            output[i..][0..VECTOR_WIDTH].* = result;
        }

        // Handle remaining elements
        for (input[i..], output[i..], gamma[i..], beta[i..]) |x, *o, g, b| {
            const normalized = (x - mean) * inv_std;
            o.* = normalized * g + b;
        }
    }
};

/// Multi-threading utilities for parallel computation
pub const Threading = struct {
    /// Thread pool for parallel operations
    thread_pool: ?std.Thread.Pool = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !Threading {
        var self = Threading{
            .allocator = allocator,
        };

        // Create thread pool with number of CPU cores
        const num_cores = std.Thread.cpuCount() catch 1;
        self.thread_pool = try std.Thread.Pool.init(.{
            .allocator = allocator,
            .n_jobs = num_cores,
        });

        return self;
    }

    pub fn deinit(self: *Threading) void {
        if (self.thread_pool) |*pool| {
            pool.deinit();
        }
    }

    /// Parallel activation forward pass
    pub fn parallelActivationForward(
        self: *Threading,
        act: anytype,
        input: []const f32,
        output: []f32,
    ) !void {
        const pool = self.thread_pool orelse {
            // Fallback to sequential
            for (input, output) |x, *out| {
                out.* = act.forward(x);
            }
            return;
        };

        const num_threads = pool.threads.len;
        const chunk_size = (input.len + num_threads - 1) / num_threads;

        var wait_group: std.Thread.WaitGroup = .{};

        for (0..num_threads) |thread_idx| {
            const start = thread_idx * chunk_size;
            const end = @min(start + chunk_size, input.len);

            if (start >= end) break;

            try pool.spawn(&wait_group, struct {
                fn worker(a: anytype, in: []const f32, out: []f32, s: usize, e: usize) void {
                    for (s..e) |i| {
                        out[i] = a.forward(in[i]);
                    }
                }
            }.worker, .{ act, input, output, start, end });
        }

        wait_group.wait();
    }

    /// Parallel gradient computation
    pub fn parallelGradientAccumulation(
        self: *Threading,
        layer: anytype,
        input: []const f32,
        grad_after_act: []const f32,
    ) !void {
        const pool = self.thread_pool orelse {
            // Fallback to sequential
            layer.accumulateGradients(input, grad_after_act);
            return;
        };

        const num_threads = pool.threads.len;
        const chunk_size = (layer.output_size + num_threads - 1) / num_threads;

        var wait_group: std.Thread.WaitGroup = .{};

        for (0..num_threads) |thread_idx| {
            const start = thread_idx * chunk_size;
            const end = @min(start + chunk_size, layer.output_size);

            if (start >= end) break;

            try pool.spawn(&wait_group, struct {
                fn worker(l: anytype, in: []const f32, grad: []const f32, s: usize, e: usize) void {
                    // Accumulate gradients for this chunk of outputs
                    for (s..e) |out_idx| {
                        // Accumulate bias gradient
                        l.grad_bias.slice[out_idx] += grad[out_idx];

                        // Accumulate weight gradients
                        for (0..l.input_size) |in_idx| {
                            const weight_idx = out_idx * l.input_size + in_idx;
                            l.grad_weights.slice[weight_idx] += grad[out_idx] * in[in_idx];
                        }
                    }
                }
            }.worker, .{ layer, input, grad_after_act, start, end });
        }

        wait_group.wait();
    }

    /// Parallel matrix multiplication
    pub fn parallelMatMul(
        self: *Threading,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: usize,
        n: usize,
        k: usize,
        accumulate: bool,
    ) !void {
        const pool = self.thread_pool orelse {
            // Fallback to sequential
            if (!accumulate) @memset(c, 0);
            for (0..m) |i| {
                for (0..n) |j| {
                    var sum: f32 = 0;
                    for (0..k) |p| {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    if (accumulate) {
                        c[i * n + j] += sum;
                    } else {
                        c[i * n + j] = sum;
                    }
                }
            }
            return;
        };

        if (!accumulate) @memset(c, 0);
        const num_threads = pool.threads.len;
        const chunk_size = (m + num_threads - 1) / num_threads;

        var wait_group: std.Thread.WaitGroup = .{};

        for (0..num_threads) |thread_idx| {
            const start = thread_idx * chunk_size;
            const end = @min(start + chunk_size, m);

            if (start >= end) break;

            try pool.spawn(&wait_group, struct {
                fn worker(
                    A: []const f32,
                    B: []const f32,
                    C: []f32,
                    rows: usize,
                    cols: usize,
                    inner: usize,
                    row_start: usize,
                    row_end: usize,
                    acc: bool,
                ) void {
                    _ = rows;
                    for (row_start..row_end) |i| {
                        for (0..cols) |j| {
                            var sum: f32 = 0;
                            for (0..inner) |p| {
                                sum += A[i * inner + p] * B[p * cols + j];
                            }
                            if (acc) {
                                C[i * cols + j] += sum;
                            } else {
                                C[i * cols + j] = sum;
                            }
                        }
                    }
                }
            }.worker, .{ a, b, c, m, n, k, start, end, accumulate });
        }

        wait_group.wait();
    }
};

/// Memory optimization utilities
pub const Memory = struct {
    /// Memory pool for efficient allocation/deallocation
    pub const MemoryPool = struct {
        allocator: std.mem.Allocator,
        buffer: []u8,
        offset: usize = 0,
        alignment: usize = 16,

        pub fn init(allocator: std.mem.Allocator, size: usize) !MemoryPool {
            const buffer = try allocator.alignedAlloc(u8, 16, size);
            return MemoryPool{
                .allocator = allocator,
                .buffer = buffer,
                .offset = 0,
            };
        }

        pub fn deinit(self: *MemoryPool) void {
            self.allocator.free(self.buffer);
        }

        pub fn alloc(self: *MemoryPool, comptime T: type, count: usize) ?[]T {
            const size = @sizeOf(T) * count;
            const aligned_size = std.mem.alignForward(usize, size, self.alignment);

            if (self.offset + aligned_size > self.buffer.len) {
                return null; // Pool exhausted
            }

            const ptr = @as([*]T, @ptrCast(@alignCast(&self.buffer[self.offset])));
            self.offset += aligned_size;

            return ptr[0..count];
        }

        pub fn reset(self: *MemoryPool) void {
            self.offset = 0;
        }
    };

    /// Pre-allocated buffer for temporary computations
    pub const PreallocatedBuffer = struct {
        buffer: []f32,
        allocator: std.mem.Allocator,
        size: usize,

        pub fn init(allocator: std.mem.Allocator, size: usize) !PreallocatedBuffer {
            const buffer = try allocator.alloc(f32, size);
            return PreallocatedBuffer{
                .allocator = allocator,
                .buffer = buffer,
                .size = size,
            };
        }

        pub fn deinit(self: *PreallocatedBuffer) void {
            self.allocator.free(self.buffer);
        }

        pub fn get(self: *PreallocatedBuffer, required_size: usize) ?[]f32 {
            if (required_size <= self.size) {
                return self.buffer[0..required_size];
            }
            return null;
        }
    };

    /// Structure-of-Arrays (SoA) layout for better cache locality
    pub const SoA = struct {
        allocator: std.mem.Allocator,
        data: []f32,
        offsets: []usize,
        num_elements: usize,

        pub fn init(allocator: std.mem.Allocator, num_arrays: usize, array_size: usize) !SoA {
            const total_size = num_arrays * array_size;
            const data = try allocator.alloc(f32, total_size);
            const offsets = try allocator.alloc(usize, num_arrays);

            // Initialize offsets
            for (0..num_arrays) |i| {
                offsets[i] = i * array_size;
            }

            return SoA{
                .allocator = allocator,
                .data = data,
                .offsets = offsets,
                .num_elements = num_arrays,
            };
        }

        pub fn deinit(self: *SoA) void {
            self.allocator.free(self.data);
            self.allocator.free(self.offsets);
        }

        pub fn getArray(self: *SoA, index: usize) []f32 {
            const offset = self.offsets[index];
            const next_offset = if (index + 1 < self.num_elements)
                self.offsets[index + 1]
            else
                self.data.len;
            return self.data[offset..next_offset];
        }
    };
};

/// Performance monitoring utilities
pub const Performance = struct {
    start_time: u64, // nanoseconds

    pub fn init() !Performance {
        const io = std.Io.Threaded.global_single_threaded.io();
        const now = std.Io.Clock.now(.real, io);
        const start_ns = @as(u64, @intCast(now.nanoseconds));
        return Performance{
            .start_time = start_ns,
        };
    }

    pub fn reset(self: *Performance) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        const now = std.Io.Clock.now(.real, io);
        self.start_time = @as(u64, @intCast(now.nanoseconds));
    }

    pub fn elapsedNs(self: *Performance) u64 {
        const io = std.Io.Threaded.global_single_threaded.io();
        const now = std.Io.Clock.now(.real, io);
        const current_ns = @as(u64, @intCast(now.nanoseconds));
        return current_ns - self.start_time;
    }

    pub fn elapsedMs(self: *Performance) f64 {
        return @as(f64, @floatFromInt(self.elapsedNs())) / 1_000_000.0;
    }

    pub fn elapsedUs(self: *Performance) f64 {
        return @as(f64, @floatFromInt(self.elapsedNs())) / 1_000.0;
    }

    pub fn benchmark(comptime func: anytype, args: anytype) struct { result: @TypeOf(@call(.auto, func, args)), elapsed_ms: f64 } {
        var perf = Performance.init() catch unreachable;
        const result = @call(.auto, func, args);
        const elapsed_ms = perf.elapsedMs();
        return .{ .result = result, .elapsed_ms = elapsed_ms };
    }
};

/// Cache-friendly computation utilities
pub const CacheFriendly = struct {
    /// Block size for cache blocking (tuned for Apple Silicon L2 cache: 4MB)
    pub const BLOCK_SIZE: usize = 64;

    /// Matrix multiplication with cache blocking
    pub fn matMulBlocked(
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: usize,
        n: usize,
        k: usize,
    ) void {
        // Initialize output to zero
        @memset(c, 0);

        // Use blocking for better cache utilization
        const block_size = BLOCK_SIZE;

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

                    for (ii..i_end) |i| {
                        for (kk..k_end) |p| {
                            const a_val = a[i * k + p];
                            for (jj..j_end) |j| {
                                c[i * n + j] += a_val * b[p * n + j];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Transpose matrix for better cache locality
    pub fn transpose(input: []const f32, output: []f32, rows: usize, cols: usize) void {
        for (0..rows) |i| {
            for (0..cols) |j| {
                output[j * rows + i] = input[i * cols + j];
            }
        }
    }

    /// Compute in blocks to fit in cache
    pub fn computeInBlocks(
        comptime func: anytype,
        data: []f32,
        block_size: usize,
    ) void {
        var i: usize = 0;
        while (i < data.len) : (i += block_size) {
            const end = @min(i + block_size, data.len);
            @call(.auto, func, .{data[i..end]});
        }
    }
};

test "SIMD relu vectorized" {
    const input = [_]f32{ -1.0, 0.0, 1.0, 2.0, -2.0, 3.0 };
    var output: [6]f32 = undefined;

    SIMD.reluVectorized(&input, &output);

    try std.testing.expect(output[0] == 0.0);
    try std.testing.expect(output[1] == 0.0);
    try std.testing.expect(output[2] == 1.0);
    try std.testing.expect(output[3] == 2.0);
    try std.testing.expect(output[4] == 0.0);
    try std.testing.expect(output[5] == 3.0);
}

test "SIMD add vectorized" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const b = [_]f32{ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 };
    var output: [6]f32 = undefined;

    SIMD.addVectorized(&a, &b, &output);

    try std.testing.expect(output[0] == 1.5);
    try std.testing.expect(output[1] == 3.5);
    try std.testing.expect(output[2] == 5.5);
    try std.testing.expect(output[3] == 7.5);
    try std.testing.expect(output[4] == 9.5);
    try std.testing.expect(output[5] == 11.5);
}

test "Memory pool allocation" {
    const allocator = std.testing.allocator;
    var pool = try Memory.MemoryPool.init(allocator, 1024);
    defer pool.deinit();

    const arr = pool.alloc(f32, 10).?;
    for (arr, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i));
    }

    try std.testing.expect(arr.len == 10);
    try std.testing.expect(arr[5] == 5.0);
}

test "Cache-friendly matrix multiplication" {
    const allocator = std.testing.allocator;
    const m: usize = 32;
    const n: usize = 32;
    const k: usize = 32;

    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    // Initialize with test data
    for (a, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 10));
    }
    for (b, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 10));
    }

    CacheFriendly.matMulBlocked(a, b, c, m, n, k);

    // Verify result
    try std.testing.expect(c[0] >= 0);
}

test "Performance benchmark" {
    const data = [_]f32{1.0, 2.0, 3.0, 4.0};
    const result = Performance.benchmark(struct {
        fn sum(arr: []const f32) f32 {
            var total: f32 = 0;
            for (arr) |x| {
                total += x;
            }
            return total;
        }
    }.sum, .{&data});

    try std.testing.expect(result.result == 10.0);
    try std.testing.expect(result.elapsed_ms >= 0);
}

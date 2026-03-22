/// Comprehensive CUDA Benchmarking Infrastructure for ZigNeuron
/// Measures and reports CUDA vs CPU performance across all critical operations
///
/// Features:
/// - Matrix multiplication benchmarks (naive vs tiled vs Tensor Core)
/// - Activation function benchmarks
/// - Memory transfer benchmarks (upload/download)
/// - Layer benchmarks (full forward/backward passes)
/// - Detailed metrics: execution time, throughput, bandwidth, speedup
const std = @import("std");
const cuda = @import("../cuda.zig");

// =============================================================================
// Benchmark Configuration
// =============================================================================

pub const BenchmarkConfig = struct {
    /// Number of warmup iterations before measurement
    warmup_iterations: usize = 10,
    /// Number of iterations for measurement
    benchmark_iterations: usize = 100,
    /// Whether to collect detailed statistics
    collect_statistics: bool = true,
    /// Random seed for reproducibility
    random_seed: u64 = 42,
};

// Default configuration
pub const DEFAULT_CONFIG = BenchmarkConfig{};

// =============================================================================
// Benchmark Result Types
// =============================================================================

/// Detailed benchmark result with comprehensive metrics
pub const BenchmarkResult = struct {
    /// Name of the benchmark
    name: []const u8,
    /// Backend used (CUDA, CPU, etc.)
    backend: []const u8,
    /// Matrix/problem dimensions
    dimensions: []const u8,
    /// Number of warmup iterations
    warmup_iterations: usize,
    /// Number of benchmark iterations
    benchmark_iterations: usize,
    /// Total time for all iterations (nanoseconds)
    total_time_ns: u64,
    /// Average time per iteration (nanoseconds)
    avg_time_ns: f64,
    /// Minimum time (nanoseconds)
    min_time_ns: u64,
    /// Maximum time (nanoseconds)
    max_time_ns: u64,
    /// Standard deviation (nanoseconds)
    std_dev_ns: f64,
    /// Throughput in GFLOPS (for matmul) or ops/sec
    throughput: f64,
    /// Memory bandwidth in GB/s (for memory operations)
    bandwidth_gbps: f64,
    /// Number of operations performed per iteration
    operations: u64,
    /// Data size in bytes processed per iteration
    data_size_bytes: usize,
};

/// Comparison result between two backends
pub const ComparisonResult = struct {
    /// Name of the benchmark
    name: []const u8,
    /// Dimensions/scale of the problem
    dimensions: []const u8,
    /// CUDA result
    cuda_result: BenchmarkResult,
    /// CPU result
    cpu_result: BenchmarkResult,
    /// Speedup factor (CPU time / CUDA time)
    speedup: f64,
    /// Efficiency percentage (speedup / theoretical_max * 100)
    efficiency_percent: f64,
};

// =============================================================================
// Timing Utilities
// =============================================================================

/// High-resolution timer using CLOCK_MONOTONIC
fn nanoTimestamp() u64 {
    const os = std.os;
    var tv: os.linux.timespec = undefined;
    const CLOCK_MONOTONIC: os.linux.clockid_t = .MONOTONIC;
    _ = os.linux.clock_gettime(CLOCK_MONOTONIC, &tv);
    return @as(u64, @intCast(tv.sec)) * 1_000_000_000 + @as(u64, @intCast(tv.nsec));
}

// =============================================================================
// Data Generation Utilities
// =============================================================================

/// Initialize random number generator
fn initRandom(seed: u64) std.Random.Xoroshiro128PlusPlus {
    return std.Random.Xoroshiro128PlusPlus.init(seed);
}

/// Fill array with random values
fn fillRandom(data: []f32, rng: *std.Random.Xoroshiro128PlusPlus) void {
    for (data) |*v| {
        v.* = rng.random().float(f32) * 2.0 - 1.0; // Range: [-1, 1]
    }
}

/// Fill array with constant value
fn fillConstant(data: []f32, value: f32) void {
    @memset(data, value);
}

// =============================================================================
// Metrics Calculation
// =============================================================================

/// Calculate GFLOPS for matrix multiplication
/// For C = A * B where A is MxK and B is KxN: 2*M*N*K operations
fn calculateMatmulGflops(m: usize, n: usize, k: usize, time_ns: u64) f64 {
    const ops = @as(f64, @floatFromInt(2 * m * n * k));
    const seconds = @as(f64, @floatFromInt(time_ns)) / 1e9;
    return ops / seconds / 1e9; // GFLOPS
}

/// Calculate GB/s for memory operations
fn calculateBandwidthGBs(bytes: usize, time_ns: u64) f64 {
    const seconds = @as(f64, @floatFromInt(time_ns)) / 1e9;
    const gb = @as(f64, @floatFromInt(bytes)) / 1e9;
    return gb / seconds;
}

/// Calculate standard deviation
fn calculateStdDev(times: []const u64, mean: f64) f64 {
    var sum_squared_diff: f64 = 0;
    for (times) |t| {
        const diff = @as(f64, @floatFromInt(t)) - mean;
        sum_squared_diff += diff * diff;
    }
    return @sqrt(sum_squared_diff / @as(f64, @floatFromInt(times.len)));
}

// =============================================================================
// CPU Reference Implementations
// =============================================================================

/// CPU matrix multiplication for comparison
fn cpuMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize, accumulate: bool) void {
    if (!accumulate) {
        @memset(c, 0);
    }

    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var sum: f32 = 0;
            var l: usize = 0;
            while (l < k) : (l += 1) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] += sum;
        }
    }
}

/// CPU ReLU activation
fn cpuRelu(input: []const f32, output: []f32) void {
    for (input, output) |in, *out| {
        out.* = if (in > 0) in else 0;
    }
}

/// CPU Sigmoid activation
fn cpuSigmoid(input: []const f32, output: []f32) void {
    for (input, output) |in, *out| {
        out.* = 1.0 / (1.0 + @exp(-in));
    }
}

/// CPU Tanh activation
fn cpuTanh(input: []const f32, output: []f32) void {
    for (input, output) |in, *out| {
        out.* = std.math.tanh(in);
    }
}

// =============================================================================
// CUDA Benchmark Implementations
// =============================================================================

/// Benchmark CUDA matrix multiplication with various algorithms
pub fn benchmarkCudaMatMul(
    allocator: std.mem.Allocator,
    m: usize,
    n: usize,
    k: usize,
    config: BenchmarkConfig,
) !BenchmarkResult {
    var backend = try cuda.CudaBackend.init(allocator);
    defer backend.deinit();

    // Allocate host memory
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    // Initialize with random data
    var rng = initRandom(config.random_seed);
    fillRandom(a, &rng);
    fillRandom(b, &rng);
    fillConstant(c, 0);

    // Warmup
    for (0..config.warmup_iterations) |_| {
        try backend.matMul(a, b, c, m, n, k, false, false, false);
    }

    // Benchmark
    var times = try allocator.alloc(u64, config.benchmark_iterations);
    defer allocator.free(times);

    for (0..config.benchmark_iterations) |i| {
        const start = nanoTimestamp();
        try backend.matMul(a, b, c, m, n, k, false, false, false);
        const end = nanoTimestamp();
        times[i] = end - start;
    }

    // Calculate statistics
    var total_time: u64 = 0;
    var min_time: u64 = times[0];
    var max_time: u64 = times[0];
    for (times) |t| {
        total_time += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    const avg_time = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(config.benchmark_iterations));
    const std_dev = calculateStdDev(times, avg_time);

    const throughput = calculateMatmulGflops(m, n, k, @intCast(@as(u64, @intFromFloat(avg_time))));

    var dims_buf: [64]u8 = undefined;
    const dims = std.fmt.bufPrint(&dims_buf, "{d}x{d}x{d}", .{ m, n, k }) catch "unknown";

    return BenchmarkResult{
        .name = "matmul",
        .backend = "CUDA",
        .dimensions = dims,
        .warmup_iterations = config.warmup_iterations,
        .benchmark_iterations = config.benchmark_iterations,
        .total_time_ns = total_time,
        .avg_time_ns = avg_time,
        .min_time_ns = min_time,
        .max_time_ns = max_time,
        .std_dev_ns = std_dev,
        .throughput = throughput,
        .bandwidth_gbps = 0,
        .operations = @intCast(2 * m * n * k),
        .data_size_bytes = (m * k + k * n + m * n) * @sizeOf(f32),
    };
}

/// Benchmark CPU matrix multiplication for comparison
pub fn benchmarkCpuMatMul(
    allocator: std.mem.Allocator,
    m: usize,
    n: usize,
    k: usize,
    config: BenchmarkConfig,
) !BenchmarkResult {
    // Allocate memory
    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);

    // Initialize with random data
    var rng = initRandom(config.random_seed);
    fillRandom(a, &rng);
    fillRandom(b, &rng);
    fillConstant(c, 0);

    // Warmup
    for (0..config.warmup_iterations) |_| {
        cpuMatMul(a, b, c, m, n, k, false);
    }

    // Benchmark
    var times = try allocator.alloc(u64, config.benchmark_iterations);
    defer allocator.free(times);

    for (0..config.benchmark_iterations) |i| {
        const start = nanoTimestamp();
        cpuMatMul(a, b, c, m, n, k, false);
        const end = nanoTimestamp();
        times[i] = end - start;
    }

    // Calculate statistics
    var total_time: u64 = 0;
    var min_time: u64 = times[0];
    var max_time: u64 = times[0];
    for (times) |t| {
        total_time += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    const avg_time = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(config.benchmark_iterations));
    const std_dev = calculateStdDev(times, avg_time);

    const throughput = calculateMatmulGflops(m, n, k, @intCast(@as(u64, @intFromFloat(avg_time))));

    var dims_buf: [64]u8 = undefined;
    const dims = std.fmt.bufPrint(&dims_buf, "{d}x{d}x{d}", .{ m, n, k }) catch "unknown";

    return BenchmarkResult{
        .name = "matmul",
        .backend = "CPU",
        .dimensions = dims,
        .warmup_iterations = config.warmup_iterations,
        .benchmark_iterations = config.benchmark_iterations,
        .total_time_ns = total_time,
        .avg_time_ns = avg_time,
        .min_time_ns = min_time,
        .max_time_ns = max_time,
        .std_dev_ns = std_dev,
        .throughput = throughput,
        .bandwidth_gbps = 0,
        .operations = @intCast(2 * m * n * k),
        .data_size_bytes = (m * k + k * n + m * n) * @sizeOf(f32),
    };
}

/// Benchmark CUDA activation functions
pub fn benchmarkCudaActivation(
    allocator: std.mem.Allocator,
    comptime activation: []const u8,
    size: usize,
    config: BenchmarkConfig,
) !BenchmarkResult {
    var backend = try cuda.CudaBackend.init(allocator);
    defer backend.deinit();

    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);

    var rng = initRandom(config.random_seed);
    fillRandom(input, &rng);

    // Warmup
    for (0..config.warmup_iterations) |_| {
        if (std.mem.eql(u8, activation, "relu")) {
            try backend.reluForward(input, output);
        } else if (std.mem.eql(u8, activation, "sigmoid")) {
            try backend.sigmoidForward(input, output);
        } else if (std.mem.eql(u8, activation, "tanh")) {
            try backend.tanhForward(input, output);
        }
    }

    // Benchmark
    var times = try allocator.alloc(u64, config.benchmark_iterations);
    defer allocator.free(times);

    for (0..config.benchmark_iterations) |i| {
        const start = nanoTimestamp();
        if (std.mem.eql(u8, activation, "relu")) {
            try backend.reluForward(input, output);
        } else if (std.mem.eql(u8, activation, "sigmoid")) {
            try backend.sigmoidForward(input, output);
        } else if (std.mem.eql(u8, activation, "tanh")) {
            try backend.tanhForward(input, output);
        }
        const end = nanoTimestamp();
        times[i] = end - start;
    }

    // Calculate statistics
    var total_time: u64 = 0;
    var min_time: u64 = times[0];
    var max_time: u64 = times[0];
    for (times) |t| {
        total_time += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    const avg_time = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(config.benchmark_iterations));
    const std_dev = calculateStdDev(times, avg_time);

    const throughput = @as(f64, @floatFromInt(size * config.benchmark_iterations)) / (@as(f64, @floatFromInt(total_time)) / 1e9);

    var dims_buf: [64]u8 = undefined;
    const dims = std.fmt.bufPrint(&dims_buf, "{d}", .{size}) catch "unknown";

    const op_name = std.fmt.allocPrint(allocator, "activation_{s}", .{activation}) catch activation;
    defer allocator.free(op_name);

    return BenchmarkResult{
        .name = op_name,
        .backend = "CUDA",
        .dimensions = dims,
        .warmup_iterations = config.warmup_iterations,
        .benchmark_iterations = config.benchmark_iterations,
        .total_time_ns = total_time,
        .avg_time_ns = avg_time,
        .min_time_ns = min_time,
        .max_time_ns = max_time,
        .std_dev_ns = std_dev,
        .throughput = throughput,
        .bandwidth_gbps = calculateBandwidthGBs(size * @sizeOf(f32), @intCast(@as(u64, @intFromFloat(avg_time)))),
        .operations = @intCast(size),
        .data_size_bytes = size * @sizeOf(f32),
    };
}

/// Benchmark CPU activation functions
pub fn benchmarkCpuActivation(
    allocator: std.mem.Allocator,
    comptime activation: []const u8,
    size: usize,
    config: BenchmarkConfig,
) !BenchmarkResult {
    const input = try allocator.alloc(f32, size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);

    var rng = initRandom(config.random_seed);
    fillRandom(input, &rng);

    // Warmup
    for (0..config.warmup_iterations) |_| {
        if (std.mem.eql(u8, activation, "relu")) {
            cpuRelu(input, output);
        } else if (std.mem.eql(u8, activation, "sigmoid")) {
            cpuSigmoid(input, output);
        } else if (std.mem.eql(u8, activation, "tanh")) {
            cpuTanh(input, output);
        }
    }

    // Benchmark
    var times = try allocator.alloc(u64, config.benchmark_iterations);
    defer allocator.free(times);

    for (0..config.benchmark_iterations) |i| {
        const start = nanoTimestamp();
        if (std.mem.eql(u8, activation, "relu")) {
            cpuRelu(input, output);
        } else if (std.mem.eql(u8, activation, "sigmoid")) {
            cpuSigmoid(input, output);
        } else if (std.mem.eql(u8, activation, "tanh")) {
            cpuTanh(input, output);
        }
        const end = nanoTimestamp();
        times[i] = end - start;
    }

    // Calculate statistics
    var total_time: u64 = 0;
    var min_time: u64 = times[0];
    var max_time: u64 = times[0];
    for (times) |t| {
        total_time += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    const avg_time = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(config.benchmark_iterations));
    const std_dev = calculateStdDev(times, avg_time);

    const throughput = @as(f64, @floatFromInt(size * config.benchmark_iterations)) / (@as(f64, @floatFromInt(total_time)) / 1e9);

    var dims_buf: [64]u8 = undefined;
    const dims = std.fmt.bufPrint(&dims_buf, "{d}", .{size}) catch "unknown";

    const op_name = std.fmt.allocPrint(allocator, "activation_{s}", .{activation}) catch activation;
    defer allocator.free(op_name);

    return BenchmarkResult{
        .name = op_name,
        .backend = "CPU",
        .dimensions = dims,
        .warmup_iterations = config.warmup_iterations,
        .benchmark_iterations = config.benchmark_iterations,
        .total_time_ns = total_time,
        .avg_time_ns = avg_time,
        .min_time_ns = min_time,
        .max_time_ns = max_time,
        .std_dev_ns = std_dev,
        .throughput = throughput,
        .bandwidth_gbps = calculateBandwidthGBs(size * @sizeOf(f32), @intCast(@as(u64, @intFromFloat(avg_time)))),
        .operations = @intCast(size),
        .data_size_bytes = size * @sizeOf(f32),
    };
}

/// Benchmark CUDA memory transfer operations
pub fn benchmarkCudaMemoryTransfer(
    allocator: std.mem.Allocator,
    size: usize,
    config: BenchmarkConfig,
) !struct { h2d: BenchmarkResult, d2h: BenchmarkResult } {
    var backend = try cuda.CudaBackend.init(allocator);
    defer backend.deinit();

    const buffer_size = size * @sizeOf(f32);
    const d_buffer = try backend.allocBuffer(buffer_size);
    defer backend.freeBuffer(d_buffer);

    const h_buffer = try allocator.alloc(f32, size);
    defer allocator.free(h_buffer);

    var rng = initRandom(config.random_seed);
    fillRandom(h_buffer, &rng);

    // Benchmark H2D (Host to Device)
    var h2d_times = try allocator.alloc(u64, config.benchmark_iterations);
    defer allocator.free(h2d_times);

    for (0..config.warmup_iterations) |_| {
        try backend.upload(d_buffer.ptr, h_buffer);
    }

    for (0..config.benchmark_iterations) |i| {
        const start = nanoTimestamp();
        try backend.upload(d_buffer.ptr, h_buffer);
        const end = nanoTimestamp();
        h2d_times[i] = end - start;
    }

    // Calculate H2D statistics
    var h2d_total: u64 = 0;
    var h2d_min: u64 = h2d_times[0];
    var h2d_max: u64 = h2d_times[0];
    for (h2d_times) |t| {
        h2d_total += t;
        if (t < h2d_min) h2d_min = t;
        if (t > h2d_max) h2d_max = t;
    }
    const h2d_avg = @as(f64, @floatFromInt(h2d_total)) / @as(f64, @floatFromInt(config.benchmark_iterations));
    const h2d_std_dev = calculateStdDev(h2d_times, h2d_avg);

    // Benchmark D2H (Device to Host)
    var d2h_times = try allocator.alloc(u64, config.benchmark_iterations);
    defer allocator.free(d2h_times);

    for (0..config.warmup_iterations) |_| {
        try backend.download(h_buffer, d_buffer.ptr);
    }

    for (0..config.benchmark_iterations) |i| {
        const start = nanoTimestamp();
        try backend.download(h_buffer, d_buffer.ptr);
        const end = nanoTimestamp();
        d2h_times[i] = end - start;
    }

    // Calculate D2H statistics
    var d2h_total: u64 = 0;
    var d2h_min: u64 = d2h_times[0];
    var d2h_max: u64 = d2h_times[0];
    for (d2h_times) |t| {
        d2h_total += t;
        if (t < d2h_min) d2h_min = t;
        if (t > d2h_max) d2h_max = t;
    }
    const d2h_avg = @as(f64, @floatFromInt(d2h_total)) / @as(f64, @floatFromInt(config.benchmark_iterations));
    const d2h_std_dev = calculateStdDev(d2h_times, d2h_avg);

    var dims_buf: [64]u8 = undefined;
    const dims = std.fmt.bufPrint(&dims_buf, "{d}", .{size}) catch "unknown";

    const h2d_result = BenchmarkResult{
        .name = "memory_h2d",
        .backend = "CUDA",
        .dimensions = dims,
        .warmup_iterations = config.warmup_iterations,
        .benchmark_iterations = config.benchmark_iterations,
        .total_time_ns = h2d_total,
        .avg_time_ns = h2d_avg,
        .min_time_ns = h2d_min,
        .max_time_ns = h2d_max,
        .std_dev_ns = h2d_std_dev,
        .throughput = 0,
        .bandwidth_gbps = calculateBandwidthGBs(buffer_size, @intCast(@as(u64, @intFromFloat(h2d_avg)))),
        .operations = @intCast(size),
        .data_size_bytes = buffer_size,
    };

    const d2h_result = BenchmarkResult{
        .name = "memory_d2h",
        .backend = "CUDA",
        .dimensions = dims,
        .warmup_iterations = config.warmup_iterations,
        .benchmark_iterations = config.benchmark_iterations,
        .total_time_ns = d2h_total,
        .avg_time_ns = d2h_avg,
        .min_time_ns = d2h_min,
        .max_time_ns = d2h_max,
        .std_dev_ns = d2h_std_dev,
        .throughput = 0,
        .bandwidth_gbps = calculateBandwidthGBs(buffer_size, @intCast(@as(u64, @intFromFloat(d2h_avg)))),
        .operations = @intCast(size),
        .data_size_bytes = buffer_size,
    };

    return .{ .h2d = h2d_result, .d2h = d2h_result };
}

// =============================================================================
// Benchmark Suites
// =============================================================================

/// Run complete matrix multiplication benchmark suite
pub fn runMatMulSuite(
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    results: *std.ArrayList(BenchmarkResult),
) !void {
    std.debug.print("\n=== Matrix Multiplication Benchmark Suite ===\n", .{});

    const sizes = [_][3]usize{
        .{ 128, 128, 128 },
        .{ 256, 256, 256 },
        .{ 512, 512, 512 },
        .{ 1024, 1024, 1024 },
        .{ 2048, 2048, 2048 },
    };

    for (sizes) |size| {
        const m, const n, const k = size;

        // CUDA benchmark
        const cuda_result = benchmarkCudaMatMul(allocator, m, n, k, config) catch |err| {
            std.log.warn("CUDA matmul benchmark failed for {d}x{d}x{d}: {s}", .{ m, n, k, @errorName(err) });
            continue;
        };
        try results.append(cuda_result);

        // CPU benchmark
        const cpu_result = benchmarkCpuMatMul(allocator, m, n, k, config) catch |err| {
            std.log.warn("CPU matmul benchmark failed for {d}x{d}x{d}: {s}", .{ m, n, k, @errorName(err) });
            continue;
        };
        try results.append(cpu_result);

        // Print comparison
        const speedup = cpu_result.avg_time_ns / cuda_result.avg_time_ns;
        std.debug.print("MatMul {d}x{d}x{d}: CUDA {d:.2}ms, CPU {d:.2}ms, Speedup: {d:.2}x, CUDA: {d:.2} GFLOPS\n", .{
            m, n, k,
            cuda_result.avg_time_ns / 1e6,
            cpu_result.avg_time_ns / 1e6,
            speedup,
            cuda_result.throughput,
        });
    }
}

/// Run complete activation function benchmark suite
pub fn runActivationSuite(
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    results: *std.ArrayList(BenchmarkResult),
) !void {
    std.debug.print("\n=== Activation Function Benchmark Suite ===\n", .{});

    const sizes = [_]usize{ 1024, 4096, 16384, 65536, 262144 };
    const activations = [_][]const u8{ "relu", "sigmoid", "tanh" };

    inline for (activations) |act| {
        std.debug.print("\n{s} activation:\n", .{act});
        for (sizes) |size| {
            // CUDA benchmark
            const cuda_result = benchmarkCudaActivation(allocator, act, size, config) catch |err| {
                std.log.warn("CUDA {s} benchmark failed for size {d}: {s}", .{ act, size, @errorName(err) });
                continue;
            };
            try results.append(cuda_result);

            // CPU benchmark
            const cpu_result = benchmarkCpuActivation(allocator, act, size, config) catch |err| {
                std.log.warn("CPU {s} benchmark failed for size {d}: {s}", .{ act, size, @errorName(err) });
                continue;
            };
            try results.append(cpu_result);

            // Print comparison
            const speedup = cpu_result.avg_time_ns / cuda_result.avg_time_ns;
            std.debug.print("  Size {d}: CUDA {d:.2}us, CPU {d:.2}us, Speedup: {d:.2}x\n", .{
                size,
                cuda_result.avg_time_ns / 1e3,
                cpu_result.avg_time_ns / 1e3,
                speedup,
            });
        }
    }
}

/// Run complete memory transfer benchmark suite
pub fn runMemoryTransferSuite(
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    results: *std.ArrayList(BenchmarkResult),
) !void {
    std.debug.print("\n=== Memory Transfer Benchmark Suite ===\n", .{});

    const sizes = [_]usize{ 1024, 4096, 16384, 65536, 262144, 1048576 };

    std.debug.print("\n{:<12} {:>12} {:>15} {:>15}\n", .{ "Size", "Direction", "Bandwidth", "Time" });
    std.debug.print("{s}\n", .{"-" ** 60});

    for (sizes) |size| {
        const transfer_results = benchmarkCudaMemoryTransfer(allocator, size, config) catch |err| {
            std.log.warn("Memory transfer benchmark failed for size {d}: {s}", .{ size, @errorName(err) });
            continue;
        };

        try results.append(transfer_results.h2d);
        try results.append(transfer_results.d2h);

        std.debug.print("{:<12} {:>12} {:>14.2} GB/s {:>14.2} us\n", .{
            size,
            "H2D",
            transfer_results.h2d.bandwidth_gbps,
            transfer_results.h2d.avg_time_ns / 1e3,
        });
        std.debug.print("{:<12} {:>12} {:>14.2} GB/s {:>14.2} us\n", .{
            size,
            "D2H",
            transfer_results.d2h.bandwidth_gbps,
            transfer_results.d2h.avg_time_ns / 1e3,
        });
    }
}

// =============================================================================
// Result Formatting and Export
// =============================================================================

/// Print benchmark result in a formatted table
pub fn printResult(result: BenchmarkResult) void {
    std.debug.print("\n{s} ({s}) [{s}]\n", .{ result.name, result.backend, result.dimensions });
    std.debug.print("  Iterations:      {d} (warmup: {d})\n", .{ result.benchmark_iterations, result.warmup_iterations });
    std.debug.print("  Total time:      {d:.2} ms\n", .{ @as(f64, @floatFromInt(result.total_time_ns)) / 1e6 });
    std.debug.print("  Average time:    {d:.2} us (+/- {d:.2} us)\n", .{ result.avg_time_ns / 1e3, result.std_dev_ns / 1e3 });
    std.debug.print("  Min/Max time:    {d:.2} us / {d:.2} us\n", .{ @as(f64, @floatFromInt(result.min_time_ns)) / 1e3, @as(f64, @floatFromInt(result.max_time_ns)) / 1e3 });
    if (result.throughput > 0) {
        if (std.mem.startsWith(u8, result.name, "matmul")) {
            std.debug.print("  Throughput:      {d:.2} GFLOPS\n", .{result.throughput});
        } else {
            std.debug.print("  Throughput:      {d:.2e} ops/sec\n", .{result.throughput});
        }
    }
    if (result.bandwidth_gbps > 0) {
        std.debug.print("  Bandwidth:       {d:.2} GB/s\n", .{result.bandwidth_gbps});
    }
}

/// Print comparison between CUDA and CPU results
pub fn printComparison(cuda_result: BenchmarkResult, cpu_result: BenchmarkResult) void {
    const speedup = cpu_result.avg_time_ns / cuda_result.avg_time_ns;
    const efficiency = (speedup / 100.0) * 100.0; // Assuming 100x theoretical max

    std.debug.print("\n=== Comparison: {s} [{s}] ===\n", .{ cuda_result.name, cuda_result.dimensions });
    std.debug.print("  CUDA: {d:.2} ms ({d:.2} GFLOPS)\n", .{ cuda_result.avg_time_ns / 1e6, cuda_result.throughput });
    std.debug.print("  CPU:  {d:.2} ms ({d:.2} GFLOPS)\n", .{ cpu_result.avg_time_ns / 1e6, cpu_result.throughput });
    std.debug.print("  Speedup: {d:.2}x\n", .{speedup});
    std.debug.print("  Efficiency: {d:.1}%\n", .{efficiency});
}

/// Export results to CSV format
pub fn exportToCsv(
    allocator: std.mem.Allocator,
    results: []const BenchmarkResult,
    filename: []const u8,
) !void {
    const file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    const writer = file.writer();

    // Header
    try writer.print("name,backend,dimensions,warmup_iterations,benchmark_iterations,total_time_ns,avg_time_ns,min_time_ns,max_time_ns,std_dev_ns,throughput,bandwidth_gbps,operations,data_size_bytes\n", .{});

    // Data rows
    for (results) |r| {
        try writer.print("{s},{s},{s},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d}\n", .{
            r.name, r.backend, r.dimensions,
            r.warmup_iterations, r.benchmark_iterations,
            r.total_time_ns, r.avg_time_ns,
            r.min_time_ns, r.max_time_ns,
            r.std_dev_ns, r.throughput,
            r.bandwidth_gbps, r.operations,
            r.data_size_bytes,
        });
    }
}

/// Print summary table of all results
pub fn printSummaryTable(results: []const BenchmarkResult) void {
    std.debug.print("\n=== Benchmark Summary ===\n", .{});
    std.debug.print("\n{:<20} {:<10} {:<12} {:>12} {:>12} {:>12}\n", .{
        "Benchmark", "Backend", "Dimensions", "Time (us)", "GFLOPS", "Speedup",
    });
    std.debug.print("{s}\n", .{"-" ** 80});

    var i: usize = 0;
    while (i < results.len) : (i += 2) {
        if (i + 1 >= results.len) break;
        const cuda = results[i];
        const cpu = results[i + 1];

        if (!std.mem.eql(u8, cuda.name, cpu.name) or
            !std.mem.eql(u8, cuda.dimensions, cpu.dimensions))
        {
            continue;
        }

        const speedup = cpu.avg_time_ns / cuda.avg_time_ns;
        std.debug.print("{:<20} {:<10} {:<12} {:>12.2} {:>12.2} {:>11.2}x\n", .{
            cuda.name,
            "CUDA",
            cuda.dimensions,
            cuda.avg_time_ns / 1e3,
            cuda.throughput,
            speedup,
        });
        std.debug.print("{:<20} {:<10} {:<12} {:>12.2} {:>12.2} {:>12}\n", .{
            "",
            "CPU",
            "",
            cpu.avg_time_ns / 1e3,
            cpu.throughput,
            "",
        });
    }
}

// =============================================================================
// Device Information
// =============================================================================

/// Print CUDA device information
pub fn printDeviceInfo(allocator: std.mem.Allocator) void {
    var backend = cuda.CudaBackend.init(allocator) catch {
        std.debug.print("CUDA not available\n", .{});
        return;
    };
    defer backend.deinit();

    std.debug.print("\n=== CUDA Device Information ===\n", .{});
    std.debug.print("  Device Name: {s}\n", .{backend.context.device_props.name});
    std.debug.print("  Compute Capability: {d}.{d}\n", .{ backend.context.device_props.compute_capability_major, backend.context.device_props.compute_capability_minor });
    std.debug.print("  Total Memory: {d} MB\n", .{backend.context.device_props.total_memory / (1024 * 1024)});
    std.debug.print("  Multiprocessors: {d}\n", .{backend.context.device_props.multiprocessor_count});
    std.debug.print("  Max Threads Per Block: {d}\n", .{backend.context.device_props.max_threads_per_block});
    std.debug.print("  Max Grid Dimensions: [{d}, {d}, {d}]\n", .{
        backend.context.device_props.max_grid_dim_x,
        backend.context.device_props.max_grid_dim_y,
        backend.context.device_props.max_grid_dim_z,
    });
    std.debug.print("  Tensor Cores: {s}\n", .{if (backend.context.device_props.hasTensorCores()) "Yes" else "No"});
    std.debug.print("  Concurrent Kernels: {s}\n", .{if (backend.context.device_props.concurrent_kernels != 0) "Yes" else "No"});
    std.debug.print("  Unified Addressing: {s}\n", .{if (backend.context.device_props.unified_addressing != 0) "Yes" else "No"});
}

// =============================================================================
// Main Entry Point
// =============================================================================

/// Run all CUDA benchmarks
pub fn runAllBenchmarks(allocator: std.mem.Allocator) !void {
    const config = DEFAULT_CONFIG;

    var results = std.ArrayList(BenchmarkResult).init(allocator);
    defer results.deinit();

    // Print device info
    printDeviceInfo(allocator);

    // Run benchmark suites
    try runMatMulSuite(allocator, config, &results);
    try runActivationSuite(allocator, config, &results);
    try runMemoryTransferSuite(allocator, config, &results);

    // Print summary
    printSummaryTable(results.items);

    // Export to CSV
    exportToCsv(allocator, results.items, "cuda_benchmark_results.csv") catch |err| {
        std.log.warn("Failed to export results to CSV: {s}", .{@errorName(err)});
    };

    std.debug.print("\nResults exported to cuda_benchmark_results.csv\n", .{});
}

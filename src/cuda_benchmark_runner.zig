/// CUDA Benchmark Runner - Command-line interface for CUDA benchmarks
const std = @import("std");
const cuda_benchmark = @import("benchmark/cuda_benchmark.zig");

const USAGE =
    \\Usage: cuda_benchmark [options] [benchmark_name]
    \\
    \Options:
    \  -h, --help              Show this help message
    \  -l, --list              List available benchmarks
    \  -c, --compare           Compare CUDA vs CPU (default: true)
    \  -o, --output <file>     Output results to CSV file
    \  -w, --warmup <n>        Number of warmup iterations (default: 10)
    \  -i, --iterations <n>    Number of benchmark iterations (default: 100)
    \  -s, --size <sizes>      Comma-separated list of sizes (e.g., 128,256,512)
    \\
    \Benchmarks:
    \  matmul                  Matrix multiplication benchmarks
    \  activation              Activation function benchmarks
    \  memory                  Memory transfer benchmarks
    \  all                     Run all benchmarks (default)
    \\
    \Examples:
    \  cuda_benchmark                      Run all benchmarks
    \  cuda_benchmark matmul               Run matrix multiplication only
    \  cuda_benchmark -o results.csv       Run all and save to CSV
    \  cuda_benchmark -s 128,256 matmul      Run matmul with specific sizes
    \\
;

const BenchmarkType = enum {
    matmul,
    activation,
    memory,
    all,
};

const RunOptions = struct {
    benchmark: BenchmarkType = .all,
    compare: bool = true,
    output_file: ?[]const u8 = null,
    warmup_iterations: usize = 10,
    benchmark_iterations: usize = 100,
    sizes: []const usize = &[_]usize{ 128, 256, 512, 1024, 2048 },
};

fn printUsage() void {
    std.debug.print("{s}\n", .{USAGE});
}

fn parseSizeList(arg: []const u8, allocator: std.mem.Allocator) ![]usize {
    var sizes = std.ArrayList(usize).init(allocator);
    defer sizes.deinit();

    var it = std.mem.split(u8, arg, ",");
    while (it.next()) |s| {
        const size = try std.fmt.parseInt(usize, std.mem.trim(u8, s, " "), 10);
        try sizes.append(size);
    }

    const result = try allocator.alloc(usize, sizes.items.len);
    @memcpy(result, sizes.items);
    return result;
}

fn parseArgs(args: []const []const u8, allocator: std.mem.Allocator) !RunOptions {
    var options = RunOptions{};
    var i: usize = 1;

    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "-l") or std.mem.eql(u8, arg, "--list")) {
            std.debug.print("Available benchmarks:\n", .{});
            std.debug.print("  matmul     - Matrix multiplication benchmarks\n", .{});
            std.debug.print("  activation - Activation function benchmarks\n", .{});
            std.debug.print("  memory     - Memory transfer benchmarks\n", .{});
            std.debug.print("  all        - Run all benchmarks\n", .{});
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "-c") or std.mem.eql(u8, arg, "--compare")) {
            options.compare = true;
        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --output requires a filename\n", .{});
                std.process.exit(1);
            }
            options.output_file = args[i];
        } else if (std.mem.eql(u8, arg, "-w") or std.mem.eql(u8, arg, "--warmup")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --warmup requires a number\n", .{});
                std.process.exit(1);
            }
            options.warmup_iterations = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "-i") or std.mem.eql(u8, arg, "--iterations")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --iterations requires a number\n", .{});
                std.process.exit(1);
            }
            options.benchmark_iterations = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--size")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --size requires a comma-separated list\n", .{});
                std.process.exit(1);
            }
            options.sizes = try parseSizeList(args[i], allocator);
        } else if (std.mem.eql(u8, arg, "matmul")) {
            options.benchmark = .matmul;
        } else if (std.mem.eql(u8, arg, "activation")) {
            options.benchmark = .activation;
        } else if (std.mem.eql(u8, arg, "memory")) {
            options.benchmark = .memory;
        } else if (std.mem.eql(u8, arg, "all")) {
            options.benchmark = .all;
        } else {
            std.debug.print("Error: Unknown argument '{s}'\n", .{arg});
            printUsage();
            std.process.exit(1);
        }
    }

    return options;
}

fn runMatMulBenchmarks(
    allocator: std.mem.Allocator,
    options: RunOptions,
    results: *std.ArrayList(cuda_benchmark.BenchmarkResult),
) !void {
    std.debug.print("\n=== Matrix Multiplication Benchmarks ===\n", .{});

    const config = cuda_benchmark.BenchmarkConfig{
        .warmup_iterations = options.warmup_iterations,
        .benchmark_iterations = options.benchmark_iterations,
    };

    for (options.sizes) |size| {
        // Use square matrices for simplicity
        const m, const n, const k = .{ size, size, size };

        // CUDA benchmark
        const cuda_result = cuda_benchmark.benchmarkCudaMatMul(allocator, m, n, k, config) catch |err| {
            std.log.warn("CUDA matmul failed for {d}x{d}x{d}: {s}", .{ m, n, k, @errorName(err) });
            continue;
        };
        try results.append(cuda_result);
        cuda_benchmark.printResult(cuda_result);

        // CPU benchmark (if comparing)
        if (options.compare) {
            const cpu_result = cuda_benchmark.benchmarkCpuMatMul(allocator, m, n, k, config) catch |err| {
                std.log.warn("CPU matmul failed for {d}x{d}x{d}: {s}", .{ m, n, k, @errorName(err) });
                continue;
            };
            try results.append(cpu_result);
            cuda_benchmark.printResult(cpu_result);

            const speedup = cpu_result.avg_time_ns / cuda_result.avg_time_ns;
            std.debug.print("Speedup: {d:.2}x\n\n", .{speedup});
        }
    }
}

fn runActivationBenchmarks(
    allocator: std.mem.Allocator,
    options: RunOptions,
    results: *std.ArrayList(cuda_benchmark.BenchmarkResult),
) !void {
    std.debug.print("\n=== Activation Function Benchmarks ===\n", .{});

    const config = cuda_benchmark.BenchmarkConfig{
        .warmup_iterations = options.warmup_iterations,
        .benchmark_iterations = options.benchmark_iterations,
    };

    const activations = [_][]const u8{ "relu", "sigmoid", "tanh" };

    inline for (activations) |act| {
        std.debug.print("\n{s}:\n", .{act});

        for (options.sizes) |size| {
            // CUDA benchmark
            const cuda_result = cuda_benchmark.benchmarkCudaActivation(allocator, act, size, config) catch |err| {
                std.log.warn("CUDA {s} failed for size {d}: {s}", .{ act, size, @errorName(err) });
                continue;
            };
            try results.append(cuda_result);

            // CPU benchmark
            if (options.compare) {
                const cpu_result = cuda_benchmark.benchmarkCpuActivation(allocator, act, size, config) catch |err| {
                    std.log.warn("CPU {s} failed for size {d}: {s}", .{ act, size, @errorName(err) });
                    continue;
                };
                try results.append(cpu_result);

                const speedup = cpu_result.avg_time_ns / cuda_result.avg_time_ns;
                std.debug.print("  Size {d}: CUDA {d:.2}us, Speedup: {d:.2}x\n", .{
                    size,
                    cuda_result.avg_time_ns / 1e3,
                    speedup,
                });
            } else {
                std.debug.print("  Size {d}: CUDA {d:.2}us\n", .{
                    size,
                    cuda_result.avg_time_ns / 1e3,
                });
            }
        }
    }
}

fn runMemoryBenchmarks(
    allocator: std.mem.Allocator,
    options: RunOptions,
    results: *std.ArrayList(cuda_benchmark.BenchmarkResult),
) !void {
    std.debug.print("\n=== Memory Transfer Benchmarks ===\n", .{});

    const config = cuda_benchmark.BenchmarkConfig{
        .warmup_iterations = options.warmup_iterations,
        .benchmark_iterations = options.benchmark_iterations,
    };

    for (options.sizes) |size| {
        const transfer_results = cuda_benchmark.benchmarkCudaMemoryTransfer(allocator, size, config) catch |err| {
            std.log.warn("Memory transfer failed for size {d}: {s}", .{ size, @errorName(err) });
            continue;
        };

        try results.append(transfer_results.h2d);
        try results.append(transfer_results.d2h);

        std.debug.print("Size {d}: H2D {d:.2} GB/s, D2H {d:.2} GB/s\n", .{
            size,
            transfer_results.h2d.bandwidth_gbps,
            transfer_results.d2h.bandwidth_gbps,
        });
    }
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const options = parseArgs(args, allocator) catch {
        printUsage();
        std.process.exit(1);
    };

    // Print device info
    cuda_benchmark.printDeviceInfo(allocator);

    // Run benchmarks
    var results = std.ArrayList(cuda_benchmark.BenchmarkResult).init(allocator);
    defer results.deinit();

    switch (options.benchmark) {
        .matmul => try runMatMulBenchmarks(allocator, options, &results),
        .activation => try runActivationBenchmarks(allocator, options, &results),
        .memory => try runMemoryBenchmarks(allocator, options, &results),
        .all => {
            try runMatMulBenchmarks(allocator, options, &results);
            try runActivationBenchmarks(allocator, options, &results);
            try runMemoryBenchmarks(allocator, options, &results);
        },
    }

    // Print summary
    cuda_benchmark.printSummaryTable(results.items);

    // Export to CSV if requested
    if (options.output_file) |filename| {
        cuda_benchmark.exportToCsv(allocator, results.items, filename) catch |err| {
            std.log.warn("Failed to export to CSV: {s}", .{@errorName(err)});
        } else {
            std.debug.print("\nResults exported to {s}\n", .{filename});
        }
    }

    std.debug.print("\n=== Benchmark Complete ===\n", .{});
}

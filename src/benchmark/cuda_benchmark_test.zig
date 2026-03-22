/// CUDA Benchmark Tests
/// Tests to validate the CUDA benchmarking infrastructure
const std = @import("std");
const cuda_benchmark = @import("benchmark/cuda_benchmark.zig");

const TEST_CONFIG = cuda_benchmark.BenchmarkConfig{
    .warmup_iterations = 2,
    .benchmark_iterations = 10,
    .collect_statistics = true,
};

test "cuda_benchmark_matmul_small" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        std.debug.print("\nCUDA not available, skipping test\n", .{});
        return;
    };
    defer backend.deinit();

    const result = try cuda_benchmark.benchmarkCudaMatMul(std.testing.allocator, 128, 128, 128, TEST_CONFIG);

    // Validate result structure
    try std.testing.expect(std.mem.eql(u8, result.name, "matmul"));
    try std.testing.expect(std.mem.eql(u8, result.backend, "CUDA"));
    try std.testing.expect(result.benchmark_iterations == 10);
    try std.testing.expect(result.total_time_ns > 0);
    try std.testing.expect(result.throughput > 0);
}

test "cuda_benchmark_cpu_matmul_small" {
    const result = try cuda_benchmark.benchmarkCpuMatMul(std.testing.allocator, 128, 128, 128, TEST_CONFIG);

    // Validate result structure
    try std.testing.expect(std.mem.eql(u8, result.name, "matmul"));
    try std.testing.expect(std.mem.eql(u8, result.backend, "CPU"));
    try std.testing.expect(result.benchmark_iterations == 10);
    try std.testing.expect(result.total_time_ns > 0);
}

test "cuda_benchmark_activation_relu" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        std.debug.print("\nCUDA not available, skipping test\n", .{});
        return;
    };
    defer backend.deinit();

    const result = try cuda_benchmark.benchmarkCudaActivation(std.testing.allocator, "relu", 1024, TEST_CONFIG);

    try std.testing.expect(std.mem.startsWith(u8, result.name, "activation_relu"));
    try std.testing.expect(std.mem.eql(u8, result.backend, "CUDA"));
    try std.testing.expect(result.total_time_ns > 0);
}

test "cuda_benchmark_memory_transfer" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        std.debug.print("\nCUDA not available, skipping test\n", .{});
        return;
    };
    defer backend.deinit();

    const results = try cuda_benchmark.benchmarkCudaMemoryTransfer(std.testing.allocator, 4096, TEST_CONFIG);

    try std.testing.expect(std.mem.eql(u8, results.h2d.name, "memory_h2d"));
    try std.testing.expect(std.mem.eql(u8, results.d2h.name, "memory_d2h"));
    try std.testing.expect(results.h2d.total_time_ns > 0);
    try std.testing.expect(results.d2h.total_time_ns > 0);
    try std.testing.expect(results.h2d.bandwidth_gbps > 0);
    try std.testing.expect(results.d2h.bandwidth_gbps > 0);
}

test "cuda_benchmark_export_csv" {
    var results = std.ArrayList(cuda_benchmark.BenchmarkResult).init(std.testing.allocator);
    defer results.deinit();

    // Create a dummy result
    const dummy_result = cuda_benchmark.BenchmarkResult{
        .name = "test",
        .backend = "TEST",
        .dimensions = "128x128",
        .warmup_iterations = 1,
        .benchmark_iterations = 1,
        .total_time_ns = 1000,
        .avg_time_ns = 1000,
        .min_time_ns = 1000,
        .max_time_ns = 1000,
        .std_dev_ns = 0,
        .throughput = 10.0,
        .bandwidth_gbps = 1.0,
        .operations = 1000,
        .data_size_bytes = 1000,
    };

    try results.append(dummy_result);

    // Export to temporary file
    const temp_file = "/tmp/cuda_benchmark_test.csv";
    try cuda_benchmark.exportToCsv(std.testing.allocator, results.items, temp_file);

    // Verify file exists and contains expected content
    const file = try std.fs.cwd().openFile(temp_file, .{});
    defer file.close();

    const content = try file.readToEndAlloc(std.testing.allocator, 1024);
    defer std.testing.allocator.free(content);

    try std.testing.expect(std.mem.containsAtLeast(u8, content, 1, "test"));
    try std.testing.expect(std.mem.containsAtLeast(u8, content, 1, "TEST"));

    // Cleanup
    try std.fs.cwd().deleteFile(temp_file);
}

test "cuda_benchmark_print_functions" {
    // These functions just print, so we verify they don't crash
    const dummy_result = cuda_benchmark.BenchmarkResult{
        .name = "test",
        .backend = "TEST",
        .dimensions = "128x128",
        .warmup_iterations = 1,
        .benchmark_iterations = 10,
        .total_time_ns = 1000000,
        .avg_time_ns = 100000,
        .min_time_ns = 90000,
        .max_time_ns = 110000,
        .std_dev_ns = 5000,
        .throughput = 10.0,
        .bandwidth_gbps = 1.0,
        .operations = 1000,
        .data_size_bytes = 1000,
    };

    cuda_benchmark.printResult(dummy_result);

    const cpu_result = cuda_benchmark.BenchmarkResult{
        .name = "test",
        .backend = "CPU",
        .dimensions = "128x128",
        .warmup_iterations = 1,
        .benchmark_iterations = 10,
        .total_time_ns = 10000000,
        .avg_time_ns = 1000000,
        .min_time_ns = 900000,
        .max_time_ns = 1100000,
        .std_dev_ns = 50000,
        .throughput = 1.0,
        .bandwidth_gbps = 0.1,
        .operations = 1000,
        .data_size_bytes = 1000,
    };

    cuda_benchmark.printComparison(dummy_result, cpu_result);
}

test "cuda_benchmark_comparison_speedup" {
    // Test that speedup calculation works correctly
    const cuda_result = cuda_benchmark.BenchmarkResult{
        .name = "matmul",
        .backend = "CUDA",
        .dimensions = "128x128x128",
        .warmup_iterations = 1,
        .benchmark_iterations = 10,
        .total_time_ns = 1000000,
        .avg_time_ns = 100000,
        .min_time_ns = 90000,
        .max_time_ns = 110000,
        .std_dev_ns = 5000,
        .throughput = 100.0,
        .bandwidth_gbps = 10.0,
        .operations = 1000,
        .data_size_bytes = 1000,
    };

    const cpu_result = cuda_benchmark.BenchmarkResult{
        .name = "matmul",
        .backend = "CPU",
        .dimensions = "128x128x128",
        .warmup_iterations = 1,
        .benchmark_iterations = 10,
        .total_time_ns = 10000000,
        .avg_time_ns = 1000000,
        .min_time_ns = 900000,
        .max_time_ns = 1100000,
        .std_dev_ns = 50000,
        .throughput = 10.0,
        .bandwidth_gbps = 1.0,
        .operations = 1000,
        .data_size_bytes = 1000,
    };

    // Speedup should be CPU time / CUDA time = 10x
    const speedup = cpu_result.avg_time_ns / cuda_result.avg_time_ns;
    try std.testing.expect(speedup == 10.0);
}

const cuda = @import("cuda.zig");

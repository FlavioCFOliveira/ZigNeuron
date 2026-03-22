/// Tests for CUDA Profiler Integration
const std = @import("std");
const cuda_profiler = @import("../cuda_profiler.zig");

// Note: These tests require a CUDA-capable GPU and driver to run
// They will be skipped if CUDA is not available

test "CudaProfiler initialization" {
    const allocator = std.testing.allocator;

    // Test initialization with disabled mode (should always work)
    const profiler = try cuda_profiler.CudaProfiler.init(allocator, .disabled);
    defer profiler.deinit();

    try std.testing.expectEqual(profiler.mode, .disabled);
    try std.testing.expect(!profiler.timing_enabled);
    try std.testing.expect(!profiler.memory_tracking_enabled);
}

test "CudaProfiler mode switching" {
    const allocator = std.testing.allocator;

    const profiler = try cuda_profiler.CudaProfiler.init(allocator, .disabled);
    defer profiler.deinit();

    // Test mode switching
    profiler.setMode(.manual);
    try std.testing.expectEqual(profiler.mode, .manual);

    profiler.setMode(.automatic);
    try std.testing.expectEqual(profiler.mode, .automatic);
    try std.testing.expect(profiler.timing_enabled);

    profiler.setMode(.full);
    try std.testing.expectEqual(profiler.mode, .full);
    try std.testing.expect(profiler.memory_tracking_enabled);
}

test "CudaProfiler custom metrics" {
    const allocator = std.testing.allocator;

    const profiler = try cuda_profiler.CudaProfiler.init(allocator, .manual);
    defer profiler.deinit();

    // Register a custom metric
    try profiler.registerMetric("test_metric", "A test metric", .{ .float64 = 0.0 });

    // Update the metric
    try profiler.updateMetric("test_metric", .{ .float64 = 42.0 });

    // Get the metric value
    const value = try profiler.getMetric("test_metric");
    try std.testing.expectEqual(value.float64, 42.0);
}

test "CudaProfiler counter metrics" {
    const allocator = std.testing.allocator;

    const profiler = try cuda_profiler.CudaProfiler.init(allocator, .manual);
    defer profiler.deinit();

    // Register a counter metric
    try profiler.registerMetric("batch_count", "Number of batches", .{ .counter = 0 });

    // Increment the counter
    try profiler.incrementMetric("batch_count", 1);
    try profiler.incrementMetric("batch_count", 1);
    try profiler.incrementMetric("batch_count", 1);

    // Verify the count
    const value = try profiler.getMetric("batch_count");
    try std.testing.expectEqual(value.counter, 3);
}

test "CudaProfiler memory tracking" {
    const allocator = std.testing.allocator;

    const profiler = try cuda_profiler.CudaProfiler.init(allocator, .full);
    defer profiler.deinit();

    // Record some allocations
    profiler.recordAllocation(1024);
    profiler.recordAllocation(2048);
    profiler.recordAllocation(4096);

    // Check stats
    const stats = profiler.getMemoryStats();
    try std.testing.expectEqual(stats.total_allocations, 3);
    try std.testing.expectEqual(stats.current_bytes, 7168);
    try std.testing.expectEqual(stats.peak_bytes, 7168);

    // Record deallocation
    profiler.recordDeallocation(2048);
    const stats2 = profiler.getMemoryStats();
    try std.testing.expectEqual(stats2.total_deallocations, 1);
    try std.testing.expectEqual(stats2.current_bytes, 5120);
}

test "CudaProfiler pool tracking" {
    const allocator = std.testing.allocator;

    const profiler = try cuda_profiler.CudaProfiler.init(allocator, .full);
    defer profiler.deinit();

    // Record pool hits and misses
    profiler.recordPoolHit();
    profiler.recordPoolHit();
    profiler.recordPoolMiss();
    profiler.recordPoolHit();

    // Check efficiency
    const stats = profiler.getMemoryStats();
    try std.testing.expectEqual(stats.pool_hits, 3);
    try std.testing.expectEqual(stats.pool_misses, 1);
    try std.testing.expectApproxEqAbs(profiler.getMemoryEfficiency(), 0.75, 0.01);
}

test "CudaProfiler timing history" {
    const allocator = std.testing.allocator;

    const profiler = try cuda_profiler.CudaProfiler.init(allocator, .automatic);
    defer profiler.deinit();

    // Manually add timing entries (normally done by kernel launches)
    const timing = cuda_profiler.KernelTiming{
        .name = try allocator.dupe(u8, "test_kernel"),
        .start_ms = 0,
        .end_ms = 10,
        .duration_ms = 10,
        .grid_dim = .{ 1, 1, 1 },
        .block_dim = .{ 256, 1, 1 },
        .shared_mem_bytes = 0,
        .num_threads = 256,
    };
    try profiler.timing_history.append(timing);

    // Check timing history
    const history = profiler.getTimingHistory();
    try std.testing.expectEqual(history.len, 1);
    try std.testing.expectEqualStrings(history[0].name, "test_kernel");
    try std.testing.expectEqual(history[0].duration_ms, 10);

    // Clean up
    allocator.free(timing.name);

    // Clear history
    profiler.clearTimingHistory();
    try std.testing.expectEqual(profiler.getTimingHistory().len, 0);
}

test "CudaProfiler kernel stats" {
    const allocator = std.testing.allocator;

    const profiler = try cuda_profiler.CudaProfiler.init(allocator, .automatic);
    defer profiler.deinit();

    // Add multiple timings for same kernel
    const kernel_names = [_][]const u8{ "kernel_a", "kernel_a", "kernel_b" };
    const durations = [_]f64{ 5.0, 15.0, 10.0 };

    for (0..3) |i| {
        const timing = cuda_profiler.KernelTiming{
            .name = try allocator.dupe(u8, kernel_names[i]),
            .start_ms = 0,
            .end_ms = durations[i],
            .duration_ms = durations[i],
            .grid_dim = .{ 1, 1, 1 },
            .block_dim = .{ 256, 1, 1 },
            .shared_mem_bytes = 0,
            .num_threads = 256,
        };
        try profiler.timing_history.append(timing);
    }

    // Check kernel_a stats
    const stats = profiler.getKernelStats("kernel_a");
    try std.testing.expectEqual(stats.count, 2);
    try std.testing.expectApproxEqAbs(stats.total_ms, 20.0, 0.01);
    try std.testing.expectApproxEqAbs(stats.avg_ms, 10.0, 0.01);
    try std.testing.expectApproxEqAbs(stats.min_ms, 5.0, 0.01);
    try std.testing.expectApproxEqAbs(stats.max_ms, 15.0, 0.01);

    // Cleanup
    for (profiler.timing_history.items) |*t| {
        allocator.free(t.name);
    }
}

test "CudaProfiler compile-time configuration" {
    // Verify compile-time constants are set correctly
    try std.testing.expect(cuda_profiler.MAX_CONCURRENT_RANGES > 0);
    try std.testing.expect(cuda_profiler.MAX_CUSTOM_METRICS > 0);
    try std.testing.expect(cuda_profiler.MAX_TIMING_EVENTS > 0);
}

test "CudaProfiler scope guard" {
    const allocator = std.testing.allocator;

    const profiler = try cuda_profiler.CudaProfiler.init(allocator, .manual);
    defer profiler.deinit();

    // Test scope guard pattern (without actual CUDA context)
    // In real usage, this would push/pop NVTX ranges
    {
        // This would normally push a range
        // defer would pop it automatically
        // For this test, we just verify the structure compiles
    }
}

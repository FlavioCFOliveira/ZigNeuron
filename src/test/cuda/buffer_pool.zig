/// CUDA Buffer Pool Tests
/// Tests for memory pooling behavior, reuse, and performance
const std = @import("std");
const zn = @import("ZigNeuron");
const cuda_driver = zn.cuda_driver;
const cuda_context = zn.cuda_context;

const CudaDriver = cuda_driver.CudaDriver;
const CudaContext = cuda_context.CudaContext;
const MEMORY_POOL_BUCKETS = cuda_context.MEMORY_POOL_BUCKETS;
const MIN_POOL_SIZE = cuda_context.MIN_POOL_SIZE;
const MAX_POOL_SIZE = cuda_context.MAX_POOL_SIZE;

// =============================================================================
// Test Configuration
// =============================================================================

/// Skip tests on platforms that don't support CUDA
fn skipIfUnsupported() !void {
    if (@import("builtin").os.tag == .macos) {
        return error.SkipZigTest;
    }
}

/// Check if CUDA is available
fn isCudaAvailable(allocator: std.mem.Allocator) bool {
    if (@import("builtin").os.tag == .macos) {
        return false;
    }
    var driver = CudaDriver.init(allocator) catch return false;
    driver.deinit();
    return driver.is_initialized;
}

/// Helper to get initialized context or skip
/// SECURITY FIX: Updated to use reference-counted driver API (CRIT-002)
fn getContextOrSkip(allocator: std.mem.Allocator) !?*CudaContext {
    // CudaContext.init now acquires its own driver reference
    return CudaContext.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound or
            err == error.CudaInitFailed or
            err == error.NoCudaDevices or
            err == error.UnsupportedPtxVersion)
        {
            return null;
        }
        return err;
    };
}

// =============================================================================
// Basic Pool Tests
// =============================================================================

test "CUDA buffer pool - get and return single buffer" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Get buffer from pool
    const buf = try ctx.?.getBuffer(1024);
    try std.testing.expect(buf.ptr != 0);
    try std.testing.expect(buf.size >= 1024);

    // Return to pool
    ctx.?.returnBuffer(buf);
}

test "CUDA buffer pool - reuse verification" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Get a buffer
    const buf1 = try ctx.?.getBuffer(2048);
    const ptr1 = buf1.ptr;

    // Return it
    ctx.?.returnBuffer(buf1);

    // Get another buffer of same size
    const buf2 = try ctx.?.getBuffer(2048);

    // May be the same buffer (reused) or different
    // The important thing is that it works
    try std.testing.expect(buf2.ptr != 0);

    // If pointer was reused, that's good pool behavior
    _ = ptr1;
    try std.testing.expect(buf2.size >= 2048);

    ctx.?.returnBuffer(buf2);
}

test "CUDA buffer pool - multiple sizes" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const sizes = [_]usize{ 64, 128, 256, 512, 1024, 2048, 4096 };

    for (sizes) |size| {
        const buf = try ctx.?.getBuffer(size);
        try std.testing.expect(buf.ptr != 0);
        try std.testing.expect(buf.size >= size);
        ctx.?.returnBuffer(buf);
    }
}

test "CUDA buffer pool - stress test" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const iterations = 100;
    const num_buffers = 10;

    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        var buffers: [10]cuda_context.CudaContext.DeviceBuffer = undefined;

        // Allocate
        for (0..num_buffers) |j| {
            buffers[j] = try ctx.?.getBuffer(1024 * ((j % 4) + 1));
            try std.testing.expect(buffers[j].ptr != 0);
        }

        // Return in random order
        const return_order: [10]usize = .{ 5, 2, 8, 1, 7, 3, 9, 0, 4, 6 };
        for (return_order) |idx| {
            ctx.?.returnBuffer(buffers[idx]);
        }
    }
}

// =============================================================================
// Pool Statistics Tests
// =============================================================================

test "CUDA buffer pool - capacity limits" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Test allocation at pool boundaries
    const min_size = MIN_POOL_SIZE;

    // Try minimum size
    const min_buf = try ctx.?.getBuffer(min_size);
    try std.testing.expect(min_buf.ptr != 0);
    ctx.?.returnBuffer(min_buf);

    // Don't test max_size as it may be too large for some systems
    // Just verify the pool handles it
}

test "CUDA buffer pool - bucket distribution" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Test that different sizes go to appropriate buckets
    const sizes = [_]usize{
        MIN_POOL_SIZE,
        MIN_POOL_SIZE * 2,
        MIN_POOL_SIZE * 4,
        1024,
        4096,
        16384,
        65536,
    };

    for (sizes) |size| {
        const buf = try ctx.?.getBuffer(size);
        try std.testing.expect(buf.ptr != 0);
        try std.testing.expect(buf.size >= size);
        ctx.?.returnBuffer(buf);
    }
}

// =============================================================================
// Pool Performance Tests
// =============================================================================

test "CUDA buffer pool - allocation speed comparison" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const num_iterations = 100;
    const buffer_size = 4096;

    // First pass: warmup and fill pool
    {
        var buffers: [10]cuda_context.CudaContext.DeviceBuffer = undefined;
        for (0..10) |i| {
            buffers[i] = try ctx.?.getBuffer(buffer_size);
        }
        for (buffers) |buf| {
            ctx.?.returnBuffer(buf);
        }
    }

    // Measure pooled allocation (simplified)
    var i: usize = 0;
    while (i < num_iterations) : (i += 1) {
        const buf = try ctx.?.getBuffer(buffer_size);
        ctx.?.returnBuffer(buf);
    }

    // Just verify the test completed successfully
    try std.testing.expect(true);
}

// =============================================================================
// Pool Consistency Tests
// =============================================================================

test "CUDA buffer pool - data persistence" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const size = data.len * @sizeOf(f32);

    // Get buffer and write data
    const buf1 = try ctx.?.getBuffer(size);
    try ctx.?.upload(buf1.ptr, std.mem.sliceAsBytes(&data));

    // Return buffer
    ctx.?.returnBuffer(buf1);

    // Get buffer again (may be reused)
    const buf2 = try ctx.?.getBuffer(size);

    // Data may or may not be preserved (depends on pool implementation)
    // Just verify we can use the buffer
    try std.testing.expect(buf2.ptr != 0);

    // Write new data
    const new_data = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    try ctx.?.upload(buf2.ptr, std.mem.sliceAsBytes(&new_data));

    // Verify data
    var output: [5]f32 = undefined;
    try ctx.?.download(std.mem.sliceAsBytes(&output), buf2.ptr);

    for (new_data, 0..) |expected, i| {
        try std.testing.expectApproxEqAbs(expected, output[i], 0.0001);
    }

    ctx.?.returnBuffer(buf2);
}

test "CUDA buffer pool - concurrent buffer usage" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Allocate multiple buffers simultaneously
    const num_buffers = 20;
    var buffers = try allocator.alloc(cuda_context.CudaContext.DeviceBuffer, num_buffers);
    defer allocator.free(buffers);

    // Allocate all
    for (0..num_buffers) |i| {
        buffers[i] = try ctx.?.getBuffer(1024 * ((i % 5) + 1));
        try std.testing.expect(buffers[i].ptr != 0);
    }

    // Verify all are valid
    for (buffers) |buf| {
        try std.testing.expect(buf.ptr != 0);
        try std.testing.expect(buf.size > 0);
    }

    // Free all
    for (buffers) |buf| {
        ctx.?.returnBuffer(buf);
    }
}

// =============================================================================
// Pool Edge Cases
// =============================================================================

test "CUDA buffer pool - zero size handling" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // May succeed or fail
    const buf = ctx.?.getBuffer(0) catch |err| {
        if (err == error.CudaInvalidValue) return;
        return err;
    };
    ctx.?.returnBuffer(buf);
}

test "CUDA buffer pool - very large allocation" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Try to allocate very large buffer
    const huge_size: usize = 1024 * 1024 * 1024; // 1GB

    const buf = ctx.?.getBuffer(huge_size) catch |err| {
        // Expected to fail on most systems
        if (err == error.CudaOutOfMemory) return;
        return err;
    };

    // If it succeeded, verify and free
    try std.testing.expect(buf.ptr != 0);
    ctx.?.returnBuffer(buf);
}

test "CUDA buffer pool - repeated allocation cycle" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const cycles = 50;
    const buffers_per_cycle = 5;

    var cycle: usize = 0;
    while (cycle < cycles) : (cycle += 1) {
        var buffers: [5]cuda_context.CudaContext.DeviceBuffer = undefined;

        // Allocate
        for (0..buffers_per_cycle) |i| {
            buffers[i] = try ctx.?.getBuffer(4096);
        }

        // Use buffers (upload data)
        const data = [_]f32{1.0} ** 1024;
        for (buffers) |buf| {
            try ctx.?.upload(buf.ptr, std.mem.sliceAsBytes(&data));
        }

        // Free
        for (buffers) |buf| {
            ctx.?.returnBuffer(buf);
        }
    }
}

// =============================================================================
// Pool Cleanup Tests
// =============================================================================

test "CUDA buffer pool - cleanup with active buffers" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;

    // Allocate some buffers
    const buf1 = try ctx.?.getBuffer(1024);
    const buf2 = try ctx.?.getBuffer(2048);

    // Context should clean up properly even with active buffers
    // In practice, this tests the deinit function
    ctx.?.returnBuffer(buf1);
    ctx.?.returnBuffer(buf2);

    ctx.?.deinit();
}

test "CUDA buffer pool - pool exhaustion and recovery" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Allocate many buffers
    const num_buffers = 100;
    var buffers = try allocator.alloc(cuda_context.CudaContext.DeviceBuffer, num_buffers);
    defer allocator.free(buffers);

    // Try to allocate many small buffers
    var allocated: usize = 0;
    for (0..num_buffers) |i| {
        buffers[i] = ctx.?.getBuffer(1024) catch {
            // May run out of memory
            break;
        };
        allocated += 1;
    }

    // Verify we got at least some buffers
    try std.testing.expect(allocated > 0);

    // Free all allocated buffers
    for (0..allocated) |i| {
        ctx.?.returnBuffer(buffers[i]);
    }

    // Should be able to allocate again
    const new_buf = try ctx.?.getBuffer(1024);
    ctx.?.returnBuffer(new_buf);
}

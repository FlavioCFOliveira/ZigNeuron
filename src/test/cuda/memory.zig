/// CUDA Memory Operations Tests
/// Tests for allocate, upload, download, copy operations
const std = @import("std");
const zn = @import("ZigNeuron");
const cuda_driver = zn.cuda_driver;
const cuda_context = zn.cuda_context;

const CudaDriver = cuda_driver.CudaDriver;
const CudaContext = cuda_context.CudaContext;
const CUdeviceptr = cuda_driver.CUdeviceptr;

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
// Memory Allocation Tests
// =============================================================================

test "CUDA memory allocation - basic" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Test basic allocation
    var buf = try ctx.?.allocBuffer(1024);
    defer ctx.?.freeBuffer(&buf);

    try std.testing.expect(buf.ptr != 0);
    try std.testing.expect(buf.size >= 1024);
}

test "CUDA memory allocation - various sizes" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const sizes = [_]usize{ 64, 256, 1024, 4096, 16384, 65536 };

    for (sizes) |size| {
        var buf = try ctx.?.allocBuffer(size);
        defer ctx.?.freeBuffer(&buf);

        try std.testing.expect(buf.ptr != 0);
        try std.testing.expect(buf.size >= size);
    }
}

test "CUDA memory allocation - zero size" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Zero-size allocation behavior may vary
    var buf = ctx.?.allocBuffer(0) catch |err| {
        // Some implementations may error on zero-size
        if (err == error.CudaInvalidValue) return;
        return err;
    };
    defer ctx.?.freeBuffer(&buf);
}

test "CUDA memory allocation - large size" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Try to allocate a larger buffer (should either succeed or fail gracefully)
    var buf = ctx.?.allocBuffer(100 * 1024 * 1024) catch |err| {
        // Expected if not enough memory
        if (err == error.CudaOutOfMemory) return;
        return err;
    };
    defer ctx.?.freeBuffer(&buf);

    try std.testing.expect(buf.ptr != 0);
}

test "CUDA memory allocation - multiple buffers" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    var buffers: [10]cuda_context.CudaContext.DeviceBuffer = undefined;

    // Allocate multiple buffers
    for (0..10) |i| {
        buffers[i] = try ctx.?.allocBuffer(1024 * (i + 1));
        try std.testing.expect(buffers[i].ptr != 0);
    }

    // Free in reverse order
    var i: usize = 10;
    while (i > 0) {
        i -= 1;
        ctx.?.freeBuffer(&buffers[i]);
    }
}

// =============================================================================
// Memory Upload Tests
// =============================================================================

test "CUDA upload - basic float array" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const size = host_data.len * @sizeOf(f32);

    var d_buf = try ctx.?.allocBuffer(size);
    defer ctx.?.freeBuffer(&d_buf);

    try ctx.?.upload(d_buf.ptr, std.mem.sliceAsBytes(&host_data));
}

test "CUDA upload - large buffer" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const count = 10000;
    const host_data = try allocator.alloc(f32, count);
    defer allocator.free(host_data);

    // Fill with test data
    for (0..count) |i| {
        host_data[i] = @floatFromInt(i);
    }

    const size = count * @sizeOf(f32);
    var d_buf = try ctx.?.allocBuffer(size);
    defer ctx.?.freeBuffer(&d_buf);

    try ctx.?.upload(d_buf.ptr, std.mem.sliceAsBytes(host_data));
}

test "CUDA upload - async" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const size = host_data.len * @sizeOf(f32);

    var d_buf = try ctx.?.allocBuffer(size);
    defer ctx.?.freeBuffer(&d_buf);

    // Async upload
    try ctx.?.uploadAsync(d_buf.ptr, std.mem.sliceAsBytes(&host_data));

    // Sync to ensure completion
    try ctx.?.synchronize();
}

// =============================================================================
// Memory Download Tests
// =============================================================================

test "CUDA download - basic float array" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const size = host_data.len * @sizeOf(f32);

    var d_buf = try ctx.?.allocBuffer(size);
    defer ctx.?.freeBuffer(&d_buf);

    // Upload
    try ctx.?.upload(d_buf.ptr, std.mem.sliceAsBytes(&host_data));

    // Download
    var output: [5]f32 = undefined;
    try ctx.?.download(std.mem.sliceAsBytes(&output), d_buf.ptr);

    // Note: Without kernel execution, we just verify download works
    // The values may not match if memory wasn't properly initialized
}

test "CUDA download - large buffer" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const count = 10000;
    const size = count * @sizeOf(f32);

    var d_buf = try ctx.?.allocBuffer(size);
    defer ctx.?.freeBuffer(&d_buf);

    const output = try allocator.alloc(f32, count);
    defer allocator.free(output);

    // Download (may contain garbage, but should not crash)
    try ctx.?.download(std.mem.sliceAsBytes(output), d_buf.ptr);
}

test "CUDA download - async" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const size = host_data.len * @sizeOf(f32);

    var d_buf = try ctx.?.allocBuffer(size);
    defer ctx.?.freeBuffer(&d_buf);

    try ctx.?.upload(d_buf.ptr, std.mem.sliceAsBytes(&host_data));

    var output: [5]f32 = undefined;
    try ctx.?.downloadAsync(std.mem.sliceAsBytes(&output), d_buf.ptr);

    // Sync to ensure completion
    try ctx.?.synchronize();
}

// =============================================================================
// Round-trip Tests (Upload + Download)
// =============================================================================

test "CUDA round-trip - simple values" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const host_data = [_]f32{ 1.5, 2.5, 3.5, 4.5, 5.5 };
    const size = host_data.len * @sizeOf(f32);

    var d_buf = try ctx.?.allocBuffer(size);
    defer ctx.?.freeBuffer(&d_buf);

    // Upload
    try ctx.?.upload(d_buf.ptr, std.mem.sliceAsBytes(&host_data));

    // Download
    var output: [5]f32 = undefined;
    try ctx.?.download(std.mem.sliceAsBytes(&output), d_buf.ptr);

    // Verify round-trip
    for (host_data, 0..) |expected, i| {
        try std.testing.expectApproxEqAbs(expected, output[i], 0.0001);
    }
}

test "CUDA round-trip - various patterns" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Test zeros
    {
        const zeros = [_]f32{0} ** 100;
        const size = zeros.len * @sizeOf(f32);

        var d_buf = try ctx.?.allocBuffer(size);
        defer ctx.?.freeBuffer(&d_buf);

        try ctx.?.upload(d_buf.ptr, std.mem.sliceAsBytes(&zeros));

        var output: [100]f32 = undefined;
        try ctx.?.download(std.mem.sliceAsBytes(&output), d_buf.ptr);

        for (zeros, 0..) |expected, i| {
            try std.testing.expectApproxEqAbs(expected, output[i], 0.0001);
        }
    }

    // Test ones
    {
        const ones = [_]f32{1.0} ** 100;
        const size = ones.len * @sizeOf(f32);

        var d_buf = try ctx.?.allocBuffer(size);
        defer ctx.?.freeBuffer(&d_buf);

        try ctx.?.upload(d_buf.ptr, std.mem.sliceAsBytes(&ones));

        var output: [100]f32 = undefined;
        try ctx.?.download(std.mem.sliceAsBytes(&output), d_buf.ptr);

        for (ones, 0..) |expected, i| {
            try std.testing.expectApproxEqAbs(expected, output[i], 0.0001);
        }
    }

    // Test negative values
    {
        const negatives = [_]f32{ -1.0, -2.0, -3.0, -4.0, -5.0 };
        const size = negatives.len * @sizeOf(f32);

        var d_buf = try ctx.?.allocBuffer(size);
        defer ctx.?.freeBuffer(&d_buf);

        try ctx.?.upload(d_buf.ptr, std.mem.sliceAsBytes(&negatives));

        var output: [5]f32 = undefined;
        try ctx.?.download(std.mem.sliceAsBytes(&output), d_buf.ptr);

        for (negatives, 0..) |expected, i| {
            try std.testing.expectApproxEqAbs(expected, output[i], 0.0001);
        }
    }
}

test "CUDA round-trip - special values" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Test with special float values
    const special_values = [_]f32{
        0.0, // Zero
        -0.0, // Negative zero
        1.0, // One
        -1.0, // Negative one
        std.math.inf(f32), // Infinity
        -std.math.inf(f32), // Negative infinity
    };

    const size = special_values.len * @sizeOf(f32);
    var d_buf = try ctx.?.allocBuffer(size);
    defer ctx.?.freeBuffer(&d_buf);

    try ctx.?.upload(d_buf.ptr, std.mem.sliceAsBytes(&special_values));

    var output: [6]f32 = undefined;
    try ctx.?.download(std.mem.sliceAsBytes(&output), d_buf.ptr);

    // Verify special values
    try std.testing.expectEqual(special_values[0], output[0]); // 0.0
    try std.testing.expectEqual(special_values[1], output[1]); // -0.0
    try std.testing.expectEqual(special_values[2], output[2]); // 1.0
    try std.testing.expectEqual(special_values[3], output[3]); // -1.0
    try std.testing.expect(std.math.isInf(output[4])); // +inf
    try std.testing.expect(std.math.isInf(output[5])); // -inf
}

// =============================================================================
// Memory Copy Tests (Device to Device)
// =============================================================================

test "CUDA device-to-device copy" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const size = host_data.len * @sizeOf(f32);

    var d_src = try ctx.?.allocBuffer(size);
    defer ctx.?.freeBuffer(&d_src);

    var d_dst = try ctx.?.allocBuffer(size);
    defer ctx.?.freeBuffer(&d_dst);

    // Upload to source
    try ctx.?.upload(d_src.ptr, std.mem.sliceAsBytes(&host_data));

    // Device to device copy
    if (ctx.?.driver.driver.memcpyDtoD) |memcpyDtoD| {
        try cuda_driver.checkCuda(memcpyDtoD(d_dst.ptr, d_src.ptr, size));
    } else {
        // Function not available - skip test
        return;
    }

    // Download from destination
    var output: [5]f32 = undefined;
    try ctx.?.download(std.mem.sliceAsBytes(&output), d_dst.ptr);

    // Verify copy
    for (host_data, 0..) |expected, i| {
        try std.testing.expectApproxEqAbs(expected, output[i], 0.0001);
    }
}

// =============================================================================
// Memory Set Tests
// =============================================================================

test "CUDA memset - zero" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    const count = 100;
    const size = count * @sizeOf(f32);

    var d_buf = try ctx.?.allocBuffer(size);
    defer ctx.?.freeBuffer(&d_buf);

    // Initialize with non-zero
    const host_data = [_]f32{1.0} ** 100;
    try ctx.?.upload(d_buf.ptr, std.mem.sliceAsBytes(&host_data));

    // Memset to zero
    try ctx.?.memset(d_buf.ptr, 0, size);

    // Download and verify
    var output: [100]f32 = undefined;
    try ctx.?.download(std.mem.sliceAsBytes(&output), d_buf.ptr);

    for (output) |val| {
        try std.testing.expectEqual(@as(f32, 0.0), val);
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

test "CUDA memory operations - null pointer handling" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // These should either error or handle gracefully
    // Note: Actual behavior depends on CUDA driver implementation

    // Test with zero-size upload (should be valid or error gracefully)
    _ = ctx.?.upload(0, &[_]u8{}) catch {};
}

test "CUDA memory operations - invalid size" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;
    const ctx = try getContextOrSkip(allocator);
    if (ctx == null) return;
    defer ctx.?.deinit();

    // Try to allocate extremely large buffer
    _ = ctx.?.allocBuffer(std.math.maxInt(usize)) catch |err| {
        // Should fail with overflow or OOM
        if (err == error.CudaOutOfMemory or
            err == error.Overflow)
        {
            return;
        }
        return err;
    };

    // If somehow it succeeded, that's unexpected but OK
}

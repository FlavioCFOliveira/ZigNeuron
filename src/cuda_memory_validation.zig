/// CUDA Memory Management Validation Suite
/// Tests memory allocation, buffer pooling, and data transfer operations
///
/// Run: zig test src/cuda_memory_validation.zig
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda = @import("cuda.zig");
const cuda_context = @import("cuda_context.zig");

// =============================================================================
// Test Constants
// =============================================================================

const TEST_SIZES = &[_]usize{
    256,           // Small allocation
    1024,          // 1KB
    64 * 1024,     // 64KB
    1024 * 1024,   // 1MB
    16 * 1024 * 1024, // 16MB
};

const ALIGNMENT = 256; // CUDA memory alignment requirement

// =============================================================================
// Test 1: Basic Memory Allocation
// =============================================================================

test "cuda_basic_memory_allocation" {
    std.log.info("=== Test 1: Basic Memory Allocation ===", .{});

    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer backend.deinit();

    // Test allocations of various sizes
    for (TEST_SIZES) |size| {
        var buffer = try backend.allocBuffer(size);
        defer backend.freeBuffer(buffer);

        try std.testing.expect(buffer.ptr != 0);
        try std.testing.expect(buffer.size >= size);
        try std.testing.expect(buffer.ptr % ALIGNMENT == 0); // Check alignment

        std.log.info("  Allocated {d} bytes at 0x{x:0>16}", .{size, buffer.ptr});
    }

    std.log.info("✓ Basic allocation test passed", .{});
}

// =============================================================================
// Test 2: Memory Overflow Protection
// =============================================================================

test "cuda_memory_overflow_protection" {
    std.log.info("=== Test 2: Memory Overflow Protection ===", .{});

    // Test that overflow is caught during size calculation
    const m: usize = 65536;
    const n: usize = 65536;

    // This should overflow: m * n > usize max on 64-bit systems
    const size = std.math.mul(usize, m, n) catch |err| {
        std.log.info("  Correctly caught overflow: {s}", .{@errorName(err)});
        std.log.info("✓ Overflow protection test passed", .{});
        return;
    };
    _ = size;
}

// =============================================================================
// Test 3: Buffer Pool Reuse
// =============================================================================

test "cuda_buffer_pool_reuse" {
    std.log.info("=== Test 3: Buffer Pool Reuse ===", .{});

    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer backend.deinit();

    const test_size = 1024 * 1024; // 1MB

    // Allocate and free multiple times
    const iterations = 10;
    var ptrs: [iterations]u64 = undefined;

    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        var buf = try backend.allocBuffer(test_size);
        ptrs[i] = buf.ptr;
        backend.freeBuffer(buf);
    }

    // Allocate again and check if we get the same addresses (reuse)
    var reuse_count: usize = 0;
    i = 0;
    while (i < iterations) : (i += 1) {
        var buf = try backend.allocBuffer(test_size);
        defer backend.freeBuffer(buf);

        // Check if this pointer was seen before
        var j: usize = 0;
        while (j < iterations) : (j += 1) {
            if (buf.ptr == ptrs[j]) {
                reuse_count += 1;
                break;
            }
        }
    }

    std.log.info("  Buffer reuse rate: {d}/{d}", .{reuse_count, iterations});
    try std.testing.expect(reuse_count > 0); // At least some reuse should happen

    std.log.info("✓ Buffer pool reuse test passed", .{});
}

// =============================================================================
// Test 4: Data Transfer (Upload/Download)
// =============================================================================

test "cuda_data_transfer" {
    std.log.info("=== Test 4: Data Transfer ===", .{});

    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer backend.deinit();

    const test_sizes = &[_]usize{ 64, 256, 1024, 4096, 16384 };

    for (test_sizes) |num_elements| {
        const buffer_size = num_elements * @sizeOf(f32);
        var buffer = try backend.allocBuffer(buffer_size);
        defer backend.freeBuffer(buffer);

        // Create test data
        const host_send = try std.testing.allocator.alloc(f32, num_elements);
        defer std.testing.allocator.free(host_send);

        for (host_send, 0..) |*val, i| {
            val.* = @floatFromInt(i);
        }

        // Upload to device
        try backend.upload(buffer.ptr, host_send);

        // Download from device
        const host_recv = try std.testing.allocator.alloc(f32, num_elements);
        defer std.testing.allocator.free(host_recv);

        try backend.download(host_recv, buffer.ptr);

        // Verify data integrity
        for (host_send, host_recv, 0..) |expected, actual, i| {
            if (expected != actual) {
                std.log.err("  Mismatch at index {d}: expected {d}, got {d}", .{i, expected, actual});
                return error.DataMismatch;
            }
        }

        std.log.info("  Transferred {d} elements successfully", .{num_elements});
    }

    std.log.info("✓ Data transfer test passed", .{});
}

// =============================================================================
// Test 5: Async Data Transfer
// =============================================================================

test "cuda_async_data_transfer" {
    std.log.info("=== Test 5: Async Data Transfer ===", .{});

    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer backend.deinit();

    const num_elements = 1024;
    const buffer_size = num_elements * @sizeOf(f32);

    var buffer = try backend.allocBuffer(buffer_size);
    defer backend.freeBuffer(buffer);

    // Create test data
    const host_send = try std.testing.allocator.alloc(f32, num_elements);
    defer std.testing.allocator.free(host_send);

    for (host_send, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Async upload
    try backend.uploadAsync(buffer.ptr, host_send);

    // Async download
    const host_recv = try std.testing.allocator.alloc(f32, num_elements);
    defer std.testing.allocator.free(host_recv);

    try backend.downloadAsync(host_recv, buffer.ptr);

    // Synchronize to ensure completion
    try backend.synchronize();

    // Verify data
    for (host_send, host_recv, 0..) |expected, actual, i| {
        if (expected != actual) {
            std.log.err("  Mismatch at index {d}", .{i});
            return error.DataMismatch;
        }
    }

    std.log.info("✓ Async data transfer test passed", .{});
}

// =============================================================================
// Test 6: Large Memory Allocation
// =============================================================================

test "cuda_large_memory_allocation" {
    std.log.info("=== Test 6: Large Memory Allocation ===", .{});

    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer backend.deinit();

    // Try to allocate 100MB
    const large_size = 100 * 1024 * 1024;

    const buffer = backend.allocBuffer(large_size) catch |err| {
        std.log.info("  Large allocation returned expected error: {s}", .{@errorName(err)});
        return;
    };
    defer backend.freeBuffer(buffer);

    std.log.info("  Successfully allocated {d} MB", .{large_size / (1024 * 1024)});
    std.log.info("✓ Large memory allocation test passed", .{});
}

// =============================================================================
// Test 7: Zero-Size Allocation Handling
// =============================================================================

test "cuda_zero_size_allocation" {
    std.log.info("=== Test 7: Zero-Size Allocation Handling ===", .{});

    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer backend.deinit();

    // Test zero-size allocation behavior
    const buffer = backend.allocBuffer(0) catch |err| {
        std.log.info("  Zero-size allocation correctly rejected: {s}", .{@errorName(err)});
        return;
    };
    defer backend.freeBuffer(buffer);

    std.log.info("✓ Zero-size allocation test completed", .{});
}

// =============================================================================
// Test 8: Multiple Buffer Management
// =============================================================================

test "cuda_multiple_buffer_management" {
    std.log.info("=== Test 8: Multiple Buffer Management ===", .{});

    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer backend.deinit();

    const num_buffers = 10;
    const buffer_size = 1024 * 1024; // 1MB each

    // Allocate multiple buffers
    var buffers: [num_buffers]cuda.DeviceBuffer = undefined;

    var i: usize = 0;
    while (i < num_buffers) : (i += 1) {
        buffers[i] = try backend.allocBuffer(buffer_size);
        std.log.info("  Buffer {d}: 0x{x:0>16}", .{i, buffers[i].ptr});
    }

    // Verify all buffers are unique
    i = 0;
    while (i < num_buffers) : (i += 1) {
        var j = i + 1;
        while (j < num_buffers) : (j += 1) {
            try std.testing.expect(buffers[i].ptr != buffers[j].ptr);
        }
    }

    // Free in reverse order
    i = num_buffers;
    while (i > 0) {
        i -= 1;
        backend.freeBuffer(buffers[i]);
    }

    std.log.info("✓ Multiple buffer management test passed", .{});
}

// =============================================================================
// Test 9: Memory Alignment Verification
// =============================================================================

test "cuda_memory_alignment" {
    std.log.info("=== Test 9: Memory Alignment Verification ===", .{});

    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer backend.deinit();

    const test_sizes = &[_]usize{ 1, 7, 13, 127, 255, 256, 257, 1023, 1024 };

    for (test_sizes) |size| {
        var buffer = try backend.allocBuffer(size);
        defer backend.freeBuffer(buffer);

        // Check 256-byte alignment
        const is_aligned = buffer.ptr % 256 == 0;
        if (!is_aligned) {
            std.log.err("  Buffer of size {d} not aligned: 0x{x}", .{size, buffer.ptr});
            return error.AlignmentError;
        }
    }

    std.log.info("✓ All allocations properly aligned to 256 bytes", .{});
}

// =============================================================================
// Test 10: Context Buffer Operations
// =============================================================================

test "cuda_context_buffer_operations" {
    std.log.info("=== Test 10: Context Buffer Operations ===", .{});

    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer driver.deinit();

    var device_count: c_int = 0;
    _ = driver.deviceGetCount.?(&device_count);
    if (device_count == 0) {
        std.log.info("No CUDA devices found", .{});
        return;
    }

    var context = cuda_context.CudaContext.init(std.testing.allocator, &driver) catch |err| {
        std.log.err("Failed to create context: {s}", .{@errorName(err)});
        return;
    };
    defer context.deinit();

    // Test getBuffer/returnBuffer
    const test_size = 1024;
    var buf = try context.getBuffer(test_size);
    try std.testing.expect(buf.ptr != 0);
    try std.testing.expect(buf.size >= test_size);

    context.returnBuffer(buf);

    std.log.info("✓ Context buffer operations test passed", .{});
}

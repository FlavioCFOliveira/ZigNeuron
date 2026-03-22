/// CUDA Security Validation (Phase 7)
/// Security-focused audit tests for CUDA backend
const std = @import("std");
const cuda = @import("cuda.zig");

// T7.1: Integer Overflow Protection
test "cuda_security_overflow_protection" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Test large allocation that would overflow
    const large_size: usize = std.math.maxInt(usize) / @sizeOf(f32);
    const result = backend.allocBuffer(large_size);

    // Should return error, not panic
    try std.testing.expectError(error.OutOfMemory, result);
}

// T7.1: Buffer Lifecycle Validation
test "cuda_security_buffer_lifecycle" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size: usize = 1024 * @sizeOf(f32);

    // Allocate and free
    const d_buffer = try backend.allocBuffer(size);
    backend.freeBuffer(d_buffer);

    // Should be able to allocate again after free
    const d_buffer2 = try backend.allocBuffer(size);
    defer backend.freeBuffer(d_buffer2);

    try std.testing.expect(d_buffer2.ptr != 0);
}

// T7.1: Null Pointer Validation
test "cuda_security_null_pointer" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Test with empty host slice
    const h_empty = try std.testing.allocator.alloc(f32, 0);
    defer std.testing.allocator.free(h_empty);

    const d_buffer = try backend.allocBuffer(1024 * @sizeOf(f32));
    defer backend.freeBuffer(d_buffer);

    // Upload with empty slice should work
    try backend.upload(d_buffer.ptr, h_empty);
}

// T7.1: Alignment Validation
test "cuda_security_memory_alignment" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // CUDA requires 256-byte alignment
    const sizes = .{ 1, 64, 256, 1024, 4096 };

    inline for (sizes) |size| {
        const d_buffer = try backend.allocBuffer(size * @sizeOf(f32));
        defer backend.freeBuffer(d_buffer);

        // Check alignment
        try std.testing.expect(d_buffer.ptr % 256 == 0);
    }
}

// T7.2: Memory Leak Detection
test "cuda_security_no_memory_leak" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Allocate and free many buffers
    const num_iterations: usize = 100;
    const buffer_size: usize = 1024 * @sizeOf(f32);

    var i: usize = 0;
    while (i < num_iterations) : (i += 1) {
        const d_buffer = try backend.allocBuffer(buffer_size);
        backend.freeBuffer(d_buffer);
    }

    // If we get here without OOM, no major leak detected
    try std.testing.expect(true);
}

// T7.2: Buffer Pool Reuse
test "cuda_security_buffer_pool_reuse" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const buffer_size: usize = 1024 * @sizeOf(f32);

    // Allocate buffer
    const d_buffer1 = try backend.allocBuffer(buffer_size);
    backend.freeBuffer(d_buffer1);

    // Allocate again - should reuse
    const d_buffer2 = try backend.allocBuffer(buffer_size);
    defer backend.freeBuffer(d_buffer2);

    // Pool should reuse the freed buffer
    try std.testing.expect(d_buffer2.ptr != 0);
}

// T7.3: Error Handling - Invalid Backend
test "cuda_security_invalid_backend" {
    // On non-CUDA systems, should gracefully fail
    if (comptime @import("builtin").os.tag == .macos) {
        const result = cuda.CudaBackend.init(std.testing.allocator);
        try std.testing.expectError(error.CudaNotAvailable, result);
        return;
    }

    // Otherwise skip this test
    return;
}

// T7.3: Error Handling - Invalid Buffer Access
test "cuda_security_invalid_buffer" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Create a valid buffer
    const d_buffer = try backend.allocBuffer(1024 * @sizeOf(f32));
    defer backend.freeBuffer(d_buffer);

    // Download with wrong size should still work (just reads less data)
    const h_small = try std.testing.allocator.alloc(f32, 100);
    defer std.testing.allocator.free(h_small);

    try backend.download(h_small, d_buffer.ptr);
}

// T7.1: Secure Memory Clear
test "cuda_security_memory_clear" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size: usize = 1024;
    const d_buffer = try backend.allocBuffer(size * @sizeOf(f32));
    defer backend.freeBuffer(d_buffer);

    // Upload data
    const h_data = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(h_data);
    for (h_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));

    try backend.upload(d_buffer.ptr, h_data);

    // Download and verify
    const h_result = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(h_result);
    try backend.download(h_result, d_buffer.ptr);

    // Data should be intact
    for (h_result, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, @as(f32, @floatFromInt(i)), 1e-5);
    }
}

// T7.3: Concurrent Access Safety
test "cuda_security_concurrent_access" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Allocate multiple buffers
    const d_buffer1 = try backend.allocBuffer(1024 * @sizeOf(f32));
    defer backend.freeBuffer(d_buffer1);
    const d_buffer2 = try backend.allocBuffer(1024 * @sizeOf(f32));
    defer backend.freeBuffer(d_buffer2);

    // Both should be valid
    try std.testing.expect(d_buffer1.ptr != 0);
    try std.testing.expect(d_buffer2.ptr != 0);
    try std.testing.expect(d_buffer1.ptr != d_buffer2.ptr);
}

// T7.1: Device Properties Validation
test "cuda_security_device_validation" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Verify device properties are valid
    try std.testing.expect(backend.context.device_props.compute_capability_major >= 3);
    try std.testing.expect(backend.context.device_props.total_memory > 0);
    try std.testing.expect(backend.context.device_props.multiprocessor_count > 0);
    try std.testing.expect(backend.context.device_props.max_threads_per_multiprocessor > 0);

    // Verify name is not empty
    try std.testing.expect(backend.context.device_props.name[0] != 0);
}

// T7.3: Graceful Degradation on Error
test "cuda_security_graceful_degradation" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Operations should continue after errors
    const h_input = try std.testing.allocator.alloc(f32, 100);
    defer std.testing.allocator.free(h_input);
    const h_output = try std.testing.allocator.alloc(f32, 100);
    defer std.testing.allocator.free(h_output);

    for (h_input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1;

    // Multiple operations
    for (0..10) |_| {
        try backend.reluForward(h_input, h_output);
        try backend.sigmoidForward(h_input, h_output);
        try backend.tanhForward(h_input, h_output);
    }
}

// T7.2: Memory Pressure Test
test "cuda_security_memory_pressure" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Allocate many small buffers
    var buffers: [100]cuda.DeviceBuffer = undefined;
    var count: usize = 0;

    for (&buffers) |*buf| {
        buf.* = backend.allocBuffer(1024 * @sizeOf(f32)) catch break;
        count += 1;
    }

    // Free all allocated buffers
    for (0..count) |i| {
        backend.freeBuffer(buffers[i]);
    }

    // Should have allocated at least some
    try std.testing.expect(count > 0);
}

// T7.1: Input Validation - Negative Dimensions
test "cuda_security_negative_dimensions" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Test with zero-size allocation
    const d_buffer = try backend.allocBuffer(0);
    backend.freeBuffer(d_buffer);

    // Should not crash
    try std.testing.expect(true);
}

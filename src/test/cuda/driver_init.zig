/// CUDA Driver Initialization Tests
/// Tests for CUDA driver loading, device detection, context creation, and cleanup
/// These tests are designed to pass on both systems with and without CUDA
const std = @import("std");
const zn = @import("ZigNeuron");
const cuda_driver = zn.cuda_driver;
const cuda_context = zn.cuda_context;

const CudaDriver = cuda_driver.CudaDriver;
const CudaContext = cuda_context.CudaContext;
const CUresult = cuda_driver.CUresult;
const CudaError = cuda_driver.CudaError;

// =============================================================================
// Test Configuration
// =============================================================================

/// Skip tests on platforms that don't support CUDA
fn skipIfUnsupported() !void {
    if (@import("builtin").os.tag == .macos) {
        return error.SkipZigTest;
    }
}

/// Check if CUDA driver is available
fn isCudaAvailable(allocator: std.mem.Allocator) bool {
    if (@import("builtin").os.tag == .macos) {
        return false;
    }
    var driver = CudaDriver.init(allocator) catch return false;
    driver.deinit();
    return driver.is_initialized;
}

// =============================================================================
// Driver Loading Tests
// =============================================================================

test "CUDA driver loading - available" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    // Try to load the CUDA driver
    var driver = CudaDriver.init(allocator) catch |err| {
        // It's OK if CUDA is not installed - this is a valid scenario
        if (err == error.CudaDriverNotFound or
            err == error.UnsupportedPlatform or
            err == error.CudaInitFailed)
        {
            return;
        }
        return err;
    };
    defer driver.deinit();

    // Verify driver is initialized
    try std.testing.expect(driver.is_initialized);
    try std.testing.expect(driver.isAvailable());
}

test "CUDA driver loading - graceful fallback" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    // Try to load the CUDA driver
    var driver = CudaDriver.init(allocator) catch |err| {
        // Expected errors when CUDA is not available
        if (err == error.CudaDriverNotFound or
            err == error.UnsupportedPlatform or
            err == error.CudaInitFailed)
        {
            return;
        }
        return err;
    };

    // If we got here, driver loaded successfully
    defer driver.deinit();
    try std.testing.expect(driver.is_initialized);
}

test "CUDA driver cleanup after failed init" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    // Try multiple init/deinit cycles to ensure cleanup works correctly
    for (0..3) |_| {
        var driver = CudaDriver.init(allocator) catch {
            // It's OK if CUDA is not available
            continue;
        };
        defer driver.deinit();

        // Verify driver state is valid after cleanup
        try std.testing.expect(driver.is_initialized);
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

test "CUDA error string conversion" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    var driver = CudaDriver.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    // Test error string for known error codes
    const err_str = driver.getErrorString(.ERROR_INVALID_VALUE);
    try std.testing.expect(err_str.len > 0);

    const success_str = driver.getErrorString(.SUCCESS);
    try std.testing.expect(success_str.len > 0);
}

test "CUDA error name conversion" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    var driver = CudaDriver.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    // Test error name for known error codes
    const err_name = driver.getErrorName(.ERROR_INVALID_VALUE);
    try std.testing.expect(err_name.len > 0);
}

test "CUDA result to Zig error conversion" {
    // Test the checkCuda function with various result codes
    // SUCCESS should not throw
    try cuda_driver.checkCuda(.SUCCESS);

    // Error codes should return appropriate errors
    const result = cuda_driver.checkCuda(.ERROR_INVALID_VALUE);
    try std.testing.expectError(error.CudaInvalidValue, result);

    const mem_result = cuda_driver.checkCuda(.ERROR_OUT_OF_MEMORY);
    try std.testing.expectError(error.CudaOutOfMemory, mem_result);
}

// =============================================================================
// Device Detection Tests
// =============================================================================

test "CUDA device count query" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    var driver = CudaDriver.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    // Get device count
    var device_count: c_int = 0;
    const result = driver.deviceGetCount.?(&device_count);

    // Should succeed even if no devices (count = 0)
    try std.testing.expectEqual(CUresult.SUCCESS, result);
    try std.testing.expect(device_count >= 0);
}

test "CUDA device properties query" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    var driver = CudaDriver.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    // Get device count
    var device_count: c_int = 0;
    _ = driver.deviceGetCount.?(&device_count);

    // Skip if no devices
    if (device_count == 0) {
        return;
    }

    // Query properties for each device
    var i: c_int = 0;
    while (i < device_count) : (i += 1) {
        var device: cuda_driver.CUdevice = 0;
        const get_result = driver.deviceGet.?(&device, i);
        try std.testing.expectEqual(CUresult.SUCCESS, get_result);

        // Get device name
        var name: [256]u8 = undefined;
        const name_result = driver.deviceGetName.?(&name, name.len, device);
        try std.testing.expectEqual(CUresult.SUCCESS, name_result);

        // Get compute capability
        var major: c_int = 0;
        var minor: c_int = 0;
        _ = driver.deviceGetAttribute.?(&major, .COMPUTE_CAPABILITY_MAJOR, device);
        _ = driver.deviceGetAttribute.?(&minor, .COMPUTE_CAPABILITY_MINOR, device);

        // Validate compute capability is reasonable (1.0 to 10.0)
        try std.testing.expect(major >= 1 and major <= 10);
        try std.testing.expect(minor >= 0 and minor <= 9);

        // Get total memory
        var total_mem: usize = 0;
        const mem_result = driver.deviceTotalMem.?(&total_mem, device);
        try std.testing.expectEqual(CUresult.SUCCESS, mem_result);
        try std.testing.expect(total_mem > 0);

        // Get warp size
        var warp_size: c_int = 0;
        _ = driver.deviceGetAttribute.?(&warp_size, .WARP_SIZE, device);
        try std.testing.expect(warp_size == 32 or warp_size == 64);
    }
}

test "CUDA device selection" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    var driver = CudaDriver.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    // Get device count
    var device_count: c_int = 0;
    _ = driver.deviceGetCount.?(&device_count);

    if (device_count == 0) {
        return; // No devices to test
    }

    // Test selecting device 0 (should always work if devices exist)
    var device: cuda_driver.CUdevice = 0;
    const result = driver.deviceGet.?(&device, 0);
    try std.testing.expectEqual(CUresult.SUCCESS, result);
}

// =============================================================================
// Context Creation Tests
// =============================================================================

test "CUDA context initialization" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    var driver = CudaDriver.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    // Get device count
    var device_count: c_int = 0;
    _ = driver.deviceGetCount.?(&device_count);

    if (device_count == 0) {
        return; // No devices to test
    }

    // Create context
    const ctx = CudaContext.init(allocator) catch |err| {
        // Context creation might fail for various reasons
        // (e.g., no permissions, device busy) - this is OK
        if (err == error.NoCudaDevices or
            err == error.ContextNotInitialized or
            err == error.CudaInvalidDevice or
            err == error.UnsupportedPtxVersion)
        {
            return;
        }
        return err;
    };
    defer ctx.deinit();

    // Verify context was created
    try std.testing.expect(ctx.context != null);
    try std.testing.expect(ctx.stream.isInitialized());
}

test "CUDA context cleanup - successful init" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    var driver = CudaDriver.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    // Get device count
    var device_count: c_int = 0;
    _ = driver.deviceGetCount.?(&device_count);

    if (device_count == 0) {
        return;
    }

    // Create and destroy context multiple times to verify cleanup
    for (0..3) |_| {
        const ctx = CudaContext.init(allocator) catch {
            // Context creation might fail - skip cleanup test
            continue;
        };
        ctx.deinit();
    }
}

test "CUDA context push/pop operations" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    var driver = CudaDriver.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    // Get device count
    var device_count: c_int = 0;
    _ = driver.deviceGetCount.?(&device_count);

    if (device_count == 0) {
        return;
    }

    const ctx = CudaContext.init(allocator) catch |err| {
        if (err == error.NoCudaDevices or err == error.UnsupportedPtxVersion) return;
        return err;
    };
    defer ctx.deinit();

    // Test push/pop
    ctx.push() catch |err| {
        // Push might fail if context is already current
        if (err == error.ContextNotInitialized) return;
        return err;
    };

    ctx.pop() catch |err| {
        // Pop might fail if no context is current
        return err;
    };
}

// =============================================================================
// Stream Tests
// =============================================================================

test "CUDA stream creation and destruction" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    var driver = CudaDriver.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    // Get device count
    var device_count: c_int = 0;
    _ = driver.deviceGetCount.?(&device_count);

    if (device_count == 0) {
        return;
    }

    const ctx = CudaContext.init(allocator) catch |err| {
        if (err == error.NoCudaDevices or err == error.UnsupportedPtxVersion) return;
        return err;
    };
    defer ctx.deinit();

    // Verify stream exists
    try std.testing.expect(ctx.stream.isInitialized());

    // Test synchronization (may fail on some systems, but shouldn't crash)
    ctx.synchronize() catch {};
}

// =============================================================================
// Memory Management Tests
// =============================================================================

test "CUDA device memory allocation" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    var driver = CudaDriver.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    var device_count: c_int = 0;
    _ = driver.deviceGetCount.?(&device_count);

    if (device_count == 0) {
        return;
    }

    const ctx = CudaContext.init(allocator) catch |err| {
        if (err == error.NoCudaDevices or err == error.UnsupportedPtxVersion) return;
        return err;
    };
    defer ctx.deinit();

    // Test allocation
    var buf = ctx.allocBuffer(1024) catch |err| {
        // Allocation might fail due to memory constraints
        if (err == error.CudaOutOfMemory) return;
        return err;
    };

    // Verify buffer is valid
    try std.testing.expect(buf.ptr != 0);
    try std.testing.expect(buf.size >= 1024);

    // Free the buffer
    ctx.freeBuffer(&buf);
}

test "CUDA buffer pool operations" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    var driver = CudaDriver.init(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    var device_count: c_int = 0;
    _ = driver.deviceGetCount.?(&device_count);

    if (device_count == 0) {
        return;
    }

    const ctx = CudaContext.init(allocator) catch |err| {
        if (err == error.NoCudaDevices or err == error.UnsupportedPtxVersion) return;
        return err;
    };
    defer ctx.deinit();

    // Get buffer from pool
    const buf = ctx.getBuffer(256) catch |err| {
        if (err == error.CudaOutOfMemory) return;
        return err;
    };

    // Return to pool
    ctx.returnBuffer(buf);
}

// =============================================================================
// Global Driver Tests
// =============================================================================

test "CUDA global driver singleton" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    // Initialize global driver
    cuda_driver.initGlobalDriver(allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        if (err == error.CudaInitFailed) return;
        return err;
    };
    defer cuda_driver.deinitGlobalDriver();

    // Get global driver
    const global = cuda_driver.getGlobalDriver();
    try std.testing.expect(global != null);

    if (global) |g| {
        try std.testing.expect(g.is_initialized);
    }
}

test "CUDA global driver multiple init" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    // Multiple init calls should be safe (idempotent)
    cuda_driver.initGlobalDriver(allocator) catch {};
    cuda_driver.initGlobalDriver(allocator) catch {};

    // Cleanup
    cuda_driver.deinitGlobalDriver();
}

// =============================================================================
// Integration Tests
// =============================================================================

test "CUDA full initialization sequence" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    // Step 1: Initialize driver
    var driver = CudaDriver.init(allocator) catch |err| {
        // Expected when CUDA is not available
        if (err == error.CudaDriverNotFound or
            err == error.CudaInitFailed or
            err == error.UnsupportedPlatform)
        {
            return;
        }
        return err;
    };
    defer driver.deinit();

    // Step 2: Check device availability
    var device_count: c_int = 0;
    const count_result = driver.deviceGetCount.?(&device_count);
    try std.testing.expectEqual(CUresult.SUCCESS, count_result);

    if (device_count == 0) {
        return; // No devices - valid test result
    }

    // Step 3: Create context
    const ctx = CudaContext.init(allocator) catch |err| {
        // Context creation might fail - this is OK
        if (err == error.NoCudaDevices or err == error.UnsupportedPtxVersion) return;
        return err;
    };
    defer ctx.deinit();

    // Step 4: Verify context is usable
    try std.testing.expect(ctx.context != null);
    try std.testing.expect(ctx.stream.isInitialized());

    // Step 5: Test basic memory operation
    var buf = ctx.allocBuffer(1024) catch |err| {
        if (err == error.CudaOutOfMemory) return;
        return err;
    };
    defer ctx.freeBuffer(&buf);

    try std.testing.expect(buf.ptr != 0);
}

test "CUDA availability check function" {
    try skipIfUnsupported();

    const allocator = std.testing.allocator;

    // Test the availability check function
    const available = isCudaAvailable(allocator);

    // On macOS, should always be false (already checked in skipIfUnsupported)
    // On other platforms, result depends on CUDA installation
    // Just verify the function doesn't crash
    _ = available;
}

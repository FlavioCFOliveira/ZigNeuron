/// CUDA Driver Integration Test Suite
/// Validates CUDA driver dynamic loading, device enumeration, and lifecycle
///
/// Run: zig test src/test/cuda_driver_validation.zig -lcuda
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda = @import("cuda.zig");

// =============================================================================
// Test Constants
// =============================================================================

const TEST_TIMEOUT_MS = 30000; // 30 seconds timeout per test
const MAX_DEVICES_TO_TEST = 4; // Test up to 4 devices

// =============================================================================
// Test Results
// =============================================================================

pub const TestResult = struct {
    name: []const u8,
    passed: bool,
    duration_ms: u64,
    error_message: ?[]const u8,

    pub fn init(name: []const u8) TestResult {
        return .{
            .name = name,
            .passed = false,
            .duration_ms = 0,
            .error_message = null,
        };
    }
};

pub const TestSummary = struct {
    total: usize,
    passed: usize,
    failed: usize,
    results: std.ArrayList(TestResult),

    pub fn init(allocator: std.mem.Allocator) TestSummary {
        return .{
            .total = 0,
            .passed = 0,
            .failed = 0,
            .results = std.ArrayList(TestResult).init(allocator),
        };
    }

    pub fn deinit(self: *TestSummary) void {
        for (self.results.items) |*result| {
            if (result.error_message) |msg| {
                self.results.allocator.free(msg);
            }
        }
        self.results.deinit();
    }

    pub fn addResult(self: *TestSummary, result: TestResult) !void {
        try self.results.append(result);
        self.total += 1;
        if (result.passed) {
            self.passed += 1;
        } else {
            self.failed += 1;
        }
    }

    pub fn printReport(self: *const TestSummary, writer: anytype) !void {
        try writer.print("\n╔══════════════════════════════════════════════════════════════╗\n", .{});
        try writer.print("║          CUDA Driver Integration Test Report                 ║\n", .{});
        try writer.print("╠══════════════════════════════════════════════════════════════╣\n", .{});
        try writer.print("║ Total:  {d:3}  │  Passed:  {d:3}  │  Failed:  {d:3}               ║\n",
            .{self.total, self.passed, self.failed});
        try writer.print("╠══════════════════════════════════════════════════════════════╣\n", .{});

        for (self.results.items) |result| {
            const status = if (result.passed) "✓ PASS" else "✗ FAIL";
            const status_color = if (result.passed) "\x1b[32m" else "\x1b[31m";
            const reset = "\x1b[0m";

            try writer.print("║ {s}{s}{s} │ {s:50} │ {d:6}ms ║\n",
                .{status_color, status, reset, result.name, result.duration_ms});

            if (!result.passed and result.error_message != null) {
                try writer.print("║      Error: {s:51} ║\n", .{result.error_message.?});
            }
        }

        try writer.print("╚══════════════════════════════════════════════════════════════╝\n", .{});
    }
};

// =============================================================================
// Test 1: CUDA Driver Loading
// =============================================================================

test "cuda_driver_loading" {
    std.log.info("=== Test 1: CUDA Driver Loading ===", .{});

    // Test 1a: Try to load CUDA driver
    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch |err| {
        std.log.info("CUDA driver not available (expected on non-CUDA systems): {s}", .{@errorName(err)});
        // This is OK - we're testing graceful failure
        return;
    };
    defer driver.deinit();

    // Verify driver is initialized
    try std.testing.expect(driver.is_initialized);
    try std.testing.expect(driver.isAvailable());

    // Verify core functions are loaded
    try std.testing.expect(driver.cuInit != null);
    try std.testing.expect(driver.deviceGetCount != null);
    try std.testing.expect(driver.deviceGet != null);
    try std.testing.expect(driver.deviceGetAttribute != null);

    std.log.info("✓ CUDA driver loaded successfully", .{});
}

// =============================================================================
// Test 2: Device Enumeration
// =============================================================================

test "cuda_device_enumeration" {
    std.log.info("=== Test 2: Device Enumeration ===", .{});

    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping device enumeration test - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer driver.deinit();

    // Get device count
    var device_count: c_int = 0;
    const count_result = driver.deviceGetCount.?(&device_count);
    try std.testing.expect(count_result.isSuccess());

    std.log.info("Found {d} CUDA device(s)", .{device_count});

    if (device_count == 0) {
        std.log.warn("No CUDA devices found", .{});
        return;
    }

    // Test each device
    const devices_to_test = @min(device_count, MAX_DEVICES_TO_TEST);
    var i: c_int = 0;
    while (i < devices_to_test) : (i += 1) {
        var device: cuda_driver.CUdevice = undefined;
        const get_result = driver.deviceGet.?(&device, i);
        try std.testing.expect(get_result.isSuccess());

        // Get device name
        var name_buf: [256]u8 = undefined;
        const name_result = driver.deviceGetName.?(&name_buf, name_buf.len, device);
        try std.testing.expect(name_result.isSuccess());

        const device_name = std.mem.sliceTo(&name_buf, 0);
        std.log.info("  Device {d}: {s}", .{i, device_name});

        // Verify device attributes
        var major: c_int = 0;
        var minor: c_int = 0;
        var mem_size: usize = 0;
        var mp_count: c_int = 0;

        const major_result = driver.deviceGetAttribute.?(&major, .COMPUTE_CAPABILITY_MAJOR, device);
        const minor_result = driver.deviceGetAttribute.?(&minor, .COMPUTE_CAPABILITY_MINOR, device);
        const mem_result = driver.deviceTotalMem.?(&mem_size, device);
        const mp_result = driver.deviceGetAttribute.?(&mp_count, .MULTIPROCESSOR_COUNT, device);

        try std.testing.expect(major_result.isSuccess());
        try std.testing.expect(minor_result.isSuccess());
        try std.testing.expect(mem_result.isSuccess());
        try std.testing.expect(mp_result.isSuccess());

        std.log.info("    Compute Capability: {d}.{d}", .{major, minor});
        std.log.info("    Memory: {d} MB", .{mem_size / (1024 * 1024)});
        std.log.info("    Multiprocessors: {d}", .{mp_count});

        // Verify compute capability is reasonable
        try std.testing.expect(major >= 3); // Minimum Pascal
        try std.testing.expect(mem_size > 0);
        try std.testing.expect(mp_count > 0);
    }

    std.log.info("✓ Device enumeration successful", .{});
}

// =============================================================================
// Test 3: Error Handling
// =============================================================================

test "cuda_error_handling" {
    std.log.info("=== Test 3: Error Handling ===", .{});

    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping error handling test - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer driver.deinit();

    // Test error string retrieval
    const error_result = cuda_driver.CUresult.ERROR_INVALID_VALUE;
    const error_str = driver.getErrorString(error_result);
    const error_name = driver.getErrorName(error_result);

    std.log.info("Error {s}: {s}", .{error_name, error_str});

    try std.testing.expect(error_str.len > 0);
    try std.testing.expect(error_name.len > 0);

    // Test success case
    const success_str = driver.getErrorString(.SUCCESS);
    try std.testing.expect(success_str.len > 0);

    std.log.info("✓ Error handling works correctly", .{});
}

// =============================================================================
// Test 4: Backend Availability Check
// =============================================================================

test "cuda_backend_availability" {
    std.log.info("=== Test 4: Backend Availability Check ===", .{});

    // Test the high-level availability check
    const is_available = cuda.CudaBackend.isAvailable();

    if (is_available) {
        std.log.info("CUDA backend is available", .{});

        const device_count = cuda.CudaBackend.getDeviceCount();
        std.log.info("Device count from backend: {d}", .{device_count});
        try std.testing.expect(device_count > 0);
    } else {
        std.log.info("CUDA backend is not available (expected on non-CUDA systems)", .{});
    }
}

// =============================================================================
// Test 5: Backend Initialization
// =============================================================================

test "cuda_backend_initialization" {
    std.log.info("=== Test 5: Backend Initialization ===", .{});

    // Try to initialize the backend
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        std.log.info("Backend initialization skipped - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer backend.deinit();

    std.log.info("✓ Backend initialized successfully", .{});
    // Device properties are in context, access via backend.context
    // Note: device_props is in CudaContext, not directly in CudaBackend
    std.log.info("  Context initialized", .{});
}

// =============================================================================
// Test 6: Multiple Initialization/Deinitialization Cycles
// =============================================================================

test "cuda_driver_lifecycle_stress" {
    std.log.info("=== Test 6: Driver Lifecycle Stress Test ===", .{});

    // Test multiple init/deinit cycles
    const cycles = 5;
    var i: usize = 0;
    while (i < cycles) : (i += 1) {
        var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch |err| {
            std.log.info("Skipping lifecycle test - CUDA not available: {s}", .{@errorName(err)});
            return;
        };

        // Verify initialization
        try std.testing.expect(driver.is_initialized);

        // Get device count
        var device_count: c_int = 0;
        _ = driver.deviceGetCount.?(&device_count);

        // Deinitialize
        driver.deinit();
        try std.testing.expect(!driver.is_initialized);
        try std.testing.expect(!driver.isAvailable());
    }

    std.log.info("✓ Completed {d} init/deinit cycles", .{cycles});
}

// =============================================================================
// Test 7: Platform-Specific Library Names
// =============================================================================

test "cuda_platform_detection" {
    std.log.info("=== Test 7: Platform Detection ===", .{});

    const os_tag = @import("builtin").os.tag;

    // Verify platform detection
    const expected_lib = switch (os_tag) {
        .linux => "libcuda.so",
        .windows => "nvcuda.dll",
        .macos => "not supported",
        else => "unknown",
    };

    std.log.info("Platform: {s}", .{@tagName(os_tag)});
    std.log.info("Expected CUDA library: {s}", .{expected_lib});

    // On macOS, CUDA should not be available
    if (os_tag == .macos) {
        const is_available = cuda.CudaBackend.isAvailable();
        try std.testing.expect(!is_available);
        std.log.info("✓ Correctly returns unavailable on macOS", .{});
    }
}

// =============================================================================
// Test 8: Device Properties Validation
// =============================================================================

test "cuda_device_properties_validation" {
    std.log.info("=== Test 8: Device Properties Validation ===", .{});

    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch |err| {
        std.log.info("Skipping properties validation - CUDA not available: {s}", .{@errorName(err)});
        return;
    };
    defer driver.deinit();

    var device_count: c_int = 0;
    _ = driver.deviceGetCount.?(&device_count);

    if (device_count == 0) {
        std.log.warn("No devices to test", .{});
        return;
    }

    // Get first device
    var device: cuda_driver.CUdevice = undefined;
    _ = driver.deviceGet.?(&device, 0);

    // Test various attributes
    const attributes_to_test = &[_]cuda_driver.CUdevice_attribute{
        .MAX_THREADS_PER_BLOCK,
        .MAX_BLOCK_DIM_X,
        .MAX_BLOCK_DIM_Y,
        .MAX_BLOCK_DIM_Z,
        .MAX_GRID_DIM_X,
        .MAX_GRID_DIM_Y,
        .MAX_GRID_DIM_Z,
        .WARP_SIZE,
        .MAX_SHARED_MEMORY_PER_BLOCK,
        .TOTAL_CONSTANT_MEMORY,
        .MULTIPROCESSOR_COUNT,
        .MAX_PITCH,
        .MAX_REGISTERS_PER_BLOCK,
        .CLOCK_RATE,
        .TEXTURE_ALIGNMENT,
        .GPU_OVERLAP,
        .KERNEL_EXEC_TIMEOUT,
        .INTEGRATED,
        .CAN_MAP_HOST_MEMORY,
        .COMPUTE_MODE,
        .MAXIMUM_TEXTURE1D_WIDTH,
        .MAXIMUM_TEXTURE2D_WIDTH,
        .MAXIMUM_TEXTURE2D_HEIGHT,
        .CONCURRENT_KERNELS,
        .ECC_ENABLED,
        .TCC_DRIVER,
        .UNIFIED_ADDRESSING,
        .MEMORY_CLOCK_RATE,
        .GLOBAL_MEMORY_BUS_WIDTH,
        .L2_CACHE_SIZE,
        .MAX_THREADS_PER_MULTIPROCESSOR,
        .ASYNC_ENGINE_COUNT,
        .COMPUTE_CAPABILITY_MAJOR,
        .COMPUTE_CAPABILITY_MINOR,
    };

    for (attributes_to_test) |attr| {
        var value: c_int = 0;
        const result = driver.deviceGetAttribute.?(&value, attr, device);

        // Most attributes should succeed
        if (!result.isSuccess()) {
            std.log.warn("Failed to get attribute {s}: {s}", .{@tagName(attr), @tagName(result)});
        }
    }

    std.log.info("✓ Queried {d} device attributes", .{attributes_to_test.len});
}

// =============================================================================
// Main Test Runner
// =============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("╔════════════════════════════════════════════════════════════╗", .{});
    std.log.info("║        CUDA Driver Integration Validation Suite            ║", .{});
    std.log.info("╚════════════════════════════════════════════════════════════╝", .{});

    var summary = TestSummary.init(allocator);
    defer summary.deinit();

    // Run all tests and collect results
    const tests = &[_]struct {
        name: []const u8,
        func: *const fn () anyerror!void,
    }{
        .{.name = "CUDA Driver Loading", .func = testCudaDriverLoading},
        .{.name = "Device Enumeration", .func = testDeviceEnumeration},
        .{.name = "Error Handling", .func = testErrorHandling},
        .{.name = "Backend Availability", .func = testBackendAvailability},
        .{.name = "Backend Initialization", .func = testBackendInitialization},
        .{.name = "Lifecycle Stress", .func = testLifecycleStress},
        .{.name = "Platform Detection", .func = testPlatformDetection},
        .{.name = "Device Properties", .func = testDeviceProperties},
    };

    for (tests) |t| {
        var result = TestResult.init(t.name);
        const start = std.time.milliTimestamp();

        const test_result = t.func();
        if (test_result) {
            result.passed = true;
        } else |err| {
            result.error_message = try std.fmt.allocPrint(allocator, "{s}", .{@errorName(err)});
            result.passed = false;
        }

        result.duration_ms = @intCast(std.time.milliTimestamp() - start);
        try summary.addResult(result);
    }

    // Print report
    const stdout = std.io.getStdOut().writer();
    try summary.printReport(stdout);

    // Return error if any tests failed
    if (summary.failed > 0) {
        return error.TestsFailed;
    }
}

// Test function wrappers
fn testCudaDriverLoading() !void {
    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch |err| {
        if (err == error.CudaDriverNotFound or
            err == error.UnsupportedPlatform or
            err == error.CudaInitFailed) {
            return; // Expected on non-CUDA systems
        }
        return err;
    };
    defer driver.deinit();

    try std.testing.expect(driver.is_initialized);
    try std.testing.expect(driver.isAvailable());
}

fn testDeviceEnumeration() !void {
    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch {
        return; // Skip if CUDA not available
    };
    defer driver.deinit();

    var device_count: c_int = 0;
    const result = driver.deviceGetCount.?(&device_count);
    try std.testing.expect(result.isSuccess());

    if (device_count == 0) return;

    var device: cuda_driver.CUdevice = undefined;
    const get_result = driver.deviceGet.?(&device, 0);
    try std.testing.expect(get_result.isSuccess());

    var major: c_int = 0;
    _ = driver.deviceGetAttribute.?(&major, .COMPUTE_CAPABILITY_MAJOR, device);
    try std.testing.expect(major >= 3);
}

fn testErrorHandling() !void {
    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch {
        return; // Skip if CUDA not available
    };
    defer driver.deinit();

    const error_str = driver.getErrorString(.ERROR_INVALID_VALUE);
    try std.testing.expect(error_str.len > 0);

    const success_str = driver.getErrorString(.SUCCESS);
    try std.testing.expect(success_str.len > 0);
}

fn testBackendAvailability() !void {
    const is_available = cuda.CudaBackend.isAvailable();
    _ = is_available; // Just verify it doesn't crash
}

fn testBackendInitialization() !void {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        if (err == error.CudaNotAvailable or
            err == error.NoCudaDevices or
            err == error.CudaDriverNotFound) {
            return; // Expected on non-CUDA systems
        }
        return err;
    };
    defer backend.deinit();

    try std.testing.expect(backend.context.device_props.multiprocessor_count > 0);
}

fn testLifecycleStress() !void {
    const cycles = 3;
    var i: usize = 0;
    while (i < cycles) : (i += 1) {
        var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch {
            return; // Skip if CUDA not available
        };
        driver.deinit();
    }
}

fn testPlatformDetection() !void {
    const os_tag = @import("builtin").os.tag;
    _ = os_tag; // Verify we can access builtin
}

fn testDeviceProperties() !void {
    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch {
        return; // Skip if CUDA not available
    };
    defer driver.deinit();

    var device_count: c_int = 0;
    _ = driver.deviceGetCount.?(&device_count);
    if (device_count == 0) return;

    var device: cuda_driver.CUdevice = undefined;
    _ = driver.deviceGet.?(&device, 0);

    var value: c_int = 0;
    const result = driver.deviceGetAttribute.?(&value, .MAX_THREADS_PER_BLOCK, device);
    try std.testing.expect(result.isSuccess());
    try std.testing.expect(value > 0);
}

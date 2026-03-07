// Test CUDA backend functionality
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda = @import("cuda.zig");

pub fn main() !void {
    std.log.info("=== CUDA Backend Tests ===", .{});
    
    // Test 1: Check CUDA availability
    std.log.info("Test 1: CUDA availability...", .{});
    const is_available = cuda_driver.CudaDriver.isAvailable();
    if (!is_available) {
        std.log.err("CUDA driver not available", .{});
        return error.CudaNotAvailable;
    }
    std.log.info("  CUDA driver available!", .{});
    
    // Test 2: Get device count
    std.log.info("Test 2: Device detection...", .{});
    var driver = try cuda_driver.CudaDriver.load(std.heap.page_allocator);
    defer driver.unload();
    
    const device_count = driver.getDeviceCount() catch |err| {
        std.log.err("Failed to get device count: {}", .{err});
        return err;
    };
    std.log.info("  Found {} CUDA device(s)", .{device_count});
    
    if (device_count == 0) {
        std.log.err("No CUDA devices found", .{});
        return error.NoCudaDevices;
    }
    
    // Test 3: Initialize CUDA backend
    std.log.info("Test 3: Backend initialization...", .{});
    var backend = try cuda.CudaBackend.init(std.heap.page_allocator);
    defer backend.deinit();
    std.log.info("  Backend initialized!", .{});
    
    // Test 4: Device properties
    std.log.info("Test 4: Device properties...", .{});
    std.log.info("  Device: {s}", .{std.mem.sliceTo(&backend.device_props.name, 0)});
    std.log.info("  Compute Capability: {}.{}", .{ backend.device_props.compute_capability_major, backend.device_props.compute_capability_minor });
    std.log.info("  Memory: {} MB", .{backend.device_props.total_memory / (1024 * 1024)});
    
    // Test 5: Memory operations
    std.log.info("Test 5: Memory allocation...", .{});
    const test_size = 1024 * @sizeOf(f32);
    var buffer = try backend.allocBuffer(test_size);
    defer backend.freeBuffer(buffer);
    std.log.info("  Allocated {} bytes", .{test_size});
    
    // Test 6: Data transfer
    std.log.info("Test 6: Data transfer...", .{});
    var host_data: [256]f32 = undefined;
    for (&host_data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    
    try backend.upload(buffer.ptr, &host_data);
    
    var host_result: [256]f32 = undefined;
    try backend.download(&host_result, buffer.ptr);
    
    var all_correct = true;
    for (host_result, 0..) |val, i| {
        if (val != @as(f32, @floatFromInt(i))) {
            all_correct = false;
            break;
        }
    }
    
    if (all_correct) {
        std.log.info("  Data transfer OK!", .{});
    } else {
        std.log.err("  Data transfer FAILED!", .{});
        return error.DataTransferFailed;
    }
    
    // Test 7: Synchronization
    std.log.info("Test 7: Synchronization...", .{});
    try backend.synchronize();
    std.log.info("  Synchronization OK!", .{});
    
    std.log.info("\n=== ALL CUDA TESTS PASSED ===", .{});
}

const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda = @import("cuda.zig");

test "CUDA driver availability" {
    // Simple check - try to load driver
    var driver = cuda_driver.CudaDriver.load(std.testing.allocator) catch {
        // CUDA not available, skip test
        return;
    };
    defer driver.unload();
    
    const count = driver.getDeviceCount() catch 0;
    try std.testing.expect(count >= 0);
}

test "CUDA backend initialization" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        // Expected if CUDA not available
        if (err == error.CudaNotAvailable or
            err == error.NoCudaDevices or
            err == error.CudaDriverNotFound)
        {
            return;
        }
        return err;
    };
    defer backend.deinit();
    
    try std.testing.expect(backend.device_props.multiprocessor_count > 0);
}

test "CUDA memory allocation" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        if (err == error.CudaNotAvailable or
            err == error.NoCudaDevices or
            err == error.CudaDriverNotFound)
        {
            return;
        }
        return err;
    };
    defer backend.deinit();
    
    var buffer = try backend.allocBuffer(1024);
    defer backend.freeBuffer(buffer);
    
    try std.testing.expect(buffer.ptr != 0);
}

test "CUDA data transfer" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        if (err == error.CudaNotAvailable or
            err == error.NoCudaDevices or
            err == error.CudaDriverNotFound)
        {
            return;
        }
        return err;
    };
    defer backend.deinit();
    
    var buffer = try backend.allocBuffer(256 * @sizeOf(f32));
    defer backend.freeBuffer(buffer);
    
    var host_data: [256]f32 = undefined;
    for (&host_data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    
    try backend.upload(buffer.ptr, &host_data);
    
    var host_result: [256]f32 = undefined;
    try backend.download(&host_result, buffer.ptr);
    
    for (host_result, 0..) |val, i| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(i)), val);
    }
}

test "CUDA synchronization" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch |err| {
        if (err == error.CudaNotAvailable or
            err == error.NoCudaDevices or
            err == error.CudaDriverNotFound)
        {
            return;
        }
        return err;
    };
    defer backend.deinit();
    
    try backend.synchronize();
}

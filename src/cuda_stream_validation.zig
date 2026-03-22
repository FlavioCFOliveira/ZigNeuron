/// CUDA Stream and Synchronization Validation Suite
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda = @import("cuda.zig");

// Test 1: Stream Creation
test "cuda_stream_creation" {
    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch {
        return; // CUDA not available
    };
    defer driver.deinit();

    var device: cuda_driver.CUdevice = undefined;
    _ = driver.deviceGet.?(&device, 0);

    var context: *cuda_driver.CUcontext = undefined;
    _ = driver.ctxCreate.?(&context, 0, device);
    defer _ = driver.ctxDestroy.?(context);

    var stream: *cuda_driver.CUstream = undefined;
    const result = driver.streamCreate.?(&stream, 0);
    try std.testing.expect(result.isSuccess());

    _ = driver.streamDestroy.?(stream);
}

// Test 2: Backend Synchronization
test "cuda_backend_synchronization" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return; // CUDA not available
    };
    defer backend.deinit();

    try backend.synchronize();
}

// Test 3: Event Creation and Timing
test "cuda_event_timing" {
    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch {
        return;
    };
    defer driver.deinit();

    var device: cuda_driver.CUdevice = undefined;
    _ = driver.deviceGet.?(&device, 0);

    var context: *cuda_driver.CUcontext = undefined;
    _ = driver.ctxCreate.?(&context, 0, device);
    defer _ = driver.ctxDestroy.?(context);

    var stream: *cuda_driver.CUstream = undefined;
    _ = driver.streamCreate.?(&stream, 0);
    defer _ = driver.streamDestroy.?(stream);

    var start_event: *cuda_driver.CUevent = undefined;
    var end_event: *cuda_driver.CUevent = undefined;

    _ = driver.eventCreate.?(&start_event, 0);
    defer _ = driver.eventDestroy.?(start_event);

    _ = driver.eventCreate.?(&end_event, 0);
    defer _ = driver.eventDestroy.?(end_event);

    _ = driver.eventRecord.?(start_event, stream);
    _ = driver.eventRecord.?(end_event, stream);
    _ = driver.eventSynchronize.?(end_event);

    var elapsed_ms: f32 = 0;
    const result = driver.eventElapsedTime.?(&elapsed_ms, start_event, end_event);
    try std.testing.expect(result.isSuccess());
    try std.testing.expect(elapsed_ms >= 0);
}

// Test 4: Async Memory Operations
test "cuda_async_memory_operations" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const num_elements = 1024;
    const buffer_size = num_elements * @sizeOf(f32);

    var buffer = try backend.allocBuffer(buffer_size);
    defer backend.freeBuffer(buffer);

    const host_data = try std.testing.allocator.alloc(f32, num_elements);
    defer std.testing.allocator.free(host_data);

    for (host_data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    try backend.uploadAsync(buffer.ptr, host_data);

    const host_result = try std.testing.allocator.alloc(f32, num_elements);
    defer std.testing.allocator.free(host_result);

    try backend.downloadAsync(host_result, buffer.ptr);
    try backend.synchronize();

    for (host_data, host_result) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }
}

// Test 5: Multiple Streams
test "cuda_multiple_streams" {
    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch {
        return;
    };
    defer driver.deinit();

    var device: cuda_driver.CUdevice = undefined;
    _ = driver.deviceGet.?(&device, 0);

    var context: *cuda_driver.CUcontext = undefined;
    _ = driver.ctxCreate.?(&context, 0, device);
    defer _ = driver.ctxDestroy.?(context);

    const num_streams = 4;
    var streams: [num_streams]*cuda_driver.CUstream = undefined;

    var i: usize = 0;
    while (i < num_streams) : (i += 1) {
        const result = driver.streamCreate.?(&streams[i], 0);
        try std.testing.expect(result.isSuccess());
    }

    i = 0;
    while (i < num_streams) : (i += 1) {
        _ = driver.streamDestroy.?(streams[i]);
    }
}

// Test 6: Concurrent Execution
test "cuda_concurrent_execution" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const num_buffers = 4;
    const buffer_size = 1024 * 1024;

    var buffers: [num_buffers]cuda.DeviceBuffer = undefined;
    var host_data: [num_buffers][]f32 = undefined;

    var i: usize = 0;
    while (i < num_buffers) : (i += 1) {
        buffers[i] = try backend.allocBuffer(buffer_size);
        host_data[i] = try std.testing.allocator.alloc(f32, buffer_size / @sizeOf(f32));
        for (host_data[i], 0..) |*val, j| {
            val.* = @floatFromInt(j + i * 1000);
        }
    }

    defer {
        i = 0;
        while (i < num_buffers) : (i += 1) {
            backend.freeBuffer(buffers[i]);
            std.testing.allocator.free(host_data[i]);
        }
    }

    i = 0;
    while (i < num_buffers) : (i += 1) {
        try backend.uploadAsync(buffers[i].ptr, host_data[i]);
    }

    try backend.synchronize();

    const host_result = try std.testing.allocator.alloc(f32, buffer_size / @sizeOf(f32));
    defer std.testing.allocator.free(host_result);

    i = 0;
    while (i < num_buffers) : (i += 1) {
        try backend.download(host_result, buffers[i].ptr);
        for (host_data[i], host_result) |expected, actual| {
            if (expected != actual) {
                return error.DataMismatch;
            }
        }
    }
}

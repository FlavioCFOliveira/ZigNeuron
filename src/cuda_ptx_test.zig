/// Test for PTX kernel loading and execution
/// This file tests if PTX kernels can be loaded successfully
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");

const CUresult = cuda_driver.CUresult;
const CUdevice = cuda_driver.CUdevice;
const CUcontext = cuda_driver.CUcontext;
const CUmodule = cuda_driver.CUmodule;
const CUfunction = cuda_driver.CUfunction;
const CUdeviceptr = cuda_driver.CUdeviceptr;

/// Simple PTX kernel for testing - compatible with sm_50+
/// This is a minimal PTX that should work on most GPUs
const TEST_PTX =
    \\.version 6.5
    \\.target sm_50
    \\.address_size 64
    \\
    \\.visible .entry fill_test(
    \\    .param .u64 data,
    \\    .param .f32 value,
    \\    .param .u32 size
    \\) {
    \\    .reg .u64 %data_ptr;
    \\    .reg .u32 %size, %idx;
    \\    .reg .u32 %tid, %ctaid, %ntid;
    \\    .reg .f32 %value;
    \\    .reg .u64 %addr;
    \\    .reg .pred %p;
    \\
    \\    ld.param.u64 %data_ptr, [data];
    \\    ld.param.f32 %value, [value];
    \\    ld.param.u32 %size, [size];
    \\
    \\    mov.u32 %tid, %tid.x;
    \\    mov.u32 %ctaid, %ctaid.x;
    \\    mov.u32 %ntid, %ntid.x;
    \\    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    \\
    \\    setp.ge.u32 %p, %idx, %size;
    \\    @%p bra END;
    \\
    \\    mul.lo.u64 %addr, %idx, 4;
    \\    add.u64 %addr, %addr, %data_ptr;
    \\    st.global.f32 [%addr], %value;
    \\
    \\END:
    \\    ret;
    \\}
;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== CUDA PTX Loading Test ===", .{});

    // Initialize CUDA driver
    var driver = cuda_driver.CudaDriver.init(allocator) catch |err| {
        std.log.err("Failed to initialize CUDA driver: {}", .{err});
        return;
    };
    defer driver.deinit();

    std.log.info("CUDA driver initialized", .{});

    // Get device
    var device_count: c_int = 0;
    try cuda_driver.checkCuda(driver.deviceGetCount.?(&device_count));
    if (device_count == 0) {
        std.log.err("No CUDA devices found", .{});
        return;
    }
    std.log.info("Found {} CUDA device(s)", .{device_count});

    var device: CUdevice = 0;
    try cuda_driver.checkCuda(driver.deviceGet.?(&device, 0));

    // Get device name
    var name: [256]u8 = undefined;
    try cuda_driver.checkCuda(driver.deviceGetName.?(&name, name.len, device));
    std.log.info("Device: {s}", .{std.mem.sliceTo(&name, 0)});

    // Create context
    var context: *CUcontext = undefined;
    try cuda_driver.checkCuda(driver.ctxCreate.?(
        &context,
        @intFromEnum(cuda_driver.CUctx_flags.SCHED_AUTO),
        device,
    ));
    defer _ = driver.ctxDestroy.?(context);

    std.log.info("CUDA context created", .{});

    // Load PTX module
    var module: *CUmodule = undefined;
    std.log.info("Loading PTX module...", .{});
    std.log.info("PTX code:\n{s}", .{TEST_PTX});

    const result = driver.moduleLoadData.?(&module, TEST_PTX.ptr);
    if (result != .SUCCESS) {
        std.log.err("Failed to load PTX module: {any}", .{result});
        return error.CudaUnknownError;
    }
    defer _ = driver.moduleUnload.?(module);

    std.log.info("PTX module loaded successfully!", .{});

    // Get function
    var function: *CUfunction = undefined;
    const func_name = "fill_test";
    try cuda_driver.checkCuda(driver.moduleGetFunction.?(
        &function,
        module,
        func_name,
    ));

    std.log.info("Function '{s}' retrieved successfully!", .{func_name});

    // Allocate device memory
    const test_size: u32 = 1024;
    const buffer_size = test_size * @sizeOf(f32);

    var d_buffer: CUdeviceptr = 0;
    try cuda_driver.checkCuda(driver.memAlloc.?(&d_buffer, buffer_size));
    defer _ = driver.memFree.?(d_buffer);

    std.log.info("Allocated {} bytes on device", .{buffer_size});

    // Launch kernel
    const block_size: c_uint = 256;
    const grid_size = (test_size + block_size - 1) / block_size;

    var kernel_params = [_]?*anyopaque{
        &d_buffer,
        @ptrFromInt(@as(usize, @bitCast(@as(f32, 3.14159)))),
        &test_size,
    };

    std.log.info("Launching kernel with grid={}, block={}", .{ grid_size, block_size });

    try cuda_driver.checkCuda(driver.launchKernel.?(
        function,
        grid_size,
        1,
        1,
        block_size,
        1,
        1,
        0,
        null,
        &kernel_params,
        null,
    ));

    std.log.info("Kernel launched successfully!", .{});

    // Synchronize
    try cuda_driver.checkCuda(driver.ctxSynchronize.?());
    std.log.info("Kernel execution completed!", .{});

    // Download and verify
    var h_buffer: [test_size]f32 = undefined;
    try cuda_driver.checkCuda(driver.memcpyDtoH.?(
        &h_buffer,
        d_buffer,
        buffer_size,
    ));

    var all_correct = true;
    for (h_buffer, 0..) |val, i| {
        if (@abs(val - 3.14159) > 0.001) {
            std.log.err("Verification failed at index {}: expected 3.14159, got {}", .{ i, val });
            all_correct = false;
            break;
        }
    }

    if (all_correct) {
        std.log.info("SUCCESS: All values verified correctly!", .{});
    }
}

test "PTX loading test" {
    if (@import("builtin").os.tag == .macos) {
        return error.SkipZigTest;
    }

    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    var device: CUdevice = 0;
    try cuda_driver.checkCuda(driver.deviceGet.?(&device, 0));

    var context: *CUcontext = undefined;
    try cuda_driver.checkCuda(driver.ctxCreate.?(
        &context,
        @intFromEnum(cuda_driver.CUctx_flags.SCHED_AUTO),
        device,
    ));
    defer _ = driver.ctxDestroy.?(context);

    // Test PTX loading
    var module: *CUmodule = undefined;
    const result = driver.moduleLoadData.?(&module, TEST_PTX.ptr);
    if (result != .SUCCESS) {
        std.log.err("PTX loading failed: {any}", .{result});
        return error.CudaUnknownError;
    }
    defer _ = driver.moduleUnload.?(module);
}

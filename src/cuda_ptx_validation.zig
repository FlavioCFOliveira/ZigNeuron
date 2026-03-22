/// CUDA NVRTC Validation Test
/// Tests that CUDA C kernels can be compiled with NVRTC and executed successfully
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda_kernels = @import("cuda_kernels.zig");
const cuda_nvrtc = @import("cuda_nvrtc.zig");

const CUresult = cuda_driver.CUresult;
const CUdevice = cuda_driver.CUdevice;
const CUcontext = cuda_driver.CUcontext;
const CUmodule = cuda_driver.CUmodule;
const CUfunction = cuda_driver.CUfunction;
const CUdeviceptr = cuda_driver.CUdeviceptr;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== CUDA PTX Kernel Validation ===", .{});
    std.log.info("Testing PTX kernel loading and execution", .{});

    // Initialize CUDA driver
    var driver = cuda_driver.CudaDriver.init(allocator) catch |err| {
        std.log.err("Failed to initialize CUDA driver: {}", .{err});
        return;
    };
    defer driver.deinit();

    std.log.info("CUDA driver initialized successfully", .{});

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

    // Get compute capability
    var major: c_int = 0;
    var minor: c_int = 0;
    try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&major,
        .COMPUTE_CAPABILITY_MAJOR, device));
    try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&minor,
        .COMPUTE_CAPABILITY_MINOR, device));
    std.log.info("Compute Capability: {}.{}", .{ major, minor });

    // Create context
    var context: *CUcontext = undefined;
    try cuda_driver.checkCuda(driver.ctxCreate.?(
        &context,
        @intFromEnum(cuda_driver.CUctx_flags.SCHED_AUTO),
        device,
    ));
    defer _ = driver.ctxDestroy.?(context);

    std.log.info("CUDA context created", .{});

    // Test kernel loading
    const test_results = try testKernels(&driver, allocator);

    // Print summary
    std.log.info("", .{});
    std.log.info("=== Test Summary ===", .{});
    std.log.info("Total kernels tested: {}", .{test_results.total});
    std.log.info("Passed: {}", .{test_results.passed});
    std.log.info("Failed: {}", .{test_results.failed});

    if (test_results.failed == 0) {
        std.log.info("SUCCESS: All PTX kernels loaded successfully!", .{});
    } else {
        std.log.err("FAILURE: Some kernels failed to load", .{});
    }
}

const TestResults = struct {
    total: usize,
    passed: usize,
    failed: usize,
};

fn testKernels(driver: *cuda_driver.CudaDriver, allocator: std.mem.Allocator) !TestResults {
    const KernelTest = struct {
        name: []const u8,
        source: []const u8,
    };

    const kernels = [_]KernelTest{
        .{ .name = "fill_constant", .source = cuda_kernels.FILL_CONSTANT_SOURCE },
        .{ .name = "relu_forward", .source = cuda_kernels.RELU_FORWARD_SOURCE },
        .{ .name = "sigmoid_forward", .source = cuda_kernels.SIGMOID_FORWARD_SOURCE },
        .{ .name = "tanh_forward", .source = cuda_kernels.TANH_FORWARD_SOURCE },
        .{ .name = "ew_add", .source = cuda_kernels.EW_ADD_SOURCE },
        .{ .name = "ew_mul", .source = cuda_kernels.EW_MUL_SOURCE },
        .{ .name = "scale_buffer", .source = cuda_kernels.SCALE_BUFFER_SOURCE },
        .{ .name = "sgd_update", .source = cuda_kernels.SGD_UPDATE_SOURCE },
        .{ .name = "add_bias", .source = cuda_kernels.ADD_BIAS_SOURCE },
        .{ .name = "matmul", .source = cuda_kernels.MATMUL_SIMPLE_SOURCE },
        .{ .name = "matmul_batch", .source = cuda_kernels.MATMUL_BATCHED_SOURCE },
        .{ .name = "matmul_transpose_b", .source = cuda_kernels.MATMUL_TRANSPOSE_B_SOURCE },
    };

    var passed: usize = 0;
    var failed: usize = 0;

    std.log.info("", .{});
    std.log.info("=== Loading Kernels ===", .{});

    for (kernels) |kernel| {
        std.log.info("Testing kernel: {s}", .{kernel.name});

        // Compile CUDA C source to PTX using NVRTC
        // FIX: Use compute_80 for PTX 8.0 compatibility with driver 535+
        const ptx = cuda_nvrtc.compileKernel(allocator, kernel.source, kernel.name, "compute_80", null) catch |err| {
            std.log.err("  FAILED to compile '{s}': {}", .{ kernel.name, err });
            failed += 1;
            continue;
        };
        defer allocator.free(ptx);

        // Load module from compiled PTX
        var module: *CUmodule = undefined;
        const result = driver.moduleLoadData.?(&module, ptx.ptr);

        if (result != .SUCCESS) {
            std.log.err("  FAILED to load '{s}': {any}", .{ kernel.name, result });
            failed += 1;
            continue;
        }

        // Try to get function
        var function: *CUfunction = undefined;
        const func_name_z = try std.heap.page_allocator.dupeZ(u8, kernel.name);
        defer std.heap.page_allocator.free(func_name_z);
        const func_result = driver.moduleGetFunction.?(
            &function,
            module,
            func_name_z,
        );

        if (func_result != .SUCCESS) {
            std.log.err("  FAILED to get function '{s}': {any}", .{ kernel.name, func_result });
            _ = driver.moduleUnload.?(module);
            failed += 1;
            continue;
        }

        // Try execution if it's fill_constant (simplest kernel)
        if (std.mem.eql(u8, kernel.name, "fill_constant")) {
            const exec_result = testFillConstant(driver, function);
            if (exec_result) {
                std.log.info("  PASSED: '{s}' loaded and executed", .{kernel.name});
                passed += 1;
            } else |err| {
                std.log.err("  FAILED execution '{s}': {}", .{ kernel.name, err });
                failed += 1;
            }
        } else {
            std.log.info("  PASSED: '{s}' loaded successfully", .{kernel.name});
            passed += 1;
        }

        _ = driver.moduleUnload.?(module);
    }

    return TestResults{
        .total = kernels.len,
        .passed = passed,
        .failed = failed,
    };
}

fn testFillConstant(driver: *cuda_driver.CudaDriver, function: *CUfunction) !void {
    // Allocate device memory
    const test_size: u32 = 256;
    const buffer_size = test_size * @sizeOf(f32);

    var d_buffer: CUdeviceptr = 0;
    try cuda_driver.checkCuda(driver.memAlloc.?(&d_buffer, buffer_size));
    defer _ = driver.memFree.?(d_buffer);

    // Launch kernel
    const block_size: c_uint = 256;
    const grid_size = (test_size + block_size - 1) / block_size;

    const value: f32 = 3.14159;
    var kernel_params = [_]?*anyopaque{
        @ptrCast(&d_buffer),
        @constCast(@ptrCast(&value)),
        @constCast(@ptrCast(&test_size)),
    };

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

    // Synchronize
    try cuda_driver.checkCuda(driver.ctxSynchronize.?());

    // Download and verify
    var h_buffer: [test_size]f32 = undefined;
    try cuda_driver.checkCuda(driver.memcpyDtoH.?(
        &h_buffer,
        d_buffer,
        buffer_size,
    ));

    // Verify results
    for (h_buffer, 0..) |val, i| {
        if (@abs(val - value) > 0.001) {
            std.log.err("Verification failed at index {}: expected {}, got {}", .{ i, value, val });
            return error.VerificationFailed;
        }
    }
}

test "PTX kernel loading" {
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

    // Test loading fill_constant kernel
    var module: *CUmodule = undefined;
    const result = driver.moduleLoadData.?(&module, cuda_kernels.FILL_CONSTANT_PTX);
    if (result != .SUCCESS) {
        return error.CudaUnknownError;
    }
    defer _ = driver.moduleUnload.?(module);

    var function: *CUfunction = undefined;
    try cuda_driver.checkCuda(driver.moduleGetFunction.?(
        &function,
        module,
        "fill_constant",
    ));
}

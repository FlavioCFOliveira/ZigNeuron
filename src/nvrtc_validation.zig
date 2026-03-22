/// NVRTC Validation Test
/// Tests runtime CUDA kernel compilation using NVRTC
const std = @import("std");
const cuda_driver = @import("cuda_driver");
const cuda_context = @import("cuda_context");
const cuda_nvrtc = @import("cuda_nvrtc");
const cuda_kernels = @import("cuda_kernels");

const NVRTC = cuda_nvrtc.NVRTC;

pub fn main() !void {
    std.log.info("NVRTC Validation Test", .{});
    std.log.info("===================", .{});

    // Check if NVRTC is available
    if (!NVRTC.isAvailable()) {
        std.log.warn("NVRTC not available on this system", .{});
        std.log.info("This is expected if CUDA is not installed", .{});
        return;
    }

    const version = NVRTC.getVersion();
    std.log.info("NVRTC version: {}.{}", .{ version.major, version.minor });

    // Test 1: Compile a simple kernel
    std.log.info("\nTest 1: Simple kernel compilation", .{});
    try testSimpleKernel();

    // Test 2: Compile matrix multiplication kernel
    std.log.info("\nTest 2: Matrix multiplication kernel", .{});
    try testMatmulKernel();

    // Test 3: Compile activation kernels
    std.log.info("\nTest 3: Activation kernels", .{});
    try testActivationKernels();

    std.log.info("\n===================", .{});
    std.log.info("All NVRTC tests passed!", .{});
}

fn testSimpleKernel() !void {
    const test_source =
        \\extern "C" __global__ void test_kernel(float* data, int n) {
        \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (idx < n) {
        \\        data[idx] = data[idx] * 2.0f;
        \\    }
        \\}
    ;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // FIX: Use virtual architecture compute_80 for PTX 8.0 compatibility
    // Virtual architectures generate PTX (portable), real architectures generate CUBIN
    // compute_80 generates PTX 8.0 compatible with driver 535+ (CUDA 12.2+)
    const ptx = NVRTC.compileKernelSimple(
        allocator,
        test_source,
        "test_kernel",
        "compute_80",
    ) catch |err| {
        std.log.err("Failed to compile simple kernel: {}", .{err});
        return err;
    };
    defer allocator.free(ptx);

    std.log.info("  Compiled successfully (PTX: {} bytes)", .{ptx.len});

    // Verify PTX is not empty
    if (ptx.len == 0) {
        return error.EmptyPTX;
    }
}

fn testMatmulKernel() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const ptx = NVRTC.compileKernelSimple(
        allocator,
        cuda_kernels.MATMUL_SIMPLE_SOURCE,
        "matmul",
        "compute_80",
    ) catch |err| {
        std.log.err("Failed to compile matmul kernel: {}", .{err});
        return err;
    };
    defer allocator.free(ptx);

    std.log.info("  matmul compiled successfully (PTX: {} bytes)", .{ptx.len});

    // Verify PTX contains expected content
    const ptx_str = std.mem.span(ptx.ptr);
    if (std.mem.indexOf(u8, ptx_str, ".version") == null) {
        std.log.warn("  PTX may be invalid: missing .version directive", .{});
    }
}

fn testActivationKernels() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const kernels = [_]struct { name: []const u8, source: []const u8 }{
        .{ .name = "relu_forward", .source = cuda_kernels.RELU_FORWARD_SOURCE },
        .{ .name = "sigmoid_forward", .source = cuda_kernels.SIGMOID_FORWARD_SOURCE },
        .{ .name = "tanh_forward", .source = cuda_kernels.TANH_FORWARD_SOURCE },
    };

    for (kernels) |kernel| {
        const ptx = NVRTC.compileKernelSimple(
            allocator,
            kernel.source,
            kernel.name,
            "compute_80",
        ) catch |err| {
            std.log.err("Failed to compile {s}: {}", .{ kernel.name, err });
            return err;
        };
        defer allocator.free(ptx);

        std.log.info("  {s} compiled (PTX: {} bytes)", .{ kernel.name, ptx.len });
    }
}

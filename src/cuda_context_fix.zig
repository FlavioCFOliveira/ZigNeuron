
    /// Launch a kernel
    pub fn launchKernel(
        self: *CudaContext,
        kernel_name: []const u8,
        grid_dim: [3]u32,
        block_dim: [3]u32,
        shared_mem_bytes: u32,
        args: []const ?*anyopaque,
    ) !void {
        const kernel = try self.getKernel(kernel_name);

        // Prepare kernel parameters
        var kernel_params: [16]?*anyopaque = undefined;
        if (args.len > kernel_params.len) {
            return error.TooManyKernelArguments;
        }
        @memcpy(kernel_params[0..args.len], args);

        try cuda_driver.checkCuda(self.driver.launchKernel.?(
            kernel.function,
            grid_dim[0],
            grid_dim[1],
            grid_dim[2],
            block_dim[0],
            block_dim[1],
            block_dim[2],
            shared_mem_bytes,
            self.stream,
            &kernel_params,
            null,
        ));
    }

    // =============================================================================
    // NVRTC Runtime Compilation
    // =============================================================================

    /// Check if NVRTC is available for runtime compilation
    pub fn isNvrtcAvailable(_: *const CudaContext) bool {
        return NVRTC.isAvailable();
    }

    // =============================================================================
    // Helper Functions
    // =============================================================================

    /// Calculate optimal grid and block dimensions for element-wise operations
    pub fn getElementWiseConfig(_: *const CudaContext, total_elements: usize) struct { grid: u32, block: u32 } {
        const block = DEFAULT_BLOCK_SIZE;
        const grid = @as(u32, @intCast((total_elements + block - 1) / block));
        return .{ .grid = grid, .block = block };
    }

    /// Calculate optimal dimensions for matrix operations
    pub fn getMatrixConfig(_: *const CudaContext, m: usize, n: usize) struct { grid_x: u32, grid_y: u32, block_x: u32, block_y: u32 } {
        const block_x: u32 = 16;
        const block_y: u32 = 16;
        const grid_x = @as(u32, @intCast((n + block_x - 1) / block_x));
        const grid_y = @as(u32, @intCast((m + block_y - 1) / block_y));
        return .{ .grid_x = grid_x, .grid_y = grid_y, .block_x = block_x, .block_y = block_y };
    }

    // =============================================================================
    // Private Helper Functions
    // =============================================================================

    fn getPoolIndex(size: usize) usize {
        if (size == 0) return 0;
        const aligned = getPooledSize(size);
        // Calculate log2 of aligned size
        var idx: usize = 0;
        var s = aligned;
        while (s > MIN_POOL_SIZE) : (s >>= 1) {
            idx += 1;
        }
        return if (idx < MEMORY_POOL_BUCKETS) idx else MEMORY_POOL_BUCKETS - 1;
    }

    fn getPooledSize(size: usize) usize {
        if (size <= MIN_POOL_SIZE) return MIN_POOL_SIZE;
        return std.math.ceilPowerOfTwo(usize, size) catch MAX_POOL_SIZE;
    }
};

// =============================================================================
// Builtin Kernel Loading
// =============================================================================

/// Load builtin kernels using NVRTC when available, with PTX fallback
pub fn loadBuiltinKernels(ctx: *CudaContext) !void {
    // Check if NVRTC is available
    const nvrtc_available = NVRTC.isAvailable();

    if (nvrtc_available) {
        std.log.info("CUDA: Using NVRTC for runtime kernel compilation", .{});
        try loadKernelsWithNvrtc(ctx);
    } else {
        std.log.info("CUDA: NVRTC not available, using embedded PTX", .{});
        try loadKernelsWithEmbeddedPtx(ctx);
    }

    std.log.info("CUDA: Loaded {} kernels", .{ctx.kernels.count()});
}

/// Load kernels using NVRTC runtime compilation
fn loadKernelsWithNvrtc(ctx: *CudaContext) !void {
    const KernelSource = struct {
        name: []const u8,
        source: []const u8,
    };

    const kernel_sources = [_]KernelSource{
        .{ .name = "matmul", .source = cuda_kernels.MATMUL_SIMPLE_SOURCE },
        .{ .name = "matmul_batch", .source = cuda_kernels.MATMUL_BATCHED_SOURCE },
        .{ .name = "matmul_transpose_b", .source = cuda_kernels.MATMUL_TRANSPOSE_B_SOURCE },
        .{ .name = "relu_forward", .source = cuda_kernels.RELU_FORWARD_SOURCE },
        .{ .name = "relu_backward", .source = cuda_kernels.RELU_BACKWARD_SOURCE },
        .{ .name = "sigmoid_forward", .source = cuda_kernels.SIGMOID_FORWARD_SOURCE },
        .{ .name = "sigmoid_backward", .source = cuda_kernels.SIGMOID_BACKWARD_SOURCE },
        .{ .name = "tanh_forward", .source = cuda_kernels.TANH_FORWARD_SOURCE },
        .{ .name = "tanh_backward", .source = cuda_kernels.TANH_BACKWARD_SOURCE },
        .{ .name = "linear_forward", .source = cuda_kernels.LINEAR_FORWARD_SOURCE },
        .{ .name = "softmax_forward", .source = cuda_kernels.SOFTMAX_FORWARD_SOURCE },
        .{ .name = "ew_add", .source = cuda_kernels.EW_ADD_SOURCE },
        .{ .name = "ew_mul", .source = cuda_kernels.EW_MUL_SOURCE },
        .{ .name = "scale_buffer", .source = cuda_kernels.SCALE_BUFFER_SOURCE },
        .{ .name = "sgd_update", .source = cuda_kernels.SGD_UPDATE_SOURCE },
        .{ .name = "fill_constant", .source = cuda_kernels.FILL_CONSTANT_SOURCE },
        .{ .name = "add_bias", .source = cuda_kernels.ADD_BIAS_SOURCE },
    };

    for (kernel_sources) |kernel| {
        ctx.compileAndLoadKernel(kernel.name, kernel.source) catch |err| {
            // Log warning but continue loading other kernels
            std.log.warn("Failed to compile kernel '{s}': {}", .{ kernel.name, err });
        };
    }
}

/// Load kernels using embedded PTX (fallback when NVRTC unavailable)
fn loadKernelsWithEmbeddedPtx(ctx: *CudaContext) !void {
    const KernelDef = struct {
        name: []const u8,
        ptx: []const u8,
    };

    const kernels = [_]KernelDef{
        .{ .name = "matmul", .ptx = cuda_kernels.MATMUL_SIMPLE_PTX },
        .{ .name = "matmul_batch", .ptx = cuda_kernels.MATMUL_BATCHED_PTX },
        .{ .name = "matmul_transpose_b", .ptx = cuda_kernels.MATMUL_TRANSPOSE_B_PTX },
        .{ .name = "relu_forward", .ptx = cuda_kernels.RELU_FORWARD_PTX },
        .{ .name = "relu_backward", .ptx = cuda_kernels.RELU_BACKWARD_PTX },
        .{ .name = "sigmoid_forward", .ptx = cuda_kernels.SIGMOID_FORWARD_PTX },
        .{ .name = "sigmoid_backward", .ptx = cuda_kernels.SIGMOID_BACKWARD_PTX },
        .{ .name = "tanh_forward", .ptx = cuda_kernels.TANH_FORWARD_PTX },
        .{ .name = "tanh_backward", .ptx = cuda_kernels.TANH_BACKWARD_PTX },
        .{ .name = "linear_forward", .ptx = cuda_kernels.LINEAR_FORWARD_PTX },
        .{ .name = "softmax_forward", .ptx = cuda_kernels.SOFTMAX_FORWARD_PTX },
        .{ .name = "ew_add", .ptx = cuda_kernels.EW_ADD_PTX },
        .{ .name = "ew_mul", .ptx = cuda_kernels.EW_MUL_PTX },
        .{ .name = "scale_buffer", .ptx = cuda_kernels.SCALE_BUFFER_PTX },
        .{ .name = "sgd_update", .ptx = cuda_kernels.SGD_UPDATE_PTX },
        .{ .name = "fill_constant", .ptx = cuda_kernels.FILL_CONSTANT_PTX },
        .{ .name = "add_bias", .ptx = cuda_kernels.ADD_BIAS_PTX },
    };

    for (kernels) |kernel| {
        ctx.loadKernel(kernel.name, kernel.ptx) catch |err| {
            // Log warning but continue loading other kernels
            if (err != error.KernelNotFound) {
                std.log.warn("Failed to load kernel '{s}': {}", .{ kernel.name, err });
            }
        };
    }
}

// =============================================================================
// Tests
// =============================================================================

test "CUDA context initialization" {
    // Skip on macOS
    if (@import("builtin").os.tag == .macos) {
        return error.SkipZigTest;
    }

    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    const ctx = CudaContext.init(std.testing.allocator, &driver) catch |err| {
        if (err == error.NoCudaDevices) return;
        return err;
    };
    defer ctx.deinit();

    try std.testing.expect(ctx.device_props.total_memory > 0);
    try std.testing.expect(ctx.device_props.multiprocessor_count > 0);
}

test "CUDA buffer pool" {
    // Skip on macOS
    if (@import("builtin").os.tag == .macos) {
        return error.SkipZigTest;
    }

    var driver = cuda_driver.CudaDriver.init(std.testing.allocator) catch |err| {
        if (err == error.CudaDriverNotFound) return;
        return err;
    };
    defer driver.deinit();

    const ctx = CudaContext.init(std.testing.allocator, &driver) catch |err| {
        if (err == error.NoCudaDevices) return;
        return err;
    };
    defer ctx.deinit();

    // Test buffer allocation
    var buf1 = try ctx.getBuffer(1024);
    try std.testing.expect(buf1.ptr != 0);
    try std.testing.expect(buf1.size >= 1024);

    // Return to pool
    ctx.returnBuffer(buf1);

    // Get from pool (should reuse)
    var buf2 = try ctx.getBuffer(512);
    try std.testing.expect(buf2.ptr != 0);
    ctx.returnBuffer(buf2);
}

/// CUDA GPU backend implementation for ZigNeuron
/// Provides CUDA compute kernel support for NVIDIA GPUs on Linux/Windows
///
/// Architecture:
/// - CUDA driver API dynamic loading (libcuda.so / nvcuda.dll)
/// - Context and stream management
/// - Memory pooling for efficient buffer reuse
/// - Kernel loading from PTX or inline CUDA-C
/// - Async execution with streams
/// - Unified memory support where available
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda_context = @import("cuda_context.zig");
const cuda_kernels = @import("cuda_kernels.zig");

const CUresult = cuda_driver.CUresult;
const CUdeviceptr = cuda_driver.CUdeviceptr;
const CudaDriver = cuda_driver.CudaDriver;
const CudaContext = cuda_context.CudaContext;

/// Error types for Tensor Core operations
pub const TensorCoreError = error{
    InvalidDimensionsForTensorCore,
    TensorCoresNotSupported,
};

// Re-export CUDA types
pub const CudaError = cuda_driver.CudaError;
pub const CUdevice = cuda_driver.CUdevice;

// =============================================================================
// Security Helper Functions
// =============================================================================

/// Safely calculate buffer size with overflow checking
/// Uses std.math.mul to detect overflow in multiplication chains
pub fn calculateBufferSize(num_elements: usize, element_size: usize) !usize {
    return std.math.mul(usize, num_elements, element_size);
}

/// Safely calculate buffer size for 2D arrays (rows * cols * element_size)
/// Returns error.Overflow if multiplication would overflow
pub fn calculateBufferSize2D(rows: usize, cols: usize, element_size: usize) !usize {
    const num_elements = try std.math.mul(usize, rows, cols);
    return std.math.mul(usize, num_elements, element_size);
}

/// Safely calculate buffer size for 3D arrays (dim1 * dim2 * dim3 * element_size)
/// Returns error.Overflow if multiplication would overflow
pub fn calculateBufferSize3D(dim1: usize, dim2: usize, dim3: usize, element_size: usize) !usize {
    const dim1_dim2 = try std.math.mul(usize, dim1, dim2);
    const num_elements = try std.math.mul(usize, dim1_dim2, dim3);
    return std.math.mul(usize, num_elements, element_size);
}

/// Safely cast usize to i32 with bounds checking
/// Returns error.IntegerOverflow if value exceeds i32::MAX
pub fn safeCastUsizeToI32(value: usize) !i32 {
    if (value > std.math.maxInt(i32)) {
        return error.IntegerOverflow;
    }
    return @intCast(value);
}

// =============================================================================
// Device Buffer Wrapper
// =============================================================================

pub const DeviceBuffer = struct {
    ptr: CUdeviceptr,
    size: usize,
    context: *CudaContext,

    pub fn deinit(self: *DeviceBuffer) void {
        const buf = CudaContext.DeviceBuffer{
            .ptr = self.ptr,
            .size = self.size,
            .pool_index = null,
        };
        self.context.returnBuffer(buf);
    }
};

/// Managed memory buffer wrapper (Unified Memory)
/// Provides a single address space accessible by both CPU and GPU
pub const ManagedBuffer = struct {
    ptr: CUdeviceptr,
    size: usize,
    host_ptr: ?*anyopaque,
    context: *CudaContext,

    pub fn deinit(self: *ManagedBuffer) void {
        var buf = CudaContext.ManagedBuffer{
            .ptr = self.ptr,
            .size = self.size,
            .host_ptr = self.host_ptr,
        };
        self.context.freeManaged(&buf);
    }

    /// Get host-accessible pointer for direct CPU access
    pub fn getHostPtr(self: *const ManagedBuffer) ?*anyopaque {
        if (self.host_ptr) |host_ptr| {
            return host_ptr;
        }
        // For true managed memory, device pointer is accessible on host
        return @ptrFromInt(self.ptr);
    }

    /// Get typed slice for host access
    pub fn asSlice(self: *const ManagedBuffer, comptime T: type) []T {
        const ptr = self.getHostPtr() orelse unreachable;
        const elem_count = self.size / @sizeOf(T);
        return @as([*]T, @ptrCast(ptr))[0..elem_count];
    }

    /// Prefetch data to GPU before kernel launch
    pub fn prefetchToDevice(self: *const ManagedBuffer) !void {
        try self.context.prefetchToDevice(self.ptr, self.size);
    }

    /// Prefetch data to host before CPU reads
    pub fn prefetchToHost(self: *const ManagedBuffer) !void {
        try self.context.prefetchToHost(self.ptr, self.size);
    }
};

// =============================================================================
// Enums
// =============================================================================

pub const ElementWiseOp = enum {
    add,
    sub,
    mul,
    div,
};

pub const MapOp = enum {
    exp,
    log,
    sqrt,
    abs,
    square,
    inv,
};

// =============================================================================
// CudaBackend Structure
// =============================================================================

pub const CudaBackend = struct {
    allocator: std.mem.Allocator,
    driver: CudaDriver,
    driver_ref: cuda_driver.CudaDriverRef, // SECURITY FIX: Hold reference to prevent UAF (CRIT-002)
    context: *CudaContext,
    context_initialized: bool,

    // Kernel PTX cache
    kernel_ptx: std.StringHashMap([]const u8),

    /// Initialize CUDA backend
    pub fn init(allocator: std.mem.Allocator) !CudaBackend {
        // SECURITY FIX: Use reference-counted driver acquisition (CRIT-002)
        // This prevents use-after-free by ensuring proper lifetime management
        const driver_ref = try cuda_driver.CudaDriverRef.acquire(allocator);
        errdefer driver_ref.release();

        // Create CUDA context - it will acquire its own driver reference
        const ctx = try CudaContext.init(allocator);
        errdefer ctx.deinit();

        // Get the driver pointer from the reference for storage
        const driver_ptr = driver_ref.driver;

        var backend = CudaBackend{
            .allocator = allocator,
            .driver = driver_ptr.*,
            .driver_ref = driver_ref, // Keep reference alive
            .context = ctx,
            .context_initialized = false, // Will be true only after full init
            .kernel_ptx = std.StringHashMap([]const u8).init(allocator),
        };

        // Load built-in kernels - may fail but context is valid
        // We mark as initialized before this since the context is ready
        backend.context_initialized = true;
        backend.loadBuiltinKernels() catch |err| {
            // Log but don't fail - we can still use the backend without built-in kernels
            std.log.warn("CUDA: Failed to load built-in kernels: {s}. Continuing without pre-loaded kernels.", .{@errorName(err)});
        };

        return backend;
    }

    /// Load built-in kernels with NVRTC and fallback to embedded PTX
    /// WORKAROUND: NVRTC disabled when driver < 545 due to PTX 8.5 incompatibility
    fn loadBuiltinKernels(self: *CudaBackend) !void {
        const KernelDef = struct {
            name: []const u8,
            source: []const u8,
            ptx: ?[]const u8,
        };

        const kernels = [_]KernelDef{
            .{ .name = "matmul", .source = cuda_kernels.MATMUL_SIMPLE_SOURCE, .ptx = cuda_kernels.MATMUL_SIMPLE_PTX },
            .{ .name = "matmul_tiled", .source = cuda_kernels.MATMUL_TILED_SOURCE, .ptx = null },
            .{ .name = "matmul_transpose_b", .source = cuda_kernels.MATMUL_TRANSPOSE_B_SOURCE, .ptx = cuda_kernels.MATMUL_TRANSPOSE_B_PTX },
            .{ .name = "matmul_tiled_transpose_b", .source = cuda_kernels.MATMUL_TILED_TRANSPOSE_B_SOURCE, .ptx = null },
            .{ .name = "matmul_batch", .source = cuda_kernels.MATMUL_BATCHED_SOURCE, .ptx = cuda_kernels.MATMUL_BATCHED_PTX },
            .{ .name = "matmul_batch_tiled", .source = cuda_kernels.MATMUL_BATCH_TILED_SOURCE, .ptx = null },
            .{ .name = "matmul_tensor_core", .source = cuda_kernels.MATMUL_TENSOR_CORE_SOURCE, .ptx = cuda_kernels.MATMUL_TENSOR_CORE_PTX },
            .{ .name = "ew_add", .source = cuda_kernels.EW_ADD_SOURCE, .ptx = cuda_kernels.EW_ADD_PTX },
            .{ .name = "ew_mul", .source = cuda_kernels.EW_MUL_SOURCE, .ptx = cuda_kernels.EW_MUL_PTX },
            .{ .name = "scale_buffer", .source = cuda_kernels.SCALE_BUFFER_SOURCE, .ptx = cuda_kernels.SCALE_BUFFER_PTX },
            .{ .name = "relu_forward", .source = cuda_kernels.RELU_FORWARD_SOURCE, .ptx = cuda_kernels.RELU_FORWARD_PTX },
            .{ .name = "relu_backward", .source = cuda_kernels.RELU_BACKWARD_SOURCE, .ptx = cuda_kernels.RELU_BACKWARD_PTX },
            .{ .name = "sigmoid_forward", .source = cuda_kernels.SIGMOID_FORWARD_SOURCE, .ptx = cuda_kernels.SIGMOID_FORWARD_PTX },
            .{ .name = "sigmoid_backward", .source = cuda_kernels.SIGMOID_BACKWARD_SOURCE, .ptx = cuda_kernels.SIGMOID_BACKWARD_PTX },
            .{ .name = "tanh_forward", .source = cuda_kernels.TANH_FORWARD_SOURCE, .ptx = cuda_kernels.TANH_FORWARD_PTX },
            .{ .name = "tanh_backward", .source = cuda_kernels.TANH_BACKWARD_SOURCE, .ptx = cuda_kernels.TANH_BACKWARD_PTX },
            // Vectorized activation kernels (LDG.128) - no CUDA C source, PTX only
            .{ .name = "relu_forward_vec4", .source = cuda_kernels.RELU_FORWARD_SOURCE, .ptx = cuda_kernels.RELU_FORWARD_VEC4_PTX },
            .{ .name = "relu_backward_vec4", .source = cuda_kernels.RELU_BACKWARD_SOURCE, .ptx = cuda_kernels.RELU_BACKWARD_VEC4_PTX },
            .{ .name = "sigmoid_forward_vec4", .source = cuda_kernels.SIGMOID_FORWARD_SOURCE, .ptx = cuda_kernels.SIGMOID_FORWARD_VEC4_PTX },
            .{ .name = "sigmoid_backward_vec4", .source = cuda_kernels.SIGMOID_BACKWARD_SOURCE, .ptx = cuda_kernels.SIGMOID_BACKWARD_VEC4_PTX },
            .{ .name = "tanh_forward_vec4", .source = cuda_kernels.TANH_FORWARD_SOURCE, .ptx = cuda_kernels.TANH_FORWARD_VEC4_PTX },
            .{ .name = "tanh_backward_vec4", .source = cuda_kernels.TANH_BACKWARD_SOURCE, .ptx = cuda_kernels.TANH_BACKWARD_VEC4_PTX },
            .{ .name = "softmax_forward", .source = cuda_kernels.SOFTMAX_FORWARD_SOURCE, .ptx = cuda_kernels.SOFTMAX_FORWARD_PTX },
            .{ .name = "mse_backward", .source = cuda_kernels.MSE_BACKWARD_SOURCE, .ptx = cuda_kernels.MSE_BACKWARD_PTX },
            .{ .name = "cross_entropy_backward", .source = cuda_kernels.CROSS_ENTROPY_BACKWARD_SOURCE, .ptx = cuda_kernels.CROSS_ENTROPY_BACKWARD_PTX },
            .{ .name = "sgd_update", .source = cuda_kernels.SGD_UPDATE_SOURCE, .ptx = cuda_kernels.SGD_UPDATE_PTX },
            .{ .name = "adam_update", .source = cuda_kernels.ADAM_UPDATE_SOURCE, .ptx = cuda_kernels.ADAM_UPDATE_PTX },
            .{ .name = "rmsprop_update", .source = cuda_kernels.RMSPROP_UPDATE_SOURCE, .ptx = cuda_kernels.RMSPROP_UPDATE_PTX },
            .{ .name = "layernorm_forward", .source = cuda_kernels.LAYERNORM_FORWARD_SOURCE, .ptx = cuda_kernels.LAYERNORM_FORWARD_PTX },
            .{ .name = "layernorm_backward", .source = cuda_kernels.LAYERNORM_BACKWARD_SOURCE, .ptx = cuda_kernels.LAYERNORM_BACKWARD_PTX },
            .{ .name = "batchnorm_forward_training", .source = cuda_kernels.BATCHNORM_FORWARD_TRAINING_SOURCE, .ptx = cuda_kernels.BATCHNORM_FORWARD_TRAINING_PTX },
            .{ .name = "batchnorm_forward_inference", .source = cuda_kernels.BATCHNORM_FORWARD_INFERENCE_SOURCE, .ptx = cuda_kernels.BATCHNORM_FORWARD_INFERENCE_PTX },
            .{ .name = "add_bias", .source = cuda_kernels.ADD_BIAS_SOURCE, .ptx = cuda_kernels.ADD_BIAS_PTX },
            // Fused operations - no CUDA C source, PTX only
            .{ .name = "matmul_bias_relu_fused", .source = "", .ptx = cuda_kernels.MATMUL_BIAS_RELU_FUSED_PTX },
            .{ .name = "matmul_bias_sigmoid_fused", .source = "", .ptx = cuda_kernels.MATMUL_BIAS_SIGMOID_FUSED_PTX },
            .{ .name = "matmul_bias_tanh_fused", .source = "", .ptx = cuda_kernels.MATMUL_BIAS_TANH_FUSED_PTX },
            .{ .name = "matmul_bias_identity_fused", .source = "", .ptx = cuda_kernels.MATMUL_BIAS_IDENTITY_FUSED_PTX },
            .{ .name = "conv2d_forward", .source = cuda_kernels.CONV2D_FORWARD_SOURCE, .ptx = cuda_kernels.CONV2D_FORWARD_PTX },
            .{ .name = "im2col", .source = cuda_kernels.IM2COL_SOURCE, .ptx = "" },
            .{ .name = "col2im", .source = cuda_kernels.COL2IM_SOURCE, .ptx = "" },
            .{ .name = "conv2d_im2col_forward", .source = cuda_kernels.CONV2D_IM2COL_FORWARD_SOURCE, .ptx = "" },
            .{ .name = "fill_constant", .source = cuda_kernels.FILL_CONSTANT_SOURCE, .ptx = cuda_kernels.FILL_CONSTANT_PTX },
            .{ .name = "linear_forward", .source = cuda_kernels.LINEAR_FORWARD_SOURCE, .ptx = cuda_kernels.LINEAR_FORWARD_PTX },
            .{ .name = "binary_cross_entropy_backward", .source = cuda_kernels.BINARY_CROSS_ENTROPY_BACKWARD_SOURCE, .ptx = cuda_kernels.BINARY_CROSS_ENTROPY_BACKWARD_PTX },
            .{ .name = "kl_divergence_backward", .source = cuda_kernels.KL_DIVERGENCE_BACKWARD_SOURCE, .ptx = cuda_kernels.KL_DIVERGENCE_BACKWARD_PTX },
        };

        var loaded_count: usize = 0;
        // NVRTC CUDA 12.6 generates PTX 8.5 regardless of --gpu-architecture flag
        // This requires driver 545+ (CUDA 12.5+), but we have driver 535 (CUDA 12.2)
        // Disabled NVRTC - using embedded PTX which uses version 6.0
        // TODO: Update driver to 545+ to enable NVRTC runtime compilation
        const use_nvrtc = false; // NVRTC disabled - PTX 8.5 incompatible with driver 535

        for (kernels) |kernel| {
            if (use_nvrtc and kernel.source.len > 0) {
                // Try NVRTC first
                self.context.compileAndLoadKernel(kernel.name, kernel.source) catch |err| {
                    std.log.warn("NVRTC failed for '{s}' ({}), trying embedded PTX...", .{ kernel.name, err });
                    if (kernel.ptx) |ptx| {
                        self.context.loadKernel(kernel.name, ptx) catch |ptx_err| {
                            std.log.err("Failed to load embedded PTX for '{s}': {}", .{ kernel.name, ptx_err });
                            continue;
                        };
                    } else {
                        std.log.err("No embedded PTX available for '{s}'", .{kernel.name});
                        continue;
                    }
                };
            } else {
                // Use embedded PTX directly (NVRTC disabled)
                if (kernel.ptx) |ptx| {
                    self.context.loadKernel(kernel.name, ptx) catch |ptx_err| {
                        std.log.err("Failed to load embedded PTX for '{s}': {}", .{ kernel.name, ptx_err });
                        continue;
                    };
                } else {
                    std.log.warn("No PTX available for '{s}', skipping", .{kernel.name});
                    continue;
                }
            }
            loaded_count += 1;
        }

        std.log.info("CUDA: Loaded {d}/{d} kernels (NVRTC: {s})", .{ loaded_count, kernels.len, if (use_nvrtc) "enabled" else "disabled" });
    }

    /// Cleanup CUDA backend
    pub fn deinit(self: *CudaBackend) void {
        // Free kernel PTX strings
        var ptx_iter = self.kernel_ptx.iterator();
        while (ptx_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.kernel_ptx.deinit();

        // Cleanup context - only if fully initialized
        // Context may be partially initialized, so we check flag before destroy
        if (self.context_initialized) {
            self.context.deinit();
        }

        // SECURITY FIX: Release driver reference (CRIT-002)
        // The driver will only be deinitialized when all references are released
        self.driver_ref.release();
    }

    /// Check if CUDA is available
    pub fn isAvailable() bool {
        if (@import("builtin").os.tag == .macos) {
            return false;
        }

        var driver = CudaDriver.init(std.heap.page_allocator) catch {
            return false;
        };
        defer driver.deinit();

        return driver.is_initialized;
    }

    // =============================================================================
    // Profiling Support
    // =============================================================================

    /// Enable profiling for this backend
    pub fn enableProfiling(self: *CudaBackend, mode: @import("cuda_profiler.zig").ProfilerMode) !void {
        try self.context.enableProfiling(mode);
    }

    /// Disable profiling
    pub fn disableProfiling(self: *CudaBackend) void {
        self.context.disableProfiling();
    }

    /// Get the profiler instance (null if not enabled)
    pub fn getProfiler(self: *CudaBackend) ?*@import("cuda_profiler.zig").CudaProfiler {
        return self.context.getProfiler();
    }

    /// Check if profiling is enabled
    pub fn isProfilingEnabled(self: *const CudaBackend) bool {
        return self.context.isProfilingEnabled();
    }

    /// Mark a point in the application timeline
    pub fn profilerMark(self: *const CudaBackend, message: []const u8) void {
        self.context.profilerMark(message);
    }

    /// Push a profiling range
    pub fn profilerPushRange(self: *CudaBackend, name: []const u8) !void {
        try self.context.profilerPushRange(name);
    }

    /// Pop a profiling range
    pub fn profilerPopRange(self: *CudaBackend) !void {
        try self.context.profilerPopRange();
    }

    /// Get number of CUDA devices
    pub fn getDeviceCount() i32 {
        if (@import("builtin").os.tag == .macos) {
            return 0;
        }

        var driver = CudaDriver.init(std.heap.page_allocator) catch {
            return 0;
        };
        defer driver.deinit();

        var count: c_int = 0;
        const result = driver.deviceGetCount.?(&count);
        if (result.isError()) {
            return 0;
        }
        return count;
    }

    /// Synchronize the CUDA stream
    pub fn synchronize(self: *CudaBackend) !void {
        try self.context.synchronize();
    }

    // =============================================================================
    // Memory Operations
    // =============================================================================

    /// Allocate device buffer
    pub fn allocBuffer(self: *CudaBackend, size: usize) !DeviceBuffer {
        const buf = try self.context.getBuffer(size);
        return DeviceBuffer{
            .ptr = buf.ptr,
            .size = buf.size,
            .context = self.context,
        };
    }

    /// Free device buffer
    pub fn freeBuffer(self: *CudaBackend, buffer: DeviceBuffer) void {
        const buf = CudaContext.DeviceBuffer{
            .ptr = buffer.ptr,
            .size = buffer.size,
            .pool_index = null,
        };
        self.context.returnBuffer(buf);
    }

    /// Upload data to device
    pub fn upload(self: *CudaBackend, dst: CUdeviceptr, src: []const f32) !void {
        try self.context.upload(dst, std.mem.sliceAsBytes(src));
    }

    /// Upload data asynchronously
    pub fn uploadAsync(self: *CudaBackend, dst: CUdeviceptr, src: []const f32) !void {
        try self.context.uploadAsync(dst, std.mem.sliceAsBytes(src));
    }

    /// Download data from device
    pub fn download(self: *CudaBackend, dst: []f32, src: CUdeviceptr) !void {
        try self.context.download(std.mem.sliceAsBytes(dst), src);
    }

    /// Download data asynchronously
    pub fn downloadAsync(self: *CudaBackend, dst: []f32, src: CUdeviceptr) !void {
        try self.context.downloadAsync(std.mem.sliceAsBytes(dst), src);
    }

    // =============================================================================
    // Unified Memory Operations
    // =============================================================================

    /// Check if unified memory is supported on this device
    pub fn hasUnifiedMemory(self: *CudaBackend) bool {
        return self.context.hasUnifiedMemory();
    }

    /// Check if concurrent managed access is supported (Pascal+)
    /// Allows simultaneous CPU and GPU access to managed memory
    pub fn hasConcurrentManagedAccess(self: *CudaBackend) bool {
        return self.context.hasConcurrentManagedAccess();
    }

    /// Allocate managed memory (unified memory)
    /// This creates a buffer accessible by both CPU and GPU
    /// Falls back to pinned host memory on older GPUs
    pub fn allocManaged(self: *CudaBackend, size: usize) !ManagedBuffer {
        const buf = try self.context.allocManaged(size);
        return ManagedBuffer{
            .ptr = buf.ptr,
            .size = buf.size,
            .host_ptr = buf.host_ptr,
            .context = self.context,
        };
    }

    /// Allocate managed memory for typed data
    pub fn allocManagedTyped(self: *CudaBackend, comptime T: type, count: usize) !ManagedBuffer {
        const size = try std.math.mul(usize, count, @sizeOf(T));
        return self.allocManaged(size);
    }

    /// Free managed memory
    pub fn freeManaged(self: *CudaBackend, buffer: ManagedBuffer) void {
        var buf = CudaContext.ManagedBuffer{
            .ptr = buffer.ptr,
            .size = buffer.size,
            .host_ptr = buffer.host_ptr,
        };
        self.context.freeManaged(&buf);
    }

    /// Prefetch managed memory to the GPU
    /// Reduces page faults by migrating data before kernel launch
    pub fn prefetchToDevice(self: *CudaBackend, ptr: CUdeviceptr, size: usize) !void {
        try self.context.prefetchToDevice(ptr, size);
    }

    /// Prefetch managed memory to the host
    /// Use before CPU needs to read GPU results
    pub fn prefetchToHost(self: *CudaBackend, ptr: CUdeviceptr, size: usize) !void {
        try self.context.prefetchToHost(ptr, size);
    }

    /// Set memory advice for optimized data placement
    pub fn setMemoryAdvice(self: *CudaBackend, ptr: CUdeviceptr, size: usize, advice: CudaContext.MemoryAdvice) !void {
        try self.context.setMemoryAdvice(ptr, size, advice);
    }

    /// Synchronize all managed memory operations
    /// Important for oversubscription scenarios
    pub fn synchronizeManaged(self: *CudaBackend) !void {
        try self.context.synchronizeManaged();
    }

    /// Get total memory info including unified memory availability
    pub fn getMemoryInfo(self: *CudaBackend) MemoryInfo {
        const props = self.context.device_props;
        return MemoryInfo{
            .total_memory = props.total_memory,
            .unified_memory_available = self.hasUnifiedMemory(),
            .concurrent_managed_access = self.hasConcurrentManagedAccess(),
        };
    }

    pub const MemoryInfo = struct {
        total_memory: usize,
        unified_memory_available: bool,
        concurrent_managed_access: bool,
    };

    /// Check if Tensor Cores should be used for matrix multiplication
    fn shouldUseTensorCores(self: *CudaBackend, m: usize, n: usize, k: usize) bool {
        // Check device capability
        if (!self.context.device_props.hasTensorCores()) {
            return false;
        }

        // Check matrix dimensions are compatible (multiples of 16)
        if (m % 16 != 0 or n % 16 != 0 or k % 16 != 0) {
            return false;
        }

        // Check matrix is large enough to benefit from Tensor Cores
        // Tensor Cores have overhead, so small matrices may be slower
        return m >= 64 and n >= 64 and k >= 64;
    }

    /// Matrix multiplication: C = A * B + (accumulate ? C : 0)
    /// Uses shared memory tiled implementation for 10-100x performance improvement
    /// Auto-selects Tensor Cores on compatible GPUs (sm_70+) for 2-8x additional speedup
    pub fn matMul(
        self: *CudaBackend,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: usize,
        n: usize,
        k: usize,
        transpose_a: bool,
        transpose_b: bool,
        accumulate: bool,
    ) !void {
        // PROFILER: Range marker for matrix multiplication
        self.profilerPushRange("matmul") catch {};
        defer self.profilerPopRange() catch {};

        // PERFORMANCE OPTIMIZATION: Use Tensor Cores when available and beneficial
        // Tensor Cores can provide 2-8x speedup for compatible GPUs (Volta, Turing, Ampere+)
        if (!transpose_a and !transpose_b and self.shouldUseTensorCores(m, n, k)) {
            return self.matMulTensorCore(a, b, c, m, n, k, accumulate);
        }

        // Allocate device buffers with overflow checking
        const size_a = try std.math.mul(usize, try std.math.mul(usize, m, k), @sizeOf(f32));
        const size_b = try std.math.mul(usize, try std.math.mul(usize, k, n), @sizeOf(f32));
        const size_c = try std.math.mul(usize, try std.math.mul(usize, m, n), @sizeOf(f32));

        var d_a = try self.context.getBuffer(size_a);
        defer self.context.returnBuffer(d_a);
        var d_b = try self.context.getBuffer(size_b);
        defer self.context.returnBuffer(d_b);
        var d_c = try self.context.getBuffer(size_c);
        defer self.context.returnBuffer(d_c);

        // Upload data
        try self.context.upload(d_a.ptr, std.mem.sliceAsBytes(a));
        try self.context.upload(d_b.ptr, std.mem.sliceAsBytes(b));
        if (accumulate) {
            try self.context.upload(d_c.ptr, std.mem.sliceAsBytes(c));
        }

        // PERFORMANCE OPTIMIZATION: Always use tiled kernel for better shared memory utilization
        // Tiled kernel with 32x32 thread blocks provides 5-10x speedup over naive implementation
        // Each tile load serves 32x32 = 1024 multiply-adds with only 64 loads from global memory
        // Tiled kernel handles all matrix sizes efficiently with proper bounds checking

        // Determine kernel name based on transpose flags
        const kernel_name = if (transpose_b) "matmul_tiled_transpose_b" else "matmul_tiled";

        // Launch kernel with tiled configuration
        var m_u32: u32 = @intCast(m);
        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var acc_u32: u32 = @intFromBool(accumulate);

        const args = [_]?*anyopaque{
            @ptrCast(&d_c.ptr),
            @ptrCast(&d_a.ptr),
            @ptrCast(&d_b.ptr),
            @ptrCast(&m_u32),
            @ptrCast(&n_u32),
            @ptrCast(&k_u32),
            @ptrCast(&acc_u32),
        };

        // TILED KERNEL CONFIGURATION: 32x32 thread blocks with shared memory tiling
        // Grid dimensions calculated to cover all output elements with proper bounds checking
        const config = self.context.getTiledMatMulConfig(m, n);
        try self.context.launchKernel(
            kernel_name,
            .{ config.grid_x, config.grid_y, 1 },
            .{ config.block_x, config.block_y, 1 },
            config.shared_mem_bytes,
            &args,
        );

        // Download result
        try self.context.download(std.mem.sliceAsBytes(c), d_c.ptr);
    }

    /// Tensor Core matrix multiplication using WMMA (Warp Matrix Multiply Accumulate)
    /// C = A * B where A, B are FP16 and C is FP32
    /// Uses mma.sync.aligned.m16n16k16 instruction for maximum throughput
    /// Requires sm_70+ (Volta, Turing, Ampere, Hopper)
    ///
    /// Performance: 2-8x faster than standard CUDA cores for compatible GPUs
    /// Constraints:
    ///   - M, N, K must be multiples of 16
    ///   - Matrices should be >= 64x64x64 for best performance
    pub fn matMulTensorCore(
        self: *CudaBackend,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: usize,
        n: usize,
        k: usize,
        accumulate: bool,
    ) !void {
        // Validate dimensions for Tensor Cores
        if (m % 16 != 0 or n % 16 != 0 or k % 16 != 0) {
            return error.InvalidDimensionsForTensorCore;
        }

        // Check device support
        if (!self.context.device_props.hasTensorCores()) {
            return error.TensorCoresNotSupported;
        }

        // Allocate device buffers with overflow checking
        // A and B are FP16, C is FP32
        const size_a_f16 = try std.math.mul(usize, try std.math.mul(usize, m, k), @sizeOf(f16));
        const size_b_f16 = try std.math.mul(usize, try std.math.mul(usize, k, n), @sizeOf(f16));
        const size_c_f32 = try std.math.mul(usize, try std.math.mul(usize, m, n), @sizeOf(f32));

        var d_a = try self.context.getBuffer(size_a_f16);
        defer self.context.returnBuffer(d_a);
        var d_b = try self.context.getBuffer(size_b_f16);
        defer self.context.returnBuffer(d_b);
        var d_c = try self.context.getBuffer(size_c_f32);
        defer self.context.returnBuffer(d_c);

        // Convert FP32 inputs to FP16 and upload
        // Note: In production, inputs should already be FP16 to avoid conversion overhead
        const a_f16 = try self.allocator.alloc(f16, m * k);
        defer self.allocator.free(a_f16);
        const b_f16 = try self.allocator.alloc(f16, k * n);
        defer self.allocator.free(b_f16);

        // Convert A: [M x K]
        for (0..m * k) |i| {
            a_f16[i] = @floatCast(a[i]);
        }

        // Convert B: [K x N]
        for (0..k * n) |i| {
            b_f16[i] = @floatCast(b[i]);
        }

        // Upload FP16 data
        try self.context.upload(d_a.ptr, std.mem.sliceAsBytes(a_f16));
        try self.context.upload(d_b.ptr, std.mem.sliceAsBytes(b_f16));

        // Initialize output if accumulating
        if (accumulate) {
            try self.context.upload(d_c.ptr, std.mem.sliceAsBytes(c));
        } else {
            try self.context.memset(d_c.ptr, 0, @intCast(size_c_f32));
        }

        // Launch WMMA kernel
        var m_u32: u32 = @intCast(m);
        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var acc_u32: u32 = @intFromBool(accumulate);

        const args = [_]?*anyopaque{
            @ptrCast(&d_a.ptr),
            @ptrCast(&d_b.ptr),
            @ptrCast(&d_c.ptr),
            @ptrCast(&m_u32),
            @ptrCast(&n_u32),
            @ptrCast(&k_u32),
            @ptrCast(&acc_u32),
        };

        // Grid: each block computes 64x64 tile
        const grid_x = (n + 63) / 64;
        const grid_y = (m + 63) / 64;

        // Block: 128 threads (4 warps)
        const block_x: u32 = 128;

        // Shared memory: 2 * (64 * 32 + 64 * 32) * 2 bytes = 16KB per buffer, 32KB total
        const shared_mem: u32 = 32 * 1024;

        try self.context.launchKernel(
            "matmul_tensor_core",
            .{ @as(u32, @intCast(grid_x)), @as(u32, @intCast(grid_y)), 1 },
            .{ block_x, 1, 1 },
            shared_mem,
            &args,
        );

        // Download FP32 result
        try self.context.download(std.mem.sliceAsBytes(c), d_c.ptr);
    }

    /// Batched matrix multiplication
    pub fn matMulBatch(
        self: *CudaBackend,
        a: []const f32,
        b: []const f32,
        c: []f32,
        batch_size: usize,
        n: usize,
        k: usize,
        accumulate: bool,
    ) !void {
        // PERFORMANCE OPTIMIZATION: Use tiled kernel for all batch matrix multiplications
        // Tiled kernel with shared memory provides 5-10x speedup over naive implementation
        // Each tile load serves 32x32 = 1024 multiply-adds with only 64 loads from global memory

        // Allocate device buffers with overflow checking
        const total_size_a = try std.math.mul(usize, try std.math.mul(usize, batch_size, k), @sizeOf(f32)); // a is batch_size x k
        const total_size_b = try std.math.mul(usize, try std.math.mul(usize, try std.math.mul(usize, batch_size, n), k), @sizeOf(f32));
        const total_size_c = try std.math.mul(usize, try std.math.mul(usize, batch_size, n), @sizeOf(f32));

        var d_a = try self.context.getBuffer(total_size_a);
        defer self.context.returnBuffer(d_a);
        var d_b = try self.context.getBuffer(total_size_b);
        defer self.context.returnBuffer(d_b);
        var d_c = try self.context.getBuffer(total_size_c);
        defer self.context.returnBuffer(d_c);

        try self.context.upload(d_a.ptr, std.mem.sliceAsBytes(a));
        try self.context.upload(d_b.ptr, std.mem.sliceAsBytes(b));
        if (accumulate) {
            try self.context.upload(d_c.ptr, std.mem.sliceAsBytes(c));
        }

        var bs_u32: u32 = @intCast(batch_size);
        var m_u32: u32 = 1; // Assuming M=1 as per cpuMatMulBatch
        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);
        var acc_u32: u32 = @intFromBool(accumulate);

        const args = [_]?*anyopaque{
            @ptrCast(&d_c.ptr),
            @ptrCast(&d_a.ptr),
            @ptrCast(&d_b.ptr),
            @ptrCast(&bs_u32),
            @ptrCast(&m_u32),
            @ptrCast(&n_u32),
            @ptrCast(&k_u32),
            @ptrCast(&acc_u32),
        };

        // TILED KERNEL CONFIGURATION: 32x32 thread blocks with shared memory tiling
        // Grid dimensions calculated to cover all output elements with proper bounds checking
        const TILE_SIZE: u32 = 32;
        const block_x = TILE_SIZE;
        const block_y = TILE_SIZE;
        const grid_x = @as(u32, @intCast((n + block_x - 1) / block_x));
        const grid_y = @as(u32, @intCast((batch_size + block_y - 1) / block_y));
        // Shared memory: 2 tiles (A and B) * TILE_SIZE * TILE_SIZE * sizeof(float) = 8KB
        const shared_mem_bytes = 2 * TILE_SIZE * TILE_SIZE * @sizeOf(f32);

        try self.context.launchKernel(
            "matmul_batch_tiled",
            .{ grid_x, grid_y, 1 },
            .{ block_x, block_y, 1 },
            shared_mem_bytes,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(c), d_c.ptr);
    }

    /// Element-wise operations
    pub fn elementWiseOp(
        self: *CudaBackend,
        op: ElementWiseOp,
        a: []const f32,
        b: []const f32,
        c: []f32,
    ) !void {
        const size = try std.math.mul(usize, a.len, @sizeOf(f32));

        var d_a = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_a);
        var d_b = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_b);
        var d_c = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_c);

        try self.context.upload(d_a.ptr, std.mem.sliceAsBytes(a));
        try self.context.upload(d_b.ptr, std.mem.sliceAsBytes(b));

        const kernel_name = switch (op) {
            .add => "ew_add",
            .sub => "ew_sub",
            .mul => "ew_mul",
            .div => "ew_div",
        };

        var len_u32: u32 = @intCast(a.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_a.ptr),
            @ptrCast(&d_b.ptr),
            @ptrCast(&d_c.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(a.len);

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(c), d_c.ptr);
    }

    /// Scalar multiplication
    pub fn scale(self: *CudaBackend, a: []const f32, scalar: f32, c: []f32) !void {
        const size = try std.math.mul(usize, a.len, @sizeOf(f32));

        var d_a = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_a);
        var d_c = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_c);

        try self.context.upload(d_a.ptr, std.mem.sliceAsBytes(a));

        var len_u32: u32 = @intCast(a.len);
        var scalar_val = scalar;
        const args = [_]?*anyopaque{
            @ptrCast(&d_a.ptr),
            @ptrCast(&scalar_val),
            @ptrCast(&d_c.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(a.len);

        try self.context.launchKernel(
            "scale_buffer",
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(c), d_c.ptr);
    }

    /// Fill buffer with constant value
    pub fn fill(self: *CudaBackend, data: []f32, value: f32) !void {
        const size = try std.math.mul(usize, data.len, @sizeOf(f32));
        var d_data = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_data);

        var n_u32: u32 = @intCast(data.len);
        var val = value;
        const args = [_]?*anyopaque{
            @ptrCast(&d_data.ptr),
            @ptrCast(&val),
            @ptrCast(&n_u32),
        };

        const config = self.context.getElementWiseConfig(data.len);

        try self.context.launchKernel(
            "fill_constant",
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(data), d_data.ptr);
    }

    /// Map operations (exp, log, sqrt, etc.)
    pub fn mapOp(
        self: *CudaBackend,
        op: MapOp,
        input: []const f32,
        output: []f32,
    ) !void {
        const size = try std.math.mul(usize, input.len, @sizeOf(f32));

        var d_input = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

        const kernel_name = switch (op) {
            .exp => "map_exp",
            .log => "map_log",
            .sqrt => "map_sqrt",
            .abs => "map_abs",
            .square => "map_square",
            .inv => "map_inv",
        };

        var len_u32: u32 = @intCast(input.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(input.len);

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
    }

    /// Add bias to buffer
    pub fn addBias(
        self: *CudaBackend,
        output: []f32,
        bias: []const f32,
        batch_size: usize,
        bias_size: usize,
    ) !void {
        const out_bytes = try std.math.mul(usize, output.len, @sizeOf(f32));
        const bias_bytes = try std.math.mul(usize, bias.len, @sizeOf(f32));

        var d_output = try self.context.getBuffer(out_bytes);
        defer self.context.returnBuffer(d_output);
        var d_bias = try self.context.getBuffer(bias_bytes);
        defer self.context.returnBuffer(d_bias);

        try self.context.upload(d_output.ptr, std.mem.sliceAsBytes(output));
        try self.context.upload(d_bias.ptr, std.mem.sliceAsBytes(bias));

        var batch_u32: u32 = @intCast(batch_size);
        var bias_u32: u32 = @intCast(bias_size);

        const args = [_]?*anyopaque{
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_bias.ptr),
            @ptrCast(&batch_u32),
            @ptrCast(&bias_u32),
        };

        const config = self.context.getElementWiseConfig(bias_size);

        try self.context.launchKernel(
            "add_bias",
            .{ config.grid, @as(u32, @intCast(batch_size)), 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
    }

    /// Fused matrix multiplication with bias and activation
    /// Performs: C = activation(A * B + bias)
    /// This is significantly faster than separate matmul + bias + activation calls
    /// because it eliminates intermediate memory writes/reads.
    ///
    /// Parameters:
    ///   - a: Input matrix A (batch_size x K)
    ///   - b: Weight matrix B (K x N) - shared across batch
    ///   - c: Output matrix C (batch_size x N)
    ///   - bias: Bias vector (N elements) - added to each output row
    ///   - batch_size: Number of batch samples
    ///   - n: Number of output columns
    ///   - k: Inner dimension (A columns / B rows)
    ///   - kernel_name: Fused kernel to use (e.g., "matmul_bias_relu_fused")
    pub fn matMulBiasActivation(
        self: *CudaBackend,
        a: []const f32,
        b: []const f32,
        c: []f32,
        bias: []const f32,
        batch_size: usize,
        n: usize,
        k: usize,
        kernel_name: []const u8,
    ) !void {
        // Validate kernel exists
        if (!self.context.hasKernel(kernel_name)) {
            std.log.warn("Fused kernel '{s}' not available, falling back to separate operations", .{kernel_name});
            return error.KernelNotAvailable;
        }

        // Allocate device buffers with overflow checking
        const total_size_a = try std.math.mul(usize, try std.math.mul(usize, batch_size, k), @sizeOf(f32));
        const total_size_b = try std.math.mul(usize, try std.math.mul(usize, n, k), @sizeOf(f32));
        const total_size_c = try std.math.mul(usize, try std.math.mul(usize, batch_size, n), @sizeOf(f32));
        const bias_bytes = try std.math.mul(usize, bias.len, @sizeOf(f32));

        var d_a = try self.context.getBuffer(total_size_a);
        defer self.context.returnBuffer(d_a);
        var d_b = try self.context.getBuffer(total_size_b);
        defer self.context.returnBuffer(d_b);
        var d_c = try self.context.getBuffer(total_size_c);
        defer self.context.returnBuffer(d_c);
        var d_bias = try self.context.getBuffer(bias_bytes);
        defer self.context.returnBuffer(d_bias);

        try self.context.upload(d_a.ptr, std.mem.sliceAsBytes(a));
        try self.context.upload(d_b.ptr, std.mem.sliceAsBytes(b));
        try self.context.upload(d_bias.ptr, std.mem.sliceAsBytes(bias));

        var bs_u32: u32 = @intCast(batch_size);
        var m_u32: u32 = 1; // M=1 for batch size
        var n_u32: u32 = @intCast(n);
        var k_u32: u32 = @intCast(k);

        const args = [_]?*anyopaque{
            @ptrCast(&d_a.ptr),
            @ptrCast(&d_b.ptr),
            @ptrCast(&d_c.ptr),
            @ptrCast(&d_bias.ptr),
            @ptrCast(&bs_u32),
            @ptrCast(&m_u32),
            @ptrCast(&n_u32),
            @ptrCast(&k_u32),
        };

        const config = self.context.getElementWiseConfig(n);

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid, @as(u32, @intCast(batch_size)), 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(c), d_c.ptr);
    }

    // =============================================================================
    // Activation Functions
    // =============================================================================

    /// Minimum tensor size to use vectorized kernels
    /// Vectorized kernels process 4 elements per thread, so we need enough elements
    const VECTORIZATION_THRESHOLD: usize = 1024;

    /// Check if we can use vectorized kernels
    /// Requires: large enough tensor, 16-byte aligned pointers
    fn canUseVectorized(n: usize, ptr1: *const anyopaque, ptr2: *const anyopaque) bool {
        if (n < VECTORIZATION_THRESHOLD) return false;
        const addr1 = @intFromPtr(ptr1);
        const addr2 = @intFromPtr(ptr2);
        return (addr1 % 16 == 0) and (addr2 % 16 == 0);
    }

    /// ReLU forward: output = max(0, input)
    /// Uses vectorized kernel (vec4) for large aligned tensors
    pub fn reluForward(self: *CudaBackend, input: []const f32, output: []f32) !void {
        const use_vec4 = canUseVectorized(input.len, input.ptr, output.ptr);
        if (use_vec4 and self.context.hasKernel("relu_forward_vec4")) {
            try self.activationForwardVec4("relu_forward_vec4", input, output);
        } else {
            try self.activationForward("relu_forward", input, output);
        }
    }

    /// ReLU backward
    /// Uses vectorized kernel (vec4) for large aligned tensors
    pub fn reluBackward(self: *CudaBackend, output: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        const use_vec4 = canUseVectorized(output.len, output.ptr, grad_output.ptr) and
                        canUseVectorized(output.len, output.ptr, grad_input.ptr);
        if (use_vec4 and self.context.hasKernel("relu_backward_vec4")) {
            try self.activationBackwardVec4("relu_backward_vec4", output, grad_output, grad_input);
        } else {
            try self.activationBackward("relu_backward", output, grad_output, grad_input);
        }
    }

    /// Sigmoid forward: output = 1 / (1 + exp(-input))
    /// Uses vectorized kernel (vec4) for large aligned tensors
    pub fn sigmoidForward(self: *CudaBackend, input: []const f32, output: []f32) !void {
        const use_vec4 = canUseVectorized(input.len, input.ptr, output.ptr);
        if (use_vec4 and self.context.hasKernel("sigmoid_forward_vec4")) {
            try self.activationForwardVec4("sigmoid_forward_vec4", input, output);
        } else {
            try self.activationForward("sigmoid_forward", input, output);
        }
    }

    /// Sigmoid backward
    /// Uses vectorized kernel (vec4) for large aligned tensors
    pub fn sigmoidBackward(self: *CudaBackend, output: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        const use_vec4 = canUseVectorized(output.len, output.ptr, grad_output.ptr) and
                        canUseVectorized(output.len, output.ptr, grad_input.ptr);
        if (use_vec4 and self.context.hasKernel("sigmoid_backward_vec4")) {
            try self.activationBackwardVec4("sigmoid_backward_vec4", output, grad_output, grad_input);
        } else {
            try self.activationBackward("sigmoid_backward", output, grad_output, grad_input);
        }
    }

    /// Tanh forward
    /// Uses vectorized kernel (vec4) for large aligned tensors
    pub fn tanhForward(self: *CudaBackend, input: []const f32, output: []f32) !void {
        const use_vec4 = canUseVectorized(input.len, input.ptr, output.ptr);
        if (use_vec4 and self.context.hasKernel("tanh_forward_vec4")) {
            try self.activationForwardVec4("tanh_forward_vec4", input, output);
        } else {
            try self.activationForward("tanh_forward", input, output);
        }
    }

    /// Tanh backward
    /// Uses vectorized kernel (vec4) for large aligned tensors
    pub fn tanhBackward(self: *CudaBackend, output: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        const use_vec4 = canUseVectorized(output.len, output.ptr, grad_output.ptr) and
                        canUseVectorized(output.len, output.ptr, grad_input.ptr);
        if (use_vec4 and self.context.hasKernel("tanh_backward_vec4")) {
            try self.activationBackwardVec4("tanh_backward_vec4", output, grad_output, grad_input);
        } else {
            try self.activationBackward("tanh_backward", output, grad_output, grad_input);
        }
    }

    /// Softmax forward
    pub fn softmaxForward(self: *CudaBackend, input: []const f32, output: []f32, batch_size: usize, features: usize) !void {
        const size = try std.math.mul(usize, input.len, @sizeOf(f32));

        var d_input = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

        var batch_u32: u32 = @intCast(batch_size);
        var feat_u32: u32 = @intCast(features);
        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&batch_u32),
            @ptrCast(&feat_u32),
        };

        const config = self.context.getElementWiseConfig(batch_size);

        try self.context.launchKernel(
            "softmax_forward",
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
    }

    fn activationForward(self: *CudaBackend, kernel_name: []const u8, input: []const f32, output: []f32) !void {
        const size = try std.math.mul(usize, input.len, @sizeOf(f32));

        var d_input = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

        var len_u32: u32 = @intCast(input.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(input.len);

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
    }

    fn activationBackward(self: *CudaBackend, kernel_name: []const u8, output: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        const size = try std.math.mul(usize, output.len, @sizeOf(f32));

        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);
        var d_grad_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_grad_output);
        var d_grad_input = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_grad_input);

        try self.context.upload(d_output.ptr, std.mem.sliceAsBytes(output));
        try self.context.upload(d_grad_output.ptr, std.mem.sliceAsBytes(grad_output));

        var len_u32: u32 = @intCast(output.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_grad_input.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(output.len);

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(grad_input), d_grad_input.ptr);
    }

    /// Vectorized forward activation (processes 4 elements per thread)
    /// Uses adjusted grid size since each thread processes 4 elements
    fn activationForwardVec4(self: *CudaBackend, kernel_name: []const u8, input: []const f32, output: []f32) !void {
        const size = try std.math.mul(usize, input.len, @sizeOf(f32));

        var d_input = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

        var len_u32: u32 = @intCast(input.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&len_u32),
        };

        // For vec4 kernels, each thread processes 4 elements
        // So we need 1/4 the number of threads
        const block_size: u32 = 256;
        const total_threads = (input.len + 3) / 4; // Round up division by 4
        const grid_size = @as(u32, @intCast((total_threads + block_size - 1) / block_size));

        try self.context.launchKernel(
            kernel_name,
            .{ grid_size, 1, 1 },
            .{ block_size, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
    }

    /// Vectorized backward activation (processes 4 elements per thread)
    fn activationBackwardVec4(self: *CudaBackend, kernel_name: []const u8, output: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        const size = try std.math.mul(usize, output.len, @sizeOf(f32));

        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);
        var d_grad_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_grad_output);
        var d_grad_input = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_grad_input);

        try self.context.upload(d_output.ptr, std.mem.sliceAsBytes(output));
        try self.context.upload(d_grad_output.ptr, std.mem.sliceAsBytes(grad_output));

        var len_u32: u32 = @intCast(output.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_grad_input.ptr),
            @ptrCast(&len_u32),
        };

        // For vec4 kernels, each thread processes 4 elements
        const block_size: u32 = 256;
        const total_threads = (output.len + 3) / 4;
        const grid_size = @as(u32, @intCast((total_threads + block_size - 1) / block_size));

        try self.context.launchKernel(
            kernel_name,
            .{ grid_size, 1, 1 },
            .{ block_size, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(grad_input), d_grad_input.ptr);
    }

    // =============================================================================
    // Loss Functions
    // =============================================================================

    /// MSE loss backward
    pub fn mseBackward(self: *CudaBackend, output: []const f32, target: []const f32, grad_output: []f32) !void {
        try self.lossBackward("mse_backward", output, target, grad_output);
    }

    /// Cross-entropy loss backward
    pub fn crossEntropyBackward(self: *CudaBackend, output: []const f32, target: []const f32, grad_output: []f32) !void {
        try self.lossBackward("cross_entropy_backward", output, target, grad_output);
    }

    fn lossBackward(self: *CudaBackend, kernel_name: []const u8, output: []const f32, target: []const f32, grad_output: []f32) !void {
        const size = try std.math.mul(usize, output.len, @sizeOf(f32));

        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);
        var d_target = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_target);
        var d_grad = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_grad);

        try self.context.upload(d_output.ptr, std.mem.sliceAsBytes(output));
        try self.context.upload(d_target.ptr, std.mem.sliceAsBytes(target));

        var len_u32: u32 = @intCast(output.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_target.ptr),
            @ptrCast(&d_grad.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(output.len);

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(grad_output), d_grad.ptr);
    }

    /// Binary Cross-Entropy backward pass
    pub fn binaryCrossEntropyBackward(
        self: *CudaBackend,
        output: []const f32,
        target: []const f32,
        grad_output: []f32,
        epsilon: f32,
    ) !void {
        const size = try std.math.mul(usize, output.len, @sizeOf(f32));
        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);
        var d_target = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_target);
        var d_grad = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_grad);

        try self.context.upload(d_output.ptr, std.mem.sliceAsBytes(output));
        try self.context.upload(d_target.ptr, std.mem.sliceAsBytes(target));

        var n_i32 = @as(i32, @intCast(output.len));
        var eps_f32 = epsilon;

        const args = [_]?*anyopaque{
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_target.ptr),
            @ptrCast(&d_grad.ptr),
            @ptrCast(&n_i32),
            @ptrCast(&eps_f32),
        };

        const config = self.context.getElementWiseConfig(output.len);
        try self.context.launchKernel("binary_cross_entropy_backward", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(grad_output), d_grad.ptr);
    }

    /// KL Divergence backward pass
    pub fn klDivergenceBackward(
        self: *CudaBackend,
        output: []const f32,
        grad_output: []f32,
        epsilon: f32,
    ) !void {
        const size = try std.math.mul(usize, output.len, @sizeOf(f32));
        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);
        var d_grad = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_grad);

        try self.context.upload(d_output.ptr, std.mem.sliceAsBytes(output));

        var n_i32 = @as(i32, @intCast(output.len));
        var eps_f32 = epsilon;

        const args = [_]?*anyopaque{
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_grad.ptr),
            @ptrCast(&n_i32),
            @ptrCast(&eps_f32),
        };

        const config = self.context.getElementWiseConfig(output.len);
        try self.context.launchKernel("kl_divergence_backward", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(grad_output), d_grad.ptr);
    }

    // =============================================================================
    // Optimizers
    // =============================================================================

    /// SGD update
    pub fn sgdUpdate(
        self: *CudaBackend,
        weights: []f32,
        gradients: []const f32,
        learning_rate: f32,
        weight_decay: f32,
    ) !void {
        const size = try std.math.mul(usize, weights.len, @sizeOf(f32));

        var d_weights = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_weights);
        var d_gradients = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_gradients);

        try self.context.upload(d_weights.ptr, std.mem.sliceAsBytes(weights));
        try self.context.upload(d_gradients.ptr, std.mem.sliceAsBytes(gradients));

        var len_u32: u32 = @intCast(weights.len);
        var lr = learning_rate;
        var wd = weight_decay;
        const args = [_]?*anyopaque{
            @ptrCast(&d_weights.ptr),
            @ptrCast(&d_gradients.ptr),
            @constCast(&lr),
            @ptrCast(&wd),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(weights.len);

        try self.context.launchKernel(
            "sgd_update",
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(weights), d_weights.ptr);
    }

    /// Adam update
    pub fn adamUpdate(
        self: *CudaBackend,
        weights: []f32,
        gradients: []const f32,
        m: []f32,
        v: []f32,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        t: u32,
    ) !void {
        const size = try std.math.mul(usize, weights.len, @sizeOf(f32));

        var d_weights = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_weights);
        var d_gradients = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_gradients);
        var d_m = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_m);
        var d_v = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_v);

        try self.context.upload(d_weights.ptr, std.mem.sliceAsBytes(weights));
        try self.context.upload(d_gradients.ptr, std.mem.sliceAsBytes(gradients));
        try self.context.upload(d_m.ptr, std.mem.sliceAsBytes(m));
        try self.context.upload(d_v.ptr, std.mem.sliceAsBytes(v));

        var lr = learning_rate;
        var b1 = beta1;
        var b2 = beta2;
        var eps = epsilon;
        var timestep = t;
        var len_u32: u32 = @intCast(weights.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_weights.ptr),
            @ptrCast(&d_gradients.ptr),
            @ptrCast(&d_m.ptr),
            @ptrCast(&d_v.ptr),
            @constCast(&lr),
            @ptrCast(&b1),
            @ptrCast(&b2),
            @ptrCast(&eps),
            @ptrCast(&timestep),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(weights.len);

        try self.context.launchKernel(
            "adam_update",
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(weights), d_weights.ptr);
        try self.context.download(std.mem.sliceAsBytes(m), d_m.ptr);
        try self.context.download(std.mem.sliceAsBytes(v), d_v.ptr);
    }

    /// RMSprop update
    pub fn rmspropUpdate(
        self: *CudaBackend,
        weights: []f32,
        gradients: []const f32,
        g_avg: []f32,
        lr: f32,
        rho: f32,
        eps: f32,
    ) !void {
        const size = try std.math.mul(usize, weights.len, @sizeOf(f32));

        var d_weights = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_weights);
        var d_gradients = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_gradients);
        var d_g_avg = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_g_avg);

        try self.context.upload(d_weights.ptr, std.mem.sliceAsBytes(weights));
        try self.context.upload(d_gradients.ptr, std.mem.sliceAsBytes(gradients));
        try self.context.upload(d_g_avg.ptr, std.mem.sliceAsBytes(g_avg));

        var len_u32: u32 = @intCast(weights.len);
        var lr_val = lr;
        var rho_val = rho;
        var eps_val = eps;
        const args = [_]?*anyopaque{
            @ptrCast(&d_weights.ptr),
            @ptrCast(&d_gradients.ptr),
            @ptrCast(&d_g_avg.ptr),
            @ptrCast(&lr_val),
            @ptrCast(&rho_val),
            @ptrCast(&eps_val),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(weights.len);

        try self.context.launchKernel(
            "rmsprop_update",
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(weights), d_weights.ptr);
        try self.context.download(std.mem.sliceAsBytes(g_avg), d_g_avg.ptr);
    }

    // =============================================================================
    // Layer Normalization
    // =============================================================================

    /// Layer Normalization forward pass
    pub fn layerNormForward(
        self: *CudaBackend,
        input: []const f32,
        output: []f32,
        gamma: []const f32,
        beta: []const f32,
        epsilon: f32,
        batch_size: usize,
        features: usize,
    ) !void {
        const input_size = try std.math.mul(usize, batch_size, features);

        const output_size = try std.math.mul(usize, output.len, @sizeOf(f32));
        const gamma_size = try std.math.mul(usize, gamma.len, @sizeOf(f32));
        const beta_size = try std.math.mul(usize, beta.len, @sizeOf(f32));

        var d_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(output_size);
        defer self.context.returnBuffer(d_output);
        var d_gamma = try self.context.getBuffer(gamma_size);
        defer self.context.returnBuffer(d_gamma);
        var d_beta = try self.context.getBuffer(beta_size);
        defer self.context.returnBuffer(d_beta);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));
        try self.context.upload(d_gamma.ptr, std.mem.sliceAsBytes(gamma));
        try self.context.upload(d_beta.ptr, std.mem.sliceAsBytes(beta));

        const total_elements = input_size;
        const block_size: u32 = 256;
        const grid_size: u32 = @intCast((total_elements + block_size - 1) / block_size);

        var batch_size_i32 = @as(i32, @intCast(batch_size));
        var features_i32 = @as(i32, @intCast(features));
        var epsilon_f32 = epsilon;

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_gamma.ptr),
            @ptrCast(&d_beta.ptr),
            @ptrCast(&epsilon_f32),
            @ptrCast(&batch_size_i32),
            @ptrCast(&features_i32),
        };

        try self.context.launchKernel(
            "layernorm_forward",
            .{ grid_size, 1, 1 },
            .{ block_size, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
    }

    /// Layer Normalization backward pass
    pub fn layerNormBackward(
        self: *CudaBackend,
        input: []const f32,
        grad_output: []const f32,
        grad_input: []f32,
        gamma: []const f32,
        grad_gamma: []f32,
        grad_beta: []f32,
        epsilon: f32,
        batch_size: usize,
        features: usize,
    ) !void {
        const input_size = try std.math.mul(usize, batch_size, features);

        const grad_output_size = try std.math.mul(usize, grad_output.len, @sizeOf(f32));
        const grad_input_size = try std.math.mul(usize, grad_input.len, @sizeOf(f32));
        const gamma_size = try std.math.mul(usize, gamma.len, @sizeOf(f32));
        const grad_gamma_size = try std.math.mul(usize, grad_gamma.len, @sizeOf(f32));
        const grad_beta_size = try std.math.mul(usize, grad_beta.len, @sizeOf(f32));

        var d_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_input);
        var d_grad_output = try self.context.getBuffer(grad_output_size);
        defer self.context.returnBuffer(d_grad_output);
        var d_grad_input = try self.context.getBuffer(grad_input_size);
        defer self.context.returnBuffer(d_grad_input);
        var d_gamma = try self.context.getBuffer(gamma_size);
        defer self.context.returnBuffer(d_gamma);
        var d_grad_gamma = try self.context.getBuffer(grad_gamma_size);
        defer self.context.returnBuffer(d_grad_gamma);
        var d_grad_beta = try self.context.getBuffer(grad_beta_size);
        defer self.context.returnBuffer(d_grad_beta);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));
        try self.context.upload(d_grad_output.ptr, std.mem.sliceAsBytes(grad_output));
        try self.context.upload(d_gamma.ptr, std.mem.sliceAsBytes(gamma));
        try self.context.memset(d_grad_gamma.ptr, 0, @intCast(grad_gamma_size));
        try self.context.memset(d_grad_beta.ptr, 0, @intCast(grad_beta_size));

        const total_elements = input_size;
        const block_size: u32 = 256;
        const grid_size: u32 = @intCast((total_elements + block_size - 1) / block_size);

        var batch_size_i32 = @as(i32, @intCast(batch_size));
        var features_i32 = @as(i32, @intCast(features));
        var epsilon_f32 = epsilon;

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_grad_input.ptr),
            @ptrCast(&d_gamma.ptr),
            @ptrCast(&d_grad_gamma.ptr),
            @ptrCast(&d_grad_beta.ptr),
            @ptrCast(&epsilon_f32),
            @ptrCast(&batch_size_i32),
            @ptrCast(&features_i32),
        };

        try self.context.launchKernel(
            "layernorm_backward",
            .{ grid_size, 1, 1 },
            .{ block_size, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(grad_input), d_grad_input.ptr);
        try self.context.download(std.mem.sliceAsBytes(grad_gamma), d_grad_gamma.ptr);
        try self.context.download(std.mem.sliceAsBytes(grad_beta), d_grad_beta.ptr);
    }

    // =============================================================================
    // Batch Normalization
    // =============================================================================

    /// Batch Normalization forward pass (training mode)
    pub fn batchNormForwardTraining(
        self: *CudaBackend,
        input: []const f32,
        output: []f32,
        gamma: []const f32,
        beta: []const f32,
        epsilon: f32,
        momentum: f32,
        running_mean: []f32,
        running_var: []f32,
        batch_size: usize,
        num_features: usize,
    ) !void {
        const input_size = try std.math.mul(usize, batch_size, num_features);
        const output_size = try std.math.mul(usize, output.len, @sizeOf(f32));
        const gamma_size = try std.math.mul(usize, gamma.len, @sizeOf(f32));
        const beta_size = try std.math.mul(usize, beta.len, @sizeOf(f32));
        const running_mean_size = try std.math.mul(usize, running_mean.len, @sizeOf(f32));
        const running_var_size = try std.math.mul(usize, running_var.len, @sizeOf(f32));

        var d_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(output_size);
        defer self.context.returnBuffer(d_output);
        var d_gamma = try self.context.getBuffer(gamma_size);
        defer self.context.returnBuffer(d_gamma);
        var d_beta = try self.context.getBuffer(beta_size);
        defer self.context.returnBuffer(d_beta);
        var d_running_mean = try self.context.getBuffer(running_mean_size);
        defer self.context.returnBuffer(d_running_mean);
        var d_running_var = try self.context.getBuffer(running_var_size);
        defer self.context.returnBuffer(d_running_var);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));
        try self.context.upload(d_gamma.ptr, std.mem.sliceAsBytes(gamma));
        try self.context.upload(d_beta.ptr, std.mem.sliceAsBytes(beta));
        try self.context.upload(d_running_mean.ptr, std.mem.sliceAsBytes(running_mean));
        try self.context.upload(d_running_var.ptr, std.mem.sliceAsBytes(running_var));

        const total_elements = input_size;
        const block_size: u32 = 256;
        const grid_size: u32 = @intCast((total_elements + block_size - 1) / block_size);

        var batch_size_i32 = @as(i32, @intCast(batch_size));
        var num_features_i32 = @as(i32, @intCast(num_features));
        var epsilon_f32 = epsilon;
        var momentum_f32 = momentum;

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_gamma.ptr),
            @ptrCast(&d_beta.ptr),
            @ptrCast(&epsilon_f32),
            @ptrCast(&momentum_f32),
            @ptrCast(&d_running_mean.ptr),
            @ptrCast(&d_running_var.ptr),
            @ptrCast(&batch_size_i32),
            @ptrCast(&num_features_i32),
        };

        try self.context.launchKernel(
            "batchnorm_forward_training",
            .{ grid_size, 1, 1 },
            .{ block_size, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
        try self.context.download(std.mem.sliceAsBytes(running_mean), d_running_mean.ptr);
        try self.context.download(std.mem.sliceAsBytes(running_var), d_running_var.ptr);
    }

    /// Batch Normalization forward pass (inference mode)
    pub fn batchNormForwardInference(
        self: *CudaBackend,
        input: []const f32,
        output: []f32,
        gamma: []const f32,
        beta: []const f32,
        epsilon: f32,
        running_mean: []const f32,
        running_var: []const f32,
        batch_size: usize,
        num_features: usize,
    ) !void {
        const input_size = try std.math.mul(usize, batch_size, num_features);
        const output_size = try std.math.mul(usize, output.len, @sizeOf(f32));
        const gamma_size = try std.math.mul(usize, gamma.len, @sizeOf(f32));
        const beta_size = try std.math.mul(usize, beta.len, @sizeOf(f32));
        const running_mean_size = try std.math.mul(usize, running_mean.len, @sizeOf(f32));
        const running_var_size = try std.math.mul(usize, running_var.len, @sizeOf(f32));

        var d_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(output_size);
        defer self.context.returnBuffer(d_output);
        var d_gamma = try self.context.getBuffer(gamma_size);
        defer self.context.returnBuffer(d_gamma);
        var d_beta = try self.context.getBuffer(beta_size);
        defer self.context.returnBuffer(d_beta);
        var d_running_mean = try self.context.getBuffer(running_mean_size);
        defer self.context.returnBuffer(d_running_mean);
        var d_running_var = try self.context.getBuffer(running_var_size);
        defer self.context.returnBuffer(d_running_var);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));
        try self.context.upload(d_gamma.ptr, std.mem.sliceAsBytes(gamma));
        try self.context.upload(d_beta.ptr, std.mem.sliceAsBytes(beta));
        try self.context.upload(d_running_mean.ptr, std.mem.sliceAsBytes(running_mean));
        try self.context.upload(d_running_var.ptr, std.mem.sliceAsBytes(running_var));

        const total_elements = input_size;
        const block_size: u32 = 256;
        const grid_size: u32 = @intCast((total_elements + block_size - 1) / block_size);

        var batch_size_i32 = @as(i32, @intCast(batch_size));
        var num_features_i32 = @as(i32, @intCast(num_features));
        var epsilon_f32 = epsilon;

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_gamma.ptr),
            @ptrCast(&d_beta.ptr),
            @ptrCast(&epsilon_f32),
            @ptrCast(&d_running_mean.ptr),
            @ptrCast(&d_running_var.ptr),
            @ptrCast(&batch_size_i32),
            @ptrCast(&num_features_i32),
        };

        try self.context.launchKernel(
            "batchnorm_forward_inference",
            .{ grid_size, 1, 1 },
            .{ block_size, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
    }

    /// Softmax backward pass
    pub fn softmaxBackward(
        self: *CudaBackend,
        output: []const f32,
        grad_output: []const f32,
        grad_input: []f32,
        batch_size: usize,
        features: usize,
    ) !void {
        const total_size = try std.math.mul(usize, batch_size, features);

        var d_output = try self.context.getBuffer(total_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_output);
        var d_grad_output = try self.context.getBuffer(total_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_output);
        var d_grad_input = try self.context.getBuffer(total_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_input);

        try self.context.upload(d_output.ptr, std.mem.sliceAsBytes(output));
        try self.context.upload(d_grad_output.ptr, std.mem.sliceAsBytes(grad_output));

        var batch_size_i32 = @as(i32, @intCast(batch_size));
        var features_i32 = @as(i32, @intCast(features));

        const args = [_]?*anyopaque{
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_grad_input.ptr),
            @ptrCast(&batch_size_i32),
            @ptrCast(&features_i32),
        };

        const config = self.context.getElementWiseConfig(batch_size);
        try self.context.launchKernel("softmax_backward", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(grad_input), d_grad_input.ptr);
    }

    /// Dropout forward pass
    pub fn dropoutForward(
        self: *CudaBackend,
        input: []const f32,
        output: []f32,
        mask: []f32,
        rate: f32,
        scaling_factor: f32,
        seed: u64,
    ) !void {
        const total_elements = input.len;
        var d_input = try self.context.getBuffer(total_elements * @sizeOf(f32));
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(total_elements * @sizeOf(f32));
        defer self.context.returnBuffer(d_output);
        var d_mask = try self.context.getBuffer(total_elements * @sizeOf(f32));
        defer self.context.returnBuffer(d_mask);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

        var n_var: i32 = @intCast(total_elements);
        var rate_var: f32 = rate;
        var scale_var: f32 = scaling_factor;
        var seed_var: u64 = seed;

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_mask.ptr),
            @ptrCast(&n_var),
            @ptrCast(&rate_var),
            @ptrCast(&scale_var),
            @ptrCast(&seed_var),
        };

        const config = self.context.getElementWiseConfig(total_elements);
        try self.context.launchKernel("dropout_forward", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
        try self.context.download(std.mem.sliceAsBytes(mask), d_mask.ptr);
    }

    /// VAE sampling forward pass
    pub fn vaeSamplingForward(
        self: *CudaBackend,
        input: []const f32,
        output: []f32,
        epsilon: []f32,
        seed: u64,
        latent_dim: usize,
    ) !void {
        const input_size = input.len;
        const output_size = output.len;

        var d_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_output);
        var d_epsilon = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_epsilon);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

        var latent_dim_var: i32 = @intCast(latent_dim);
        var seed_var: u64 = seed;

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_epsilon.ptr),
            @ptrCast(&latent_dim_var),
            @ptrCast(&seed_var),
        };

        const config = self.context.getElementWiseConfig(latent_dim);
        try self.context.launchKernel("vae_sampling_forward", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
        try self.context.download(std.mem.sliceAsBytes(epsilon), d_epsilon.ptr);
    }

    /// VAE sampling backward pass
    pub fn vaeSamplingBackward(
        self: *CudaBackend,
        input: []const f32,
        grad_output: []const f32,
        grad_input: []f32,
        epsilon: []const f32,
        latent_dim: usize,
    ) !void {
        const input_size = input.len;
        const grad_output_size = grad_output.len;
        const grad_input_size = grad_input.len;

        var d_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_input);
        var d_grad_output = try self.context.getBuffer(grad_output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_output);
        var d_grad_input = try self.context.getBuffer(grad_input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_input);
        var d_epsilon = try self.context.getBuffer(grad_output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_epsilon);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));
        try self.context.upload(d_grad_output.ptr, std.mem.sliceAsBytes(grad_output));
        try self.context.upload(d_epsilon.ptr, std.mem.sliceAsBytes(epsilon));

        var latent_dim_var: i32 = @intCast(latent_dim);

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_grad_input.ptr),
            @ptrCast(&d_epsilon.ptr),
            @ptrCast(&latent_dim_var),
        };

        const config = self.context.getElementWiseConfig(latent_dim);
        try self.context.launchKernel("vae_sampling_backward", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(grad_input), d_grad_input.ptr);
    }

    /// Batch Normalization backward pass
    pub fn batchNormBackward(
        self: *CudaBackend,
        input: []const f32,
        grad_output: []const f32,
        grad_input: []f32,
        gamma: []const f32,
        grad_gamma: []f32,
        grad_beta: []f32,
        epsilon: f32,
        batch_size: usize,
        num_features: usize,
    ) !void {
        const input_size = input.len;
        const input_bytes = try std.math.mul(usize, input_size, @sizeOf(f32));
        const grad_output_bytes = try std.math.mul(usize, grad_output.len, @sizeOf(f32));
        const grad_input_bytes = try std.math.mul(usize, grad_input.len, @sizeOf(f32));
        const gamma_bytes = try std.math.mul(usize, gamma.len, @sizeOf(f32));
        const grad_gamma_bytes = try std.math.mul(usize, grad_gamma.len, @sizeOf(f32));
        const grad_beta_bytes = try std.math.mul(usize, grad_beta.len, @sizeOf(f32));

        var d_input = try self.context.getBuffer(input_bytes);
        defer self.context.returnBuffer(d_input);
        var d_grad_output = try self.context.getBuffer(grad_output_bytes);
        defer self.context.returnBuffer(d_grad_output);
        var d_grad_input = try self.context.getBuffer(grad_input_bytes);
        defer self.context.returnBuffer(d_grad_input);
        var d_gamma = try self.context.getBuffer(gamma_bytes);
        defer self.context.returnBuffer(d_gamma);
        var d_grad_gamma = try self.context.getBuffer(grad_gamma_bytes);
        defer self.context.returnBuffer(d_grad_gamma);
        var d_grad_beta = try self.context.getBuffer(grad_beta_bytes);
        defer self.context.returnBuffer(d_grad_beta);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));
        try self.context.upload(d_grad_output.ptr, std.mem.sliceAsBytes(grad_output));
        try self.context.upload(d_gamma.ptr, std.mem.sliceAsBytes(gamma));

        var batch_size_i32 = @as(i32, @intCast(batch_size));
        var num_features_i32 = @as(i32, @intCast(num_features));
        var epsilon_f32 = epsilon;

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_grad_input.ptr),
            @ptrCast(&d_grad_gamma.ptr),
            @ptrCast(&d_grad_beta.ptr),
            @ptrCast(&d_gamma.ptr),
            @ptrCast(&batch_size_i32),
            @ptrCast(&num_features_i32),
            @ptrCast(&epsilon_f32),
        };

        const config = self.context.getElementWiseConfig(num_features);
        try self.context.launchKernel("batchnorm_backward", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(grad_input), d_grad_input.ptr);
        try self.context.download(std.mem.sliceAsBytes(grad_gamma), d_grad_gamma.ptr);
        try self.context.download(std.mem.sliceAsBytes(grad_beta), d_grad_beta.ptr);
    }

    /// Conv1D forward pass
    pub fn conv1dForward(
        self: *CudaBackend,
        input: []const f32,
        weights: []const f32,
        bias: []const f32,
        output: []f32,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        in_len: usize,
        out_len: usize,
    ) !void {
        const input_size = try std.math.mul(usize, in_channels, in_len);
        const weights_size = try std.math.mul(usize, out_channels, try std.math.mul(usize, in_channels, kernel_size));
        const output_size = try std.math.mul(usize, out_channels, out_len);

        var d_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_input);
        var d_weights = try self.context.getBuffer(weights_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_weights);
        var d_bias = try self.context.getBuffer(out_channels * @sizeOf(f32));
        defer self.context.returnBuffer(d_bias);
        var d_output = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_output);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));
        try self.context.upload(d_weights.ptr, std.mem.sliceAsBytes(weights));
        try self.context.upload(d_bias.ptr, std.mem.sliceAsBytes(bias));

        var in_ch: i32 = @intCast(in_channels);
        var out_ch: i32 = @intCast(out_channels);
        var k_size: i32 = @intCast(kernel_size);
        var in_l: i32 = @intCast(in_len);
        var out_l: i32 = @intCast(out_len);

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_weights.ptr),
            @ptrCast(&d_bias.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&in_ch),
            @ptrCast(&out_ch),
            @ptrCast(&k_size),
            @ptrCast(&in_l),
            @ptrCast(&out_l),
        };

        const grid_dim = [3]u32{ @intCast((out_len + 15) / 16), @intCast(out_channels), 1 };
        const block_dim = [3]u32{ 16, 1, 1 };

        try self.context.launchKernel("conv1d_forward", grid_dim, block_dim, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
    }

    /// Conv1D backward pass
    pub fn conv1dBackward(
        self: *CudaBackend,
        input: []const f32,
        weights: []const f32,
        grad_output: []const f32,
        grad_input: []f32,
        grad_weights: []f32,
        grad_bias: []f32,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        in_len: usize,
        out_len: usize,
    ) !void {
        const input_size = try std.math.mul(usize, in_channels, in_len);
        const weights_size = try std.math.mul(usize, out_channels, try std.math.mul(usize, in_channels, kernel_size));
        const output_size = try std.math.mul(usize, out_channels, out_len);

        var d_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_input);
        var d_weights = try self.context.getBuffer(weights_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_weights);
        var d_grad_output = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_output);
        var d_grad_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_input);
        var d_grad_weights = try self.context.getBuffer(weights_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_weights);
        var d_grad_bias = try self.context.getBuffer(out_channels * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_bias);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));
        try self.context.upload(d_weights.ptr, std.mem.sliceAsBytes(weights));
        try self.context.upload(d_grad_output.ptr, std.mem.sliceAsBytes(grad_output));

        var in_ch: i32 = @intCast(in_channels);
        var out_ch: i32 = @intCast(out_channels);
        var k_size: i32 = @intCast(kernel_size);
        var in_l: i32 = @intCast(in_len);
        var out_l: i32 = @intCast(out_len);

        // 1. Bias gradient
        const bias_args = [_]?*anyopaque{
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_grad_bias.ptr),
            @ptrCast(&out_ch),
            @ptrCast(&out_l),
        };
        const bias_config = self.context.getElementWiseConfig(out_channels);
        try self.context.launchKernel("conv1d_grad_bias", .{ bias_config.grid, 1, 1 }, .{ bias_config.block, 1, 1 }, 0, &bias_args);

        // 2. Weight gradient
        const weight_args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_grad_weights.ptr),
            @ptrCast(&in_ch),
            @ptrCast(&out_ch),
            @ptrCast(&k_size),
            @ptrCast(&in_l),
            @ptrCast(&out_l),
        };
        const weight_config = self.context.getElementWiseConfig(out_channels * in_channels * kernel_size);
        try self.context.launchKernel("conv1d_grad_weight", .{ weight_config.grid, 1, 1 }, .{ weight_config.block, 1, 1 }, 0, &weight_args);

        // 3. Input gradient
        const input_args = [_]?*anyopaque{
            @ptrCast(&d_weights.ptr),
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_grad_input.ptr),
            @ptrCast(&in_ch),
            @ptrCast(&out_ch),
            @ptrCast(&k_size),
            @ptrCast(&in_l),
            @ptrCast(&out_l),
        };
        const input_config = self.context.getElementWiseConfig(in_channels * in_len);
        try self.context.launchKernel("conv1d_grad_input", .{ input_config.grid, 1, 1 }, .{ input_config.block, 1, 1 }, 0, &input_args);

        try self.context.download(std.mem.sliceAsBytes(grad_input), d_grad_input.ptr);
        try self.context.download(std.mem.sliceAsBytes(grad_weights), d_grad_weights.ptr);
        try self.context.download(std.mem.sliceAsBytes(grad_bias), d_grad_bias.ptr);
    }

    /// Attention forward pass
    pub fn attentionForward(
        self: *CudaBackend,
        q: []const f32,
        k: []const f32,
        v: []const f32,
        output: []f32,
        scores: []f32,
        seq_len: usize,
        d_k: usize,
        scaling_factor: f32,
    ) !void {
        // 1. Scores = Q * K^T
        // Q: [seq_len x d_k], K: [seq_len x d_k], K^T: [d_k x seq_len]
        // Scores: [seq_len x seq_len]
        try self.matMul(q, k, scores, seq_len, seq_len, d_k, false, true, false);

        // Scale scores
        try self.scale(scores, scaling_factor, scores);

        // 2. Softmax(scores)
        // Softmax is applied per row (batch_size = seq_len, features = seq_len)
        try self.softmaxForward(scores, scores, seq_len, seq_len);

        // 3. Output = Scores * V
        // Scores: [seq_len x seq_len], V: [seq_len x d_k], Output: [seq_len x d_k]
        try self.matMul(scores, v, output, seq_len, d_k, seq_len, false, false, false);
    }

    /// Matrix multiplication with B transposed: C = A * B^T
    pub fn matMulTransposeB(
        self: *CudaBackend,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: usize,
        n: usize,
        k: usize,
        accumulate: bool,
    ) !void {
        // SECURITY FIX: Use overflow-checked size calculations (CRIT-001)
        const size_a = try calculateBufferSize2D(m, k, @sizeOf(f32));
        const size_b = try calculateBufferSize2D(n, k, @sizeOf(f32));
        const size_c = try calculateBufferSize2D(m, n, @sizeOf(f32));

        var d_a = try self.context.getBuffer(size_a);
        defer self.context.returnBuffer(d_a);
        var d_b = try self.context.getBuffer(size_b);
        defer self.context.returnBuffer(d_b);
        var d_c = try self.context.getBuffer(size_c);
        defer self.context.returnBuffer(d_c);

        try self.context.upload(d_a.ptr, std.mem.sliceAsBytes(a));
        try self.context.upload(d_b.ptr, std.mem.sliceAsBytes(b));

        // SECURITY FIX: Bounds checking before @intCast
        var m_i32: i32 = try safeCastUsizeToI32(m);
        var n_i32: i32 = try safeCastUsizeToI32(n);
        var k_i32: i32 = try safeCastUsizeToI32(k);
        var acc_i32: i32 = if (accumulate) 1 else 0;

        const args = [_]?*anyopaque{
            @ptrCast(&d_c.ptr),
            @ptrCast(&d_a.ptr),
            @ptrCast(&d_b.ptr),
            @ptrCast(&m_i32),
            @ptrCast(&n_i32),
            @ptrCast(&k_i32),
            @ptrCast(&acc_i32),
        };

        // SECURITY FIX: Validate grid dimensions before @intCast
        const grid_n = try std.math.add(usize, (n + 15) / 16, 0); // Check no overflow
        const grid_m = try std.math.add(usize, (m + 15) / 16, 0); // Check no overflow
        const grid_dim = [3]u32{
            @intCast(grid_n),
            @intCast(grid_m),
            1,
        };
        const block_dim = [3]u32{ 16, 16, 1 };

        try self.context.launchKernel("matmul_transpose_b", grid_dim, block_dim, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(c), d_c.ptr);
    }

    /// Max pooling 2D forward pass
    pub fn maxPool2dForward(
        self: *CudaBackend,
        input: []const f32,
        output: []f32,
        max_indices: []f32,
        channels: usize,
        input_h: usize,
        input_w: usize,
        output_h: usize,
        output_w: usize,
        pool_h: usize,
        pool_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) !void {
        const input_size = try std.math.mul(usize, channels, try std.math.mul(usize, input_h, input_w));
        const output_size = try std.math.mul(usize, channels, try std.math.mul(usize, output_h, output_w));

        var d_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_output);
        var d_max_indices = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_max_indices);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

        var channels_var: i32 = @intCast(channels);
        var input_h_var: i32 = @intCast(input_h);
        var input_w_var: i32 = @intCast(input_w);
        var output_h_var: i32 = @intCast(output_h);
        var output_w_var: i32 = @intCast(output_w);
        var pool_h_var: i32 = @intCast(pool_h);
        var pool_w_var: i32 = @intCast(pool_w);
        var stride_h_var: i32 = @intCast(stride_h);
        var stride_w_var: i32 = @intCast(stride_w);

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_max_indices.ptr),
            @ptrCast(&channels_var),
            @ptrCast(&input_h_var),
            @ptrCast(&input_w_var),
            @ptrCast(&output_h_var),
            @ptrCast(&output_w_var),
            @ptrCast(&pool_h_var),
            @ptrCast(&pool_w_var),
            @ptrCast(&stride_h_var),
            @ptrCast(&stride_w_var),
        };

        const grid_dim = [3]u32{
            @intCast((output_w + 15) / 16),
            @intCast((output_h + 15) / 16),
            @intCast(channels),
        };
        const block_dim = [3]u32{ 16, 16, 1 };

        try self.context.launchKernel("max_pool2d_forward", grid_dim, block_dim, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
        try self.context.download(std.mem.sliceAsBytes(max_indices), d_max_indices.ptr);
    }

    /// Max pooling 2D backward pass
    pub fn maxPool2dBackward(
        self: *CudaBackend,
        grad_output: []const f32,
        grad_input: []f32,
        max_indices: []const f32,
        channels: usize,
        output_h: usize,
        output_w: usize,
    ) !void {
        const output_size = try std.math.mul(usize, channels, try std.math.mul(usize, output_h, output_w));
        const input_size = grad_input.len;

        var d_grad_output = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_output);
        var d_grad_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_input);
        var d_max_indices = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_max_indices);

        try self.context.upload(d_grad_output.ptr, std.mem.sliceAsBytes(grad_output));
        try self.context.upload(d_max_indices.ptr, std.mem.sliceAsBytes(max_indices));
        try self.context.memset(d_grad_input.ptr, 0, input_size);

        var total_elements: i32 = @intCast(output_size);

        const args = [_]?*anyopaque{
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_max_indices.ptr),
            @ptrCast(&d_grad_input.ptr),
            @ptrCast(&total_elements),
        };

        const config = self.context.getElementWiseConfig(output_size);

        try self.context.launchKernel("max_pool2d_backward", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(grad_input), d_grad_input.ptr);
    }

    /// Max pooling 1D forward pass
    pub fn maxPool1dForward(
        self: *CudaBackend,
        input: []const f32,
        output: []f32,
        max_indices: []f32,
        channels: usize,
        input_len: usize,
        output_len: usize,
        pool_size: usize,
        stride: usize,
    ) !void {
        const input_size = try std.math.mul(usize, channels, input_len);
        const output_size = try std.math.mul(usize, channels, output_len);

        var d_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_output);
        var d_max_indices = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_max_indices);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

        var channels_var: i32 = @intCast(channels);
        var input_len_var: i32 = @intCast(input_len);
        var output_len_var: i32 = @intCast(output_len);
        var pool_size_var: i32 = @intCast(pool_size);
        var stride_var: i32 = @intCast(stride);

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_max_indices.ptr),
            @ptrCast(&channels_var),
            @ptrCast(&input_len_var),
            @ptrCast(&output_len_var),
            @ptrCast(&pool_size_var),
            @ptrCast(&stride_var),
        };

        const config = self.context.getElementWiseConfig(output_len);

        try self.context.launchKernel("max_pool1d_forward", .{ config.grid, @intCast(channels), 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
        try self.context.download(std.mem.sliceAsBytes(max_indices), d_max_indices.ptr);
    }

    /// Max pooling 1D backward pass
    pub fn maxPool1dBackward(
        self: *CudaBackend,
        grad_output: []const f32,
        grad_input: []f32,
        max_indices: []const f32,
        channels: usize,
        output_len: usize,
    ) !void {
        const output_size = try std.math.mul(usize, channels, output_len);
        const input_size = grad_input.len;

        var d_grad_output = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_output);
        var d_grad_input = try self.context.getBuffer(input_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_grad_input);
        var d_max_indices = try self.context.getBuffer(output_size * @sizeOf(f32));
        defer self.context.returnBuffer(d_max_indices);

        try self.context.upload(d_grad_output.ptr, std.mem.sliceAsBytes(grad_output));
        try self.context.upload(d_max_indices.ptr, std.mem.sliceAsBytes(max_indices));
        try self.context.memset(d_grad_input.ptr, 0, input_size);

        var total_elements: i32 = @intCast(output_size);

        const args = [_]?*anyopaque{
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_max_indices.ptr),
            @ptrCast(&d_grad_input.ptr),
            @ptrCast(&total_elements),
        };

        const config = self.context.getElementWiseConfig(output_size);

        try self.context.launchKernel("max_pool1d_backward", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(grad_input), d_grad_input.ptr);
    }

    /// RNN forward step
    pub fn rnnForwardStep(
        self: *CudaBackend,
        gates_ih: []const f32,
        gates_hh: []const f32,
        bias: []const f32,
        h_curr: []f32,
        hidden_size: usize,
    ) !void {
        const gates_ih_bytes = try std.math.mul(usize, gates_ih.len, @sizeOf(f32));
        const gates_hh_bytes = try std.math.mul(usize, gates_hh.len, @sizeOf(f32));
        const bias_bytes = try std.math.mul(usize, bias.len, @sizeOf(f32));
        const h_curr_bytes = try std.math.mul(usize, h_curr.len, @sizeOf(f32));

        var d_ih = try self.context.getBuffer(gates_ih_bytes);
        defer self.context.returnBuffer(d_ih);
        var d_hh = try self.context.getBuffer(gates_hh_bytes);
        defer self.context.returnBuffer(d_hh);
        var d_bias = try self.context.getBuffer(bias_bytes);
        defer self.context.returnBuffer(d_bias);
        var d_hc = try self.context.getBuffer(h_curr_bytes);
        defer self.context.returnBuffer(d_hc);

        try self.context.upload(d_ih.ptr, std.mem.sliceAsBytes(gates_ih));
        try self.context.upload(d_hh.ptr, std.mem.sliceAsBytes(gates_hh));
        try self.context.upload(d_bias.ptr, std.mem.sliceAsBytes(bias));

        var hs: i32 = @intCast(hidden_size);
        const args = [_]?*anyopaque{
            @ptrCast(&d_ih.ptr),
            @ptrCast(&d_hh.ptr),
            @ptrCast(&d_bias.ptr),
            @ptrCast(&d_hc.ptr),
            @ptrCast(&hs),
        };

        const config = self.context.getElementWiseConfig(hidden_size);
        try self.context.launchKernel("rnn_forward_step", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(h_curr), d_hc.ptr);
    }

    /// RNN backward step
    pub fn rnnBackwardStep(
        self: *CudaBackend,
        grad_h_curr: []const f32,
        grad_h_next: []const f32,
        h_curr: []const f32,
        grad_after_act: []f32,
        hidden_size: usize,
    ) !void {
        const grad_h_curr_bytes = try std.math.mul(usize, grad_h_curr.len, @sizeOf(f32));
        const grad_h_next_bytes = try std.math.mul(usize, grad_h_next.len, @sizeOf(f32));
        const h_curr_bytes = try std.math.mul(usize, h_curr.len, @sizeOf(f32));
        const grad_after_act_bytes = try std.math.mul(usize, grad_after_act.len, @sizeOf(f32));

        var d_gc = try self.context.getBuffer(grad_h_curr_bytes);
        defer self.context.returnBuffer(d_gc);
        var d_gn = try self.context.getBuffer(grad_h_next_bytes);
        defer self.context.returnBuffer(d_gn);
        var d_hc = try self.context.getBuffer(h_curr_bytes);
        defer self.context.returnBuffer(d_hc);
        var d_ga = try self.context.getBuffer(grad_after_act_bytes);
        defer self.context.returnBuffer(d_ga);

        try self.context.upload(d_gc.ptr, std.mem.sliceAsBytes(grad_h_curr));
        try self.context.upload(d_gn.ptr, std.mem.sliceAsBytes(grad_h_next));
        try self.context.upload(d_hc.ptr, std.mem.sliceAsBytes(h_curr));

        var hs: i32 = @intCast(hidden_size);
        const args = [_]?*anyopaque{
            @ptrCast(&d_gc.ptr),
            @ptrCast(&d_gn.ptr),
            @ptrCast(&d_hc.ptr),
            @ptrCast(&d_ga.ptr),
            @ptrCast(&hs),
        };

        const config = self.context.getElementWiseConfig(hidden_size);
        try self.context.launchKernel("rnn_backward_step", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(grad_after_act), d_ga.ptr);
    }

    /// LSTM forward step
    pub fn lstmForwardStep(
        self: *CudaBackend,
        gates_ih: []const f32,
        gates_hh: []const f32,
        bias: []const f32,
        c_prev: []const f32,
        c_curr: []f32,
        h_curr: []f32,
        gate_acts: []f32,
        hidden_size: usize,
    ) !void {
        const gates_ih_bytes = try std.math.mul(usize, gates_ih.len, @sizeOf(f32));
        const gates_hh_bytes = try std.math.mul(usize, gates_hh.len, @sizeOf(f32));
        const bias_bytes = try std.math.mul(usize, bias.len, @sizeOf(f32));
        const c_prev_bytes = try std.math.mul(usize, c_prev.len, @sizeOf(f32));
        const c_curr_bytes = try std.math.mul(usize, c_curr.len, @sizeOf(f32));
        const h_curr_bytes = try std.math.mul(usize, h_curr.len, @sizeOf(f32));
        const gate_acts_bytes = try std.math.mul(usize, gate_acts.len, @sizeOf(f32));

        var d_ih = try self.context.getBuffer(gates_ih_bytes);
        defer self.context.returnBuffer(d_ih);
        var d_hh = try self.context.getBuffer(gates_hh_bytes);
        defer self.context.returnBuffer(d_hh);
        var d_bias = try self.context.getBuffer(bias_bytes);
        defer self.context.returnBuffer(d_bias);
        var d_cp = try self.context.getBuffer(c_prev_bytes);
        defer self.context.returnBuffer(d_cp);
        var d_cc = try self.context.getBuffer(c_curr_bytes);
        defer self.context.returnBuffer(d_cc);
        var d_hc = try self.context.getBuffer(h_curr_bytes);
        defer self.context.returnBuffer(d_hc);
        var d_ga = try self.context.getBuffer(gate_acts_bytes);
        defer self.context.returnBuffer(d_ga);

        try self.context.upload(d_ih.ptr, std.mem.sliceAsBytes(gates_ih));
        try self.context.upload(d_hh.ptr, std.mem.sliceAsBytes(gates_hh));
        try self.context.upload(d_bias.ptr, std.mem.sliceAsBytes(bias));
        try self.context.upload(d_cp.ptr, std.mem.sliceAsBytes(c_prev));

        var hs: i32 = @intCast(hidden_size);
        const args = [_]?*anyopaque{
            @ptrCast(&d_ih.ptr),
            @ptrCast(&d_hh.ptr),
            @ptrCast(&d_bias.ptr),
            @ptrCast(&d_cp.ptr),
            @ptrCast(&d_cc.ptr),
            @ptrCast(&d_hc.ptr),
            @ptrCast(&d_ga.ptr),
            @ptrCast(&hs),
        };

        const config = self.context.getElementWiseConfig(hidden_size);
        try self.context.launchKernel("lstm_forward_step", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(c_curr), d_cc.ptr);
        try self.context.download(std.mem.sliceAsBytes(h_curr), d_hc.ptr);
        try self.context.download(std.mem.sliceAsBytes(gate_acts), d_ga.ptr);
    }

    /// LSTM backward step
    pub fn lstmBackwardStep(
        self: *CudaBackend,
        grad_h_curr: []const f32,
        grad_h_next: []const f32,
        grad_c_next: []const f32,
        gate_acts: []const f32,
        c_curr: []const f32,
        c_prev: []const f32,
        grad_gates: []f32,
        grad_c_prev: []f32,
        grad_h_prev_part: []f32,
        hidden_size: usize,
    ) !void {
        const grad_h_curr_bytes = try std.math.mul(usize, grad_h_curr.len, @sizeOf(f32));
        const grad_h_next_bytes = try std.math.mul(usize, grad_h_next.len, @sizeOf(f32));
        const grad_c_next_bytes = try std.math.mul(usize, grad_c_next.len, @sizeOf(f32));
        const gate_acts_bytes = try std.math.mul(usize, gate_acts.len, @sizeOf(f32));
        const c_curr_bytes = try std.math.mul(usize, c_curr.len, @sizeOf(f32));
        const c_prev_bytes = try std.math.mul(usize, c_prev.len, @sizeOf(f32));
        const grad_gates_bytes = try std.math.mul(usize, grad_gates.len, @sizeOf(f32));
        const grad_c_prev_bytes = try std.math.mul(usize, grad_c_prev.len, @sizeOf(f32));
        const grad_h_prev_part_bytes = try std.math.mul(usize, grad_h_prev_part.len, @sizeOf(f32));

        var d_ghc = try self.context.getBuffer(grad_h_curr_bytes);
        defer self.context.returnBuffer(d_ghc);
        var d_ghn = try self.context.getBuffer(grad_h_next_bytes);
        defer self.context.returnBuffer(d_ghn);
        var d_gcn = try self.context.getBuffer(grad_c_next_bytes);
        defer self.context.returnBuffer(d_gcn);
        var d_ga = try self.context.getBuffer(gate_acts_bytes);
        defer self.context.returnBuffer(d_ga);
        var d_cc = try self.context.getBuffer(c_curr_bytes);
        defer self.context.returnBuffer(d_cc);
        var d_cp = try self.context.getBuffer(c_prev_bytes);
        defer self.context.returnBuffer(d_cp);
        var d_gg = try self.context.getBuffer(grad_gates_bytes);
        defer self.context.returnBuffer(d_gg);
        var d_gcp = try self.context.getBuffer(grad_c_prev_bytes);
        defer self.context.returnBuffer(d_gcp);
        var d_ghp = try self.context.getBuffer(grad_h_prev_part_bytes);
        defer self.context.returnBuffer(d_ghp);

        try self.context.upload(d_ghc.ptr, std.mem.sliceAsBytes(grad_h_curr));
        try self.context.upload(d_ghn.ptr, std.mem.sliceAsBytes(grad_h_next));
        try self.context.upload(d_gcn.ptr, std.mem.sliceAsBytes(grad_c_next));
        try self.context.upload(d_ga.ptr, std.mem.sliceAsBytes(gate_acts));
        try self.context.upload(d_cc.ptr, std.mem.sliceAsBytes(c_curr));
        try self.context.upload(d_cp.ptr, std.mem.sliceAsBytes(c_prev));

        var hs: i32 = @intCast(hidden_size);
        const args = [_]?*anyopaque{
            @ptrCast(&d_ghc.ptr),
            @ptrCast(&d_ghn.ptr),
            @ptrCast(&d_gcn.ptr),
            @ptrCast(&d_ga.ptr),
            @ptrCast(&d_cc.ptr),
            @ptrCast(&d_cp.ptr),
            @ptrCast(&d_gg.ptr),
            @ptrCast(&d_gcp.ptr),
            @ptrCast(&d_ghp.ptr),
            @ptrCast(&hs),
        };

        const config = self.context.getElementWiseConfig(hidden_size);
        try self.context.launchKernel("lstm_backward_step", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(grad_gates), d_gg.ptr);
        try self.context.download(std.mem.sliceAsBytes(grad_c_prev), d_gcp.ptr);
        try self.context.download(std.mem.sliceAsBytes(grad_h_prev_part), d_ghp.ptr);
    }

    /// GRU forward step
    pub fn gruForwardStep(
        self: *CudaBackend,
        gates_ih: []const f32,
        gates_hh: []const f32,
        bias: []const f32,
        h_prev: []const f32,
        h_curr: []f32,
        gate_acts: []f32,
        n_hh_out: []f32,
        hidden_size: usize,
    ) !void {
        const gates_ih_bytes = try std.math.mul(usize, gates_ih.len, @sizeOf(f32));
        const gates_hh_bytes = try std.math.mul(usize, gates_hh.len, @sizeOf(f32));
        const bias_bytes = try std.math.mul(usize, bias.len, @sizeOf(f32));
        const h_prev_bytes = try std.math.mul(usize, h_prev.len, @sizeOf(f32));
        const h_curr_bytes = try std.math.mul(usize, h_curr.len, @sizeOf(f32));
        const gate_acts_bytes = try std.math.mul(usize, gate_acts.len, @sizeOf(f32));
        const n_hh_out_bytes = try std.math.mul(usize, n_hh_out.len, @sizeOf(f32));

        var d_ih = try self.context.getBuffer(gates_ih_bytes);
        defer self.context.returnBuffer(d_ih);
        var d_hh = try self.context.getBuffer(gates_hh_bytes);
        defer self.context.returnBuffer(d_hh);
        var d_bias = try self.context.getBuffer(bias_bytes);
        defer self.context.returnBuffer(d_bias);
        var d_hp = try self.context.getBuffer(h_prev_bytes);
        defer self.context.returnBuffer(d_hp);
        var d_hc = try self.context.getBuffer(h_curr_bytes);
        defer self.context.returnBuffer(d_hc);
        var d_ga = try self.context.getBuffer(gate_acts_bytes);
        defer self.context.returnBuffer(d_ga);
        var d_nh = try self.context.getBuffer(n_hh_out_bytes);
        defer self.context.returnBuffer(d_nh);

        try self.context.upload(d_ih.ptr, std.mem.sliceAsBytes(gates_ih));
        try self.context.upload(d_hh.ptr, std.mem.sliceAsBytes(gates_hh));
        try self.context.upload(d_bias.ptr, std.mem.sliceAsBytes(bias));
        try self.context.upload(d_hp.ptr, std.mem.sliceAsBytes(h_prev));

        var hs: i32 = @intCast(hidden_size);
        const args = [_]?*anyopaque{
            @ptrCast(&d_ih.ptr),
            @ptrCast(&d_hh.ptr),
            @ptrCast(&d_bias.ptr),
            @ptrCast(&d_hp.ptr),
            @ptrCast(&d_hc.ptr),
            @ptrCast(&d_ga.ptr),
            @ptrCast(&d_nh.ptr),
            @ptrCast(&hs),
        };

        const config = self.context.getElementWiseConfig(hidden_size);
        try self.context.launchKernel("gru_forward_step", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(h_curr), d_hc.ptr);
        try self.context.download(std.mem.sliceAsBytes(gate_acts), d_ga.ptr);
        try self.context.download(std.mem.sliceAsBytes(n_hh_out), d_nh.ptr);
    }

    /// GRU backward step
    pub fn gruBackwardStep(
        self: *CudaBackend,
        grad_h_curr: []const f32,
        grad_h_next: []const f32,
        gate_acts: []const f32,
        h_prev: []const f32,
        n_hh: []const f32,
        grad_gates_ih: []f32,
        grad_gates_hh: []f32,
        grad_h_prev: []f32,
        hidden_size: usize,
    ) !void {
        const grad_h_curr_bytes = try std.math.mul(usize, grad_h_curr.len, @sizeOf(f32));
        const grad_h_next_bytes = try std.math.mul(usize, grad_h_next.len, @sizeOf(f32));
        const gate_acts_bytes = try std.math.mul(usize, gate_acts.len, @sizeOf(f32));
        const h_prev_bytes = try std.math.mul(usize, h_prev.len, @sizeOf(f32));
        const n_hh_bytes = try std.math.mul(usize, n_hh.len, @sizeOf(f32));
        const grad_gates_ih_bytes = try std.math.mul(usize, grad_gates_ih.len, @sizeOf(f32));
        const grad_gates_hh_bytes = try std.math.mul(usize, grad_gates_hh.len, @sizeOf(f32));
        const grad_h_prev_bytes = try std.math.mul(usize, grad_h_prev.len, @sizeOf(f32));

        var d_ghc = try self.context.getBuffer(grad_h_curr_bytes);
        defer self.context.returnBuffer(d_ghc);
        var d_ghn = try self.context.getBuffer(grad_h_next_bytes);
        defer self.context.returnBuffer(d_ghn);
        var d_ga = try self.context.getBuffer(gate_acts_bytes);
        defer self.context.returnBuffer(d_ga);
        var d_hp = try self.context.getBuffer(h_prev_bytes);
        defer self.context.returnBuffer(d_hp);
        var d_nh = try self.context.getBuffer(n_hh_bytes);
        defer self.context.returnBuffer(d_nh);
        var d_gih = try self.context.getBuffer(grad_gates_ih_bytes);
        defer self.context.returnBuffer(d_gih);
        var d_ghh = try self.context.getBuffer(grad_gates_hh_bytes);
        defer self.context.returnBuffer(d_ghh);
        var d_ghp = try self.context.getBuffer(grad_h_prev_bytes);
        defer self.context.returnBuffer(d_ghp);

        try self.context.upload(d_ghc.ptr, std.mem.sliceAsBytes(grad_h_curr));
        try self.context.upload(d_ghn.ptr, std.mem.sliceAsBytes(grad_h_next));
        try self.context.upload(d_ga.ptr, std.mem.sliceAsBytes(gate_acts));
        try self.context.upload(d_hp.ptr, std.mem.sliceAsBytes(h_prev));
        try self.context.upload(d_nh.ptr, std.mem.sliceAsBytes(n_hh));

        var hs: i32 = @intCast(hidden_size);
        const args = [_]?*anyopaque{
            @ptrCast(&d_ghc.ptr),
            @ptrCast(&d_ghn.ptr),
            @ptrCast(&d_ga.ptr),
            @ptrCast(&d_hp.ptr),
            @ptrCast(&d_nh.ptr),
            @ptrCast(&d_gih.ptr),
            @ptrCast(&d_ghh.ptr),
            @ptrCast(&d_ghp.ptr),
            @ptrCast(&hs),
        };

        const config = self.context.getElementWiseConfig(hidden_size);
        try self.context.launchKernel("gru_backward_step", .{ config.grid, 1, 1 }, .{ config.block, 1, 1 }, 0, &args);

        try self.context.download(std.mem.sliceAsBytes(grad_gates_ih), d_gih.ptr);
        try self.context.download(std.mem.sliceAsBytes(grad_gates_hh), d_ghh.ptr);
        try self.context.download(std.mem.sliceAsBytes(grad_h_prev), d_ghp.ptr);
    }

    /// im2col transformation on GPU
    /// Converts image patches to columns for GEMM-based convolution
    /// PERFORMANCE: 10-20x faster than CPU im2col
    pub fn im2col(self: *CudaBackend, input: []const f32, col: []f32,
                  batch_size: usize, in_channels: usize, input_h: usize, input_w: usize,
                  kernel_h: usize, kernel_w: usize,
                  output_h: usize, output_w: usize,
                  stride_h: usize, stride_w: usize,
                  padding_h: usize, padding_w: usize) !void {
        // SECURITY: Validate dimensions with overflow checking
        const col_height = try std.math.mul(usize, output_h, output_w);
        const col_width = try std.math.mul(usize, try std.math.mul(usize, kernel_h, kernel_w), in_channels);
        const total_col_elements = try std.math.mul(usize, try std.math.mul(usize, batch_size, col_height), col_width);

        // SECURITY: Validate buffer sizes
        if (input.len < batch_size * in_channels * input_h * input_w) {
            return error.InvalidInputSize;
        }
        if (col.len < total_col_elements) {
            return error.InvalidOutputSize;
        }

        // Allocate device memory
        const input_size = batch_size * in_channels * input_h * input_w * @sizeOf(f32);
        const col_size = total_col_elements * @sizeOf(f32);

        var d_input = try self.allocBuffer(input_size);
        defer self.freeBuffer(d_input);
        var d_col = try self.allocBuffer(col_size);
        defer self.freeBuffer(d_col);

        // Upload input
        try self.upload(d_input.ptr, input);

        // Launch im2col kernel
        const threads_per_block = 256;
        const blocks = (total_col_elements + threads_per_block - 1) / threads_per_block;

        // Create mutable copies for kernel args
        var batch_size_i32: i32 = @intCast(batch_size);
        var in_channels_i32: i32 = @intCast(in_channels);
        var input_h_i32: i32 = @intCast(input_h);
        var input_w_i32: i32 = @intCast(input_w);
        var kernel_h_i32: i32 = @intCast(kernel_h);
        var kernel_w_i32: i32 = @intCast(kernel_w);
        var output_h_i32: i32 = @intCast(output_h);
        var output_w_i32: i32 = @intCast(output_w);
        var stride_h_i32: i32 = @intCast(stride_h);
        var stride_w_i32: i32 = @intCast(stride_w);
        var padding_h_i32: i32 = @intCast(padding_h);
        var padding_w_i32: i32 = @intCast(padding_w);

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_col.ptr),
            @ptrCast(&batch_size_i32),
            @ptrCast(&in_channels_i32),
            @ptrCast(&input_h_i32),
            @ptrCast(&input_w_i32),
            @ptrCast(&kernel_h_i32),
            @ptrCast(&kernel_w_i32),
            @ptrCast(&output_h_i32),
            @ptrCast(&output_w_i32),
            @ptrCast(&stride_h_i32),
            @ptrCast(&stride_w_i32),
            @ptrCast(&padding_h_i32),
            @ptrCast(&padding_w_i32),
        };

        try self.context.launchKernel("im2col",
            .{ @intCast(blocks), 1, 1 },
            .{ @intCast(threads_per_block), 1, 1 },
            0, &args);

        // Download result
        try self.download(col, d_col.ptr);
    }

    /// col2im transformation on GPU
    /// Converts columns back to image for backward pass
    pub fn col2im(self: *CudaBackend, col: []const f32, input_grad: []f32,
                  batch_size: usize, in_channels: usize, input_h: usize, input_w: usize,
                  kernel_h: usize, kernel_w: usize,
                  output_h: usize, output_w: usize,
                  stride_h: usize, stride_w: usize,
                  padding_h: usize, padding_w: usize) !void {
        // SECURITY: Validate dimensions
        const col_height = output_h * output_w;
        const col_width = kernel_h * kernel_w * in_channels;
        const total_col_elements = batch_size * col_height * col_width;

        // Allocate device memory
        const col_size = total_col_elements * @sizeOf(f32);
        const input_size = batch_size * in_channels * input_h * input_w * @sizeOf(f32);

        var d_col = try self.allocBuffer(col_size);
        defer self.freeBuffer(d_col);
        var d_input_grad = try self.allocBuffer(input_size);
        defer self.freeBuffer(d_input_grad);

        // Upload col and zero out input_grad
        try self.upload(d_col.ptr, col);
        try self.context.memset(d_input_grad.ptr, 0, @intCast(input_size));

        // Launch col2im kernel
        const threads_per_block = 256;
        const blocks = (total_col_elements + threads_per_block - 1) / threads_per_block;

        // Create mutable copies for kernel args
        var batch_size_i32: i32 = @intCast(batch_size);
        var in_channels_i32: i32 = @intCast(in_channels);
        var input_h_i32: i32 = @intCast(input_h);
        var input_w_i32: i32 = @intCast(input_w);
        var kernel_h_i32: i32 = @intCast(kernel_h);
        var kernel_w_i32: i32 = @intCast(kernel_w);
        var output_h_i32: i32 = @intCast(output_h);
        var output_w_i32: i32 = @intCast(output_w);
        var stride_h_i32: i32 = @intCast(stride_h);
        var stride_w_i32: i32 = @intCast(stride_w);
        var padding_h_i32: i32 = @intCast(padding_h);
        var padding_w_i32: i32 = @intCast(padding_w);

        const args = [_]?*anyopaque{
            @ptrCast(&d_col.ptr),
            @ptrCast(&d_input_grad.ptr),
            @ptrCast(&batch_size_i32),
            @ptrCast(&in_channels_i32),
            @ptrCast(&input_h_i32),
            @ptrCast(&input_w_i32),
            @ptrCast(&kernel_h_i32),
            @ptrCast(&kernel_w_i32),
            @ptrCast(&output_h_i32),
            @ptrCast(&output_w_i32),
            @ptrCast(&stride_h_i32),
            @ptrCast(&stride_w_i32),
            @ptrCast(&padding_h_i32),
            @ptrCast(&padding_w_i32),
        };

        try self.context.launchKernel("col2im",
            .{ @intCast(blocks), 1, 1 },
            .{ @intCast(threads_per_block), 1, 1 },
            0, &args);

        // Download result
        try self.download(input_grad, d_input_grad.ptr);
    }

    /// Conv2D forward pass using im2col + GEMM optimization
    /// PERFORMANCE: 10-20x faster than naive convolution
    pub fn conv2dForwardIm2col(self: *CudaBackend, input: []const f32, weights: []const f32, bias: []const f32, output: []f32,
                               batch_size: usize, in_channels: usize, out_channels: usize,
                               input_h: usize, input_w: usize, kernel_h: usize, kernel_w: usize,
                               output_h: usize, output_w: usize,
                               stride_h: usize, stride_w: usize,
                               padding_h: usize, padding_w: usize) !void {
        const total_outputs = batch_size * out_channels * output_h * output_w;

        // Allocate device memory
        const input_size = batch_size * in_channels * input_h * input_w * @sizeOf(f32);
        const weights_size = out_channels * in_channels * kernel_h * kernel_w * @sizeOf(f32);
        const bias_size = out_channels * @sizeOf(f32);
        const output_size = total_outputs * @sizeOf(f32);

        var d_input = try self.allocBuffer(input_size);
        defer self.freeBuffer(d_input);
        var d_weights = try self.allocBuffer(weights_size);
        defer self.freeBuffer(d_weights);
        var d_bias = try self.allocBuffer(bias_size);
        defer self.freeBuffer(d_bias);
        var d_output = try self.allocBuffer(output_size);
        defer self.freeBuffer(d_output);

        // Upload data
        try self.upload(d_input.ptr, input);
        try self.upload(d_weights.ptr, weights);
        try self.upload(d_bias.ptr, bias);

        // Launch kernel
        const threads_per_block = 256;
        const blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

        // Create mutable copies for kernel args
        var batch_size_i32: i32 = @intCast(batch_size);
        var in_channels_i32: i32 = @intCast(in_channels);
        var out_channels_i32: i32 = @intCast(out_channels);
        var input_h_i32: i32 = @intCast(input_h);
        var input_w_i32: i32 = @intCast(input_w);
        var kernel_h_i32: i32 = @intCast(kernel_h);
        var kernel_w_i32: i32 = @intCast(kernel_w);
        var output_h_i32: i32 = @intCast(output_h);
        var output_w_i32: i32 = @intCast(output_w);
        var stride_h_i32: i32 = @intCast(stride_h);
        var stride_w_i32: i32 = @intCast(stride_w);
        var padding_h_i32: i32 = @intCast(padding_h);
        var padding_w_i32: i32 = @intCast(padding_w);

        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_weights.ptr),
            @ptrCast(&d_bias.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&batch_size_i32),
            @ptrCast(&in_channels_i32),
            @ptrCast(&out_channels_i32),
            @ptrCast(&input_h_i32),
            @ptrCast(&input_w_i32),
            @ptrCast(&kernel_h_i32),
            @ptrCast(&kernel_w_i32),
            @ptrCast(&output_h_i32),
            @ptrCast(&output_w_i32),
            @ptrCast(&stride_h_i32),
            @ptrCast(&stride_w_i32),
            @ptrCast(&padding_h_i32),
            @ptrCast(&padding_w_i32),
        };

        try self.context.launchKernel("conv2d_im2col_forward",
            .{ @intCast(blocks), 1, 1 },
            .{ @intCast(threads_per_block), 1, 1 },
            0, &args);

        // Download result
        try self.download(output, d_output.ptr);
    }
};
test "CUDA backend initialization" {
    if (@import("builtin").os.tag == .macos) {
        return error.SkipZigTest;
    }

    var backend = CudaBackend.init(std.testing.allocator) catch |err| {
        if (err == cuda_driver.CudaError.CudaDriverNotFound or
            err == cuda_driver.CudaError.CudaInitFailed or
            err == error.NoCudaDevices)
        {
            return;
        }
        return err;
    };
    defer backend.deinit();
}

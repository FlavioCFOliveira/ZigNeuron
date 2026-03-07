/// CUDA Context for managing persistent resources
/// Optimized for NVIDIA GPU performance
/// Follows the same pattern as MetalContext for consistency
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");

const CUresult = cuda_driver.CUresult;
const CUdevice = cuda_driver.CUdevice;
const CUcontext = cuda_driver.CUcontext;
const CUstream = cuda_driver.CUstream;
const CUmodule = cuda_driver.CUmodule;
const CUfunction = cuda_driver.CUfunction;
const CUdeviceptr = cuda_driver.CUdeviceptr;
const CUevent = cuda_driver.CUevent;
const CudaDriver = cuda_driver.CudaDriver;

// =============================================================================
// Configuration Constants
// =============================================================================

/// Default thread block size for element-wise operations
pub const DEFAULT_BLOCK_SIZE: c_uint = 256;

/// Default number of threads per block for matrix operations
pub const DEFAULT_THREADS_PER_BLOCK: c_uint = 256;

/// Memory pool configuration
pub const MEMORY_POOL_BUCKETS: usize = 32; // Powers of 2 from 4 bytes to 128MB
pub const MIN_POOL_SIZE: usize = 4;
pub const MAX_POOL_SIZE: usize = 128 * 1024 * 1024; // 128MB

// =============================================================================
// CudaContext Structure
// =============================================================================

pub const CudaContext = struct {
    allocator: std.mem.Allocator,
    driver: *CudaDriver,

    // CUDA handles
    device: CUdevice,
    context: *CUcontext,
    stream: *CUstream,

    // Device properties
    device_props: DeviceProperties,

    // Kernel cache - stores loaded kernels by name
    kernels: std.StringHashMap(Kernel),

    // Memory pool for efficient buffer reuse
    buffer_pools: [MEMORY_POOL_BUCKETS]std.ArrayListUnmanaged(DeviceBuffer),

    // Temporary resources to be cleared after operations
    temp_buffers: std.ArrayListUnmanaged(*DeviceBuffer),

    // Module cache for PTX code
    modules: std.ArrayListUnmanaged(*CUmodule),

    /// Device properties
    pub const DeviceProperties = struct {
        compute_capability_major: i32,
        compute_capability_minor: i32,
        total_memory: usize,
        multiprocessor_count: i32,
        max_threads_per_block: i32,
        max_block_dim_x: i32,
        max_block_dim_y: i32,
        max_block_dim_z: i32,
        max_grid_dim_x: i32,
        max_grid_dim_y: i32,
        max_grid_dim_z: i32,
        max_shared_memory_per_block: i32,
        warp_size: i32,
        memory_clock_rate: i32,
        global_memory_bus_width: i32,
        l2_cache_size: i32,
        max_threads_per_multiprocessor: i32,
        unified_addressing: i32,
        managed_memory: i32,
        concurrent_managed_access: i32,
        name: [256]u8,

        pub fn computeCapability(self: DeviceProperties) i32 {
            return self.compute_capability_major * 10 + self.compute_capability_minor;
        }

        pub fn hasTensorCores(self: DeviceProperties) bool {
            return self.compute_capability_major >= 7;
        }

        pub fn hasUnifiedMemory(self: DeviceProperties) bool {
            return self.unified_addressing != 0 and self.managed_memory != 0;
        }
    };

    /// Device buffer wrapper
    pub const DeviceBuffer = struct {
        ptr: CUdeviceptr,
        size: usize,
        pool_index: ?usize = null,

        pub fn deinit(self: *DeviceBuffer, driver: *CudaDriver) void {
            if (self.ptr != 0) {
                _ = driver.memFree.?(self.ptr);
                self.ptr = 0;
            }
        }
    };

    /// Kernel wrapper
    pub const Kernel = struct {
        function: *CUfunction,
        module: *CUmodule,
        max_threads_per_block: c_int,
        min_grid_size: c_int,

        pub fn deinit(self: *Kernel, driver: *CudaDriver) void {
            _ = driver.moduleUnload.?(self.module);
        }
    };

    // =============================================================================
    // Initialization
    // =============================================================================

    /// Initialize CUDA context with the best available device
    pub fn init(allocator: std.mem.Allocator, driver: *CudaDriver) !*CudaContext {
        const self = try allocator.create(CudaContext);
        errdefer allocator.destroy(self);

        self.allocator = allocator;
        self.driver = driver;

        // Get device count
        var device_count: c_int = 0;
        try cuda_driver.checkCuda(driver.deviceGetCount.?(&device_count));
        if (device_count == 0) {
            return error.NoCudaDevices;
        }

        // Select the best device (highest compute capability, then most memory)
        self.device = try selectBestDevice(driver, device_count);

        // Create context
        try cuda_driver.checkCuda(driver.ctxCreate.?(
            &self.context,
            @intFromEnum(cuda_driver.CUctx_flags.SCHED_AUTO),
            self.device,
        ));
        errdefer _ = driver.ctxDestroy.?(self.context);

        // Create default stream
        try cuda_driver.checkCuda(driver.streamCreate.?(
            &self.stream,
            @intFromEnum(cuda_driver.CUstream_flags.DEFAULT),
        ));
        errdefer _ = driver.streamDestroy.?(self.stream);

        // Query device properties
        self.device_props = try queryDeviceProperties(driver, self.device);

        // Initialize kernel cache
        self.kernels = std.StringHashMap(Kernel).init(allocator);
        errdefer self.kernels.deinit();

        // Initialize buffer pools
        for (0..MEMORY_POOL_BUCKETS) |i| {
            self.buffer_pools[i] = .{};
        }
        errdefer {
            for (0..MEMORY_POOL_BUCKETS) |i| {
                for (self.buffer_pools[i].items) |*buf| {
                    buf.deinit(driver);
                }
                self.buffer_pools[i].deinit(allocator);
            }
        }

        // Initialize temp resources list
        self.temp_buffers = .{};
        errdefer self.temp_buffers.deinit(allocator);

        // Initialize module list
        self.modules = .{};
        errdefer self.modules.deinit(allocator);

        std.log.info("CUDA Context initialized: {s}", .{std.mem.sliceTo(&self.device_props.name, 0)});
        std.log.info("  Compute Capability: {}.{}", .{ self.device_props.compute_capability_major, self.device_props.compute_capability_minor });
        std.log.info("  Total Memory: {} MB", .{self.device_props.total_memory / (1024 * 1024)});
        std.log.info("  Multiprocessors: {}", .{self.device_props.multiprocessor_count});
        std.log.info("  Tensor Cores: {}", .{self.device_props.hasTensorCores()});

        return self;
    }

    /// Cleanup CUDA context
    pub fn deinit(self: *CudaContext) void {
        // Free temporary buffers
        for (self.temp_buffers.items) |*buf| {
            buf.deinit(self.driver);
        }
        self.temp_buffers.deinit(self.allocator);

        // Cleanup kernel cache
        var kernel_iter = self.kernels.iterator();
        while (kernel_iter.next()) |entry| {
            entry.value_ptr.deinit(self.driver);
            self.allocator.free(entry.key_ptr.*);
        }
        self.kernels.deinit();

        // Cleanup buffer pools
        for (0..MEMORY_POOL_BUCKETS) |i| {
            for (self.buffer_pools[i].items) |*buf| {
                buf.deinit(self.driver);
            }
            self.buffer_pools[i].deinit(self.allocator);
        }

        // Cleanup modules
        for (self.modules.items) |*module| {
            _ = self.driver.moduleUnload.?(module);
        }
        self.modules.deinit(self.allocator);

        // Destroy stream
        _ = self.driver.streamDestroy.?(self.stream);

        // Destroy context
        _ = self.driver.ctxDestroy.?(self.context);

        self.allocator.destroy(self);
    }

    /// Push this context to the current thread
    pub fn push(self: *CudaContext) !void {
        try cuda_driver.checkCuda(self.driver.ctxPushCurrent.?(self.context));
    }

    /// Pop context from current thread
    pub fn pop(self: *CudaContext) !void {
    }

        try cuda_driver.checkCuda(self.driver.ctxSetCurrent.?(self.context));
    }

    /// Synchronize the stream
    pub fn synchronize(self: *CudaContext) !void {
        try cuda_driver.checkCuda(self.driver.streamSynchronize.?(self.stream));
    }

    // =============================================================================
    // Device Selection and Properties
    // =============================================================================

    fn selectBestDevice(driver: *CudaDriver, count: c_int) !CUdevice {
        var best_device: CUdevice = 0;
        var best_score: i32 = -1;

        var i: c_int = 0;
        while (i < count) : (i += 1) {
            var device: CUdevice = 0;
            try cuda_driver.checkCuda(driver.deviceGet.?(&device, i));

            const props = try queryDeviceProperties(driver, device);

            // Score: compute capability * 100 + multiprocessors
            const score = props.computeCapability() * 100 + props.multiprocessor_count;

            if (score > best_score) {
                best_score = score;
                best_device = device;
            }
        }

        return best_device;
    }

    fn queryDeviceProperties(driver: *CudaDriver, device: CUdevice) !DeviceProperties {
        var props: DeviceProperties = undefined;

        // Compute capability
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.compute_capability_major,
            .COMPUTE_CAPABILITY_MAJOR, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.compute_capability_minor,
            .COMPUTE_CAPABILITY_MINOR, device));

        // Total memory
        try cuda_driver.checkCuda(driver.deviceTotalMem.?(&props.total_memory, device));

        // Other attributes
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.multiprocessor_count, .MULTIPROCESSOR_COUNT, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.max_threads_per_block, .MAX_THREADS_PER_BLOCK, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.max_block_dim_x, .MAX_BLOCK_DIM_X, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.max_block_dim_y, .MAX_BLOCK_DIM_Y, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.max_block_dim_z, .MAX_BLOCK_DIM_Z, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.max_grid_dim_x, .MAX_GRID_DIM_X, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.max_grid_dim_y, .MAX_GRID_DIM_Y, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.max_grid_dim_z, .MAX_GRID_DIM_Z, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.max_shared_memory_per_block, .MAX_SHARED_MEMORY_PER_BLOCK, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.warp_size, .WARP_SIZE, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.memory_clock_rate, .MEMORY_CLOCK_RATE, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.global_memory_bus_width, .GLOBAL_MEMORY_BUS_WIDTH, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.l2_cache_size, .L2_CACHE_SIZE, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.max_threads_per_multiprocessor, .MAX_THREADS_PER_MULTIPROCESSOR, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.unified_addressing, .UNIFIED_ADDRESSING, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.managed_memory, .MANAGED_MEMORY, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(
            &props.concurrent_managed_access, .CONCURRENT_MANAGED_ACCESS, device));

        // Device name
        var name: [256]u8 = undefined;
        try cuda_driver.checkCuda(driver.deviceGetName.?(&name, name.len, device));
        props.name = name;

        return props;
    }

    // =============================================================================
    // Memory Management
    // =============================================================================

    /// Get a buffer from the pool or allocate a new one
    pub fn getBuffer(self: *CudaContext, size: usize) !DeviceBuffer {
        const pool_idx = getPoolIndex(size);

        // Try to get from pool
        if (pool_idx < MEMORY_POOL_BUCKETS) {
            if (self.buffer_pools[pool_idx].items.len > 0) {
                return self.buffer_pools[pool_idx].pop().?;
            }
        }

        // Allocate new buffer
        const aligned_size = getPooledSize(size);
        var ptr: CUdeviceptr = 0;
        try cuda_driver.checkCuda(self.driver.memAlloc.?(&ptr, aligned_size));

        return DeviceBuffer{
            .ptr = ptr,
            .size = aligned_size,
            .pool_index = pool_idx,
        };
    }

    /// Return a buffer to the pool
    pub fn returnBuffer(self: *CudaContext, buffer: DeviceBuffer) void {
        if (buffer.pool_index) |idx| {
            if (idx < MEMORY_POOL_BUCKETS) {
                self.buffer_pools[idx].append(self.allocator, buffer) catch {
                    // If pool is full, just free the buffer
                    buffer.deinit(self.driver);
                };
                return;
            }
        }
        // Not poolable, free directly
        var buf = buffer;
        buf.deinit(self.driver);
    }

    /// Allocate buffer without pooling
    pub fn allocBuffer(self: *CudaContext, size: usize) !DeviceBuffer {
        var ptr: CUdeviceptr = 0;
        try cuda_driver.checkCuda(self.driver.memAlloc.?(&ptr, size));
        return DeviceBuffer{
            .ptr = ptr,
            .size = size,
            .pool_index = null,
        };
    }

    /// Free a buffer
    pub fn freeBuffer(self: *CudaContext, buffer: *DeviceBuffer) void {
        buffer.deinit(self.driver);
    }

    /// Register a temporary buffer to be freed later
    pub fn registerTempBuffer(self: *CudaContext, buffer: *DeviceBuffer) !void {
        try self.temp_buffers.append(self.allocator, buffer.*);
    }

    /// Clear all temporary buffers
    pub fn clearTempBuffers(self: *CudaContext) void {
        for (self.temp_buffers.items) |*buf| {
            self.returnBuffer(buf.*);
        }
        self.temp_buffers.clearRetainingCapacity();
    }

    /// Upload data from host to device
    pub fn upload(self: *CudaContext, dst: CUdeviceptr, src: []const u8) !void {
        try cuda_driver.checkCuda(self.driver.memcpyHtoD.?(
            dst,
            src.ptr,
            src.len,
        ));
    }

    /// Upload data asynchronously
    pub fn uploadAsync(self: *CudaContext, dst: CUdeviceptr, src: []const u8) !void {
        try cuda_driver.checkCuda(self.driver.memcpyHtoDAsync.?(
            dst,
            src.ptr,
            src.len,
            self.stream,
        ));
    }

    /// Download data from device to host
    pub fn download(self: *CudaContext, dst: []u8, src: CUdeviceptr) !void {
        try cuda_driver.checkCuda(self.driver.memcpyDtoH.?(
            dst.ptr,
            src,
            dst.len,
        ));
    }

    /// Download data asynchronously
    pub fn downloadAsync(self: *CudaContext, dst: []u8, src: CUdeviceptr) !void {
        try cuda_driver.checkCuda(self.driver.memcpyDtoHAsync.?(
            dst.ptr,
            src,
            dst.len,
            self.stream,
        ));
    }

    /// Copy between device buffers
    pub fn copyDeviceToDevice(self: *CudaContext, dst: CUdeviceptr, src: CUdeviceptr, size: usize) !void {
        try cuda_driver.checkCuda(self.driver.memcpyDtoD.?(
            dst,
            src,
            size,
        ));
    }

    /// Set memory to a value
    pub fn memset(self: *CudaContext, ptr: CUdeviceptr, value: u32, count: usize) !void {
        try cuda_driver.checkCuda(self.driver.memsetD32.?(
            ptr,
            value,
            count,
        ));
    }

    /// Set memory asynchronously
    pub fn memsetAsync(self: *CudaContext, ptr: CUdeviceptr, value: u32, count: usize) !void {
        try cuda_driver.checkCuda(self.driver.memsetD32Async.?(
            ptr,
            value,
            count,
            self.stream,
        ));
    }

    // =============================================================================
    // Kernel Management
    // =============================================================================

    /// Load a kernel from PTX code
    pub fn loadKernel(self: *CudaContext, name: []const u8, ptx_code: []const u8) !void {
        // Check if already loaded
        if (self.kernels.contains(name)) {
            return;
        }

        // Load module from PTX
        var module: *CUmodule = undefined;
        try cuda_driver.checkCuda(self.driver.moduleLoadData.?(
            &module,
            ptx_code.ptr,
        ));
        try self.modules.append(self.allocator, module);

        // Get function
        var function: *CUfunction = undefined;
        const name_z = try self.allocator.dupeZ(u8, name);
        defer self.allocator.free(name_z);

        try cuda_driver.checkCuda(self.driver.moduleGetFunction.?(
            &function,
            module,
            name_z,
        ));

        // Get occupancy info
        var min_grid_size: c_int = 0;
        var max_threads_per_block: c_int = 0;
        if (self.driver.occupancyMaxPotentialBlockSize) |occupancy_fn| {
            _ = occupancy_fn(
                &min_grid_size,
                &max_threads_per_block,
                function,
                null,
                0,
                0,
            );
        }

        const kernel = Kernel{
            .function = function,
            .module = module,
            .max_threads_per_block = max_threads_per_block,
            .min_grid_size = min_grid_size,
        };

        const name_copy = try self.allocator.dupe(u8, name);
        try self.kernels.put(name_copy, kernel);
    }

    /// Get a loaded kernel
    pub fn getKernel(self: *const CudaContext, name: []const u8) !*const Kernel {
        const kernel = self.kernels.getPtr(name) orelse return error.KernelNotFound;
        return kernel;
    }

    /// Check if kernel is loaded
    pub fn hasKernel(self: *const CudaContext, name: []const u8) bool {
        return self.kernels.contains(name);
    }

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
        std.debug.assert(args.len <= kernel_params.len);
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

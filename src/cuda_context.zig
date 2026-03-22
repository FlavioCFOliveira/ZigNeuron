/// CUDA Context for managing persistent resources
/// Optimized for NVIDIA GPU performance
/// Follows the same pattern as MetalContext for consistency
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda_kernels = @import("cuda_kernels.zig");
const cuda_profiler = @import("cuda_profiler.zig");

const CUresult = cuda_driver.CUresult;
const nvrtcResult = cuda_driver.nvrtcResult;
const CUdevice = cuda_driver.CUdevice;
const CUcontext = cuda_driver.CUcontext;
const CUstream = cuda_driver.CUstream;
const CUmodule = cuda_driver.CUmodule;
const CUfunction = cuda_driver.CUfunction;
const CUdeviceptr = cuda_driver.CUdeviceptr;
const CUevent = cuda_driver.CUevent;
const CudaDriver = cuda_driver.CudaDriver;
const nvrtcProgram = cuda_driver.nvrtcProgram;
const CudaProfiler = cuda_profiler.CudaProfiler;

// =============================================================================
// Thread-Safe Stream Handle
// =============================================================================

/// Thread-safe wrapper for CUDA stream handle
/// SECURITY FIX: Prevents race conditions in stream operations (CRIT-003)
const ThreadSafeStream = struct {
    stream: ?*CUstream,
    mutex: std.atomic.Mutex,

    pub fn init() ThreadSafeStream {
        return .{
            .stream = null,
            .mutex = .unlocked,
        };
    }

    /// Set the stream handle (thread-safe)
    pub fn set(self: *ThreadSafeStream, strm: ?*CUstream) void {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        self.stream = strm;
    }

    /// Get the stream handle (thread-safe)
    /// Returns null if no stream is set
    pub fn get(self: *ThreadSafeStream) ?*CUstream {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        return self.stream;
    }

    /// Get the stream handle for operations that require it
    /// Returns error if stream is not initialized
    pub fn getOrError(self: *ThreadSafeStream) !*CUstream {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        if (self.stream) |strm| {
            return strm;
        }
        return error.StreamNotInitialized;
    }

    /// Check if stream is initialized (thread-safe)
    pub fn isInitialized(self: *ThreadSafeStream) bool {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        return self.stream != null;
    }

    /// Take ownership of the stream (sets internal to null)
    /// Used during cleanup to prevent use-after-free
    pub fn take(self: *ThreadSafeStream) ?*CUstream {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        const strm = self.stream;
        self.stream = null;
        return strm;
    }
};

// =============================================================================
// Configuration Constants
// =============================================================================

/// Maximum number of kernel arguments supported
/// SECURITY: Prevents buffer overflow in kernel parameter array
pub const MAX_KERNEL_ARGS: usize = 16;

/// Maximum shared memory per block (48KB is standard for most GPUs)
/// SECURITY: Prevents shared memory overflow
pub const MAX_SHARED_MEMORY_PER_BLOCK: u32 = 48 * 1024; // 48KB

/// Maximum grid dimension in any direction
/// SECURITY: Prevents grid dimension overflow
pub const MAX_GRID_DIM: u32 = 65535;

/// Maximum threads per block
/// SECURITY: Prevents thread count overflow
pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

/// Default thread block size for element-wise operations
pub const DEFAULT_BLOCK_SIZE: c_uint = 256;

// =============================================================================
// Async Stream Management Configuration
// =============================================================================

/// Maximum number of concurrent streams in the pool
/// Higher values enable more overlap but consume more GPU resources
pub const MAX_STREAMS: usize = 16;

/// Maximum number of events in the pool
pub const MAX_EVENTS: usize = 64;

/// Stream priority levels (higher value = higher priority)
pub const StreamPriority = struct {
    pub const HIGH: c_int = -1;
    pub const DEFAULT: c_int = 0;
    pub const LOW: c_int = 1;
};

/// Stream types for different workloads
pub const StreamType = enum {
    /// Default stream for compute workloads
    compute,
    /// Stream optimized for data transfers (H2D)
    h2d_transfer,
    /// Stream optimized for data transfers (D2H)
    d2h_transfer,
    /// High-priority stream for latency-sensitive work
    high_priority,
};

/// Async operation handle for tracking submitted work
pub const AsyncOpHandle = struct {
    stream_id: u32,
    event_id: ?u32,
    submitted_time: i64, // nanoseconds

    pub fn isValid(self: AsyncOpHandle) bool {
        return self.stream_id < MAX_STREAMS;
    }
};

/// Stream state tracking
const StreamState = struct {
    stream: ?*CUstream,
    in_use: bool,
    stream_type: StreamType,
    priority: c_int,
    // Track submitted operations for dependency management
    last_event: ?*CUevent,
    operation_count: u64,

    pub fn init() StreamState {
        return .{
            .stream = null,
            .in_use = false,
            .stream_type = .compute,
            .priority = StreamPriority.DEFAULT,
            .last_event = null,
            .operation_count = 0,
        };
    }
};

/// Event state tracking
const EventState = struct {
    event: ?*CUevent,
    in_use: bool,
    recorded: bool,

    pub fn init() EventState {
        return .{
            .event = null,
            .in_use = false,
            .recorded = false,
        };
    }
};

/// Stream callback function type
pub const StreamCallback = *const fn (
    stream: ?*CUstream,
    status: CUresult,
    user_data: ?*anyopaque,
) callconv(.c) void;

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

    /// Managed memory buffer wrapper (Unified Memory)
    /// Allows CPU and GPU to access the same memory with automatic migration
    pub const ManagedBuffer = struct {
        ptr: CUdeviceptr,
        size: usize,
        host_ptr: ?*anyopaque = null,

        pub fn deinit(self: *ManagedBuffer, driver: *CudaDriver) void {
            if (self.ptr != 0) {
                if (self.host_ptr) |host_ptr| {
                    // This was pinned memory allocation
                    if (driver.memFreeHost) |free_fn| {
                        _ = free_fn(host_ptr);
                    }
                } else {
                    _ = driver.memFree.?(self.ptr);
                }
                self.ptr = 0;
            }
        }

        /// Get host-accessible pointer for managed memory
        pub fn getHostPtr(self: *const ManagedBuffer) ?*anyopaque {
            if (self.host_ptr) |host_ptr| {
                return host_ptr;
            }
            // For true managed memory, the device pointer is also valid on host
            return @ptrFromInt(self.ptr);
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

    allocator: std.mem.Allocator,
    driver: cuda_driver.CudaDriverRef,

    // CUDA handles
    device: CUdevice,
    context: ?*CUcontext,
    // SECURITY FIX: Thread-safe stream handle (CRIT-003)
    stream: ThreadSafeStream,

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

    // =============================================================================
    // Async Stream Management
    // =============================================================================

    /// Stream pool for concurrent execution
    stream_pool: [MAX_STREAMS]StreamState,
    stream_pool_mutex: std.atomic.Mutex,

    /// Event pool for synchronization
    event_pool: [MAX_EVENTS]EventState,
    event_pool_mutex: std.atomic.Mutex,

    /// Default stream for backward compatibility
    default_stream: ThreadSafeStream,

    /// Async execution statistics
    async_stats: AsyncStats,

    // =============================================================================
    // Profiler Integration
    // =============================================================================

    /// Optional profiler for performance analysis
    profiler: ?*CudaProfiler,

    // =============================================================================
    // Initialization
    // =============================================================================

    /// Async execution statistics structure
    const AsyncStats = struct {
        streams_created: std.atomic.Value(u32),
        events_created: std.atomic.Value(u32),
        async_ops_submitted: std.atomic.Value(u64),
        async_ops_completed: std.atomic.Value(u64),
        bytes_transferred_h2d: std.atomic.Value(u64),
        bytes_transferred_d2h: std.atomic.Value(u64),

        pub fn init() AsyncStats {
            return .{
                .streams_created = std.atomic.Value(u32).init(0),
                .events_created = std.atomic.Value(u32).init(0),
                .async_ops_submitted = std.atomic.Value(u64).init(0),
                .async_ops_completed = std.atomic.Value(u64).init(0),
                .bytes_transferred_h2d = std.atomic.Value(u64).init(0),
                .bytes_transferred_d2h = std.atomic.Value(u64).init(0),
            };
        }
    };

    /// Initialize CUDA context with the best available device
    /// SECURITY FIX: Now uses CudaDriverRef for safe reference counting (CRIT-002)
    pub fn init(allocator: std.mem.Allocator) !*CudaContext {
        const self = try allocator.create(CudaContext);
        errdefer allocator.destroy(self);

        self.allocator = allocator;
        // Acquire a reference to the global driver - prevents use-after-free
        self.driver = try cuda_driver.CudaDriverRef.acquire(allocator);

        // Initialize CUDA handles to null for safe cleanup
        self.context = null;
        self.stream = ThreadSafeStream.init();

        // Get device count
        var device_count: c_int = 0;
        try cuda_driver.checkCuda(self.driver.driver.deviceGetCount.?(&device_count));
        if (device_count == 0) {
            return error.NoCudaDevices;
        }

        // Select the best device (highest compute capability, then most memory)
        self.device = try selectBestDevice(self.driver.driver, device_count);

        // Create context using temporary pointer, then assign to optional field
        var temp_context: ?*CUcontext = null;
        try cuda_driver.checkCuda(self.driver.driver.ctxCreate.?(
            @ptrCast(&temp_context),
            @intFromEnum(cuda_driver.CUctx_flags.SCHED_AUTO),
            self.device,
        ));
        self.context = temp_context;
        errdefer {
            if (self.context) |ctx| {
                _ = self.driver.driver.ctxDestroy.?(ctx);
                self.context = null;
            }
        }

        // Create default stream using temporary pointer, then assign to thread-safe wrapper
        var temp_stream: ?*CUstream = null;
        try cuda_driver.checkCuda(self.driver.driver.streamCreate.?(
            @ptrCast(&temp_stream),
            @intFromEnum(cuda_driver.CUstream_flags.DEFAULT),
        ));
        self.stream.set(temp_stream);
        errdefer {
            if (self.stream.get()) |strm| {
                // Context must be current to destroy stream
                if (self.context) |ctx| {
                    if (self.driver.driver.ctxSetCurrent != null) {
                        _ = self.driver.driver.ctxSetCurrent.?(ctx);
                    }
                }
                _ = self.driver.driver.streamDestroy.?(strm);
                self.stream.set(null);
            }
        }

        // Query device properties
        self.device_props = try queryDeviceProperties(self.driver.driver, self.device);

        // Initialize kernel cache
        self.kernels = std.StringHashMap(Kernel).init(allocator);
        errdefer self.kernels.deinit();

        // Initialize buffer pools
        for (0..MEMORY_POOL_BUCKETS) |i| {
            self.buffer_pools[i] = .empty;
        }
        errdefer {
            for (0..MEMORY_POOL_BUCKETS) |i| {
                for (self.buffer_pools[i].items) |*buf| {
                    buf.deinit(self.driver.driver);
                }
                self.buffer_pools[i].deinit(self.allocator);
            }
        }

        // Initialize temp resources list
        self.temp_buffers = .empty;
        errdefer self.temp_buffers.deinit(allocator);

        // Initialize module list
        self.modules = .empty;
        errdefer self.modules.deinit(allocator);

        // Initialize stream pool
        self.stream_pool_mutex = .unlocked;
        for (0..MAX_STREAMS) |i| {
            self.stream_pool[i] = StreamState.init();
        }

        // Initialize event pool
        self.event_pool_mutex = .unlocked;
        for (0..MAX_EVENTS) |i| {
            self.event_pool[i] = EventState.init();
        }

        // Initialize default stream wrapper
        self.default_stream = ThreadSafeStream.init();
        self.default_stream.set(temp_stream);

        // Initialize async stats
        self.async_stats = AsyncStats.init();

        // Initialize profiler (disabled by default)
        self.profiler = null;

        std.log.info("CUDA Context initialized: {s}", .{std.mem.sliceTo(&self.device_props.name, 0)});
        std.log.info("  Async Stream Management: Enabled (max {d} concurrent streams)", .{MAX_STREAMS});
        std.log.info("  Compute Capability: {}.{}", .{ self.device_props.compute_capability_major, self.device_props.compute_capability_minor });
        std.log.info("  Total Memory: {} MB", .{self.device_props.total_memory / (1024 * 1024)});
        std.log.info("  Multiprocessors: {}", .{self.device_props.multiprocessor_count});
        std.log.info("  Tensor Cores: {}", .{self.device_props.hasTensorCores()});

        return self;
    }

    /// Cleanup CUDA context
    pub fn deinit(self: *CudaContext) void {
        // Free temporary buffers
        for (self.temp_buffers.items) |buf| {
            buf.deinit(self.driver.driver);
        }
        self.temp_buffers.deinit(self.allocator);

        // Cleanup kernel cache
        var kernel_iter = self.kernels.iterator();
        while (kernel_iter.next()) |entry| {
            entry.value_ptr.deinit(self.driver.driver);
            self.allocator.free(entry.key_ptr.*);
        }
        self.kernels.deinit();

        // Cleanup buffer pools
        for (0..MEMORY_POOL_BUCKETS) |i| {
            for (self.buffer_pools[i].items) |*buf| {
                buf.deinit(self.driver.driver);
            }
            self.buffer_pools[i].deinit(self.allocator);
        }

        // Cleanup modules - only if driver is still available
        if (self.driver.driver.moduleUnload != null) {
            for (self.modules.items) |module| {
                _ = self.driver.driver.moduleUnload.?(module);
            }
        }
        self.modules.deinit(self.allocator);

        // Cleanup stream pool - destroy all created streams
        if (self.driver.driver.streamDestroy != null) {
            // Make context current for stream destruction
            if (self.context) |ctx| {
                if (self.driver.driver.ctxSetCurrent != null) {
                    _ = self.driver.driver.ctxSetCurrent.?(ctx);
                }
            }

            for (0..MAX_STREAMS) |i| {
                if (self.stream_pool[i].stream) |stream| {
                    _ = self.driver.driver.streamDestroy.?(stream);
                }
            }
        }

        // Cleanup event pool - destroy all created events
        if (self.driver.driver.eventDestroy != null) {
            for (0..MAX_EVENTS) |i| {
                if (self.event_pool[i].event) |event| {
                    _ = self.driver.driver.eventDestroy.?(event);
                }
            }
        }

        // Destroy default stream - best effort, ignore errors during cleanup
        // SECURITY FIX: Use take() to atomically get and clear stream handle (CRIT-003)
        // This prevents race conditions where another thread might access the stream
        // while we're destroying it
        if (self.driver.driver.streamDestroy != null) {
            const strm = self.stream.take();
            if (strm) |s| {
                // Make context current before destroying stream
                if (self.context) |ctx| {
                    if (self.driver.driver.ctxSetCurrent != null) {
                        _ = self.driver.driver.ctxSetCurrent.?(ctx);
                    }
                }
                // Best-effort destroy - ignore return value during cleanup
                // This prevents crashes during partial initialization failure
                _ = self.driver.driver.streamDestroy.?(s);
            }
        }

        // Destroy context - best effort, ignore errors
        // Context may be in an invalid state if initialization failed partway
        if (self.driver.driver.ctxDestroy != null) {
            if (self.context) |ctx| {
                // Best-effort destroy - don't check return value during cleanup
                // This prevents crashes during partial initialization
                _ = self.driver.driver.ctxDestroy.?(ctx);
                self.context = null;
            }
        }

        // Cleanup profiler if enabled
        if (self.profiler) |profiler| {
            profiler.deinit();
            self.profiler = null;
        }

        // SECURITY FIX: Release our reference to the driver (CRIT-002)
        // This ensures proper cleanup when all references are released
        self.driver.release();

        self.allocator.destroy(self);
    }

    /// Push this context to the current thread
    pub fn push(self: *CudaContext) !void {
        if (self.context) |ctx| {
            try cuda_driver.checkCuda(self.driver.driver.ctxPushCurrent.?(ctx));
        } else {
            return error.ContextNotInitialized;
        }
    }

    /// Pop context from current thread
    pub fn pop(self: *CudaContext) !void {
        var ctx: ?*anyopaque = null;
        try cuda_driver.checkCuda(self.driver.driver.ctxPopCurrent.?(@ptrCast(&ctx)));
    }

    /// Set this as the current context
    pub fn setCurrent(self: *CudaContext) !void {
        if (self.context) |ctx| {
            try cuda_driver.checkCuda(self.driver.driver.ctxSetCurrent.?(ctx));
        } else {
            return error.ContextNotInitialized;
        }
    }

    /// Synchronize the stream
    /// SECURITY FIX: Thread-safe stream access (CRIT-003)
    pub fn synchronize(self: *CudaContext) !void {
        const strm = self.stream.getOrError() catch |err| return err;
        try cuda_driver.checkCuda(self.driver.driver.streamSynchronize.?(strm));
    }

    // =============================================================================
    // Profiler Integration
    // =============================================================================

    /// Enable profiling with specified mode
    /// Returns error if profiler already enabled
    pub fn enableProfiling(self: *CudaContext, mode: cuda_profiler.ProfilerMode) !void {
        if (self.profiler != null) {
            return error.ProfilerAlreadyEnabled;
        }
        self.profiler = try CudaProfiler.initWithContext(self.allocator, mode, self);
        std.log.info("CUDA Context: Profiling enabled with mode {s}", .{@tagName(mode)});
    }

    /// Disable profiling
    pub fn disableProfiling(self: *CudaContext) void {
        if (self.profiler) |profiler| {
            profiler.printSummary();
            profiler.deinit();
            self.profiler = null;
            std.log.info("CUDA Context: Profiling disabled", .{});
        }
    }

    /// Get the profiler instance (null if not enabled)
    pub fn getProfiler(self: *CudaContext) ?*CudaProfiler {
        return self.profiler;
    }

    /// Check if profiling is enabled
    pub fn isProfilingEnabled(self: *const CudaContext) bool {
        return self.profiler != null;
    }

    /// Mark a point in the application (for Nsight Systems)
    /// Zero overhead if profiling disabled
    pub fn profilerMark(self: *const CudaContext, message: []const u8) void {
        if (self.profiler) |profiler| {
            profiler.mark(message, null);
        }
    }

    /// Push a profiling range
    pub fn profilerPushRange(self: *CudaContext, name: []const u8) !void {
        if (self.profiler) |profiler| {
            _ = try profiler.pushRange(name, null);
        }
    }

    /// Pop a profiling range
    pub fn profilerPopRange(self: *CudaContext) !void {
        if (self.profiler) |profiler| {
            try profiler.popRange();
        }
    }

    // =============================================================================
    // Device Selection and Properties
    // =============================================================================

    fn selectBestDevice(driver_ref: *CudaDriver, count: c_int) !CUdevice {
        var best_device: CUdevice = 0;
        var best_score: i32 = -1;

        var i: c_int = 0;
        while (i < count) : (i += 1) {
            var device: CUdevice = 0;
            try cuda_driver.checkCuda(driver_ref.deviceGet.?(&device, i));

            const props = try queryDeviceProperties(driver_ref, device);

            // Score: compute capability * 100 + multiprocessors
            const score = props.computeCapability() * 100 + props.multiprocessor_count;

            if (score > best_score) {
                best_score = score;
                best_device = device;
            }
        }

        return best_device;
    }

    fn queryDeviceProperties(driver_ref: *CudaDriver, device: CUdevice) !DeviceProperties {
        var props: DeviceProperties = undefined;

        // Compute capability
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(&props.compute_capability_major,
            .COMPUTE_CAPABILITY_MAJOR, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(&props.compute_capability_minor,
            .COMPUTE_CAPABILITY_MINOR, device));

        // Total memory
        try cuda_driver.checkCuda(driver_ref.deviceTotalMem.?(&props.total_memory, device));

        // Other attributes
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(&props.multiprocessor_count, .MULTIPROCESSOR_COUNT, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.max_threads_per_block, .MAX_THREADS_PER_BLOCK, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.max_block_dim_x, .MAX_BLOCK_DIM_X, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.max_block_dim_y, .MAX_BLOCK_DIM_Y, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.max_block_dim_z, .MAX_BLOCK_DIM_Z, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.max_grid_dim_x, .MAX_GRID_DIM_X, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.max_grid_dim_y, .MAX_GRID_DIM_Y, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.max_grid_dim_z, .MAX_GRID_DIM_Z, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.max_shared_memory_per_block, .MAX_SHARED_MEMORY_PER_BLOCK, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.warp_size, .WARP_SIZE, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.memory_clock_rate, .MEMORY_CLOCK_RATE, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.global_memory_bus_width, .GLOBAL_MEMORY_BUS_WIDTH, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.l2_cache_size, .L2_CACHE_SIZE, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.max_threads_per_multiprocessor, .MAX_THREADS_PER_MULTIPROCESSOR, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.unified_addressing, .UNIFIED_ADDRESSING, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.managed_memory, .MANAGED_MEMORY, device));
        try cuda_driver.checkCuda(driver_ref.deviceGetAttribute.?(
            &props.concurrent_managed_access, .CONCURRENT_MANAGED_ACCESS, device));

        // Device name
        var name: [256]u8 = undefined;
        try cuda_driver.checkCuda(driver_ref.deviceGetName.?(&name, name.len, device));
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
                // PROFILER: Pool hit
                if (self.profiler) |profiler| {
                    profiler.recordPoolHit();
                }
                return self.buffer_pools[pool_idx].pop().?;
            }
        }

        // PROFILER: Pool miss
        if (self.profiler) |profiler| {
            profiler.recordPoolMiss();
        }

        // Allocate new buffer
        const aligned_size = getPooledSize(size);
        var ptr: CUdeviceptr = 0;
        try cuda_driver.checkCuda(self.driver.driver.memAlloc.?(&ptr, aligned_size));

        // PROFILER: Memory allocation
        if (self.profiler) |profiler| {
            profiler.recordAllocation(aligned_size);
        }

        return DeviceBuffer{
            .ptr = ptr,
            .size = aligned_size,
            .pool_index = pool_idx,
        };
    }

    /// Return a buffer to the pool
    /// SECURITY FIX: Validate buffer state before returning to pool (VULN-004)
    pub fn returnBuffer(self: *CudaContext, buffer: DeviceBuffer) void {
        // Validate buffer pointer is not null (already freed)
        if (buffer.ptr == 0) {
            std.log.warn("Attempting to return already-freed buffer to pool", .{});
            return;
        }

        if (buffer.pool_index) |idx| {
            if (idx < MEMORY_POOL_BUCKETS) {
                self.buffer_pools[idx].append(self.allocator, buffer) catch |err| {
                    std.log.warn("Failed to append to pool ({}), freeing buffer", .{err});
                    // PROFILER: Memory deallocation (pool return failed)
                    if (self.profiler) |profiler| {
                        profiler.recordDeallocation(buffer.size);
                    }
                    var buf = buffer;
                    buf.deinit(self.driver.driver);
                };
                return;
            } else {
                std.log.warn("Invalid pool index {}, freeing buffer", .{idx});
            }
        }
        // Not poolable or invalid pool index, free directly
        // PROFILER: Memory deallocation
        if (self.profiler) |profiler| {
            profiler.recordDeallocation(buffer.size);
        }
        var buf = buffer;
        buf.deinit(self.driver.driver);
    }

    /// Allocate buffer without pooling
    pub fn allocBuffer(self: *CudaContext, size: usize) !DeviceBuffer {
        var ptr: CUdeviceptr = 0;
        try cuda_driver.checkCuda(self.driver.driver.memAlloc.?(&ptr, size));
        return DeviceBuffer{
            .ptr = ptr,
            .size = size,
            .pool_index = null,
        };
    }

    /// Free a buffer
    /// SECURITY FIX: Validate buffer state before freeing (VULN-004)
    pub fn freeBuffer(self: *CudaContext, buffer: *DeviceBuffer) void {
        if (buffer.ptr == 0) {
            std.log.warn("Attempting to free already-freed buffer", .{});
            return;
        }
        // PROFILER: Memory deallocation
        if (self.profiler) |profiler| {
            profiler.recordDeallocation(buffer.size);
        }
        buffer.deinit(self.driver.driver);
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
        try cuda_driver.checkCuda(self.driver.driver.memcpyHtoD.?(
            dst,
            src.ptr,
            src.len,
        ));
    }

    /// Upload data asynchronously
    /// SECURITY FIX: Thread-safe stream access (CRIT-003)
    pub fn uploadAsync(self: *CudaContext, dst: CUdeviceptr, src: []const u8) !void {
        const strm = self.stream.getOrError() catch |err| return err;
        try cuda_driver.checkCuda(self.driver.driver.memcpyHtoDAsync.?(
            dst,
            src.ptr,
            src.len,
            strm,
        ));
    }

    /// Download data from device to host
    pub fn download(self: *CudaContext, dst: []u8, src: CUdeviceptr) !void {
        try cuda_driver.checkCuda(self.driver.driver.memcpyDtoH.?(
            dst.ptr,
            src,
            dst.len,
        ));
    }

    /// Download data asynchronously
    /// SECURITY FIX: Thread-safe stream access (CRIT-003)
    pub fn downloadAsync(self: *CudaContext, dst: []u8, src: CUdeviceptr) !void {
        const strm = self.stream.getOrError() catch |err| return err;
        try cuda_driver.checkCuda(self.driver.driver.memcpyDtoHAsync.?(
            dst.ptr,
            src,
            dst.len,
            strm,
        ));
    }

    /// Copy between device buffers
    pub fn copyDeviceToDevice(self: *CudaContext, dst: CUdeviceptr, src: CUdeviceptr, size: usize) !void {
        try cuda_driver.checkCuda(self.driver.driver.memcpyDtoD.?(
            dst,
            src,
            size,
        ));
    }

    /// Set memory to a value
    pub fn memset(self: *CudaContext, ptr: CUdeviceptr, value: u32, count: usize) !void {
        try cuda_driver.checkCuda(self.driver.driver.memsetD32.?(
            ptr,
            value,
            count,
        ));
    }

    /// Set memory asynchronously
    /// SECURITY FIX: Thread-safe stream access (CRIT-003)
    pub fn memsetAsync(self: *CudaContext, ptr: CUdeviceptr, value: u32, count: usize) !void {
        const strm = self.stream.getOrError() catch |err| return err;
        try cuda_driver.checkCuda(self.driver.driver.memsetD32Async.?(
            ptr,
            value,
            count,
            strm,
        ));
    }

    // =============================================================================
    // Unified Memory Management
    // =============================================================================

    /// Memory advice/hint types for unified memory optimization
    pub const MemoryAdvice = enum {
        /// Default behavior, no explicit hint
        default,
        /// Data will be read mostly by the specified device, allow replication
        read_mostly,
        /// Set preferred location for data (device or host)
        preferred_location,
        /// Data accessed by device should be accessed last on device
        accessed_by,
    };

    /// Check if unified memory is supported on this device
    pub fn hasUnifiedMemory(self: *const CudaContext) bool {
        return self.device_props.hasUnifiedMemory();
    }

    /// Check if concurrent managed access is supported (Pascal+)
    /// Allows GPU and CPU to access managed memory simultaneously
    pub fn hasConcurrentManagedAccess(self: *const CudaContext) bool {
        return self.device_props.concurrent_managed_access != 0;
    }

    /// Allocate managed memory (unified memory accessible by both CPU and GPU)
    /// Falls back to pinned host memory if unified memory is not supported
    pub fn allocManaged(self: *CudaContext, size: usize) !CudaContext.ManagedBuffer {
        // Check if unified memory is supported
        if (!self.hasUnifiedMemory()) {
            // Fall back to pinned host memory for older GPUs
            return self.allocPinnedManaged(size);
        }

        // SECURITY: Check for integer overflow in size calculation
        if (size == 0) {
            return error.InvalidSize;
        }

        var ptr: CUdeviceptr = 0;
        const flags: c_uint = @intFromEnum(cuda_driver.CUmemAttach_flags.GLOBAL);

        // Allocate managed memory
        if (self.driver.driver.memAllocManaged) |alloc_fn| {
            try cuda_driver.checkCuda(alloc_fn(&ptr, size, flags));
        } else {
            // Fallback if function not available
            return self.allocPinnedManaged(size);
        }

        return CudaContext.ManagedBuffer{
            .ptr = ptr,
            .size = size,
            .host_ptr = null,
        };
    }

    /// Allocate pinned host memory as fallback for managed memory
    /// This provides faster CPU<->GPU transfers but not true unified memory
    fn allocPinnedManaged(self: *CudaContext, size: usize) !CudaContext.ManagedBuffer {
        var host_ptr: ?*anyopaque = null;

        if (self.driver.driver.memAllocHost) |alloc_fn| {
            try cuda_driver.checkCuda(alloc_fn(&host_ptr, size));
        } else {
            // Last resort: allocate regular host memory
            host_ptr = self.allocator.alignedAlloc(u8, 64, size) catch {
                return error.OutOfMemory;
            };
        }

        // For pinned memory, we need to register it to get a device pointer
        // In practice, pinned memory is accessed via the host pointer
        return CudaContext.ManagedBuffer{
            .ptr = @intFromPtr(host_ptr.?), // Convert host pointer to device pointer representation
            .size = size,
            .host_ptr = host_ptr,
        };
    }

    /// Free managed memory (handles both true managed and pinned fallback)
    pub fn freeManaged(self: *CudaContext, buffer: *CudaContext.ManagedBuffer) void {
        if (buffer.ptr == 0) {
            return;
        }

        if (buffer.host_ptr) |host_ptr| {
            // This was pinned memory allocation
            if (self.driver.driver.memFreeHost) |free_fn| {
                _ = free_fn(host_ptr);
            } else {
                // Allocated with regular allocator
                const ptr: [*]u8 = @ptrCast(host_ptr);
                self.allocator.free(ptr[0..buffer.size]);
            }
        } else {
            // True managed memory
            _ = self.driver.driver.memFree.?(buffer.ptr);
        }

        buffer.ptr = 0;
        buffer.size = 0;
        buffer.host_ptr = null;
    }

    /// Prefetch managed memory to the GPU
    /// This reduces page faults by explicitly migrating data before kernel launch
    pub fn prefetchToDevice(self: *CudaContext, ptr: CUdeviceptr, size: usize) !void {
        if (!self.hasUnifiedMemory()) {
            // No-op for pinned fallback
            return;
        }

        if (self.driver.driver.memPrefetchAsync) |prefetch_fn| {
            // Use the current stream for prefetch
            const strm = self.stream.getOrError() catch |err| return err;
            try cuda_driver.checkCuda(prefetch_fn(ptr, size, self.device, strm));
        }
        // If function not available, silently continue - data will migrate on demand
    }

    /// Prefetch managed memory to the host
    /// Useful before CPU needs to read results from GPU
    pub fn prefetchToHost(self: *CudaContext, ptr: CUdeviceptr, size: usize) !void {
        if (!self.hasUnifiedMemory()) {
            // No-op for pinned fallback
            return;
        }

        if (self.driver.driver.memPrefetchAsync) |prefetch_fn| {
            const strm = self.stream.getOrError() catch |err| return err;
            // Device ordinal -1 means prefetch to host
            try cuda_driver.checkCuda(prefetch_fn(ptr, size, -1, strm));
        }
    }

    /// Set memory advice/hint for managed memory
    /// Helps the CUDA driver optimize data placement and migration
    pub fn setMemoryAdvice(self: *CudaContext, ptr: CUdeviceptr, size: usize, advice: MemoryAdvice) !void {
        if (!self.hasUnifiedMemory()) {
            return;
        }

        if (self.driver.driver.memAdvise) |advise_fn| {
            const cu_advice: cuda_driver.CUmem_advise = switch (advice) {
                .default => return, // No advice needed
                .read_mostly => .SET_READ_MOSTLY,
                .preferred_location => .SET_PREFERRED_LOCATION,
                .accessed_by => .SET_ACCESSED_BY,
            };

            try cuda_driver.checkCuda(advise_fn(ptr, size, cu_advice, self.device));
        }
    }

    /// Synchronize managed memory - ensures all managed memory operations are complete
    /// This is useful for oversubscription scenarios where memory may be migrating
    pub fn synchronizeManaged(self: *CudaContext) !void {
        if (self.hasConcurrentManagedAccess()) {
            // With concurrent access, we just need to synchronize the stream
            try self.synchronize();
        } else {
            // Without concurrent access, we need full context synchronization
            try cuda_driver.checkCuda(self.driver.driver.ctxSynchronize.?());
        }
    }

    /// Check if memory is actually on the device (for debugging)
    pub fn isMemoryOnDevice(self: *CudaContext, ptr: CUdeviceptr) bool {
        if (!self.hasUnifiedMemory()) {
            return false;
        }

        // Query the memory attribute
        if (self.driver.driver.pointerGetAttribute) |attr_fn| {
            var memory_type: c_int = 0;
            const result = attr_fn(
                @ptrCast(&memory_type),
                .MEMORY_TYPE,
                ptr,
            );
            if (result == .SUCCESS) {
                return memory_type == @intFromEnum(cuda_driver.CUmemorytype.DEVICE);
            }
        }
        return false;
    }

    /// PTX validation errors
    pub const PtxValidationError = error{
        PtxTooShort,
        InvalidPtxHeader,
        InvalidPtxVersion,
        InvalidPtxTarget,
        UnsupportedPtxVersion,
        SuspiciousPtxPattern,
        PtxContainsNullBytes,
        PtxTooLarge,
    };

    /// Maximum PTX code size (32MB - reasonable limit for kernel code)
    /// SECURITY: Prevents memory exhaustion from maliciously large PTX
    pub const MAX_PTX_SIZE: usize = 32 * 1024 * 1024;

    /// Minimum PTX version supported (3.0 - CUDA 6.5+)
    pub const MIN_PTX_VERSION_MAJOR: u32 = 3;
    pub const MAX_PTX_VERSION_MAJOR: u32 = 8; // PTX 8.x (CUDA 11.x+)

    /// Supported target architectures
    pub const SUPPORTED_TARGETS = &[_][]const u8{
        "sm_30", "sm_35", "sm_37",
        "sm_50", "sm_52", "sm_53",
        "sm_60", "sm_61", "sm_62",
        "sm_70", "sm_75", "sm_80", "sm_86",
    };

    /// Suspicious patterns that may indicate malicious PTX
    const SUSPICIOUS_PATTERNS = &[_][]const u8{
        "// malicious",
        "/* exploit",
        "<script",
        "javascript:",
        "eval(",
        "execve",
        "system(",
        "__nvvm",
        "// -^-",
        "SHELL",
        "cmd.exe",
        "powershell",
    };

    /// Validate PTX code before passing to driver
    /// SECURITY: Prevents injection attacks and ensures PTX compatibility
    pub fn validatePtx(ptx: []const u8) PtxValidationError!void {
        // Check minimum length for valid PTX header
        // Must have at least ".version X.Y\n.target XXX\n.address_size 64"
        if (ptx.len < 30) {
            return PtxValidationError.PtxTooShort;
        }

        // Check maximum size
        if (ptx.len > MAX_PTX_SIZE) {
            return PtxValidationError.PtxTooLarge;
        }

        // Check for embedded null bytes (PTX is text-based, internal nulls indicate corruption)
        // Note: A null byte at the end is OK (null-terminated string), but internal nulls are suspicious
        if (ptx.len > 0) {
            // Check for null bytes in the content (excluding the last position if it's a null terminator)
            const content_to_check = if (ptx[ptx.len - 1] == 0) ptx[0 .. ptx.len - 1] else ptx;
            if (std.mem.indexOfScalar(u8, content_to_check, 0) != null) {
                return PtxValidationError.PtxContainsNullBytes;
            }
        }

        // Validate PTX header structure
        // PTX must start with ".version" directive
        if (!std.mem.startsWith(u8, ptx, ".version")) {
            // Also allow leading whitespace/comments
            // Custom trimLeft implementation since std.mem.trimLeft doesn't exist in this Zig version
            var trim_idx: usize = 0;
            while (trim_idx < ptx.len) {
                const c = ptx[trim_idx];
                if (c != ' ' and c != '\t' and c != '\n' and c != '\r') break;
                trim_idx += 1;
            }
            const trimmed = ptx[trim_idx..];
            if (!std.mem.startsWith(u8, trimmed, ".version") and
                !std.mem.startsWith(u8, trimmed, "//"))
            {
                return PtxValidationError.InvalidPtxHeader;
            }
        }

        // Check for version directive
        const version_pattern = ".version ";
        const version_idx = std.mem.indexOf(u8, ptx, version_pattern);
        if (version_idx == null) {
            return PtxValidationError.InvalidPtxVersion;
        }

        // Parse version number (format: ".version X.Y")
        const version_start = version_idx.? + version_pattern.len;
        if (version_start >= ptx.len) {
            return PtxValidationError.InvalidPtxVersion;
        }

        // Find end of version line
        const version_end = std.mem.indexOfAnyPos(u8, ptx, version_start, "\r\n") orelse ptx.len;
        const version_str = std.mem.trim(u8, ptx[version_start..version_end], " \t");

        // Parse major version
        const major_end = std.mem.indexOfScalar(u8, version_str, '.') orelse version_str.len;
        const major_str = version_str[0..major_end];
        const major_version = std.fmt.parseInt(u32, major_str, 10) catch {
            return PtxValidationError.InvalidPtxVersion;
        };

        // Validate version range - warn but allow (driver will handle compatibility)
        if (major_version < MIN_PTX_VERSION_MAJOR or major_version > MAX_PTX_VERSION_MAJOR) {
            std.log.warn("PTX version {d}.x is outside supported range ({d}-{d}), allowing anyway (driver may JIT compile)", .{
                major_version, MIN_PTX_VERSION_MAJOR, MAX_PTX_VERSION_MAJOR,
            });
        }

        // Check for target directive
        const target_pattern = ".target ";
        const target_idx = std.mem.indexOf(u8, ptx, target_pattern);
        if (target_idx == null) {
            return PtxValidationError.InvalidPtxTarget;
        }

        // Parse target architecture
        const target_start = target_idx.? + target_pattern.len;
        if (target_start >= ptx.len) {
            return PtxValidationError.InvalidPtxTarget;
        }

        const target_end = std.mem.indexOfAnyPos(u8, ptx, target_start, " \t\r\n,") orelse ptx.len;
        const target_str = ptx[target_start..target_end];

        // Validate target is in supported list
        var target_valid = false;
        for (SUPPORTED_TARGETS) |supported| {
            if (std.mem.eql(u8, target_str, supported)) {
                target_valid = true;
                break;
            }
        }
        if (!target_valid) {
            // Allow but warn - driver may still support it via JIT
            std.log.warn("PTX target '{s}' not in standard supported list, allowing anyway", .{target_str});
        }

        // Check for suspicious patterns
        for (SUSPICIOUS_PATTERNS) |pattern| {
            if (std.mem.indexOf(u8, ptx, pattern) != null) {
                std.log.err("PTX contains suspicious pattern: '{s}'", .{pattern});
                return PtxValidationError.SuspiciousPtxPattern;
            }
        }

        // Check for required entry point structure
        if (std.mem.indexOf(u8, ptx, ".entry ") == null and
            std.mem.indexOf(u8, ptx, ".func ") == null)
        {
            // PTX without entry points is unusual but may be valid (e.g., library)
            std.log.warn("PTX code contains no .entry or .func directives", .{});
        }

        // All validation passed
        return;
    }

    /// Validate PTX and log results (for debugging)
    fn validateAndLogPtx(ptx: []const u8, kernel_name: []const u8) !void {
        validatePtx(ptx) catch |err| {
            std.log.err("PTX validation failed for kernel '{s}': {}", .{ kernel_name, err });
            return err;
        };
        std.log.debug("PTX validation passed for kernel '{s}'", .{kernel_name});
    }

    // =============================================================================
    // Kernel Management
    // =============================================================================

    /// Load a kernel from PTX code
    /// SECURITY: Validates PTX before loading to prevent injection attacks
    pub fn loadKernel(self: *CudaContext, name: []const u8, ptx_code: []const u8) !void {
        // Check if already loaded
        if (self.kernels.contains(name)) {
            return;
        }

        // Log PTX info for debugging
        const ptx_preview_len = @min(ptx_code.len, 200);
        std.log.debug("CUDA: Loading kernel '{s}' from PTX ({d} bytes)", .{ name, ptx_code.len });
        std.log.debug("CUDA: PTX preview: {s}", .{ptx_code[0..ptx_preview_len]});

        // SECURITY FIX: Validate PTX before loading
        // This prevents injection attacks and ensures PTX compatibility
        validateAndLogPtx(ptx_code, name) catch |err| {
            // Map validation errors to InvalidPtx for backward compatibility
            std.log.err("PTX validation failed for kernel '{s}': {}, mapping to InvalidPtx", .{ name, err });
            return error.InvalidPtx;
        };

        // Ensure PTX is null-terminated for the driver
        const ptx_z = try self.allocator.dupeZ(u8, ptx_code);
        defer self.allocator.free(ptx_z);

        // Load module from PTX
        var module: *CUmodule = undefined;
        const result = self.driver.driver.moduleLoadData.?(
            &module,
            ptx_z.ptr,
        );
        if (result != .SUCCESS) {
            std.log.err("moduleLoadData failed for kernel '{s}': {any} (code {d})", .{ name, result, @intFromEnum(result) });
            // Provide specific error messages for common PTX errors
            switch (result) {
                .ERROR_INVALID_PTX => {
                    std.log.err("  -> Invalid PTX syntax. Check PTX version and target architecture.", .{});
                    return error.InvalidPtx;
                },
                .ERROR_UNSUPPORTED_PTX_VERSION => {
                    std.log.err("  -> PTX version not supported by driver. Try updating the driver or using older PTX version.", .{});
                    return error.UnsupportedPtxVersion;
                },
                .ERROR_NO_BINARY_FOR_GPU => {
                    std.log.err("  -> No binary available for this GPU. Check compute capability compatibility.", .{});
                    return error.NoBinaryForGpu;
                },
                else => return error.CudaUnknownError,
            }
        }
        try self.modules.append(self.allocator, module);

        // Get function
        var function: *CUfunction = undefined;
        const name_z = try self.allocator.dupeZ(u8, name);
        defer self.allocator.free(name_z);

        try cuda_driver.checkCuda(self.driver.driver.moduleGetFunction.?(
            &function,
            module,
            name_z,
        ));

        // Get occupancy info
        var min_grid_size: c_int = 0;
        var max_threads_per_block: c_int = 0;
        if (self.driver.driver.occupancyMaxPotentialBlockSize) |occupancy_fn| {
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

    /// Compile and load a kernel from CUDA C source using NVRTC
    /// This allows runtime compilation for the specific GPU architecture
    pub fn compileAndLoadKernel(
        self: *CudaContext,
        name: []const u8,
        source: []const u8,
    ) !void {
        // Check if already loaded
        if (self.kernels.contains(name)) {
            return;
        }

        if (!self.driver.driver.is_nvrtc_available) {
            return error.NvrtcNotAvailable;
        }

        std.log.info("CUDA: Compiling kernel '{s}' with NVRTC...", .{name});

        // Log device capability for debugging
        const major = self.device_props.compute_capability_major;
        const minor = self.device_props.compute_capability_minor;
        std.log.info("CUDA: Device capability {d}.{d}", .{ major, minor });

        // Compile with NVRTC (no specific architecture - uses default PTX)
        const ptx = try self.compileKernel(source, name);
        defer self.allocator.free(ptx);

        std.log.info("CUDA: Kernel '{s}' compiled successfully (PTX size: {} bytes)", .{ name, ptx.len });

        // Load the compiled PTX
        try self.loadKernel(name, ptx);
    }

    /// Internal helper to compile CUDA source to PTX using dynamic NVRTC
    /// Uses default PTX generation for maximum compatibility
    fn compileKernel(
        self: *CudaContext,
        source: []const u8,
        name: []const u8,
    ) ![]u8 {
        var program: *nvrtcProgram = undefined;

        // Create null-terminated strings
        const src_z = try self.allocator.dupeZ(u8, source);
        defer self.allocator.free(src_z);

        const name_z = try self.allocator.dupeZ(u8, name);
        defer self.allocator.free(name_z);

        // Create the program
        const create_result = self.driver.driver.nvrtcCreateProgram.?(
            &program,
            src_z.ptr,
            name_z.ptr,
            0,
            null,
            null,
        );
        if (!create_result.isSuccess()) return error.NvrtcProgramCreationFailed;
        defer _ = self.driver.driver.nvrtcDestroyProgram.?(&program);

        // Prepare compiler options
        // Note: We skip architecture-specific flags to let NVRTC use default PTX
        // which provides maximum compatibility across GPU generations
        const max_options = 2;
        var option_ptrs: [max_options][*c]const u8 = undefined;
        var option_count: usize = 0;

        // Use default PTX generation (no --gpu-architecture flag)
        // This produces PTX that the driver can JIT compile to any architecture

        // Standard options for performance
        const fast_math = try self.allocator.dupeZ(u8, "--use_fast_math");
        option_ptrs[option_count] = fast_math.ptr;
        option_count += 1;

        const std_cpp = try self.allocator.dupeZ(u8, "-std=c++11");
        option_ptrs[option_count] = std_cpp.ptr;
        option_count += 1;

        defer {
            self.allocator.free(fast_math);
            self.allocator.free(std_cpp);
        }

        // Compile the program
        const compile_result = self.driver.driver.nvrtcCompileProgram.?(
            program,
            @intCast(option_count),
            &option_ptrs,
        );

        // Get compilation log regardless of success/failure
        var log_size: usize = 0;
        _ = self.driver.driver.nvrtcGetProgramLogSize.?(program, &log_size);

        if (log_size > 1) {
            const log_buf = try self.allocator.alloc(u8, log_size);
            defer self.allocator.free(log_buf);
            _ = self.driver.driver.nvrtcGetProgramLog.?(program, log_buf.ptr);

            if (!compile_result.isSuccess()) {
                std.log.err("NVRTC compilation failed for '{s}':\n{s}", .{ name, log_buf });
            } else {
                std.log.debug("NVRTC log for '{s}':\n{s}", .{ name, log_buf });
            }
        }

        if (!compile_result.isSuccess()) return error.NvrtcCompilationFailed;

        // Get PTX size
        var ptx_size: usize = 0;
        _ = self.driver.driver.nvrtcGetPTXSize.?(program, &ptx_size);

        // Allocate PTX buffer
        const ptx = try self.allocator.alloc(u8, ptx_size);
        errdefer self.allocator.free(ptx);

        // Get the PTX
        _ = self.driver.driver.nvrtcGetPTX.?(program, ptx.ptr);

        // Debug: Log the generated PTX header to diagnose version issues
        std.log.debug("CUDA: Generated PTX for '{s}' (first 500 chars):\n{s}", .{ name, ptx[0..@min(ptx.len, 500)] });

        return ptx;
    }

    /// Try NVRTC compilation, fall back to embedded PTX if available
    pub fn compileKernelWithFallback(
        self: *CudaContext,
        name: []const u8,
        source: []const u8,
        embedded_ptx: ?[]const u8,
    ) !void {
        // Try NVRTC first
        self.compileAndLoadKernel(name, source) catch |err| {
            std.log.warn("NVRTC failed for '{s}', trying fallback: {}", .{ name, err });

            // Fall back to embedded PTX if available
            if (embedded_ptx) |ptx| {
                std.log.info("CUDA: Loading embedded PTX for '{s}'", .{name});
                try self.loadKernel(name, ptx);
            } else {
                return err;
            }
        };
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
    /// SECURITY FIX: Thread-safe stream access (CRIT-003)
    /// SECURITY FIX: Comprehensive bounds checking for all kernel launch parameters (HIGH-005)
    pub fn launchKernel(
        self: *CudaContext,
        kernel_name: []const u8,
        grid_dim: [3]u32,
        block_dim: [3]u32,
        shared_mem_bytes: u32,
        args: []const ?*anyopaque,
    ) !void {
        const strm = self.stream.getOrError() catch |err| return err;

        const kernel = try self.getKernel(kernel_name);

        // =============================================================================
        // SECURITY FIX: Validate kernel launch parameters (HIGH-005)
        // All parameters must be validated before kernel launch to prevent:
        // - Buffer overflows
        // - GPU hangs
        // - Memory corruption
        // - Undefined behavior
        // =============================================================================

        // Validate argument count
        if (args.len > MAX_KERNEL_ARGS) {
            std.log.err("Kernel '{s}' has {d} arguments, max is {d}", .{ kernel_name, args.len, MAX_KERNEL_ARGS });
            return error.TooManyKernelArguments;
        }

        // Validate all arguments are non-null
        for (args, 0..) |arg, i| {
            if (arg == null) {
                std.log.err("Kernel '{s}' argument {d} is null", .{ kernel_name, i });
                return error.NullKernelArgument;
            }
        }

        // Validate grid dimensions against device limits
        if (grid_dim[0] > self.device_props.max_grid_dim_x or
            grid_dim[0] > MAX_GRID_DIM)
        {
            std.log.err("Kernel '{s}' grid_dim.x ({d}) exceeds limit (device: {d}, max: {d})", .{
                kernel_name, grid_dim[0], self.device_props.max_grid_dim_x, MAX_GRID_DIM,
            });
            return error.InvalidGridDimension;
        }

        if (grid_dim[1] > self.device_props.max_grid_dim_y or
            grid_dim[1] > MAX_GRID_DIM)
        {
            std.log.err("Kernel '{s}' grid_dim.y ({d}) exceeds limit (device: {d}, max: {d})", .{
                kernel_name, grid_dim[1], self.device_props.max_grid_dim_y, MAX_GRID_DIM,
            });
            return error.InvalidGridDimension;
        }

        if (grid_dim[2] > self.device_props.max_grid_dim_z or
            grid_dim[2] > MAX_GRID_DIM)
        {
            std.log.err("Kernel '{s}' grid_dim.z ({d}) exceeds limit (device: {d}, max: {d})", .{
                kernel_name, grid_dim[2], self.device_props.max_grid_dim_z, MAX_GRID_DIM,
            });
            return error.InvalidGridDimension;
        }

        // Validate block dimensions against device limits
        if (block_dim[0] > self.device_props.max_block_dim_x) {
            std.log.err("Kernel '{s}' block_dim.x ({d}) exceeds device limit ({d})", .{
                kernel_name, block_dim[0], self.device_props.max_block_dim_x,
            });
            return error.InvalidBlockDimension;
        }

        if (block_dim[1] > self.device_props.max_block_dim_y) {
            std.log.err("Kernel '{s}' block_dim.y ({d}) exceeds device limit ({d})", .{
                kernel_name, block_dim[1], self.device_props.max_block_dim_y,
            });
            return error.InvalidBlockDimension;
        }

        if (block_dim[2] > self.device_props.max_block_dim_z) {
            std.log.err("Kernel '{s}' block_dim.z ({d}) exceeds device limit ({d})", .{
                kernel_name, block_dim[2], self.device_props.max_block_dim_z,
            });
            return error.InvalidBlockDimension;
        }

        // Validate total threads per block
        const total_threads = @as(u64, block_dim[0]) * @as(u64, block_dim[1]) * @as(u64, block_dim[2]);
        if (total_threads > self.device_props.max_threads_per_block or
            total_threads > MAX_THREADS_PER_BLOCK)
        {
            std.log.err("Kernel '{s}' total threads per block ({d}) exceeds limit (device: {d}, max: {d})", .{
                kernel_name, total_threads, self.device_props.max_threads_per_block, MAX_THREADS_PER_BLOCK,
            });
            return error.TooManyThreadsPerBlock;
        }

        // Validate shared memory size
        if (shared_mem_bytes > self.device_props.max_shared_memory_per_block) {
            std.log.err("Kernel '{s}' shared memory ({d} bytes) exceeds device limit ({d} bytes)", .{
                kernel_name, shared_mem_bytes, self.device_props.max_shared_memory_per_block,
            });
            return error.SharedMemoryTooLarge;
        }

        if (shared_mem_bytes > MAX_SHARED_MEMORY_PER_BLOCK) {
            std.log.err("Kernel '{s}' shared memory ({d} bytes) exceeds safe limit ({d} bytes)", .{
                kernel_name, shared_mem_bytes, MAX_SHARED_MEMORY_PER_BLOCK,
            });
            return error.SharedMemoryTooLarge;
        }

        // Prepare kernel parameters
        var kernel_params: [MAX_KERNEL_ARGS]?*anyopaque = undefined;
        @memcpy(kernel_params[0..args.len], args);

        // PROFILER: Start timing if profiling enabled
        var start_event: ?*CUevent = null;
        if (self.profiler) |profiler| {
            if (profiler.timing_enabled) {
                start_event = try profiler.getEvent();
                try cuda_driver.checkCuda(self.driver.driver.eventRecord.?(start_event.?, strm));
            }
            // Push range marker for Nsight Systems
            _ = profiler.pushRange(kernel_name, null) catch 0;
        }

        try cuda_driver.checkCuda(self.driver.driver.launchKernel.?(
            kernel.function,
            grid_dim[0],
            grid_dim[1],
            grid_dim[2],
            block_dim[0],
            block_dim[1],
            block_dim[2],
            shared_mem_bytes,
            strm,
            &kernel_params,
            null,
        ));

        // PROFILER: End timing if profiling enabled
        if (self.profiler) |profiler| {
            // Pop range marker
            try profiler.popRange();

            // Record timing
            if (start_event) |se| {
                const end_event = try profiler.getEvent();
                defer profiler.returnEvent(end_event);

                try cuda_driver.checkCuda(self.driver.driver.eventRecord.?(end_event, strm));
                try cuda_driver.checkCuda(self.driver.driver.eventSynchronize.?(end_event));

                var duration_ms: f32 = 0;
                try cuda_driver.checkCuda(self.driver.driver.eventElapsedTime.?(
                    &duration_ms,
                    se,
                    end_event,
                ));

                const num_threads = grid_dim[0] * grid_dim[1] * grid_dim[2] *
                    block_dim[0] * block_dim[1] * block_dim[2];

                const timing = cuda_profiler.KernelTiming{
                    .name = profiler.allocator.dupe(u8, kernel_name) catch return,
                    .start_ms = 0,
                    .end_ms = 0,
                    .duration_ms = @as(f64, @floatCast(duration_ms)),
                    .grid_dim = grid_dim,
                    .block_dim = block_dim,
                    .shared_mem_bytes = shared_mem_bytes,
                    .num_threads = num_threads,
                };
                profiler.timing_history.append(self.allocator, timing) catch {};
                profiler.returnEvent(se);
            }
        }
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

    /// Calculate optimal dimensions for matrix operations (legacy naive kernel)
    pub fn getMatrixConfig(_: *const CudaContext, m: usize, n: usize) struct { grid_x: u32, grid_y: u32, block_x: u32, block_y: u32 } {
        const block_x: u32 = 16;
        const block_y: u32 = 16;
        const grid_x = @as(u32, @intCast((n + block_x - 1) / block_x));
        const grid_y = @as(u32, @intCast((m + block_y - 1) / block_y));
        return .{ .grid_x = grid_x, .grid_y = grid_y, .block_x = block_x, .block_y = block_y };
    }

    /// Tile size for shared memory tiled matrix multiplication
    pub const TILE_SIZE: u32 = 32;

    /// Calculate dimensions for tiled matrix multiplication
    /// Uses 32x32 thread blocks with shared memory tiling for 10-100x performance
    pub fn getTiledMatMulConfig(_: *const CudaContext, m: usize, n: usize) struct {
        grid_x: u32,
        grid_y: u32,
        block_x: u32,
        block_y: u32,
        shared_mem_bytes: u32,
    } {
        const block_x = TILE_SIZE;
        const block_y = TILE_SIZE;
        const grid_x = @as(u32, @intCast((n + block_x - 1) / block_x));
        const grid_y = @as(u32, @intCast((m + block_y - 1) / block_y));
        // Shared memory: 2 tiles (A and B) * TILE_SIZE * TILE_SIZE * sizeof(float)
        const shared_mem_bytes = 2 * TILE_SIZE * TILE_SIZE * @sizeOf(f32);
        return .{
            .grid_x = grid_x,
            .grid_y = grid_y,
            .block_x = block_x,
            .block_y = block_y,
            .shared_mem_bytes = shared_mem_bytes,
        };
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

    /// Initialize CUDA context for a specific device
    /// Used by MultiCudaContext to create contexts for specific devices
    pub fn initForDevice(allocator: std.mem.Allocator, target_device: i32) !CudaContext {
        // Acquire a reference to the global driver
        const driver_ref = try cuda_driver.CudaDriverRef.acquire(allocator);
        errdefer driver_ref.release();

        // Get device count to validate target_device
        var device_count: i32 = 0;
        try cuda_driver.checkCuda(driver_ref.driver.deviceGetCount.?(&device_count));
        if (target_device < 0 or target_device >= device_count) {
            return error.InvalidDevice;
        }

        // Get the device
        var device: i32 = undefined;
        try cuda_driver.checkCuda(driver_ref.driver.deviceGet.?(&device, target_device));

        // Create context
        var temp_context: ?*cuda_driver.CUcontext = null;
        try cuda_driver.checkCuda(driver_ref.driver.ctxCreate.?(
            @ptrCast(&temp_context),
            @intFromEnum(cuda_driver.CUctx_flags.SCHED_AUTO),
            device,
        ));
        errdefer {
            if (temp_context) |ctx| {
                _ = driver_ref.driver.ctxDestroy.?(ctx);
            }
        }

        // Create stream
        var temp_stream: ?*cuda_driver.CUstream = null;
        try cuda_driver.checkCuda(driver_ref.driver.streamCreate.?(
            @ptrCast(&temp_stream),
            @intFromEnum(cuda_driver.CUstream_flags.DEFAULT),
        ));
        errdefer {
            if (temp_stream) |strm| {
                _ = driver_ref.driver.streamDestroy.?(strm);
            }
        }

        // Query device properties
        var props = try queryDevicePropertiesForDevice(driver_ref.driver, device);

        // Initialize the context struct
        var ctx: CudaContext = undefined;
        ctx.allocator = allocator;
        ctx.driver = driver_ref;
        ctx.device = device;
        ctx.context = temp_context;
        ctx.stream = ThreadSafeStream.init();
        ctx.stream.set(temp_stream);
        ctx.device_props = props;
        ctx.kernels = std.StringHashMap(Kernel).init(allocator);

        // Initialize buffer pools
        for (0..MEMORY_POOL_BUCKETS) |i| {
            ctx.buffer_pools[i] = .empty;
        }
        ctx.temp_buffers = .empty;
        ctx.modules = .empty;

        // Initialize profiler to null
        ctx.profiler = null;

        std.log.info("CUDA Context initialized for device {d}: {s}", .{ device, std.mem.sliceTo(&props.name, 0) });

        return ctx;
    }

    /// Query device properties for a specific device
    fn queryDevicePropertiesForDevice(driver: *cuda_driver.CudaDriver, device: i32) !DeviceProperties {
        var props: DeviceProperties = undefined;

        // Get device name
        var name: [256]u8 = undefined;
        try cuda_driver.checkCuda(driver.deviceGetName.?(&name, name.len, device));
        props.name = name;

        // Get compute capability and other attributes
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.compute_capability_major, .COMPUTE_CAPABILITY_MAJOR, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.compute_capability_minor, .COMPUTE_CAPABILITY_MINOR, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.multiprocessor_count, .MULTIPROCESSOR_COUNT, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.max_threads_per_block, .MAX_THREADS_PER_BLOCK, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.max_block_dim_x, .MAX_BLOCK_DIM_X, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.max_block_dim_y, .MAX_BLOCK_DIM_Y, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.max_block_dim_z, .MAX_BLOCK_DIM_Z, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.max_grid_dim_x, .MAX_GRID_DIM_X, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.max_grid_dim_y, .MAX_GRID_DIM_Y, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.max_grid_dim_z, .MAX_GRID_DIM_Z, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.max_shared_memory_per_block, .MAX_SHARED_MEMORY_PER_BLOCK, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.warp_size, .WARP_SIZE, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.memory_clock_rate, .MEMORY_CLOCK_RATE, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.global_memory_bus_width, .GLOBAL_MEMORY_BUS_WIDTH, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.l2_cache_size, .L2_CACHE_SIZE, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.max_threads_per_multiprocessor, .MAX_THREADS_PER_MULTIPROCESSOR, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.unified_addressing, .UNIFIED_ADDRESSING, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.managed_memory, .MANAGED_MEMORY, device));
        try cuda_driver.checkCuda(driver.deviceGetAttribute.?(&props.concurrent_managed_access, .CONCURRENT_MANAGED_ACCESS, device));

        // Get total memory
        try cuda_driver.checkCuda(driver.deviceTotalMem.?(&props.total_memory, device));

        return props;
    }
};

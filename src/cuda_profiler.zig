/// CUDA Profiler Integration for ZigNeuron
/// Provides Nsight Systems/Compute integration with minimal overhead when disabled
///
/// Architecture:
/// - Zero overhead when profiling is disabled (compile-time and runtime checks)
/// - NVTX range markers for Nsight Systems visualization
/// - CUDA events for precise kernel timing
/// - Custom metrics collection for library-specific performance data
/// - Memory profiling for tracking GPU memory usage patterns
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda_context = @import("cuda_context.zig");

const CUevent = cuda_driver.CUevent;
const CUevent_flags = cuda_driver.CUevent_flags;
const CUstream = cuda_driver.CUstream;
const CudaContext = cuda_context.CudaContext;

// Get current time in nanoseconds (Zig 0.16+ compatible)
fn nanoTimestamp() u64 {
    var tv: std.os.linux.timespec = undefined;
    const CLOCK_MONOTONIC: std.os.linux.clockid_t = .MONOTONIC;
    _ = std.os.linux.clock_gettime(CLOCK_MONOTONIC, &tv);
    return @as(u64, @intCast(tv.sec)) * 1_000_000_000 + @as(u64, @intCast(tv.nsec));
}

// =============================================================================
// Compile-Time Configuration
// =============================================================================

/// Enable profiling at compile time (default: false for release builds)
/// Set with: -Dcuda_profiler_enable=true
pub const PROFILING_ENABLED = if (@hasDecl(@import("root"), "cuda_profiler_enable"))
    @import("root").cuda_profiler_enable
else
    !std.builtin.mode.isRelease();

/// Enable NVTX markers (requires NVTX library at runtime)
pub const NVTX_ENABLED = PROFILING_ENABLED;

/// Maximum number of concurrent ranges
pub const MAX_CONCURRENT_RANGES: usize = 256;

/// Maximum number of custom metrics
pub const MAX_CUSTOM_METRICS: usize = 64;

/// Maximum events in timing pool
pub const MAX_TIMING_EVENTS: usize = 1024;

// =============================================================================
// NVTX Types and Functions (Dynamically Loaded)
// =============================================================================

/// NVTX domain handle
const nvtxDomainHandle = *opaque {};

/// NVTX range ID
const nvtxRangeId = u64;

/// NVTX color types
const nvtxColorType = enum(c_int) {
    UNKNOWN = 0,
    ARGB = 1,
};

/// NVTX message types
const nvtxMessageType = enum(c_int) {
    UNKNOWN = 0,
    ASCII = 1,
    UNICODE = 2,
    REGISTERED = 3,
};

/// NVTX payload types
const nvtxPayloadType = enum(c_int) {
    UNKNOWN = 0,
    UNSIGNED_INT64 = 1,
    INT64 = 2,
    DOUBLE = 3,
    UNSIGNED_INT32 = 4,
    INT32 = 5,
    FLOAT = 6,
};

/// NVTX event attributes
const nvtxEventAttributes = extern struct {
    version: u16 = 2,
    size: u16 = @sizeOf(nvtxEventAttributes),
    category: u32 = 0,
    colorType: nvtxColorType = .UNKNOWN,
    color: u32 = 0,
    payloadType: nvtxPayloadType = .UNKNOWN,
    payload: extern union {
        ull: u64,
        ll: i64,
        d: f64,
        ui: u32,
        i: i32,
        f: f32,
    } = .{ .ull = 0 },
    messageType: nvtxMessageType = .UNKNOWN,
    message: extern union {
        ascii: [*c]const u8,
        unicode: [*c]const u16,
        registered: u32,
    } = .{ .ascii = null },
};

// =============================================================================
// Profiler State
// =============================================================================

/// Profiling mode
pub const ProfilerMode = enum {
    /// Profiling disabled - zero overhead
    disabled,
    /// Manual profiling - explicit start/stop
    manual,
    /// Automatic profiling - tracks all operations
    automatic,
    /// Full profiling - includes memory tracking
    full,
};

/// Timing result for kernel execution
pub const KernelTiming = struct {
    /// Kernel name
    name: []const u8,
    /// Start time in milliseconds
    start_ms: f64,
    /// End time in milliseconds
    end_ms: f64,
    /// Duration in milliseconds
    duration_ms: f64,
    /// Grid dimensions
    grid_dim: [3]u32,
    /// Block dimensions
    block_dim: [3]u32,
    /// Shared memory used (bytes)
    shared_mem_bytes: u32,
    /// Number of threads
    num_threads: u32,
};

/// Memory statistics
pub const MemoryStats = struct {
    /// Total allocations
    total_allocations: u64 = 0,
    /// Total deallocations
    total_deallocations: u64 = 0,
    /// Current allocated bytes
    current_bytes: u64 = 0,
    /// Peak allocated bytes
    peak_bytes: u64 = 0,
    /// Total bytes allocated (cumulative)
    total_bytes_allocated: u64 = 0,
    /// Total bytes freed (cumulative)
    total_bytes_freed: u64 = 0,
    /// Number of pool hits
    pool_hits: u64 = 0,
    /// Number of pool misses
    pool_misses: u64 = 0,
};

/// Custom metric value
pub const MetricValue = union(enum) {
    int64: i64,
    uint64: u64,
    float64: f64,
    counter: u64,
    timestamp: u64,
};

/// Custom metric definition
pub const CustomMetric = struct {
    /// Metric name
    name: []const u8,
    /// Metric description
    description: []const u8,
    /// Current value
    value: MetricValue,
    /// Total samples
    sample_count: u64,
    /// Running average (for numeric types)
    running_average: f64,
    /// Minimum value
    min_value: f64,
    /// Maximum value
    max_value: f64,
};

/// Active range information
const ActiveRange = struct {
    /// Range name
    name: []const u8,
    /// NVTX range ID (if using NVTX)
    nvtx_id: ?u64,
    /// Start event (for CUDA timing)
    start_event: ?*CUevent,
    /// Start timestamp
    start_timestamp: i64,
};

/// Event pool entry
const EventPoolEntry = struct {
    /// CUDA event
    event: *CUevent,
    /// Is available
    available: bool,
};

/// CUDA Profiler structure
pub const CudaProfiler = struct {
    /// Current profiling mode
    mode: ProfilerMode,
    /// Is initialized
    initialized: bool,
    /// NVTX library handle (null if not available)
    nvtx_handle: ?std.DynLib,
    /// NVTX domain
    nvtx_domain: ?*nvtxDomainHandle,
    /// NVTX function pointers
    nvtx_funcs: NVTXFunctions,
    /// Active ranges stack
    active_ranges: std.ArrayList(ActiveRange),
    /// Timing history
    timing_history: std.ArrayList(KernelTiming),
    /// Memory statistics
    memory_stats: MemoryStats,
    /// Custom metrics
    custom_metrics: std.StringHashMap(CustomMetric),
    /// Event pool for timing
    event_pool: std.ArrayList(EventPoolEntry),
    /// Next event index
    next_event_idx: usize,
    /// Allocator
    allocator: std.mem.Allocator,
    /// Context reference
    context: ?*CudaContext,
    /// Timing enabled
    timing_enabled: bool,
    /// Memory tracking enabled
    memory_tracking_enabled: bool,

    /// NVTX function signatures
    const NVTXFunctions = struct {
        /// Mark a point in the application
        nvtxMarkA: ?*const fn ([*c]const nvtxEventAttributes) callconv(.c) void,
        /// Start a range
        nvtxRangeStartEx: ?*const fn ([*c]const nvtxEventAttributes) callconv(.c) nvtxRangeId,
        /// End a range
        nvtxRangeEnd: ?*const fn (nvtxRangeId) callconv(.c) void,
        /// Push a range onto the stack
        nvtxRangePushEx: ?*const fn ([*c]const nvtxEventAttributes) callconv(.c) c_int,
        /// Pop a range from the stack
        nvtxRangePop: ?*const fn () callconv(.c) c_int,
        /// Create a domain
        nvtxDomainCreateA: ?*const fn ([*c]const u8) callconv(.c) ?*nvtxDomainHandle,
        /// Destroy a domain
        nvtxDomainDestroy: ?*const fn (*nvtxDomainHandle) callconv(.c) void,
        /// Mark in domain
        nvtxDomainMarkEx: ?*const fn (?*nvtxDomainHandle, [*c]const nvtxEventAttributes) callconv(.c) void,
        /// Start range in domain
        nvtxDomainRangeStartEx: ?*const fn (?*nvtxDomainHandle, [*c]const nvtxEventAttributes) callconv(.c) nvtxRangeId,
        /// Push range in domain
        nvtxDomainRangePushEx: ?*const fn (?*nvtxDomainHandle, [*c]const nvtxEventAttributes) callconv(.c) c_int,
    };

    // =============================================================================
    // Initialization
    // =============================================================================

    /// Initialize the profiler
    /// Returns a profiler instance (may be in disabled mode if profiling not available)
    pub fn init(allocator: std.mem.Allocator, mode: ProfilerMode) !*CudaProfiler {
        const self = try allocator.create(CudaProfiler);
        errdefer allocator.destroy(self);

        self.* = .{
            .mode = mode,
            .initialized = false,
            .nvtx_handle = null,
            .nvtx_domain = null,
            .nvtx_funcs = .{
                .nvtxMarkA = null,
                .nvtxRangeStartEx = null,
                .nvtxRangeEnd = null,
                .nvtxRangePushEx = null,
                .nvtxRangePop = null,
                .nvtxDomainCreateA = null,
                .nvtxDomainDestroy = null,
                .nvtxDomainMarkEx = null,
                .nvtxDomainRangeStartEx = null,
                .nvtxDomainRangePushEx = null,
            },
            .active_ranges = std.ArrayList(ActiveRange).init(allocator),
            .timing_history = std.ArrayList(KernelTiming).init(allocator),
            .memory_stats = .{},
            .custom_metrics = std.StringHashMap(CustomMetric).init(allocator),
            .event_pool = std.ArrayList(EventPoolEntry).init(allocator),
            .next_event_idx = 0,
            .allocator = allocator,
            .context = null,
            .timing_enabled = mode != .disabled,
            .memory_tracking_enabled = mode == .full,
        };

        // If profiling disabled, return immediately
        if (mode == .disabled) {
            return self;
        }

        // Try to load NVTX library
        try self.loadNVTX();

        // Create NVTX domain if available
        if (self.nvtx_funcs.nvtxDomainCreateA != null) {
            const domain_name = try allocator.dupeZ(u8, "ZigNeuron");
            defer allocator.free(domain_name);
            self.nvtx_domain = self.nvtx_funcs.nvtxDomainCreateA.?(domain_name.ptr);
        }

        self.initialized = true;
        return self;
    }

    /// Initialize profiler with a CUDA context
    pub fn initWithContext(
        allocator: std.mem.Allocator,
        mode: ProfilerMode,
        context: *CudaContext,
    ) !*CudaProfiler {
        const self = try init(allocator, mode);
        self.context = context;

        // Pre-allocate event pool if timing enabled
        if (self.timing_enabled) {
            try self.allocateEventPool(32);
        }

        return self;
    }

    /// Load NVTX library dynamically
    fn loadNVTX(self: *CudaProfiler) !void {
        const lib_name = switch (@import("builtin").os.tag) {
            .linux => "libnvToolsExt.so.1",
            .windows => "nvToolsExt64_1.dll",
            else => return error.UnsupportedPlatform,
        };

        self.nvtx_handle = std.DynLib.open(lib_name) catch |err| {
            std.log.debug("NVTX library not found: {s}", .{@errorName(err)});
            return;
        };

        // Load function pointers
        self.loadNVTXFunction("nvtxMarkA", &self.nvtx_funcs.nvtxMarkA);
        self.loadNVTXFunction("nvtxRangeStartEx", &self.nvtx_funcs.nvtxRangeStartEx);
        self.loadNVTXFunction("nvtxRangeEnd", &self.nvtx_funcs.nvtxRangeEnd);
        self.loadNVTXFunction("nvtxRangePushEx", &self.nvtx_funcs.nvtxRangePushEx);
        self.loadNVTXFunction("nvtxRangePop", &self.nvtx_funcs.nvtxRangePop);
        self.loadNVTXFunction("nvtxDomainCreateA", &self.nvtx_funcs.nvtxDomainCreateA);
        self.loadNVTXFunction("nvtxDomainDestroy", &self.nvtx_funcs.nvtxDomainDestroy);
        self.loadNVTXFunction("nvtxDomainMarkEx", &self.nvtx_funcs.nvtxDomainMarkEx);
        self.loadNVTXFunction("nvtxDomainRangeStartEx", &self.nvtx_funcs.nvtxDomainRangeStartEx);
        self.loadNVTXFunction("nvtxDomainRangePushEx", &self.nvtx_funcs.nvtxDomainRangePushEx);

        std.log.info("CUDA Profiler: NVTX loaded successfully", .{});
    }

    fn loadNVTXFunction(self: *CudaProfiler, name: [:0]const u8, ptr: anytype) void {
        if (self.nvtx_handle) |*h| {
            const func = h.lookup(*const anyopaque, @ptrCast(name)) orelse return;
            ptr.* = @ptrCast(func);
        }
    }

    /// Pre-allocate CUDA events for timing
    fn allocateEventPool(self: *CudaProfiler, count: usize) !void {
        if (self.context == null) return;

        const driver = self.context.?.driver.driver;

        for (0..count) |_| {
            var event: *CUevent = undefined;
            try cuda_driver.checkCuda(driver.eventCreate.?(
                &event,
                @intFromEnum(CUevent_flags.DEFAULT),
            ));
            try self.event_pool.append(.{
                .event = event,
                .available = true,
            });
        }
    }

    /// Cleanup profiler resources
    pub fn deinit(self: *CudaProfiler) void {
        // Destroy NVTX domain
        if (self.nvtx_domain) |domain| {
            if (self.nvtx_funcs.nvtxDomainDestroy) |destroy| {
                destroy(domain);
            }
        }

        // Close NVTX library
        if (self.nvtx_handle) |*h| {
            h.close();
        }

        // Destroy CUDA events
        if (self.context) |ctx| {
            const driver = ctx.driver.driver;
            for (self.event_pool.items) |entry| {
                _ = driver.eventDestroy.?(entry.event);
            }
        }
        self.event_pool.deinit(self.allocator);

        // Free timing history
        for (self.timing_history.items) |*timing| {
            self.allocator.free(timing.name);
        }
        self.timing_history.deinit(self.allocator);

        // Free active ranges
        for (self.active_ranges.items) |*range| {
            self.allocator.free(range.name);
        }
        self.active_ranges.deinit(self.allocator);

        // Free custom metrics
        var metric_iter = self.custom_metrics.iterator();
        while (metric_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.name);
            self.allocator.free(entry.value_ptr.description);
        }
        self.custom_metrics.deinit();

        self.allocator.destroy(self);
    }

    // =============================================================================
    // Mode Control
    // =============================================================================

    /// Set profiling mode at runtime
    pub fn setMode(self: *CudaProfiler, mode: ProfilerMode) void {
        if (self.mode == mode) return;

        self.mode = mode;
        self.timing_enabled = mode != .disabled;
        self.memory_tracking_enabled = mode == .full;

        std.log.info("CUDA Profiler: Mode changed to {s}", .{@tagName(mode)});
    }

    /// Enable timing
    pub fn enableTiming(self: *CudaProfiler) void {
        self.timing_enabled = true;
    }

    /// Disable timing
    pub fn disableTiming(self: *CudaProfiler) void {
        self.timing_enabled = false;
    }

    /// Enable memory tracking
    pub fn enableMemoryTracking(self: *CudaProfiler) void {
        self.memory_tracking_enabled = true;
    }

    /// Disable memory tracking
    pub fn disableMemoryTracking(self: *CudaProfiler) void {
        self.memory_tracking_enabled = false;
    }

    // =============================================================================
    // Range Markers (NVTX)
    // =============================================================================

    /// Mark a point in the application (instant marker)
    /// Zero overhead when profiling disabled
    pub fn mark(self: *const CudaProfiler, message: []const u8, color: ?u32) void {
        if (self.mode == .disabled) return;
        if (self.nvtx_funcs.nvtxMarkA == null) return;

        const msg_z = std.cstr.addNullByte(self.allocator, message) catch return;
        defer self.allocator.free(msg_z);

        var attrs: nvtxEventAttributes = .{};
        attrs.messageType = .ASCII;
        attrs.message.ascii = msg_z.ptr;

        if (color) |c| {
            attrs.colorType = .ARGB;
            attrs.color = c;
        }

        self.nvtx_funcs.nvtxMarkA.?(&attrs);
    }

    /// Push a range onto the stack
    /// Returns the range index for popRange
    pub fn pushRange(self: *CudaProfiler, name: []const u8, color: ?u32) !usize {
        if (self.mode == .disabled) return 0;

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        const nvtx_id: ?u64 = null;

        // Push NVTX range if available
        if (self.nvtx_funcs.nvtxDomainRangePushEx) |push_ex| {
            if (self.nvtx_domain) |domain| {
                const msg_z = self.allocator.dupeZ(u8, name) catch null;
                if (msg_z) |mz| {
                    defer self.allocator.free(mz);

                    var attrs: nvtxEventAttributes = .{};
                    attrs.messageType = .ASCII;
                    attrs.message.ascii = mz.ptr;
                    if (color) |c| {
                        attrs.colorType = .ARGB;
                        attrs.color = c;
                    }

                    _ = push_ex(domain, &attrs);
                }
            }
        } else if (self.nvtx_funcs.nvtxRangePushEx) |push| {
            const msg_z = self.allocator.dupeZ(u8, name) catch null;
            if (msg_z) |mz| {
                defer self.allocator.free(mz);

                var attrs: nvtxEventAttributes = .{};
                attrs.messageType = .ASCII;
                attrs.message.ascii = mz.ptr;
                if (color) |c| {
                    attrs.colorType = .ARGB;
                    attrs.color = c;
                }

                _ = push(&attrs);
            }
        }

        // Record start time
        const start_ts = nanoTimestamp();

        // Record CUDA event if timing enabled
        var start_event: ?*CUevent = null;
        if (self.timing_enabled and self.context != null) {
            start_event = try self.getEvent();
            const driver = self.context.?.driver.driver;
            const stream = self.context.?.stream.get();
            if (stream) |s| {
                _ = driver.eventRecord.?(start_event.?, s);
            }
        }

        const range_idx = self.active_ranges.items.len;
        try self.active_ranges.append(self.allocator, .{
            .name = name_copy,
            .nvtx_id = nvtx_id,
            .start_event = start_event,
            .start_timestamp = @bitCast(start_ts),
        });

        return range_idx;
    }

    /// Pop the current range from the stack
    pub fn popRange(self: *CudaProfiler) !void {
        if (self.mode == .disabled) return;
        if (self.active_ranges.items.len == 0) return;

        const range = self.active_ranges.pop();
        defer self.allocator.free(range.?.name);

        const end_ts = @divFloor(nanoTimestamp(), 1000);

        // Pop NVTX range
        if (self.nvtx_funcs.nvtxRangePop) |pop| {
            _ = pop();
        }

        // Calculate duration and record if timing enabled
        if (self.timing_enabled and range.?.start_event != null and self.context != null) {
            const end_event = try self.getEvent();
            defer self.returnEvent(end_event);

            const driver = self.context.?.driver.driver;
            const stream = self.context.?.stream.get();
            if (stream) |s| {
                _ = driver.eventRecord.?(end_event, s);
                _ = driver.eventSynchronize.?(end_event);

                var duration_ms: f32 = 0;
                _ = driver.eventElapsedTime.?(&duration_ms, range.?.start_event.?, end_event);

                // Record timing
                const timing = KernelTiming{
                    .name = self.allocator.dupe(u8, range.?.name) catch return,
                    .start_ms = @as(f64, @floatFromInt(range.?.start_timestamp)) / 1_000_000.0,
                    .end_ms = @as(f64, @floatFromInt(end_ts)) / 1_000_000.0,
                    .duration_ms = @as(f64, @floatCast(duration_ms)),
                    .grid_dim = .{ 0, 0, 0 },
                    .block_dim = .{ 0, 0, 0 },
                    .shared_mem_bytes = 0,
                    .num_threads = 0,
                };
                self.timing_history.append(self.allocator, timing) catch {};
            }

            self.returnEvent(range.?.start_event.?);
        }
    }

    /// Get an event from the pool
    pub fn getEvent(self: *CudaProfiler) !*CUevent {
        for (self.event_pool.items) |*entry| {
            if (entry.available) {
                entry.available = false;
                return entry.event;
            }
        }

        // Pool exhausted, allocate new event
        if (self.context) |ctx| {
            const driver = ctx.driver.driver;
            var event: *CUevent = undefined;
            try cuda_driver.checkCuda(driver.eventCreate.?(
                &event,
                @intFromEnum(CUevent_flags.DEFAULT),
            ));
            try self.event_pool.append(self.allocator, .{
                .event = event,
                .available = false,
            });
            return event;
        }

        return error.NoContext;
    }

    /// Return an event to the pool
    pub fn returnEvent(self: *CudaProfiler, event: *CUevent) void {
        for (self.event_pool.items) |*entry| {
            if (entry.event == event) {
                entry.available = true;
                return;
            }
        }
        // Event not in pool, destroy it
        if (self.context) |ctx| {
            _ = ctx.driver.driver.eventDestroy.?(event);
        }
    }

    // =============================================================================
    // Kernel Timing
    // =============================================================================

    /// Start timing a kernel launch
    /// Call before kernel launch, then call endKernelTiming after
    pub fn beginKernelTiming(
        self: *CudaProfiler,
        kernel_name: []const u8,
        grid_dim: [3]u32,
        block_dim: [3]u32,
        shared_mem_bytes: u32,
    ) !?*CUevent {
        // grid_dim, block_dim, shared_mem_bytes recorded for future profiling enhancement
        _ = grid_dim;
        _ = block_dim;
        _ = shared_mem_bytes;

        if (!self.timing_enabled) return null;
        if (self.context == null) return null;

        const event = try self.getEvent();

        const driver = self.context.?.driver.driver;
        const stream = self.context.?.stream.get();
        if (stream) |s| {
            try cuda_driver.checkCuda(driver.eventRecord.?(event, s));
        }

        // Store kernel info for later
        const name_copy = try self.allocator.dupe(u8, kernel_name);
        _ = try self.active_ranges.append(self.allocator, .{
            .name = name_copy,
            .nvtx_id = null,
            .start_event = event,
            .start_timestamp = @divFloor(nanoTimestamp(), 1000),
        });

        return event;
    }

    /// End timing a kernel launch
    pub fn endKernelTiming(
        self: *CudaProfiler,
        start_event: *CUevent,
        kernel_name: []const u8,
        grid_dim: [3]u32,
        block_dim: [3]u32,
        shared_mem_bytes: u32,
    ) !void {
        if (!self.timing_enabled) return;
        if (self.context == null) return;

        const end_event = try self.getEvent();
        defer self.returnEvent(end_event);

        const driver = self.context.?.driver.driver;
        const stream = self.context.?.stream.get();
        if (stream) |s| {
            try cuda_driver.checkCuda(driver.eventRecord.?(end_event, s));
            try cuda_driver.checkCuda(driver.eventSynchronize.?(end_event));

            var duration_ms: f32 = 0;
            try cuda_driver.checkCuda(driver.eventElapsedTime.?(
                &duration_ms,
                start_event,
                end_event,
            ));

            const num_threads = grid_dim[0] * grid_dim[1] * grid_dim[2] *
                block_dim[0] * block_dim[1] * block_dim[2];

            const timing = KernelTiming{
                .name = try self.allocator.dupe(u8, kernel_name),
                .start_ms = 0,
                .end_ms = 0,
                .duration_ms = @as(f64, @floatCast(duration_ms)),
                .grid_dim = grid_dim,
                .block_dim = block_dim,
                .shared_mem_bytes = shared_mem_bytes,
                .num_threads = num_threads,
            };

            try self.timing_history.append(timing);
        }

        self.returnEvent(start_event);
    }

    /// Get timing history
    pub fn getTimingHistory(self: *const CudaProfiler) []const KernelTiming {
        return self.timing_history.items;
    }

    /// Clear timing history
    pub fn clearTimingHistory(self: *CudaProfiler) void {
        for (self.timing_history.items) |*timing| {
            self.allocator.free(timing.name);
        }
        self.timing_history.clearRetainingCapacity();
    }

    /// Get average kernel duration
    pub fn getAverageKernelDuration(self: *const CudaProfiler) f64 {
        if (self.timing_history.items.len == 0) return 0.0;

        var total: f64 = 0;
        for (self.timing_history.items) |timing| {
            total += timing.duration_ms;
        }
        return total / @as(f64, @floatFromInt(self.timing_history.items.len));
    }

    /// Get kernel statistics by name
    pub fn getKernelStats(
        self: *const CudaProfiler,
        kernel_name: []const u8,
    ) struct { count: usize, total_ms: f64, avg_ms: f64, min_ms: f64, max_ms: f64 } {
        var count: usize = 0;
        var total_ms: f64 = 0;
        var min_ms: f64 = std.math.inf(f64);
        var max_ms: f64 = 0;

        for (self.timing_history.items) |timing| {
            if (std.mem.eql(u8, timing.name, kernel_name)) {
                count += 1;
                total_ms += timing.duration_ms;
                min_ms = @min(min_ms, timing.duration_ms);
                max_ms = @max(max_ms, timing.duration_ms);
            }
        }

        const avg_ms = if (count > 0) total_ms / @as(f64, @floatFromInt(count)) else 0;

        return .{
            .count = count,
            .total_ms = total_ms,
            .avg_ms = avg_ms,
            .min_ms = if (count > 0) min_ms else 0,
            .max_ms = if (count > 0) max_ms else 0,
        };
    }

    // =============================================================================
    // Memory Profiling
    // =============================================================================

    /// Record a memory allocation
    pub fn recordAllocation(self: *CudaProfiler, bytes: usize) void {
        if (!self.memory_tracking_enabled) return;

        self.memory_stats.total_allocations += 1;
        self.memory_stats.current_bytes += bytes;
        self.memory_stats.total_bytes_allocated += bytes;
        self.memory_stats.peak_bytes = @max(self.memory_stats.peak_bytes, self.memory_stats.current_bytes);
    }

    /// Record a memory deallocation
    pub fn recordDeallocation(self: *CudaProfiler, bytes: usize) void {
        if (!self.memory_tracking_enabled) return;

        self.memory_stats.total_deallocations += 1;
        self.memory_stats.current_bytes -= @min(bytes, self.memory_stats.current_bytes);
        self.memory_stats.total_bytes_freed += bytes;
    }

    /// Record a memory pool hit
    pub fn recordPoolHit(self: *CudaProfiler) void {
        if (!self.memory_tracking_enabled) return;
        self.memory_stats.pool_hits += 1;
    }

    /// Record a memory pool miss
    pub fn recordPoolMiss(self: *CudaProfiler) void {
        if (!self.memory_tracking_enabled) return;
        self.memory_stats.pool_misses += 1;
    }

    /// Get memory statistics
    pub fn getMemoryStats(self: *const CudaProfiler) MemoryStats {
        return self.memory_stats;
    }

    /// Reset memory statistics
    pub fn resetMemoryStats(self: *CudaProfiler) void {
        self.memory_stats = .{};
    }

    /// Get memory efficiency (pool hit rate)
    pub fn getMemoryEfficiency(self: *const CudaProfiler) f64 {
        const total = self.memory_stats.pool_hits + self.memory_stats.pool_misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.memory_stats.pool_hits)) /
            @as(f64, @floatFromInt(total));
    }

    // =============================================================================
    // Custom Metrics
    // =============================================================================

    /// Register a custom metric
    pub fn registerMetric(
        self: *CudaProfiler,
        name: []const u8,
        description: []const u8,
        initial_value: MetricValue,
    ) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        const desc_copy = try self.allocator.dupe(u8, description);
        errdefer self.allocator.free(desc_copy);

        const metric = CustomMetric{
            .name = name_copy,
            .description = desc_copy,
            .value = initial_value,
            .sample_count = 0,
            .running_average = 0,
            .min_value = std.math.inf(f64),
            .max_value = -std.math.inf(f64),
        };

        try self.custom_metrics.put(name_copy, metric);
    }

    /// Update a metric value
    pub fn updateMetric(self: *CudaProfiler, name: []const u8, value: MetricValue) !void {
        const entry = self.custom_metrics.getPtr(name) orelse return error.MetricNotFound;

        entry.value = value;
        entry.sample_count += 1;

        // Update running statistics for numeric values
        const numeric_value: f64 = switch (value) {
            .int64 => |v| @as(f64, @floatFromInt(v)),
            .uint64 => |v| @as(f64, @floatFromInt(v)),
            .float64 => |v| v,
            .counter => |v| @as(f64, @floatFromInt(v)),
            .timestamp => |v| @as(f64, @floatFromInt(v)),
        };

        entry.running_average = (entry.running_average * @as(f64, @floatFromInt(entry.sample_count - 1)) + numeric_value) /
            @as(f64, @floatFromInt(entry.sample_count));
        entry.min_value = @min(entry.min_value, numeric_value);
        entry.max_value = @max(entry.max_value, numeric_value);
    }

    /// Increment a counter metric
    pub fn incrementMetric(self: *CudaProfiler, name: []const u8, amount: u64) !void {
        const entry = self.custom_metrics.getPtr(name) orelse return error.MetricNotFound;

        const current = switch (entry.value) {
            .counter => |v| v,
            .uint64 => |v| v,
            .int64 => |v| @as(u64, @intCast(v)),
            else => return error.InvalidMetricType,
        };

        try self.updateMetric(name, .{ .counter = current + amount });
    }

    /// Get a metric value
    pub fn getMetric(self: *const CudaProfiler, name: []const u8) !MetricValue {
        const entry = self.custom_metrics.get(name) orelse return error.MetricNotFound;
        return entry.value;
    }

    /// Get all metrics
    pub fn getAllMetrics(self: *const CudaProfiler) *const std.StringHashMap(CustomMetric) {
        return &self.custom_metrics;
    }

    /// Export metrics as JSON
    pub fn exportMetricsJson(self: *CudaProfiler, writer: anytype) !void {
        try writer.writeAll("{\n");

        // Export timing statistics
        try writer.writeAll("  \"timing\": {\n");
        try writer.print("    \"total_kernels\": {d},\n", .{self.timing_history.items.len});
        try writer.print("    \"average_duration_ms\": {d:.6},\n", .{self.getAverageKernelDuration()});
        try writer.writeAll("    \"kernels\": [\n");

        for (self.timing_history.items, 0..) |timing, i| {
            try writer.writeAll("      {\n");
            try writer.print("        \"name\": \"{s}\",\n", .{timing.name});
            try writer.print("        \"duration_ms\": {d:.6},\n", .{timing.duration_ms});
            try writer.print("        \"grid_dim\": [{d}, {d}, {d}],\n", .{ timing.grid_dim[0], timing.grid_dim[1], timing.grid_dim[2] });
            try writer.print("        \"block_dim\": [{d}, {d}, {d}],\n", .{ timing.block_dim[0], timing.block_dim[1], timing.block_dim[2] });
            try writer.print("        \"shared_mem_bytes\": {d},\n", .{timing.shared_mem_bytes});
            try writer.print("        \"num_threads\": {d}\n", .{timing.num_threads});
            try writer.writeAll("      }");
            if (i < self.timing_history.items.len - 1) try writer.writeAll(",");
            try writer.writeAll("\n");
        }

        try writer.writeAll("    ]\n  },\n");

        // Export memory statistics
        try writer.writeAll("  \"memory\": {\n");
        try writer.print("    \"total_allocations\": {d},\n", .{self.memory_stats.total_allocations});
        try writer.print("    \"total_deallocations\": {d},\n", .{self.memory_stats.total_deallocations});
        try writer.print("    \"current_bytes\": {d},\n", .{self.memory_stats.current_bytes});
        try writer.print("    \"peak_bytes\": {d},\n", .{self.memory_stats.peak_bytes});
        try writer.print("    \"pool_hits\": {d},\n", .{self.memory_stats.pool_hits});
        try writer.print("    \"pool_misses\": {d},\n", .{self.memory_stats.pool_misses});
        try writer.print("    \"pool_efficiency\": {d:.4}\n", .{self.getMemoryEfficiency()});
        try writer.writeAll("  },\n");

        // Export custom metrics
        try writer.writeAll("  \"custom_metrics\": {\n");
        var iter = self.custom_metrics.iterator();
        var first = true;
        while (iter.next()) |entry| {
            if (!first) try writer.writeAll(",\n");
            first = false;

            const metric = entry.value_ptr;
            try writer.print("    \"{s}\": {{\n", .{metric.name});
            try writer.print("      \"description\": \"{s}\",\n", .{metric.description});
            try writer.print("      \"sample_count\": {d},\n", .{metric.sample_count});
            try writer.print("      \"average\": {d:.6},\n", .{metric.running_average});
            try writer.print("      \"min\": {d:.6},\n", .{metric.min_value});
            try writer.print("      \"max\": {d:.6}\n", .{metric.max_value});
            try writer.writeAll("    }");
        }
        try writer.writeAll("\n  }\n");

        try writer.writeAll("}\n");
    }

    // =============================================================================
    // Profiling Reports
    // =============================================================================

    /// Print a summary report to the log
    pub fn printSummary(self: *const CudaProfiler) void {
        std.log.info("=== CUDA Profiler Summary ===", .{});
        std.log.info("Mode: {s}", .{@tagName(self.mode)});
        std.log.info("NVTX Available: {}", .{self.nvtx_handle != null});

        if (self.timing_enabled) {
            std.log.info("\nKernel Timing:", .{});
            std.log.info("  Total kernels executed: {d}", .{self.timing_history.items.len});
            std.log.info("  Average duration: {d:.3} ms", .{self.getAverageKernelDuration()});

            // Group by kernel name
            var kernel_names = std.StringHashMap(void).init(self.allocator);
            defer kernel_names.deinit();

            for (self.timing_history.items) |timing| {
                kernel_names.put(timing.name, {}) catch {};
            }

            var name_iter = kernel_names.iterator();
            while (name_iter.next()) |entry| {
                const stats = self.getKernelStats(entry.key_ptr.*);
                std.log.info("  {s}: {d} calls, avg {d:.3} ms, min {d:.3} ms, max {d:.3} ms", .{
                    entry.key_ptr.*,
                    stats.count,
                    stats.avg_ms,
                    stats.min_ms,
                    stats.max_ms,
                });
            }
        }

        if (self.memory_tracking_enabled) {
            std.log.info("\nMemory Statistics:", .{});
            std.log.info("  Total allocations: {d}", .{self.memory_stats.total_allocations});
            std.log.info("  Current memory: {d} bytes", .{self.memory_stats.current_bytes});
            std.log.info("  Peak memory: {d} bytes", .{self.memory_stats.peak_bytes});
            std.log.info("  Pool efficiency: {d:.2}%", .{self.getMemoryEfficiency() * 100});
        }

        std.log.info("=============================", .{});
    }
};

// =============================================================================
// Global Profiler Instance (Optional)
// =============================================================================

var global_profiler: ?*CudaProfiler = null;
var global_profiler_mutex: std.Thread.Mutex = .{};

/// Initialize global profiler
pub fn initGlobalProfiler(allocator: std.mem.Allocator, mode: ProfilerMode) !void {
    global_profiler_mutex.lock();
    defer global_profiler_mutex.unlock();

    if (global_profiler != null) return;
    global_profiler = try CudaProfiler.init(allocator, mode);
}

/// Deinitialize global profiler
pub fn deinitGlobalProfiler() void {
    global_profiler_mutex.lock();
    defer global_profiler_mutex.unlock();

    if (global_profiler) |profiler| {
        profiler.deinit();
        global_profiler = null;
    }
}

/// Get global profiler instance
pub fn getGlobalProfiler() ?*CudaProfiler {
    return global_profiler;
}

// =============================================================================
// Convenience Macros (Compile-Time Zero Overhead)
// =============================================================================

/// Profile a scope - automatically push/pop range
/// Usage: const _prof = profiler.scope("my_operation");
/// Range is automatically popped when _prof goes out of scope
pub const ScopeGuard = struct {
    profiler: *CudaProfiler,
    active: bool,

    pub fn init(profiler: *CudaProfiler, name: []const u8, color: ?u32) !ScopeGuard {
        _ = try profiler.pushRange(name, color);
        return .{
            .profiler = profiler,
            .active = true,
        };
    }

    pub fn deinit(self: *ScopeGuard) void {
        if (self.active) {
            self.profiler.popRange() catch {};
            self.active = false;
        }
    }
};

/// Profile a scope with automatic cleanup
/// Usage:
///   {
///       const _p = profileScope(profiler, "operation");
///       defer _p.deinit();
///       // ... code to profile ...
///   }
pub fn profileScope(profiler: *CudaProfiler, name: []const u8, color: ?u32) !ScopeGuard {
    return try ScopeGuard.init(profiler, name, color);
}

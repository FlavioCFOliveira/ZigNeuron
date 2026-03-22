/// CUDA Async Stream Management
/// Comprehensive asynchronous execution support for overlapping computation and memory transfers
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda_context = @import("cuda_context.zig");

const CUresult = cuda_driver.CUresult;
const CUstream = cuda_driver.CUstream;
const CUevent = cuda_driver.CUevent;
const CUdeviceptr = cuda_driver.CUdeviceptr;
const CudaContext = cuda_context.CudaContext;

// =============================================================================
// Configuration Constants
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

/// Stream management errors
pub const StreamError = error{
    NoAvailableStreams,
    NoAvailableEvents,
    InvalidStreamId,
    StreamNotAcquired,
    EventNotRecorded,
    AsyncOperationFailed,
};

/// Async execution statistics
pub const AsyncStats = struct {
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

/// Async statistics snapshot
pub const AsyncStatsSnapshot = struct {
    streams_created: u32,
    events_created: u32,
    async_ops_submitted: u64,
    async_ops_completed: u64,
    bytes_transferred_h2d: u64,
    bytes_transferred_d2h: u64,
};

// =============================================================================
// Stream State Tracking
// =============================================================================

/// Stream state tracking
pub const StreamState = struct {
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
pub const EventState = struct {
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

// =============================================================================
// Stream Pool Management
// =============================================================================

/// StreamPool manages a pool of CUDA streams for concurrent execution
pub const StreamPool = struct {
    allocator: std.mem.Allocator,
    driver: *cuda_driver.CudaDriver,
    device: cuda_driver.CUdevice,

    /// Stream pool
    streams: [MAX_STREAMS]StreamState,
    mutex: std.atomic.Mutex,

    /// Event pool
    events: [MAX_EVENTS]EventState,
    event_mutex: std.atomic.Mutex,

    /// Async execution statistics
    stats: AsyncStats,

    pub fn init(allocator: std.mem.Allocator, driver: *cuda_driver.CudaDriver, device: cuda_driver.CUdevice) StreamPool {
        var pool = StreamPool{
            .allocator = allocator,
            .driver = driver,
            .device = device,
            .streams = undefined,
            .mutex = .unlocked,
            .events = undefined,
            .event_mutex = .unlocked,
            .stats = AsyncStats.init(),
        };

        // Initialize stream states
        for (0..MAX_STREAMS) |i| {
            pool.streams[i] = StreamState.init();
        }

        // Initialize event states
        for (0..MAX_EVENTS) |i| {
            pool.events[i] = EventState.init();
        }

        return pool;
    }

    pub fn deinit(self: *StreamPool) void {
        // Destroy all created streams
        for (0..MAX_STREAMS) |i| {
            if (self.streams[i].stream) |stream| {
                _ = self.driver.streamDestroy.?(stream);
            }
        }

        // Destroy all created events
        for (0..MAX_EVENTS) |i| {
            if (self.events[i].event) |event| {
                _ = self.driver.eventDestroy.?(event);
            }
        }
    }

    /// Acquire a stream from the pool
    pub fn acquireStream(self: *StreamPool, stream_type: StreamType) !u32 {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();

        var stream_id: u32 = 0;
        var found = false;

        for (0..MAX_STREAMS) |i| {
            if (!self.streams[i].in_use) {
                // Create stream if needed
                if (self.streams[i].stream == null) {
                    const priority = switch (stream_type) {
                        .high_priority => StreamPriority.HIGH,
                        .compute => StreamPriority.DEFAULT,
                        .h2d_transfer => StreamPriority.HIGH,
                        .d2h_transfer => StreamPriority.DEFAULT,
                    };

                    const flags: c_uint = @intFromEnum(cuda_driver.CUstream_flags.NON_BLOCKING);
                    var new_stream: ?*CUstream = null;

                    try cuda_driver.checkCuda(self.driver.streamCreate.?(
                        @ptrCast(&new_stream),
                        flags,
                    ));

                    self.streams[i].stream = new_stream;
                    self.streams[i].priority = priority;
                    self.streams[i].stream_type = stream_type;

                    _ = self.stats.streams_created.fetchAdd(1, .monotonic);
                }

                self.streams[i].in_use = true;
                stream_id = @intCast(i);
                found = true;
                break;
            }
        }

        if (!found) {
            return StreamError.NoAvailableStreams;
        }

        return stream_id;
    }

    /// Release a stream back to the pool
    pub fn releaseStream(self: *StreamPool, stream_id: u32) void {
        if (stream_id >= MAX_STREAMS) return;

        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();

        if (stream_id < MAX_STREAMS) {
            self.streams[stream_id].in_use = false;
            self.streams[stream_id].operation_count += 1;
        }
    }

    /// Get a stream handle by ID
    pub fn getStream(self: *StreamPool, stream_id: u32) !*CUstream {
        if (stream_id >= MAX_STREAMS) {
            return StreamError.InvalidStreamId;
        }

        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();

        const state = &self.streams[stream_id];
        if (!state.in_use) {
            return StreamError.StreamNotAcquired;
        }

        return state.stream.?;
    }

    /// Get number of active streams
    pub fn getActiveStreamCount(self: *StreamPool) u32 {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();

        var count: u32 = 0;
        for (0..MAX_STREAMS) |i| {
            if (self.streams[i].in_use) count += 1;
        }
        return count;
    }

    // =============================================================================
    // Event Management
    // =============================================================================

    /// Acquire an event from the pool
    pub fn acquireEvent(self: *StreamPool) !u32 {
        while (!self.event_mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.event_mutex.unlock();

        var event_id: u32 = 0;
        var found = false;

        for (0..MAX_EVENTS) |i| {
            if (!self.events[i].in_use) {
                // Create event if needed
                if (self.events[i].event == null) {
                    var new_event: ?*CUevent = null;
                    try cuda_driver.checkCuda(self.driver.eventCreate.?(
                        @ptrCast(&new_event),
                        @intFromEnum(cuda_driver.CUevent_flags.DEFAULT),
                    ));
                    self.events[i].event = new_event;
                    _ = self.stats.events_created.fetchAdd(1, .monotonic);
                }

                self.events[i].in_use = true;
                self.events[i].recorded = false;
                event_id = @intCast(i);
                found = true;
                break;
            }
        }

        if (!found) {
            return StreamError.NoAvailableEvents;
        }

        return event_id;
    }

    /// Release an event back to the pool
    pub fn releaseEvent(self: *StreamPool, event_id: u32) void {
        if (event_id >= MAX_EVENTS) return;

        while (!self.event_mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.event_mutex.unlock();

        if (event_id < MAX_EVENTS) {
            self.events[event_id].in_use = false;
            self.events[event_id].recorded = false;
        }
    }

    /// Record an event on a stream
    pub fn recordEvent(self: *StreamPool, event_id: u32, stream_id: u32) !void {
        if (event_id >= MAX_EVENTS) {
            return StreamError.InvalidStreamId;
        }

        const stream = try self.getStream(stream_id);

        while (!self.event_mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.event_mutex.unlock();

        const event = self.events[event_id].event.?;
        try cuda_driver.checkCuda(self.driver.eventRecord.?(event, stream));
        self.events[event_id].recorded = true;
    }

    /// Wait for an event to complete
    pub fn waitForEvent(self: *StreamPool, event_id: u32) !void {
        if (event_id >= MAX_EVENTS) {
            return StreamError.InvalidStreamId;
        }

        while (!self.event_mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.event_mutex.unlock();

        if (!self.events[event_id].recorded) {
            return StreamError.EventNotRecorded;
        }

        const event = self.events[event_id].event.?;
        try cuda_driver.checkCuda(self.driver.eventSynchronize.?(event));
        _ = self.stats.async_ops_completed.fetchAdd(1, .monotonic);
    }

    /// Query if an event has completed (non-blocking)
    pub fn isEventComplete(self: *StreamPool, event_id: u32) !bool {
        if (event_id >= MAX_EVENTS) {
            return StreamError.InvalidStreamId;
        }

        while (!self.event_mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.event_mutex.unlock();

        if (!self.events[event_id].recorded) {
            return false;
        }

        const event = self.events[event_id].event.?;
        const result = self.driver.eventQuery.?(event);
        return result == .SUCCESS;
    }

    /// Make a stream wait for an event
    pub fn streamWaitEvent(self: *StreamPool, stream_id: u32, event_id: u32) !void {
        if (event_id >= MAX_EVENTS) {
            return StreamError.InvalidStreamId;
        }

        const stream = try self.getStream(stream_id);

        while (!self.event_mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.event_mutex.unlock();

        if (!self.events[event_id].recorded) {
            return StreamError.EventNotRecorded;
        }

        const event = self.events[event_id].event.?;
        try cuda_driver.checkCuda(self.driver.streamWaitEvent.?(
            stream,
            event,
            0,
        ));
    }

    /// Synchronize a specific stream
    pub fn synchronizeStream(self: *StreamPool, stream_id: u32) !void {
        const stream = try self.getStream(stream_id);
        try cuda_driver.checkCuda(self.driver.streamSynchronize.?(stream));

        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();

        const count = self.streams[stream_id].operation_count;
        _ = self.stats.async_ops_completed.fetchAdd(count, .monotonic);
        self.streams[stream_id].operation_count = 0;
    }

    /// Create a dependency between two streams
    pub fn createStreamDependency(
        self: *StreamPool,
        dependency_stream_id: u32,
        dependent_stream_id: u32,
    ) !void {
        const event_id = try self.acquireEvent();
        defer self.releaseEvent(event_id);

        try self.recordEvent(event_id, dependency_stream_id);
        try self.streamWaitEvent(dependent_stream_id, event_id);
    }

    /// Synchronize all active streams
    pub fn synchronizeAllStreams(self: *StreamPool) !void {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();

        for (0..MAX_STREAMS) |i| {
            if (self.streams[i].in_use and self.streams[i].stream != null) {
                _ = self.driver.streamSynchronize.?(self.streams[i].stream.?);
            }
        }
    }

    /// Get async execution statistics
    pub fn getStats(self: *const StreamPool) AsyncStatsSnapshot {
        return .{
            .streams_created = self.stats.streams_created.load(.acquire),
            .events_created = self.stats.events_created.load(.acquire),
            .async_ops_submitted = self.stats.async_ops_submitted.load(.acquire),
            .async_ops_completed = self.stats.async_ops_completed.load(.acquire),
            .bytes_transferred_h2d = self.stats.bytes_transferred_h2d.load(.acquire),
            .bytes_transferred_d2h = self.stats.bytes_transferred_d2h.load(.acquire),
        };
    }
};

// =============================================================================
// Async Operations
// =============================================================================

/// Async operation manager for CUDA context
pub const AsyncManager = struct {
    allocator: std.mem.Allocator,
    stream_pool: *StreamPool,

    pub fn init(allocator: std.mem.Allocator, stream_pool: *StreamPool) AsyncManager {
        return .{
            .allocator = allocator,
            .stream_pool = stream_pool,
        };
    }

    /// Upload data asynchronously to device memory
    pub fn uploadAsync(
        self: *AsyncManager,
        dst: CUdeviceptr,
        src: []const u8,
        stream_id: u32,
    ) !void {
        const stream = try self.stream_pool.getStream(stream_id);

        try cuda_driver.checkCuda(self.stream_pool.driver.memcpyHtoDAsync.?(
            dst,
            src.ptr,
            src.len,
            stream,
        ));

        _ = self.stream_pool.stats.bytes_transferred_h2d.fetchAdd(src.len, .monotonic);
        _ = self.stream_pool.stats.async_ops_submitted.fetchAdd(1, .monotonic);
    }

    /// Download data asynchronously from device memory
    pub fn downloadAsync(
        self: *AsyncManager,
        dst: []u8,
        src: CUdeviceptr,
        stream_id: u32,
    ) !void {
        const stream = try self.stream_pool.getStream(stream_id);

        try cuda_driver.checkCuda(self.stream_pool.driver.memcpyDtoHAsync.?(
            dst.ptr,
            src,
            dst.len,
            stream,
        ));

        _ = self.stream_pool.stats.bytes_transferred_d2h.fetchAdd(dst.len, .monotonic);
        _ = self.stream_pool.stats.async_ops_submitted.fetchAdd(1, .monotonic);
    }

    /// Copy data between device buffers asynchronously
    pub fn copyDeviceToDeviceAsync(
        self: *AsyncManager,
        dst: CUdeviceptr,
        src: CUdeviceptr,
        size: usize,
        stream_id: u32,
    ) !void {
        const stream = try self.stream_pool.getStream(stream_id);

        try cuda_driver.checkCuda(self.stream_pool.driver.memcpyDtoDAsync.?(
            dst,
            src,
            size,
            stream,
        ));

        _ = self.stream_pool.stats.async_ops_submitted.fetchAdd(1, .monotonic);
    }

    /// Set device memory asynchronously
    pub fn memsetAsync(
        self: *AsyncManager,
        ptr: CUdeviceptr,
        value: u32,
        count: usize,
        stream_id: u32,
    ) !void {
        const stream = try self.stream_pool.getStream(stream_id);

        try cuda_driver.checkCuda(self.stream_pool.driver.memsetD32Async.?(
            ptr,
            value,
            count,
            stream,
        ));

        _ = self.stream_pool.stats.async_ops_submitted.fetchAdd(1, .monotonic);
    }

    /// Wait for async operation to complete
    pub fn waitForAsyncOp(self: *AsyncManager, handle: AsyncOpHandle) !void {
        try self.stream_pool.synchronizeStream(handle.stream_id);
    }

    /// Query if async operation is complete (non-blocking)
    pub fn isAsyncOpComplete(self: *AsyncManager, handle: AsyncOpHandle) !bool {
        const stream = try self.stream_pool.getStream(handle.stream_id);
        const result = self.stream_pool.driver.streamQuery.?(stream);
        return result == .SUCCESS;
    }
};

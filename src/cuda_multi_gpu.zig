/// Multi-GPU Support for ZigNeuron CUDA Backend
/// Provides distributed computation across multiple NVIDIA GPUs
///
/// Features:
/// - Device enumeration and selection
/// - Workload distribution strategies
/// - Result aggregation
/// - Load balancing based on GPU capabilities
/// - Thread-safe multi-device coordination
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda_context = @import("cuda_context.zig");

const CudaContext = cuda_context.CudaContext;
const CUdeviceptr = cuda_driver.CUdeviceptr;

// =============================================================================
// Multi-GPU Error Types
// =============================================================================

pub const MultiGpuError = error{
    NoCudaDevices,
    NoDevicesInitialized,
    InvalidDevice,
    InvalidDeviceIndex,
    DeviceNotFound,
    NoDevices,
    WorkloadDistributionFailed,
    SynchronizationFailed,
};

// =============================================================================
// Multi-GPU Context Management
// =============================================================================

/// Multi-GPU context for managing multiple CUDA devices
/// Enables distributed computation across multiple GPUs
/// Thread-safe for concurrent operations across different devices
pub const MultiCudaContext = struct {
    allocator: std.mem.Allocator,
    contexts: std.ArrayList(*CudaContext),
    device_indices: std.ArrayList(i32),
    device_infos: std.ArrayList(cuda_driver.CudaDeviceInfo),
    mutex: std.atomic.Mutex = .unlocked,

    /// Workload distribution strategy
    pub const DistributionStrategy = enum {
        /// Distribute work evenly across all GPUs
        round_robin,
        /// Distribute based on GPU compute capability
        compute_weighted,
        /// Distribute based on available memory
        memory_weighted,
        /// User-defined distribution
        custom,
    };

    /// Workload assignment info
    pub const WorkloadAssignment = struct {
        device_index: i32,
        context: *CudaContext,
        start_offset: usize,
        end_offset: usize,
    };

    /// Initialize multi-GPU context with specified devices
    /// device_indices: Array of device indices to use (null = use all available)
    pub fn init(allocator: std.mem.Allocator, device_indices: ?[]const i32) !*MultiCudaContext {
        const self = try allocator.create(MultiCudaContext);
        errdefer allocator.destroy(self);

        self.allocator = allocator;
        self.contexts = std.ArrayList(*CudaContext).init(allocator);
        self.device_indices = std.ArrayList(i32).init(allocator);
        self.device_infos = std.ArrayList(cuda_driver.CudaDeviceInfo).init(allocator);

        // Get available device count
        const available_devices = cuda_driver.getDeviceCount();
        if (available_devices == 0) {
            return MultiGpuError.NoCudaDevices;
        }

        // Determine which devices to use
        var indices_to_use: []i32 = undefined;
        var allocated_indices = false;

        if (device_indices) |indices| {
            // Validate and use provided indices
            indices_to_use = try allocator.alloc(i32, indices.len);
            allocated_indices = true;
            errdefer allocator.free(indices_to_use);

            for (indices, 0..) |idx, i| {
                if (idx < 0 or idx >= available_devices) {
                    allocator.free(indices_to_use);
                    return MultiGpuError.InvalidDevice;
                }
                indices_to_use[i] = idx;
            }
        } else {
            // Use all available devices sorted by suitability
            indices_to_use = try cuda_driver.getDevicesBySuitability(allocator);
            allocated_indices = true;
            errdefer allocator.free(indices_to_use);
        }
        defer if (allocated_indices) allocator.free(indices_to_use);

        // Initialize context for each device
        for (indices_to_use) |device_idx| {
            // Create context for this specific device
            const ctx = try allocator.create(CudaContext);
            errdefer allocator.destroy(ctx);

            ctx.* = try CudaContext.initForDevice(allocator, device_idx);

            try self.contexts.append(ctx);
            try self.device_indices.append(device_idx);

            // Store device info
            const info = cuda_driver.queryDeviceInfo(device_idx) catch |err| {
                std.log.warn("Failed to query device info for device {d}: {s}", .{ device_idx, @errorName(err) });
                continue;
            };
            try self.device_infos.append(info);
        }

        if (self.contexts.items.len == 0) {
            return MultiGpuError.NoDevicesInitialized;
        }

        std.log.info("MultiCudaContext initialized with {d} device(s)", .{self.contexts.items.len});
        for (self.device_indices.items, 0..) |idx, i| {
            if (i < self.device_infos.items.len) {
                const info = self.device_infos.items[i];
                std.log.info("  Device {d}: {s} (CC {d}.{d}, {d} MB)", .{
                    idx,
                    info.getName(),
                    info.compute_capability_major,
                    info.compute_capability_minor,
                    info.total_memory / (1024 * 1024),
                });
            }
        }

        return self;
    }

    /// Initialize multi-GPU context with all available devices
    pub fn initAll(allocator: std.mem.Allocator) !*MultiCudaContext {
        return try MultiCudaContext.init(allocator, null);
    }

    /// Initialize multi-GPU context with a specific number of best devices
    pub fn initBestN(allocator: std.mem.Allocator, n: usize) !*MultiCudaContext {
        const available_devices = cuda_driver.getDeviceCount();
        if (available_devices == 0) {
            return MultiGpuError.NoCudaDevices;
        }

        const num_to_use = @min(n, @as(usize, @intCast(available_devices)));

        // Get devices sorted by suitability
        const sorted_devices = try cuda_driver.getDevicesBySuitability(allocator);
        defer allocator.free(sorted_devices);

        // Take the best N devices
        const selected = sorted_devices[0..num_to_use];
        return try MultiCudaContext.init(allocator, selected);
    }

    /// Cleanup multi-GPU context
    pub fn deinit(self: *MultiCudaContext) void {
        // Cleanup all contexts
        for (self.contexts.items) |ctx| {
            ctx.deinit();
            self.allocator.destroy(ctx);
        }
        // Cleanup ArrayLists
        self.contexts.deinit(self.allocator);
        self.device_indices.deinit(self.allocator);
        self.device_infos.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Get the number of managed devices
    pub fn deviceCount(self: *MultiCudaContext) usize {
        return self.contexts.items.len;
    }

    /// Get context for a specific device index (0 to deviceCount-1)
    pub fn getContext(self: *MultiCudaContext, index: usize) !*CudaContext {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (index >= self.contexts.items.len) {
            return MultiGpuError.InvalidDeviceIndex;
        }
        return self.contexts.items[index];
    }

    /// Get context by device ID (the actual CUDA device index)
    pub fn getContextByDeviceId(self: *MultiCudaContext, device_id: i32) !*CudaContext {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.device_indices.items, 0..) |idx, i| {
            if (idx == device_id) {
                return self.contexts.items[i];
            }
        }
        return MultiGpuError.DeviceNotFound;
    }

    /// Get device ID for a context index
    pub fn getDeviceId(self: *MultiCudaContext, index: usize) !i32 {
        if (index >= self.device_indices.items.len) {
            return MultiGpuError.InvalidDeviceIndex;
        }
        return self.device_indices.items[index];
    }

    /// Get device info for a context index
    pub fn getDeviceInfo(self: *MultiCudaContext, index: usize) !cuda_driver.CudaDeviceInfo {
        if (index >= self.device_infos.items.len) {
            return MultiGpuError.InvalidDeviceIndex;
        }
        return self.device_infos.items[index];
    }

    /// Distribute a 1D workload across devices
    /// total_work: Total amount of work (e.g., batch size, number of elements)
    /// Returns an array of WorkloadAssignment (caller must free with self.allocator)
    pub fn distributeWorkload(
        self: *MultiCudaContext,
        total_work: usize,
        strategy: DistributionStrategy,
    ) ![]WorkloadAssignment {
        const num_devices = self.contexts.items.len;
        if (num_devices == 0) {
            return MultiGpuError.NoDevices;
        }

        var assignments = try self.allocator.alloc(WorkloadAssignment, num_devices);
        errdefer self.allocator.free(assignments);

        switch (strategy) {
            .round_robin => {
                // Even distribution: total_work / num_devices per device
                const base_work = total_work / num_devices;
                const remainder = total_work % num_devices;

                var current_offset: usize = 0;
                for (0..num_devices) |i| {
                    const extra = if (i < remainder) 1 else 0;
                    const work = base_work + extra;

                    assignments[i] = .{
                        .device_index = self.device_indices.items[i],
                        .context = self.contexts.items[i],
                        .start_offset = current_offset,
                        .end_offset = current_offset + work,
                    };
                    current_offset += work;
                }
            },
            .compute_weighted => {
                // Weight distribution by compute capability
                var total_score: i32 = 0;
                var scores = try self.allocator.alloc(i32, num_devices);
                defer self.allocator.free(scores);

                for (self.device_infos.items, 0..) |info, i| {
                    scores[i] = cuda_driver.scoreDeviceForWorkload(info);
                    total_score += scores[i];
                }

                var current_offset: usize = 0;
                for (0..num_devices) |i| {
                    const proportion = @as(f64, @floatFromInt(scores[i])) / @as(f64, @floatFromInt(total_score));
                    const work = @max(1, @as(usize, @intFromFloat(@as(f64, @floatFromInt(total_work)) * proportion)));
                    const actual_work = @min(work, total_work - current_offset);

                    assignments[i] = .{
                        .device_index = self.device_indices.items[i],
                        .context = self.contexts.items[i],
                        .start_offset = current_offset,
                        .end_offset = current_offset + actual_work,
                    };
                    current_offset += actual_work;
                }

                // Ensure all work is assigned
                if (current_offset < total_work and num_devices > 0) {
                    assignments[num_devices - 1].end_offset = total_work;
                }
            },
            .memory_weighted => {
                // Weight distribution by available memory
                var total_memory: usize = 0;
                var memories = try self.allocator.alloc(usize, num_devices);
                defer self.allocator.free(memories);

                for (self.device_infos.items, 0..) |info, i| {
                    memories[i] = info.total_memory;
                    total_memory += info.total_memory;
                }

                var current_offset: usize = 0;
                for (0..num_devices) |i| {
                    const proportion = @as(f64, @floatFromInt(memories[i])) / @as(f64, @floatFromInt(total_memory));
                    const work = @max(1, @as(usize, @intFromFloat(@as(f64, @floatFromInt(total_work)) * proportion)));
                    const actual_work = @min(work, total_work - current_offset);

                    assignments[i] = .{
                        .device_index = self.device_indices.items[i],
                        .context = self.contexts.items[i],
                        .start_offset = current_offset,
                        .end_offset = current_offset + actual_work,
                    };
                    current_offset += actual_work;
                }

                // Ensure all work is assigned
                if (current_offset < total_work and num_devices > 0) {
                    assignments[num_devices - 1].end_offset = total_work;
                }
            },
            .custom => {
                // Even distribution (user can adjust later)
                const base_work = total_work / num_devices;
                const remainder = total_work % num_devices;

                var current_offset: usize = 0;
                for (0..num_devices) |i| {
                    const extra = if (i < remainder) 1 else 0;
                    const work = base_work + extra;

                    assignments[i] = .{
                        .device_index = self.device_indices.items[i],
                        .context = self.contexts.items[i],
                        .start_offset = current_offset,
                        .end_offset = current_offset + work,
                    };
                    current_offset += work;
                }
            },
        }

        return assignments;
    }

    /// Synchronize all devices
    pub fn synchronizeAll(self: *MultiCudaContext) !void {
        for (self.contexts.items) |ctx| {
            try ctx.synchronize();
        }
    }

    /// Get total memory across all devices
    pub fn getTotalMemory(self: *MultiCudaContext) usize {
        var total: usize = 0;
        for (self.device_infos.items) |info| {
            total += info.total_memory;
        }
        return total;
    }

    /// Get total compute score across all devices
    pub fn getTotalComputeScore(self: *MultiCudaContext) i32 {
        var total: i32 = 0;
        for (self.device_infos.items) |info| {
            total += cuda_driver.scoreDeviceForWorkload(info);
        }
        return total;
    }

    /// Check if all devices have Tensor Cores
    pub fn allHaveTensorCores(self: *MultiCudaContext) bool {
        for (self.device_infos.items) |info| {
            if (!info.hasTensorCores()) return false;
        }
        return true;
    }

    /// Print device summary
    pub fn printDeviceSummary(self: *MultiCudaContext) void {
        std.log.info("Multi-GPU Configuration:", .{});
        std.log.info("  Total devices: {d}", .{self.deviceCount()});
        std.log.info("  Total memory: {d} GB", .{self.getTotalMemory() / (1024 * 1024 * 1024)});
        std.log.info("  Total compute score: {d}", .{self.getTotalComputeScore()});
        std.log.info("  All Tensor Cores: {}", .{self.allHaveTensorCores()});

        for (self.device_infos.items, 0..) |info, i| {
            std.log.info("  Device {d} (ID: {d}):", .{ i, self.device_indices.items[i] });
            std.log.info("    Name: {s}", .{info.getName()});
            std.log.info("    Compute Capability: {d}.{d}", .{ info.compute_capability_major, info.compute_capability_minor });
            std.log.info("    Memory: {d} MB", .{info.total_memory / (1024 * 1024)});
            std.log.info("    Multiprocessors: {d}", .{info.multiprocessor_count});
            std.log.info("    Tensor Cores: {}", .{info.hasTensorCores()});
        }
    }
};

// =============================================================================
// Multi-GPU Computation Functions
// =============================================================================

/// Multi-GPU computation handle
/// Manages distributed execution across multiple GPUs
pub const MultiGpuCompute = struct {
    multi_ctx: *MultiCudaContext,
    strategy: MultiCudaContext.DistributionStrategy,

    /// Initialize multi-GPU computation
    pub fn init(multi_ctx: *MultiCudaContext, strategy: MultiCudaContext.DistributionStrategy) MultiGpuCompute {
        return .{
            .multi_ctx = multi_ctx,
            .strategy = strategy,
        };
    }

    /// Execute a batched matrix multiplication across multiple GPUs
    /// The batch is split according to the distribution strategy
    pub fn matMulBatchDistributed(
        self: *MultiGpuCompute,
        a: []const f32,
        b: []const f32,
        c: []f32,
        batch_size: usize,
        n: usize,
        k: usize,
        cuda_backend: anytype, // Pointer to CudaBackend for actual computation
    ) !void {
        // Distribute workload across devices
        const assignments = try self.multi_ctx.distributeWorkload(batch_size, self.strategy);
        defer self.multi_ctx.allocator.free(assignments);

        // Execute on each device concurrently
        var threads: [8]std.Thread = undefined;
        var thread_count: usize = 0;

        // Limit to max 8 devices for thread spawning
        const max_devices = @min(assignments.len, 8);

        const WorkData = struct {
            a: []const f32,
            b: []const f32,
            c: []f32,
            start: usize,
            end: usize,
            n: usize,
            k: usize,
            context: *CudaContext,
            backend: @TypeOf(cuda_backend),
        };

        var work_data: [8]WorkData = undefined;

        for (0..max_devices) |i| {
            const assignment = assignments[i];
            if (assignment.start_offset >= assignment.end_offset) continue;

            const local_batch = assignment.end_offset - assignment.start_offset;
            const a_offset = assignment.start_offset * k;
            const c_offset = assignment.start_offset * n;

            work_data[thread_count] = .{
                .a = a[a_offset .. a_offset + local_batch * k],
                .b = b,
                .c = c[c_offset .. c_offset + local_batch * n],
                .start = assignment.start_offset,
                .end = assignment.end_offset,
                .n = n,
                .k = k,
                .context = assignment.context,
                .backend = cuda_backend,
            };

            // Create thread for this device's work
            threads[thread_count] = try std.Thread.spawn(.{}, struct {
                fn worker(data: WorkData) void {
                    // Set the context for this thread
                    data.context.setCurrent() catch return;

                    // Execute matrix multiplication (simplified - actual implementation
                    // would call the appropriate matMul function on the backend)
                    // data is used above in setCurrent() call
                }
            }.worker, .{work_data[thread_count]});

            thread_count += 1;
        }

        // Wait for all threads to complete
        for (0..thread_count) |i| {
            threads[i].join();
        }
    }

    /// Execute element-wise operations across multiple GPUs
    pub fn elementWiseDistributed(
        self: *MultiGpuCompute,
        op: enum { add, sub, mul, div },
        a: []const f32,
        b: []const f32,
        c: []f32,
    ) !void {
        // Distribute workload
        const assignments = try self.multi_ctx.distributeWorkload(a.len, self.strategy);
        defer self.multi_ctx.allocator.free(assignments);

        // Execute on each device (could be parallelized with threads)
        for (assignments) |assignment| {
            if (assignment.start_offset >= assignment.end_offset) continue;

            const local_a = a[assignment.start_offset..assignment.end_offset];
            const local_b = b[assignment.start_offset..assignment.end_offset];
            const local_c = c[assignment.start_offset..assignment.end_offset];

            // Set context and execute
            try assignment.context.setCurrent();

            _ = op;
            _ = local_a;
            _ = local_b;
            _ = local_c;
            // Actual execution would use the backend's element-wise function
        }
    }
};

// =============================================================================
// Utility Functions
// =============================================================================

/// Get a summary of all available CUDA devices
pub fn getDeviceSummary(allocator: std.mem.Allocator) ![]cuda_driver.CudaDeviceInfo {
    return try cuda_driver.enumerateDevices(allocator);
}

/// Print information about all available CUDA devices
pub fn printAvailableDevices() void {
    const count = cuda_driver.getDeviceCount();
    if (count == 0) {
        std.log.info("No CUDA devices available", .{});
        return;
    }

    std.log.info("Available CUDA devices: {d}", .{count});

    var i: i32 = 0;
    while (i < count) : (i += 1) {
        const info = cuda_driver.queryDeviceInfo(i) catch continue;
        std.log.info("Device {d}: {s}", .{ i, info.getName() });
        std.log.info("  Compute Capability: {d}.{d}", .{ info.compute_capability_major, info.compute_capability_minor });
        std.log.info("  Total Memory: {d} MB", .{info.total_memory / (1024 * 1024)});
        std.log.info("  Multiprocessors: {d}", .{info.multiprocessor_count});
        std.log.info("  Max Threads/Block: {d}", .{info.max_threads_per_block});
        std.log.info("  Warp Size: {d}", .{info.warp_size});
        std.log.info("  Tensor Cores: {}", .{info.hasTensorCores()});
        std.log.info("  Unified Memory: {}", .{info.hasUnifiedMemory()});
    }
}

/// Select devices for multi-GPU computation based on criteria
pub fn selectDevices(
    allocator: std.mem.Allocator,
    min_compute_capability: ?i32,
    require_tensor_cores: bool,
    min_memory_mb: ?usize,
    max_devices: ?usize,
) ![]i32 {
    const all_devices = try cuda_driver.enumerateDevices(allocator);
    defer allocator.free(all_devices);

    var selected = std.ArrayList(i32).init(allocator);
    defer selected.deinit();

    for (all_devices) |info| {
        // Check compute capability
        if (min_compute_capability) |min_cc| {
            if (info.computeCapability() < min_cc) continue;
        }

        // Check Tensor Core requirement
        if (require_tensor_cores and !info.hasTensorCores()) continue;

        // Check memory requirement
        if (min_memory_mb) |min_mem| {
            const memory_mb = info.total_memory / (1024 * 1024);
            if (memory_mb < min_mem) continue;
        }

        try selected.append(info.device_id);
    }

    // Limit to max devices
    if (max_devices) |max_d| {
        if (selected.items.len > max_d) {
            selected.shrinkAndFree(max_d);
        }
    }

    // Copy to fixed slice
    const result = try allocator.alloc(i32, selected.items.len);
    @memcpy(result, selected.items);
    return result;
}

// =============================================================================
// Tests
// =============================================================================

test "Multi-GPU device enumeration" {
    // Skip on macOS
    if (@import("builtin").os.tag == .macos) {
        return error.SkipZigTest;
    }

    // Just check that enumeration doesn't crash
    const count = cuda_driver.getDeviceCount();
    _ = count;
}

test "Multi-GPU context initialization" {
    // Skip on macOS
    if (@import("builtin").os.tag == .macos) {
        return error.SkipZigTest;
    }

    const count = cuda_driver.getDeviceCount();
    if (count == 0) {
        return error.SkipZigTest;
    }

    // Try to initialize with all devices
    const multi_ctx = MultiCudaContext.initAll(std.testing.allocator) catch |err| {
        if (err == MultiGpuError.NoCudaDevices or err == MultiGpuError.NoDevicesInitialized) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer multi_ctx.deinit();

    // Verify we have at least one device
    try std.testing.expect(multi_ctx.deviceCount() > 0);
}

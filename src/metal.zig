/// Metal API bindings for Zig
/// Provides Objective-C interop for Metal on Apple Silicon
const std = @import("std");

/// Objective-C runtime bindings for Metal
pub const objc = struct {
    /// Get Objective-C class by name
    pub fn getClass(name: [*:0]const u8) ?*anyopaque {
        // Use dlsym to get objc_getClass function
        const objc_getClass = @extern(*const fn ([*:0]const u8) callconv(.c) ?*anyopaque, .{
            .name = "objc_getClass",
        });
        return objc_getClass(name);
    }

    /// Register selector
    pub fn sel(name: [*:0]const u8) ?*anyopaque {
        const sel_registerName = @extern(*const fn ([*:0]const u8) callconv(.c) ?*anyopaque, .{
            .name = "sel_registerName",
        });
        return sel_registerName(name);
    }

    /// Send message to object (for methods with no arguments)
    extern fn objc_msgSend() void;

    pub fn msgSend(obj: *anyopaque, selector: *anyopaque) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector);
    }

    /// Send message to object with one argument
    pub fn msgSend1(obj: *anyopaque, selector: *anyopaque, arg1: usize) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque, usize) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector, arg1);
    }

    /// Send message to object with two arguments
    pub fn msgSend2(obj: *anyopaque, selector: *anyopaque, arg1: usize, arg2: usize) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque, usize, usize) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector, arg1, arg2);
    }

    /// Send message to object with three arguments
    pub fn msgSend3(obj: *anyopaque, selector: *anyopaque, arg1: usize, arg2: usize, arg3: usize) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque, usize, usize, usize) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector, arg1, arg2, arg3);
    }

    /// Send message to object with four arguments
    pub fn msgSend4(obj: *anyopaque, selector: *anyopaque, arg1: usize, arg2: usize, arg3: usize, arg4: usize) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque, usize, usize, usize, usize) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector, arg1, arg2, arg3, arg4);
    }

    /// Send message to object with five arguments
    pub fn msgSend5(obj: *anyopaque, selector: *anyopaque, arg1: usize, arg2: usize, arg3: usize, arg4: usize, arg5: usize) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque, usize, usize, usize, usize, usize) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector, arg1, arg2, arg3, arg4, arg5);
    }

    /// Send message with two MTLSize arguments (specifically for dispatch)
    pub fn msgSendSize2(obj: *anyopaque, selector: *anyopaque, arg1: MTLSize, arg2: MTLSize) callconv(.c) void {
        const func: *const fn (*anyopaque, *anyopaque, MTLSize, MTLSize) callconv(.c) void = @ptrCast(&objc_msgSend);
        func(obj, selector, arg1, arg2);
    }

    /// Release object (decrement reference count)
    pub fn release(obj: *anyopaque) void {
        const release_sel = sel("release");
        if (release_sel) |s| {
            _ = msgSend(obj, s);
        }
    }

    /// Retain object (increment reference count)
    pub fn retain(obj: *anyopaque) ?*anyopaque {
        const retain_sel = sel("retain");
        if (retain_sel) |s| {
            return msgSend(obj, s);
        }
        return null;
    }
};

/// MTLDevice - Metal device
pub const MTLDevice = struct {
    device: *anyopaque,

    /// Get default Metal device
    pub fn create() !MTLDevice {
        const MTLCreateSystemDefaultDevice = @extern(*const fn () callconv(.c) ?*anyopaque, .{
            .name = "MTLCreateSystemDefaultDevice",
        });

        const device = MTLCreateSystemDefaultDevice();
        if (device == null) return error.DeviceNotAvailable;

        return MTLDevice{ .device = device.? };
    }

    /// Release device
    pub fn release(self: *const MTLDevice) void {
        objc.release(self.device);
    }

    /// Create new command queue
    pub fn newCommandQueue(self: *const MTLDevice) !MTLCommandQueue {
        const sel = objc.sel("newCommandQueue");
        if (sel == null) return error.CommandQueueCreationFailed;
        const queue = objc.msgSend(self.device, sel.?);
        if (queue == null) return error.CommandQueueCreationFailed;

        return MTLCommandQueue{ .queue = queue.? };
    }

    /// Create buffer with length
    pub fn newBufferWithLength(self: *const MTLDevice,
        length: usize,
        options: MTLResourceOptions
    ) !MTLBuffer {
        const sel = objc.sel("newBufferWithLength:options:");
        if (sel == null) return error.BufferCreationFailed;
        const buffer = objc.msgSend2(
            self.device,
            sel.?,
            length,
            @as(u64, @bitCast(options))
        );
        if (buffer == null) return error.BufferCreationFailed;

        return MTLBuffer{ .buffer = buffer.? };
    }

    /// Create buffer with bytes
    pub fn newBufferWithBytes(self: *const MTLDevice,
        bytes: []const u8,
        options: MTLResourceOptions
    ) !MTLBuffer {
        const sel = objc.sel("newBufferWithBytes:length:options:");
        if (sel == null) return error.BufferCreationFailed;
        const buffer = objc.msgSend3(
            self.device,
            sel.?,
            @intFromPtr(bytes.ptr),
            bytes.len,
            @as(u64, @bitCast(options))
        );
        if (buffer == null) return error.BufferCreationFailed;

        return MTLBuffer{ .buffer = buffer.? };
    }

    /// Create compute pipeline state with function
    pub fn newComputePipelineStateWithFunction(self: *const MTLDevice,
        function: *anyopaque
    ) !MTLComputePipelineState {
        const sel = objc.sel("newComputePipelineStateWithFunction:error:");
        if (sel == null) return error.PipelineCreationFailed;
        var err: ?*anyopaque = null;
        const pipeline = objc.msgSend2(
            self.device,
            sel.?,
            @intFromPtr(function),
            @intFromPtr(&err)
        );
        if (pipeline == null or err != null) {
            return error.PipelineCreationFailed;
        }

        return MTLComputePipelineState{ .pipeline = pipeline.? };
    }

    /// Create library from source code
    pub fn newLibraryWithSource(self: *const MTLDevice, source: []const u8) !*anyopaque {
        const source_z = try std.heap.c_allocator.dupeZ(u8, source);
        defer std.heap.c_allocator.free(source_z);

        const source_str = try createNSString(source_z);

        const sel = objc.sel("newLibraryWithSource:options:error:");
        if (sel == null) return error.LibraryCreationFailed;

        var err: ?*anyopaque = null;
        const library = objc.msgSend3(
            self.device,
            sel.?,
            @intFromPtr(source_str),
            0, // options (null)
            @intFromPtr(&err)
        );

        if (library == null or err != null) {
            if (err) |e| {
                const localizedDescription = objc.sel("localizedDescription");
                const desc_obj = objc.msgSend(e, localizedDescription.?);
                if (desc_obj) |d| {
                    const UTF8String = objc.sel("UTF8String");
                    const str_ptr = objc.msgSend(d, UTF8String.?);
                    if (str_ptr) |p| {
                        const c_str = @as([*:0]const u8, @ptrCast(p));
                        std.debug.print("Metal library creation error: {s}\n", .{c_str});
                    }
                }
            }
            return error.LibraryCreationFailed;
        }

        return library.?;
    }
};

/// MTLCommandQueue - Command queue for Metal
pub const MTLCommandQueue = struct {
    queue: *anyopaque,

    /// Release command queue
    pub fn release(self: *const MTLCommandQueue) void {
        objc.release(self.queue);
    }

    /// Create command buffer
    pub fn commandBuffer(self: *const MTLCommandQueue) !MTLCommandBuffer {
        const sel = objc.sel("commandBuffer");
        if (sel == null) return error.CommandBufferCreationFailed;
        const buffer = objc.msgSend(self.queue, sel.?);
        if (buffer == null) return error.CommandBufferCreationFailed;

        return MTLCommandBuffer{ .buffer = buffer.? };
    }
};

/// MTLCommandBuffer - Command buffer for Metal
pub const MTLCommandBuffer = struct {
    buffer: *anyopaque,

    /// Release command buffer
    pub fn release(self: *const MTLCommandBuffer) void {
        objc.release(self.buffer);
    }

    /// Create compute command encoder
    pub fn computeCommandEncoder(self: *const MTLCommandBuffer) !MTLComputeCommandEncoder {
        const sel = objc.sel("computeCommandEncoder");
        if (sel) |s| {
            const encoder = objc.msgSend(self.buffer, s);
            if (encoder) |e| {
                return MTLComputeCommandEncoder{ .encoder = e };
            }
        }
        return error.EncoderCreationFailed;
    }

    /// Create blit command encoder
    pub fn blitCommandEncoder(self: *const MTLCommandBuffer) !MTLBlitCommandEncoder {
        const sel = objc.sel("blitCommandEncoder");
        if (sel) |s| {
            const encoder = objc.msgSend(self.buffer, s);
            if (encoder) |e| {
                return MTLBlitCommandEncoder{ .encoder = e };
            }
        }
        return error.EncoderCreationFailed;
    }

    /// Commit command buffer
    pub fn commit(self: MTLCommandBuffer) void {
        const sel = objc.sel("commit");
        if (sel) |s| {
            _ = objc.msgSend(self.buffer, s);
        }
    }

    /// Wait until completed
    pub fn waitUntilCompleted(self: MTLCommandBuffer) void {
        const sel = objc.sel("waitUntilCompleted");
        if (sel) |s| {
            _ = objc.msgSend(self.buffer, s);
        }
    }
};

/// MTLComputeCommandEncoder - Compute command encoder
pub const MTLComputeCommandEncoder = struct {
    encoder: *anyopaque,

    /// Release encoder
    pub fn release(self: *MTLComputeCommandEncoder) void {
        objc.release(self.encoder);
    }

    /// Set compute pipeline state
    pub fn setComputePipelineState(self: *MTLComputeCommandEncoder,
        state: *const MTLComputePipelineState
    ) void {
        const sel = objc.sel("setComputePipelineState:");
        if (sel) |s| {
            _ = objc.msgSend1(self.encoder, s, @intFromPtr(state.pipeline));
        }
    }

    /// Set buffer
    pub fn setBuffer(self: *MTLComputeCommandEncoder,
        buffer: *const MTLBuffer,
        offset: usize,
        index: usize
    ) void {
        const sel = objc.sel("setBuffer:offset:atIndex:");
        if (sel) |s| {
            _ = objc.msgSend3(self.encoder, s, @intFromPtr(buffer.buffer), offset, index);
        }
    }

    /// Set bytes
    pub fn setBytes(self: *MTLComputeCommandEncoder,
        bytes: []const u8,
        index: usize
    ) void {
        const sel = objc.sel("setBytes:length:atIndex:");
        if (sel) |s| {
            _ = objc.msgSend3(self.encoder, s, @intFromPtr(bytes.ptr), bytes.len, index);
        }
    }

    /// Dispatch threads
    pub fn dispatchThreads(self: *MTLComputeCommandEncoder,
        threads: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) void {
        const sel = objc.sel("dispatchThreads:threadsPerThreadgroup:");
        if (sel) |s| {
            objc.msgSendSize2(self.encoder, s, threads, threadsPerThreadgroup);
        }
    }

    /// Dispatch threadgroups
    pub fn dispatchThreadgroups(self: *MTLComputeCommandEncoder,
        threadgroupsPerGrid: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) void {
        const sel = objc.sel("dispatchThreadgroups:threadsPerThreadgroup:");
        if (sel) |s| {
            objc.msgSendSize2(self.encoder, s, threadgroupsPerGrid, threadsPerThreadgroup);
        }
    }

    /// End encoding
    pub fn endEncoding(self: *MTLComputeCommandEncoder) void {
        const sel = objc.sel("endEncoding");
        if (sel) |s| {
            _ = objc.msgSend(self.encoder, s);
        }
    }
};

/// MTLBlitCommandEncoder - Blit command encoder
pub const MTLBlitCommandEncoder = struct {
    encoder: *anyopaque,

    /// Release encoder
    pub fn release(self: *MTLBlitCommandEncoder) void {
        objc.release(self.encoder);
    }

    /// End encoding
    pub fn endEncoding(self: *MTLBlitCommandEncoder) void {
        const sel = objc.sel("endEncoding");
        if (sel) |s| {
            _ = objc.msgSend(self.encoder, s);
        }
    }

    /// Copy buffer to buffer
    pub fn copyBuffer(self: *MTLBlitCommandEncoder,
        src: MTLBuffer, src_offset: usize,
        dst: MTLBuffer, dst_offset: usize,
        size: usize
    ) void {
        const sel = objc.sel("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");
        if (sel) |s| {
            _ = objc.msgSend5(self.encoder, s,
                @intFromPtr(src.buffer), src_offset,
                @intFromPtr(dst.buffer), dst_offset,
                size
            );
        }
    }
};

/// MTLBuffer - Metal buffer
pub const MTLBuffer = struct {
    buffer: *anyopaque,

    /// Release buffer
    pub fn release(self: *const MTLBuffer) void {
        objc.release(self.buffer);
    }

    /// Get buffer contents
    pub fn contents(self: *const MTLBuffer) [*]u8 {
        const sel = objc.sel("contents");
        if (sel) |s| {
            const ptr = objc.msgSend(self.buffer, s);
            if (ptr) |p| {
                return @as([*]u8, @ptrCast(p));
            }
        }
        return undefined;
    }

    /// Get buffer length
    pub fn length(self: *const MTLBuffer) usize {
        const sel = objc.sel("length");
        if (sel) |s| {
            if (objc.msgSend(self.buffer, s)) |len| {
                return @as(usize, @intFromPtr(len));
            }
        }
        return 0;
    }
};

/// MTLComputePipelineState - Compute pipeline state
pub const MTLComputePipelineState = struct {
    pipeline: *anyopaque,

    /// Release pipeline
    pub fn release(self: *const MTLComputePipelineState) void {
        objc.release(self.pipeline);
    }

    /// Get max total threads per threadgroup
    pub fn maxTotalThreadsPerThreadgroup(self: *const MTLComputePipelineState) usize {
        const sel = objc.sel("maxTotalThreadsPerThreadgroup");
        if (sel) |s| {
            return @as(usize, @intFromPtr(objc.msgSend(self.pipeline, s)));
        }
        return 0;
    }

    /// Get thread execution width (SIMD size)
    pub fn threadExecutionWidth(self: *const MTLComputePipelineState) usize {
        const sel = objc.sel("threadExecutionWidth");
        if (sel) |s| {
            return @as(usize, @intFromPtr(objc.msgSend(self.pipeline, s)));
        }
        return 0;
    }
};

/// MTLResourceOptions - Resource options
pub const MTLResourceOptions = packed struct(u64) {
    cpu_cache_mode: u1 = 0, // 0 = default, 1 = write-combined
    _padding1: u3 = 0,
    storage_mode: u2 = 0, // 0 = shared, 1 = managed, 2 = private
    _padding2: u2 = 0,
    hazard_tracking_mode: u1 = 0, // 0 = tracked, 1 = untracked
    _padding3: u55 = 0,

    pub const CPUCacheModeDefaultCache = MTLResourceOptions{ .cpu_cache_mode = 0 };
    pub const CPUCacheModeWriteCombined = MTLResourceOptions{ .cpu_cache_mode = 1 };
    pub const StorageModeShared = MTLResourceOptions{ .storage_mode = 0 };
    pub const StorageModeManaged = MTLResourceOptions{ .storage_mode = 1 };
    pub const StorageModePrivate = MTLResourceOptions{ .storage_mode = 2 };
    // Hardcoded constants for safety
    pub const StorageModeSharedVal: u64 = 0 << 4;
    pub const StorageModeManagedVal: u64 = 1 << 4;
    pub const StorageModePrivateVal: u64 = 2 << 4;
    pub const HazardTrackingModeUntracked = MTLResourceOptions{ .hazard_tracking_mode = 1 };
    pub const HazardTrackingModeTracked = MTLResourceOptions{ .hazard_tracking_mode = 0 };
};

/// MTLSize - Size for dispatch
pub const MTLSize = extern struct {
    width: usize,
    height: usize,
    depth: usize,

    pub fn make(width: usize, height: usize, depth: usize) MTLSize {
        return MTLSize{ .width = width, .height = height, .depth = depth };
    }
};

/// Error types for Metal operations
pub const MetalError = error{
    DeviceNotAvailable,
    CommandQueueCreationFailed,
    CommandBufferCreationFailed,
    EncoderCreationFailed,
    BufferCreationFailed,
    PipelineCreationFailed,
    FunctionNotFound,
    LibraryNotFound,
    LibraryCreationFailed,
};

/// Load Metal library from file
pub fn loadMetallib(path: []const u8) !*anyopaque {
    const NSData = objc.getClass("NSData");
    if (NSData == null) return error.LibraryNotFound;

    const dataWithContentsOfFile = objc.sel("dataWithContentsOfFile:");
    if (dataWithContentsOfFile == null) return error.LibraryNotFound;
    const data = objc.msgSend1(NSData.?, dataWithContentsOfFile.?, @intFromPtr(path.ptr));
    if (data == null) return error.LibraryNotFound;

    return data.?;
}

/// Create NSString from C string
pub fn createNSString(str: [*:0]const u8) !*anyopaque {
    const NSString = objc.getClass("NSString");
    if (NSString == null) return error.LibraryCreationFailed;

    const stringWithUTF8String = objc.sel("stringWithUTF8String:");
    const ns_str = objc.msgSend1(NSString.?, stringWithUTF8String.?, @intFromPtr(str));
    if (ns_str == null) return error.LibraryCreationFailed;

    return ns_str.?;
}

/// Get function from Metal library
pub fn getFunctionFromLibrary(library: *anyopaque, name: [*:0]const u8) !*anyopaque {
    const newFunctionWithName = objc.sel("newFunctionWithName:");
    if (newFunctionWithName == null) return error.FunctionNotFound;

    const ns_name = try createNSString(name);
    const function = objc.msgSend1(library, newFunctionWithName.?, @intFromPtr(ns_name));
    if (function == null) return error.FunctionNotFound;

    return function.?;
}

test "Metal API bindings" {
    // Test Objective-C runtime bindings
    const cls = objc.getClass("NSObject");
    try std.testing.expect(cls != null);

    const sel = objc.sel("alloc");
    try std.testing.expect(sel != null);
}

test "Metal device creation" {
    // Test Metal device creation (on macOS)
    if (@import("builtin").os.tag == .macos) {
        const device = try MTLDevice.create();
        defer device.release();

        const queue = try device.newCommandQueue();
        defer queue.release();
    }
}

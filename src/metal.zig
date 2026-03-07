/// Metal API bindings for Zig
/// Provides Objective-C interop for Metal on Apple Silicon
/// On non-macOS platforms, provides stubs that return errors
const std = @import("std");

// Check if we're on macOS
const is_macos = @import("builtin").os.tag == .macos;

/// Objective-C runtime bindings for Metal
/// On non-macOS platforms, these are stubs that return null/error
pub const objc = if (is_macos) struct {
    /// Get Objective-C class by name
    pub fn getClass(name: [*:0]const u8) ?*anyopaque {
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

    /// Send message to object
    extern fn objc_msgSend() void;

    pub fn msgSend(obj: *anyopaque, selector: *anyopaque) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector);
    }

    pub fn msgSend1(obj: *anyopaque, selector: *anyopaque, arg1: usize) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque, usize) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector, arg1);
    }

    pub fn msgSend2(obj: *anyopaque, selector: *anyopaque, arg1: usize, arg2: usize) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque, usize, usize) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector, arg1, arg2);
    }

    pub fn msgSend3(obj: *anyopaque, selector: *anyopaque, arg1: usize, arg2: usize, arg3: usize) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque, usize, usize, usize) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector, arg1, arg2, arg3);
    }

    pub fn msgSend4(obj: *anyopaque, selector: *anyopaque, arg1: usize, arg2: usize, arg3: usize, arg4: usize) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque, usize, usize, usize, usize) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector, arg1, arg2, arg3, arg4);
    }

    pub fn msgSend5(obj: *anyopaque, selector: *anyopaque, arg1: usize, arg2: usize, arg3: usize, arg4: usize, arg5: usize) callconv(.c) ?*anyopaque {
        const func: *const fn (*anyopaque, *anyopaque, usize, usize, usize, usize, usize) callconv(.c) ?*anyopaque = @ptrCast(&objc_msgSend);
        return func(obj, selector, arg1, arg2, arg3, arg4, arg5);
    }

    pub fn msgSendSize2(obj: *anyopaque, selector: *anyopaque, arg1: MTLSize, arg2: MTLSize) callconv(.c) void {
        const func: *const fn (*anyopaque, *anyopaque, MTLSize, MTLSize) callconv(.c) void = @ptrCast(&objc_msgSend);
        func(obj, selector, arg1, arg2);
    }

    /// Release object
    pub fn release(obj: *anyopaque) void {
        const release_sel = sel("release");
        if (release_sel) |s| {
            _ = msgSend(obj, s);
        }
    }

    /// Retain object
    pub fn retain(obj: *anyopaque) ?*anyopaque {
        const retain_sel = sel("retain");
        if (retain_sel) |s| {
            return msgSend(obj, s);
        }
        return null;
    }
} else struct {
    // Stubs for non-macOS platforms
    pub fn getClass(_: [*:0]const u8) ?*anyopaque { return null; }
    pub fn sel(_: [*:0]const u8) ?*anyopaque { return null; }
    pub fn msgSend(_: *anyopaque, _: *anyopaque) callconv(.c) ?*anyopaque { return null; }
    pub fn msgSend1(_: *anyopaque, _: *anyopaque, _: usize) callconv(.c) ?*anyopaque { return null; }
    pub fn msgSend2(_: *anyopaque, _: *anyopaque, _: usize, _: usize) callconv(.c) ?*anyopaque { return null; }
    pub fn msgSend3(_: *anyopaque, _: *anyopaque, _: usize, _: usize, _: usize) callconv(.c) ?*anyopaque { return null; }
    pub fn msgSend4(_: *anyopaque, _: *anyopaque, _: usize, _: usize, _: usize, _: usize) callconv(.c) ?*anyopaque { return null; }
    pub fn msgSend5(_: *anyopaque, _: *anyopaque, _: usize, _: usize, _: usize, _: usize, _: usize) callconv(.c) ?*anyopaque { return null; }
    pub fn msgSendSize2(_: *anyopaque, _: *anyopaque, _: MTLSize, _: MTLSize) callconv(.c) void {}
    pub fn release(_: *anyopaque) void {}
    pub fn retain(_: *anyopaque) ?*anyopaque { return null; }
};

/// MTLDevice - Metal device (stub on non-macOS)
pub const MTLDevice = struct {
    device: *anyopaque,

    pub fn create() !MTLDevice {
        if (!is_macos) return error.DeviceNotAvailable;

        const MTLCreateSystemDefaultDevice = @extern(*const fn () callconv(.c) ?*anyopaque, .{
            .name = "MTLCreateSystemDefaultDevice",
        });

        const device = MTLCreateSystemDefaultDevice();
        if (device == null) return error.DeviceNotAvailable;

        return MTLDevice{ .device = device.? };
    }

    pub fn release(self: *const MTLDevice) void {
        if (!is_macos) return;
        objc.release(self.device);
    }

    pub fn newCommandQueue(self: *const MTLDevice) !MTLCommandQueue {
        if (!is_macos) return error.CommandQueueCreationFailed;

        const sel = objc.sel("newCommandQueue");
        if (sel == null) return error.CommandQueueCreationFailed;
        const queue = objc.msgSend(self.device, sel.?);
        if (queue == null) return error.CommandQueueCreationFailed;

        return MTLCommandQueue{ .queue = queue.? };
    }

    pub fn newBufferWithLength(self: *const MTLDevice, length: usize, options: MTLResourceOptions) !MTLBuffer {
        if (!is_macos) return error.BufferCreationFailed;

        const sel = objc.sel("newBufferWithLength:options:");
        if (sel == null) return error.BufferCreationFailed;
        const buffer = objc.msgSend2(self.device, sel.?, length, @as(u64, @bitCast(options)));
        if (buffer == null) return error.BufferCreationFailed;

        return MTLBuffer{ .buffer = buffer.? };
    }

    pub fn newBufferWithBytes(self: *const MTLDevice, bytes: []const u8, options: MTLResourceOptions) !MTLBuffer {
        if (!is_macos) return error.BufferCreationFailed;

        const sel = objc.sel("newBufferWithBytes:length:options:");
        if (sel == null) return error.BufferCreationFailed;
        const buffer = objc.msgSend3(self.device, sel.?, @intFromPtr(bytes.ptr), bytes.len, @as(u64, @bitCast(options)));
        if (buffer == null) return error.BufferCreationFailed;

        return MTLBuffer{ .buffer = buffer.? };
    }

    pub fn newComputePipelineStateWithFunction(self: *const MTLDevice, function: *anyopaque) !MTLComputePipelineState {
        if (!is_macos) return error.PipelineCreationFailed;

        const sel = objc.sel("newComputePipelineStateWithFunction:error:");
        if (sel == null) return error.PipelineCreationFailed;
        var err: ?*anyopaque = null;
        const pipeline = objc.msgSend2(self.device, sel.?, @intFromPtr(function), @intFromPtr(&err));
        if (pipeline == null or err != null) {
            return error.PipelineCreationFailed;
        }

        return MTLComputePipelineState{ .pipeline = pipeline.? };
    }

    pub fn newLibraryWithSource(self: *const MTLDevice, source: []const u8) !*anyopaque {
        if (!is_macos) return error.LibraryCreationFailed;

        const source_z = try std.heap.c_allocator.dupeZ(u8, source);
        defer std.heap.c_allocator.free(source_z);

        const source_str = try createNSString(source_z);

        const sel = objc.sel("newLibraryWithSource:options:error:");
        if (sel == null) return error.LibraryCreationFailed;

        var err: ?*anyopaque = null;
        const library = objc.msgSend3(self.device, sel.?, @intFromPtr(source_str), 0, @intFromPtr(&err));

        if (library == null or err != null) return error.LibraryCreationFailed;

        return library.?;
    }
};

/// MTLCommandQueue
pub const MTLCommandQueue = struct {
    queue: *anyopaque,

    pub fn release(self: *const MTLCommandQueue) void {
        if (!is_macos) return;
        objc.release(self.queue);
    }

    pub fn commandBuffer(self: *const MTLCommandQueue) !MTLCommandBuffer {
        if (!is_macos) return error.CommandBufferCreationFailed;

        const sel = objc.sel("commandBuffer");
        if (sel == null) return error.CommandBufferCreationFailed;
        const buffer = objc.msgSend(self.queue, sel.?);
        if (buffer == null) return error.CommandBufferCreationFailed;

        return MTLCommandBuffer{ .buffer = buffer.? };
    }
};

/// MTLCommandBuffer
pub const MTLCommandBuffer = struct {
    buffer: *anyopaque,

    pub fn release(self: *const MTLCommandBuffer) void {
        if (!is_macos) return;
        objc.release(self.buffer);
    }

    pub fn computeCommandEncoder(self: *const MTLCommandBuffer) !MTLComputeCommandEncoder {
        if (!is_macos) return error.EncoderCreationFailed;

        const sel = objc.sel("computeCommandEncoder");
        if (sel) |s| {
            const encoder = objc.msgSend(self.buffer, s);
            if (encoder) |e| {
                return MTLComputeCommandEncoder{ .encoder = e };
            }
        }
        return error.EncoderCreationFailed;
    }

    pub fn blitCommandEncoder(self: *const MTLCommandBuffer) !MTLBlitCommandEncoder {
        if (!is_macos) return error.EncoderCreationFailed;

        const sel = objc.sel("blitCommandEncoder");
        if (sel) |s| {
            const encoder = objc.msgSend(self.buffer, s);
            if (encoder) |e| {
                return MTLBlitCommandEncoder{ .encoder = e };
            }
        }
        return error.EncoderCreationFailed;
    }

    pub fn commit(self: MTLCommandBuffer) void {
        if (!is_macos) return;
        const sel = objc.sel("commit");
        if (sel) |s| {
            _ = objc.msgSend(self.buffer, s);
        }
    }

    pub fn waitUntilCompleted(self: MTLCommandBuffer) void {
        if (!is_macos) return;
        const sel = objc.sel("waitUntilCompleted");
        if (sel) |s| {
            _ = objc.msgSend(self.buffer, s);
        }
    }
};

/// MTLComputeCommandEncoder
pub const MTLComputeCommandEncoder = struct {
    encoder: *anyopaque,

    pub fn release(self: *MTLComputeCommandEncoder) void {
        if (!is_macos) return;
        objc.release(self.encoder);
    }

    pub fn setComputePipelineState(self: *MTLComputeCommandEncoder, state: *const MTLComputePipelineState) void {
        if (!is_macos) return;
        const sel = objc.sel("setComputePipelineState:");
        if (sel) |s| {
            _ = objc.msgSend1(self.encoder, s, @intFromPtr(state.pipeline));
        }
    }

    pub fn setBuffer(self: *MTLComputeCommandEncoder, buffer: *const MTLBuffer, offset: usize, index: usize) void {
        if (!is_macos) return;
        const sel = objc.sel("setBuffer:offset:atIndex:");
        if (sel) |s| {
            _ = objc.msgSend3(self.encoder, s, @intFromPtr(buffer.buffer), offset, index);
        }
    }

    pub fn setBytes(self: *MTLComputeCommandEncoder, bytes: []const u8, index: usize) void {
        if (!is_macos) return;
        const sel = objc.sel("setBytes:length:atIndex:");
        if (sel) |s| {
            _ = objc.msgSend3(self.encoder, s, @intFromPtr(bytes.ptr), bytes.len, index);
        }
    }

    pub fn dispatchThreads(self: *MTLComputeCommandEncoder, threads: MTLSize, threadsPerThreadgroup: MTLSize) void {
        if (!is_macos) return;
        const sel = objc.sel("dispatchThreads:threadsPerThreadgroup:");
        if (sel) |s| {
            objc.msgSendSize2(self.encoder, s, threads, threadsPerThreadgroup);
        }
    }

    pub fn dispatchThreadgroups(self: *MTLComputeCommandEncoder, threadgroupsPerGrid: MTLSize, threadsPerThreadgroup: MTLSize) void {
        if (!is_macos) return;
        const sel = objc.sel("dispatchThreadgroups:threadsPerThreadgroup:");
        if (sel) |s| {
            objc.msgSendSize2(self.encoder, s, threadgroupsPerGrid, threadsPerThreadgroup);
        }
    }

    pub fn endEncoding(self: *MTLComputeCommandEncoder) void {
        if (!is_macos) return;
        const sel = objc.sel("endEncoding");
        if (sel) |s| {
            _ = objc.msgSend(self.encoder, s);
        }
    }
};

/// MTLBlitCommandEncoder
pub const MTLBlitCommandEncoder = struct {
    encoder: *anyopaque,

    pub fn release(self: *MTLBlitCommandEncoder) void {
        if (!is_macos) return;
        objc.release(self.encoder);
    }

    pub fn endEncoding(self: *MTLBlitCommandEncoder) void {
        if (!is_macos) return;
        const sel = objc.sel("endEncoding");
        if (sel) |s| {
            _ = objc.msgSend(self.encoder, s);
        }
    }

    pub fn copyBuffer(self: *MTLBlitCommandEncoder, src: MTLBuffer, src_offset: usize, dst: MTLBuffer, dst_offset: usize, size: usize) void {
        if (!is_macos) return;
        const sel = objc.sel("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");
        if (sel) |s| {
            _ = objc.msgSend5(self.encoder, s, @intFromPtr(src.buffer), src_offset, @intFromPtr(dst.buffer), dst_offset, size);
        }
    }
};

/// MTLBuffer
pub const MTLBuffer = struct {
    buffer: *anyopaque,

    pub fn release(self: *const MTLBuffer) void {
        if (!is_macos) return;
        objc.release(self.buffer);
    }

    pub fn contents(self: *const MTLBuffer) [*]u8 {
        if (!is_macos) return undefined;
        const sel = objc.sel("contents");
        if (sel) |s| {
            const ptr = objc.msgSend(self.buffer, s);
            if (ptr) |p| {
                return @as([*]u8, @ptrCast(p));
            }
        }
        return undefined;
    }

    pub fn length(self: *const MTLBuffer) usize {
        if (!is_macos) return 0;
        const sel = objc.sel("length");
        if (sel) |s| {
            if (objc.msgSend(self.buffer, s)) |len| {
                return @as(usize, @intFromPtr(len));
            }
        }
        return 0;
    }
};

/// MTLComputePipelineState
pub const MTLComputePipelineState = struct {
    pipeline: *anyopaque,

    pub fn release(self: *const MTLComputePipelineState) void {
        if (!is_macos) return;
        objc.release(self.pipeline);
    }

    pub fn maxTotalThreadsPerThreadgroup(self: *const MTLComputePipelineState) usize {
        if (!is_macos) return 0;
        const sel = objc.sel("maxTotalThreadsPerThreadgroup");
        if (sel) |s| {
            return @as(usize, @intFromPtr(objc.msgSend(self.pipeline, s)));
        }
        return 0;
    }

    pub fn threadExecutionWidth(self: *const MTLComputePipelineState) usize {
        if (!is_macos) return 0;
        const sel = objc.sel("threadExecutionWidth");
        if (sel) |s| {
            return @as(usize, @intFromPtr(objc.msgSend(self.pipeline, s)));
        }
        return 0;
    }
};

/// MTLResourceOptions
pub const MTLResourceOptions = packed struct(u64) {
    cpu_cache_mode: u1 = 0,
    _padding1: u3 = 0,
    storage_mode: u2 = 0,
    _padding2: u2 = 0,
    hazard_tracking_mode: u1 = 0,
    _padding3: u55 = 0,

    pub const CPUCacheModeDefaultCache = MTLResourceOptions{ .cpu_cache_mode = 0 };
    pub const CPUCacheModeWriteCombined = MTLResourceOptions{ .cpu_cache_mode = 1 };
    pub const StorageModeShared = MTLResourceOptions{ .storage_mode = 0 };
    pub const StorageModeManaged = MTLResourceOptions{ .storage_mode = 1 };
    pub const StorageModePrivate = MTLResourceOptions{ .storage_mode = 2 };
    pub const StorageModeSharedVal: u64 = 0 << 4;
    pub const StorageModeManagedVal: u64 = 1 << 4;
    pub const StorageModePrivateVal: u64 = 2 << 4;
    pub const HazardTrackingModeUntracked = MTLResourceOptions{ .hazard_tracking_mode = 1 };
    pub const HazardTrackingModeTracked = MTLResourceOptions{ .hazard_tracking_mode = 0 };
};

/// MTLSize
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

/// Load Metal library from file (stub on non-macOS)
pub fn loadMetallib(path: []const u8) !*anyopaque {
    if (!is_macos) return error.LibraryNotFound;

    const NSData = objc.getClass("NSData");
    if (NSData == null) return error.LibraryNotFound;

    const dataWithContentsOfFile = objc.sel("dataWithContentsOfFile:");
    if (dataWithContentsOfFile == null) return error.LibraryNotFound;
    const data = objc.msgSend1(NSData.?, dataWithContentsOfFile.?, @intFromPtr(path.ptr));
    if (data == null) return error.LibraryNotFound;

    return data.?;
}

/// Create NSString from C string (stub on non-macOS)
pub fn createNSString(str: [*:0]const u8) !*anyopaque {
    if (!is_macos) return error.LibraryCreationFailed;

    const NSString = objc.getClass("NSString");
    if (NSString == null) return error.LibraryCreationFailed;

    const stringWithUTF8String = objc.sel("stringWithUTF8String:");
    const ns_str = objc.msgSend1(NSString.?, stringWithUTF8String.?, @intFromPtr(str));
    if (ns_str == null) return error.LibraryCreationFailed;

    return ns_str.?;
}

/// Get function from Metal library (stub on non-macOS)
pub fn getFunctionFromLibrary(library: *anyopaque, name: [*:0]const u8) !*anyopaque {
    if (!is_macos) return error.FunctionNotFound;

    const newFunctionWithName = objc.sel("newFunctionWithName:");
    if (newFunctionWithName == null) return error.FunctionNotFound;

    const ns_name = try createNSString(name);
    const function = objc.msgSend1(library, newFunctionWithName.?, @intFromPtr(ns_name));
    if (function == null) return error.FunctionNotFound;

    return function.?;
}

// Tests
test "Metal API bindings" {
    // Test Objective-C runtime bindings
    const cls = objc.getClass("NSObject");
    // On non-macOS, this returns null which is expected
    _ = cls;
}

test "Metal device creation" {
    if (is_macos) {
        const device = try MTLDevice.create();
        defer device.release();

        const queue = try device.newCommandQueue();
        defer queue.release();
    }
}

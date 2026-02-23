/// Tensor management with Unified Memory support for Metal
const std = @import("std");
const backend_module = @import("backend.zig");
const metal = @import("metal.zig");

pub const Tensor = struct {
    allocator: std.mem.Allocator,
    size: usize,
    shape: []const usize,
    slice: []f32,
    mtl_buffer: ?metal.MTLBuffer = null,
    backend_type: backend_module.Backend.BackendType,

    pub fn init(allocator: std.mem.Allocator, shape: []const usize, backend: backend_module.Backend) !Tensor {
        var size: usize = 1;
        for (shape) |dim| size *= dim;

        const shape_copy = try allocator.alloc(usize, shape.len);
        @memcpy(shape_copy, shape);

        var self = Tensor{
            .allocator = allocator,
            .size = size,
            .shape = shape_copy,
            .slice = undefined,
            .mtl_buffer = null,
            .backend_type = backend.type,
        };

        if (backend.type == .gpu and backend.type.gpu == .metal) {
            if (backend.metal_ctx) |ctx| {
                // Allocate Unified Memory buffer on Metal using the pool
                const byte_length = size * @sizeOf(f32);
                var buffer = try ctx.allocBuffer(
                    byte_length,
                    .StorageModeShared
                );
                self.mtl_buffer = buffer;

                // If a command batch is active, register this buffer so it's not released
                // until the batch completes, even if the Tensor is deinitialized.
                if (ctx.active_command_buffer != null) {
                    try ctx.registerTempResource(buffer);
                    // Retain it because Tensor.deinit() will call release()
                    _ = metal.objc.retain(buffer.buffer);
                }

                // Map the buffer to a Zig slice
                const ptr = @as([*]f32, @ptrCast(@alignCast(buffer.contents())));
                self.slice = ptr[0..size];

                // Zero initialize
                @memset(self.slice, 0);
            } else {
                return error.MetalContextMissing;
            }
        } else {
            // Standard CPU allocation
            self.slice = try allocator.alloc(f32, size);
            @memset(self.slice, 0);
        }

        return self;
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.shape);
        if (self.mtl_buffer) |buf| {
            buf.release();
        } else {
            self.allocator.free(self.slice);
        }
    }

    /// Sync CPU data to GPU (noop for Unified Memory)
    pub fn syncToDevice(self: *Tensor) void {
        _ = self;
        // In Shared Storage Mode on Apple Silicon, this is a noop
    }

    /// Get MTLBuffer pointer if available
    pub fn getMtlBuffer(self: *const Tensor) ?*const metal.MTLBuffer {
        if (self.mtl_buffer) |*buf| {
            return buf;
        }
        return null;
    }

    /// Helper for 3D indexing: [batch, seq, feature]
    pub fn index3D(self: *const Tensor, b: usize, s: usize, f: usize) usize {
        std.debug.assert(self.shape.len == 3);
        return (b * self.shape[1] * self.shape[2]) + (s * self.shape[2]) + f;
    }

    /// Helper for 2D indexing: [rows, cols]
    pub fn index2D(self: *const Tensor, r: usize, c: usize) usize {
        std.debug.assert(self.shape.len == 2);
        return (r * self.shape[1]) + c;
    }
};

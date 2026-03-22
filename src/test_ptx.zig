const std = @import("std");
const cuda_kernels = @import("cuda_kernels.zig");

pub fn main() !void {
    // Print first 200 chars of matmul PTX to verify it's valid
    const ptx = cuda_kernels.MATMUL_SIMPLE_PTX;
    std.debug.print("PTX length: {d}\n", .{ptx.len});
    std.debug.print("PTX header:\n{s}\n", .{ptx[0..@min(ptx.len, 200)]});
}

const std = @import("std");
const cuda_kernels = @import("cuda_kernels.zig");

pub fn main() !void {
    // Print PTX bytes to check for any issues
    const ptx = cuda_kernels.FILL_CONSTANT_PTX;

    std.debug.print("PTX length: {}\n", .{ptx.len});
    std.debug.print("PTX hex dump (first 200 bytes):\n", .{});

    for (ptx[0..@min(200, ptx.len)], 0..) |b, i| {
        if (i % 16 == 0) {
            std.debug.print("\n{d:4}: ", .{i});
        }
        std.debug.print("{x:02} ", .{b});
        if (b >= 32 and b < 127) {
            std.debug.print("{c} ", .{b});
        } else {
            std.debug.print(". ", .{});
        }
    }
    std.debug.print("\n", .{});

    // Check for any non-printable characters
    std.debug.print("\nChecking for non-printable characters:\n", .{});
    var has_issues = false;
    for (ptx, 0..) |b, i| {
        if (b != '\n' and b != '\t' and (b < 32 or b > 126)) {
            std.debug.print("  Found non-printable char 0x{x:02} at position {}\n", .{ b, i });
            has_issues = true;
        }
    }

    if (!has_issues) {
        std.debug.print("  No issues found!\n", .{});
    }
}

const std = @import("std");
pub fn main() !void {
    const file = try std.fs.cwd().openFile("examples/comprehensive_suite/data/GOOG.csv", .{});
    defer file.close();
    var buf_reader = std.io.bufferedReader(file.reader());
    _ = buf_reader;
}

const std = @import("std");
pub fn main() void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var list = std.ArrayList(u32).init(allocator);
    list.deinit();
}

const std = @import("std");
const zn = @import("ZigNeuron");
const common = @import("common.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const window_size = 10;
    const target_size = 5;
    const dataset = try common.loadStockData(allocator, "examples/comprehensive_suite/data/GOOG.csv", window_size, target_size);
    defer dataset.deinit();

    const backend = try zn.backend.Backend.init(allocator);
    var net = try zn.network.Network.init(allocator, backend);
    defer net.deinit();

    // 13. GRU Seq2seq
    _ = try net.addGRU(1, 64, window_size);
    _ = try net.addDense(64, target_size, .linear);

    std.debug.print("\n--- Training GRU Seq2seq ---\n", .{});
    try net.train(dataset.x, dataset.y, 50, 0.01, .{ .mse = {} }, null, null);
}

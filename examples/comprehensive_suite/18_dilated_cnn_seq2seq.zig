const std = @import("std");
const zn = @import("ZigNeuron");
const common = @import("common.zig");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    const window_size = 12;
    const target_size = 1;
    const dataset = try common.loadStockData(allocator, "examples/comprehensive_suite/data/GOOG.csv", window_size, target_size);
    defer dataset.deinit();

    const backend = try zn.backend.Backend.init(allocator);
    var net = try zn.network.Network.init(allocator, backend);
    defer net.deinit();

    // 18. Dilated-CNN-Seq2seq
    _ = try net.addConv1D(1, 8, 3, window_size, .relu);
    _ = try net.addDense(80, 16, .relu);
    _ = try net.addDense(16, target_size, .linear);

    std.debug.print("\n--- Training Dilated CNN Seq2seq ---\n", .{});
    try net.train(dataset.x, dataset.y, 50, 0.01, .{ .mse = {} }, null, null);
}

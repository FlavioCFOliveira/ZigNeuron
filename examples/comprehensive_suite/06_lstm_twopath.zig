const std = @import("std");
const zn = @import("ZigNeuron");
const common = @import("common.zig");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    const window_size = 10;
    const dataset = try common.loadStockData(allocator, "examples/comprehensive_suite/data/GOOG.csv", window_size, 1);
    defer dataset.deinit();

    const backend = try zn.backend.Backend.init(allocator);
    var net = try zn.network.Network.init(allocator, backend);
    defer net.deinit();

    // 6. LSTM 2-Path
    _ = try net.addTwoPath(.lstm, 16, .lstm, 16, 1, window_size, .tanh);
    _ = try net.addDense(32, 1, .linear);

    std.debug.print("\n--- Training LSTM 2-Path ---\n", .{});
    try net.train(dataset.x, dataset.y, 50, 0.01, .{ .mse = {} }, null, null);
}

const std = @import("std");
const zn = @import("ZigNeuron");
const common = @import("common.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const window_size = 10;
    const target_size = 10;
    const dataset = try common.loadStockData(allocator, "examples/comprehensive_suite/data/GOOG.csv", window_size, target_size);
    defer dataset.deinit();

    const backend = try zn.backend.Backend.init(allocator);
    var net = try zn.network.Network.init(allocator, backend);
    defer net.deinit();

    const latent_dim = 8;

    // 12. LSTM Seq2seq VAE
    _ = try net.addLSTM(1, 32, window_size);
    _ = try net.addDense(32, latent_dim * 2, .linear);
    _ = try net.addSampling(latent_dim * 2);
    _ = try net.addDense(latent_dim, 32, .relu);
    _ = try net.addDense(32, window_size, .linear);

    std.debug.print("\n--- Training LSTM Seq2seq VAE ---\n", .{});
    try net.train(dataset.x, dataset.x, 50, 0.01, .{ .mse = {} });
}

const std = @import("std");
const zn = @import("ZigNeuron");
const utils = @import("utils.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // 1. Generate synthetic stock data
    const data_len = 200;
    var raw_data = try allocator.alloc(f32, data_len);
    defer allocator.free(raw_data);
    for (0..data_len) |i| {
        raw_data[i] = @sin(@as(f32, @floatFromInt(i)) * 0.1) + 1.5;
    }

    // 2. Scale data
    var scaler = utils.MinMaxScaler.init(0, 1);
    scaler.fit(raw_data);
    const scaled_data = try allocator.alloc(f32, data_len);
    defer allocator.free(scaled_data);
    scaler.transform(raw_data, scaled_data);

    // 3. Create windows
    const window_size = 12;
    const windows = try utils.createWindows(allocator, scaled_data, window_size);
    defer {
        for (windows.x) |x| allocator.free(x);
        for (windows.y) |y| allocator.free(y);
        allocator.free(windows.x);
        allocator.free(windows.y);
    }

    // 4. Initialize Network
    const backend = try zn.backend.Backend.init(allocator);
    var net = try zn.network.Network.init(allocator, backend);
    defer net.deinit();

    // CNN + Dense architecture for time series
    // Input: window_size=12
    // Conv1D: in_channels=1, out_channels=4, kernel_size=3 => out_len = (12-3)+1 = 10. Total size = 40
    _ = try net.addConv1D(1, 4, 3, window_size, .relu);
    _ = try net.addDense(40, 16, .relu);
    _ = try net.addDense(16, 1, .linear);

    // 5. Train
    std.debug.print("Starting training CNN stock predictor...\n", .{});
    const epochs = 100;
    const learning_rate: f32 = 0.01;
    try net.train(windows.x, windows.y, epochs, learning_rate, .{ .mse = {} }, null, null);

    // 6. Predict
    const last_window = windows.x[windows.x.len - 1];
    var prediction = [_]f32{0};
    try net.forward(last_window, &prediction);

    var unscaled_pred = [_]f32{0};
    scaler.inverseTransform(&prediction, &unscaled_pred);
    std.debug.print("Final prediction (unscaled): {d:.4}, Target: {d:.4}\n", .{ unscaled_pred[0], raw_data[data_len - 1] });
}

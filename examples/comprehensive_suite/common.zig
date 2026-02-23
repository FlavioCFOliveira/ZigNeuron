const std = @import("std");
const zn = @import("ZigNeuron");

pub const Dataset = struct {
    x: [][]f32,
    y: [][]f32,
    scaler: MinMaxScaler,
    allocator: std.mem.Allocator,

    pub fn deinit(self: Dataset) void {
        for (self.x) |slice| self.allocator.free(slice);
        for (self.y) |slice| self.allocator.free(slice);
        self.allocator.free(self.x);
        self.allocator.free(self.y);
    }
};

pub const MinMaxScaler = struct {
    data_min: f32,
    data_max: f32,

    pub fn fitTransform(data: []f32) MinMaxScaler {
        var min = data[0];
        var max = data[0];
        for (data) |v| {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        const range = max - min;
        for (data) |*v| {
            v.* = (v.* - min) / range;
        }
        return .{ .data_min = min, .data_max = max };
    }
};

pub fn loadStockData(allocator: std.mem.Allocator, path: []const u8, window_size: usize, target_size: usize) !Dataset {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var buf: [4096]u8 = undefined;
    var reader = file.reader(&buf);

    // Skip header
    _ = try reader.interface.takeDelimiter('\n');

    var prices = std.ArrayList(f32).empty;
    defer prices.deinit(allocator);

    while (try reader.interface.takeDelimiter('\n')) |line| {
        var it = std.mem.splitSequence(u8, line, ",");
        _ = it.next(); // Date
        _ = it.next(); // Open
        _ = it.next(); // High
        _ = it.next(); // Low
        const close_str = it.next() orelse continue;
        const close = std.fmt.parseFloat(f32, close_str) catch continue;
        try prices.append(allocator, close);
    }

    const scaler = MinMaxScaler.fitTransform(prices.items);

    const num_samples = prices.items.len - window_size - target_size + 1;
    var x = try allocator.alloc([]f32, num_samples);
    var y = try allocator.alloc([]f32, num_samples);

    for (0..num_samples) |i| {
        // LSTM/RNN expects sequence of features. If target_size is 1, y is [1]
        // Prices items are already scaled.
        x[i] = try allocator.alloc(f32, window_size);
        y[i] = try allocator.alloc(f32, target_size);
        @memcpy(x[i], prices.items[i .. i + window_size]);
        @memcpy(y[i], prices.items[i + window_size .. i + window_size + target_size]);
    }

    return Dataset{
        .x = x,
        .y = y,
        .scaler = scaler,
        .allocator = allocator,
    };
}

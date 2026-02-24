const std = @import("std");

pub const MinMaxScaler = struct {
    min: f32,
    max: f32,
    data_min: f32,
    data_max: f32,

    pub fn init(min: f32, max: f32) MinMaxScaler {
        return .{
            .min = min,
            .max = max,
            .data_min = 0,
            .data_max = 0,
        };
    }

    pub fn fit(self: *MinMaxScaler, data: []const f32) void {
        var min_val = data[0];
        var max_val = data[0];
        for (data) |v| {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        self.data_min = min_val;
        self.data_max = max_val;
    }

    pub fn transform(self: MinMaxScaler, data: []const f32, output: []f32) void {
        const range = self.data_max - self.data_min;
        const target_range = self.max - self.min;
        for (data, 0..) |v, i| {
            output[i] = ((v - self.data_min) / range) * target_range + self.min;
        }
    }

    pub fn inverseTransform(self: MinMaxScaler, data: []const f32, output: []f32) void {
        const range = self.data_max - self.data_min;
        const target_range = self.max - self.min;
        for (data, 0..) |v, i| {
            output[i] = ((v - self.min) / target_range) * range + self.data_min;
        }
    }
};

pub fn createWindows(allocator: std.mem.Allocator, data: []const f32, window_size: usize) !struct { x: [][]f32, y: [][]f32 } {
    const num_windows = data.len - window_size;
    var x = try allocator.alloc([]f32, num_windows);
    var y = try allocator.alloc([]f32, num_windows);

    for (0..num_windows) |i| {
        x[i] = try allocator.alloc(f32, window_size);
        @memcpy(x[i], data[i .. i + window_size]);
        y[i] = try allocator.alloc(f32, 1);
        y[i][0] = data[i + window_size];
    }

    return .{ .x = x, .y = y };
}

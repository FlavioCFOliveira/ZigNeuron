const std = @import("std");
const zn = @import("ZigNeuron");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const training_data = [_][]const f32{
        &.{0, 0},
        &.{0, 1},
        &.{1, 0},
        &.{1, 1},
    };
    const training_targets = [_][]const f32{
        &.{0},
        &.{1},
        &.{1},
        &.{0},
    };

    const network = try zn.network.Network.init(allocator);
    defer network.deinit();

    _ = try network.addDense(2, 3, .relu);
    _ = try network.addDense(3, 1, .sigmoid);

    const epochs: usize = 1000;
    const learning_rate: f32 = 0.1;
    const loss_fn = zn.loss.Loss{ .mse = {} };

    // Train and print loss each epoch
    for (0..epochs) |epoch| {
        var total_loss: f32 = 0;
        for (training_data, training_targets) |sample, target| {
            const sample_loss = try network.trainStep(sample, target, learning_rate, loss_fn);
            total_loss += sample_loss;
        }
        total_loss /= @as(f32, @floatFromInt(training_data.len));

        if (epoch % 100 == 0) {
            std.debug.print("Epoch {}: Loss = {d:.6}\n", .{ epoch, total_loss });
        }
    }

    std.debug.print("\nTesting:\n", .{});
    for (training_data, training_targets) |data, target| {
        var output: [1]f32 = undefined;
        _ = try network.forward(data, &output);
        std.debug.print("Input: [{d}, {d}] -> Output: {d:.4} (Expected: {d})\n", .{
            data[0], data[1], output[0], target[0],
        });
    }
}

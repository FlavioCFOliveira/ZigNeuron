/// Examples for ZigNeuron library - XOR problem with backpropagation
const std = @import("std");
const zn = @import("ZigNeuron");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Get default backend (GPU if available, CPU fallback)
    var backend = try zn.backend.Backend.init(allocator);
    backend.type = .cpu;
    errdefer backend.deinit();
    std.debug.print("Using backend: ", .{});
    switch (backend.type) {
        .gpu => |gpu| switch (gpu) {
            .metal => std.debug.print("Metal (Apple Silicon GPU)\n", .{}),
            .vulkan => std.debug.print("Vulkan (Cross-platform GPU)\n", .{}),
        },
        .cpu => std.debug.print("CPU (fallback)\n", .{}),
    }

    // Training data: XOR problem
    // Input: [a, b], Output: a XOR b
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

    std.debug.print("\nXOR Neural Network with Backpropagation Training\n", .{});
    std.debug.print("=================================================\n\n", .{});

    // Create network: 2 inputs -> 3 hidden -> 1 output
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();

    _ = try network.addDense(2, 3, .tanh);
    _ = try network.addDense(3, 1, .sigmoid);

    std.debug.print("Network created with 2 inputs, 3 hidden neurons, 1 output\n\n", .{});

    // Train the network using backpropagation with SGD
    const epochs: usize = 1000;
    const learning_rate: f32 = 0.5;
    const loss_fn = zn.loss.Loss{ .mse = {} };

    std.debug.print("Training for {} epochs with learning rate {}...\n\n", .{ epochs, learning_rate });

    try network.train(training_data[0..], training_targets[0..], epochs, learning_rate, loss_fn);

    std.debug.print("\nTesting after training:\n", .{});

    // Test the network
    for (training_data, training_targets) |data, target| {
        var output: [1]f32 = undefined;
        _ = try network.forward(data, &output);
        std.debug.print("Input: [{d}, {d}] -> Output: {d:.4} (Expected: {d})\n", .{
            data[0],
            data[1],
            output[0],
            target[0],
        });
    }
}

/// Memory leak detection tests

const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const layer = zn.layer;
const activation = zn.activation;
const loss = zn.loss;
const network = zn.network;
const backend = zn.backend;
const optimizer = zn.optimizer;

test "memory leak: Dense layer init/deinit" {
    const allocator = testing.allocator;

    // Allocate many layers and verify no leaks
    for (0..100) |_| {
        const lyr = try layer.Dense.init(allocator, 4, 8, .relu, backend.Backend{ .cpu = {} });
        lyr.deinit();
    }
}

test "memory leak: Network init/deinit" {
    const allocator = testing.allocator;

    // Create and destroy many networks
    for (0..100) |_| {
        const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
        net.deinit();
    }
}

test "memory leak: Network with layers" {
    const allocator = testing.allocator;

    for (0..50) |_| {
        const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });

        _ = try net.addDense(4, 8, .relu);
        _ = try net.addDense(8, 4, .relu);
        _ = try net.addDense(4, 1, .sigmoid);

        net.deinit();
    }
}

test "memory leak: Forward pass allocations" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    _ = try net.addDense(4, 8, .relu);
    _ = try net.addDense(8, 1, .sigmoid);

    const input: []const f32 = &.{ 0.1, 0.2, 0.3, 0.4 };
    var output: [1]f32 = undefined;

    // Run many forward passes
    for (0..1000) |_| {
        _ = try net.forward(input, &output);
    }

    // Network should still be valid
    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: Training allocations" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    _ = try net.addDense(4, 8, .relu);
    _ = try net.addDense(8, 1, .sigmoid);

    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0, 0.0, 0.0 },
        &.{ 1.0, 1.0, 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Train for multiple epochs
    for (0..100) |_| {
        try net.train(training_data, training_targets, 1, 0.01, loss_fn);
    }

    // Network should still work
    var output: [1]f32 = undefined;
    _ = try net.forward(training_data[0], &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: SGD optimizer" {
    const allocator = testing.allocator;

    for (0..50) |_| {
        var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

        const lyr = try layer.Dense.init(allocator, 4, 8, .relu, backend.Backend{ .cpu = {} });
        defer lyr.deinit();

        try sgd.init(allocator, &lyr);
        sgd.deinit(allocator);
    }
}

test "memory leak: Adam optimizer" {
    const allocator = testing.allocator;

    for (0..50) |_| {
        var adam: optimizer.Adam = .{};

        const lyr = try layer.Dense.init(allocator, 4, 8, .relu, backend.Backend{ .cpu = {} });
        defer lyr.deinit();

        try adam.init(allocator, &lyr);
        adam.deinit(allocator);
    }
}

test "memory leak: RMSprop optimizer" {
    const allocator = testing.allocator;

    for (0..50) |_| {
        var rmsprop: optimizer.Rmsprop = .{};

        const lyr = try layer.Dense.init(allocator, 4, 8, .relu, backend.Backend{ .cpu = {} });
        defer lyr.deinit();

        try rmsprop.init(allocator, &lyr);
        rmsprop.deinit(allocator);
    }
}

test "memory leak: Network with optimizer" {
    const allocator = testing.allocator;

    for (0..20) |_| {
        const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });

        _ = try net.addDense(4, 8, .relu);
        _ = try net.addDense(8, 1, .linear);

        const opt = optimizer.Optimizer{ .sgd = optimizer.Sgd{ .momentum = 0.9 } };
        try net.initOptimizer(&opt);

        net.deinitOptimizer(&opt);
        net.deinit();
    }
}

test "memory leak: Deep network" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    // Deep network: 1-64-64-64-64-1
    _ = try net.addDense(1, 64, .relu);
    _ = try net.addDense(64, 64, .relu);
    _ = try net.addDense(64, 64, .relu);
    _ = try net.addDense(64, 64, .relu);
    _ = try net.addDense(64, 1, .linear);

    const input: []const f32 = &.{ 0.5 };
    var output: [1]f32 = undefined;

    // Run many forward passes
    for (0..1000) |_| {
        _ = try net.forward(input, &output);
    }

    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: Batch training" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    _ = try net.addDense(4, 8, .relu);
    _ = try net.addDense(8, 1, .sigmoid);

    // Create a larger batch
    var training_data: [20][]const f32 = undefined;
    var training_targets: [20][]const f32 = undefined;

    for (0..20) |i| {
        training_data[i] = &.{ @as(f32, @floatFromInt(i % 4)) / 3.0,
            @as(f32, @floatFromInt((i + 1) % 4)) / 3.0,
            @as(f32, @floatFromInt((i + 2) % 4)) / 3.0,
            @as(f32, @floatFromInt((i + 3) % 4)) / 3.0 };
        training_targets[i] = &.{ @as(f32, @floatFromInt(i % 2)) / 1.0 };
    }

    const loss_fn = loss.Loss{ .mse = {} };
    try net.train(&training_data, &training_targets, 10, 0.01, loss_fn);

    var output: [1]f32 = undefined;
    _ = try net.forward(training_data[0], &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: Multiple networks simultaneously" {
    const allocator = testing.allocator;

    // Create multiple networks at once
    var networks: [20]*network.Network = undefined;
    for (0..20) |i| {
        networks[i] = try network.Network.init(allocator, backend.Backend{ .cpu = {} });

        _ = try networks[i].addDense(2, 4, .relu);
        _ = try networks[i].addDense(4, 1, .sigmoid);
    }

    // Use them
    var input: [2]f32 = .{ 0.5, 0.5 };
    var output: [1]f32 = undefined;
    for (networks) |net| {
        _ = try net.forward(&input, &output);
        try testing.expect(std.math.isFinite(output[0]));
    }

    // Clean up
    for (networks) |net| {
        net.deinit();
    }
}

test "memory leak: Sequential training" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    _ = try net.addDense(1, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Train for many epochs
    for (0..500) |_| {
        try net.train(training_data, training_targets, 1, 0.01, loss_fn);
    }

    // Should still work
    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: Layer operations" {
    const allocator = testing.allocator;

    // Test forward/backward cycles
    for (0..100) |_| {
        const lyr = try layer.Dense.init(allocator, 4, 8, .relu, backend.Backend{ .cpu = {} });

        const input: []const f32 = &.{ 0.1, 0.2, 0.3, 0.4 };
        var output: [8]f32 = undefined;
        try lyr.forward(input, &output);

        const grad_output: []const f32 = &.{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
        var grad_input: [4]f32 = undefined;
        try lyr.backward(input, grad_output, &grad_input);

        lyr.deinit();
    }
}

test "memory leak: Activation operations" {
    const relu = activation.Activation{ .relu = {} };

    // Test many forward/backward calls
    for (0..1000) |_| {
        var input: [64]f32 = undefined;
        var output: [64]f32 = undefined;

        for (0..64) |i| {
            input[i] = @as(f32, @floatFromInt(i % 100)) / 10.0 - 5.0;
        }

        try relu.forwardBatch(&input, &output);
        try relu.backwardBatch(&input, &output, &input);

        // Just verify it doesn't crash
        _ = output[0];
    }
}

test "memory leak: Loss operations" {
    const loss_fn = loss.Loss{ .mse = {} };

    // Test many forward/backward calls
    for (0..1000) |_| {
        var output: [32]f32 = undefined;
        var target: [32]f32 = undefined;
        var grad: [32]f32 = undefined;

        for (0..32) |i| {
            output[i] = @as(f32, @floatFromInt(i % 100)) / 10.0;
            target[i] = @as(f32, @floatFromInt((i + 50) % 100)) / 10.0;
        }

        const loss_value = try loss_fn.forward(&output, &target);
        try loss_fn.backward(&output, &target, &grad);

        try testing.expect(std.math.isFinite(loss_value));
    }
}

test "memory leak: Large network" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    // Large network
    _ = try net.addDense(64, 128, .relu);
    _ = try net.addDense(128, 128, .relu);
    _ = try net.addDense(128, 64, .relu);
    _ = try net.addDense(64, 1, .linear);

    const input: []const f32 = &.{ [64]f32{0.1} ** 64 };
    var output: [1]f32 = undefined;

    // Run forward passes
    for (0..100) |_| {
        _ = try net.forward(input, &output);
    }

    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: Gradient computation" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    _ = try net.addDense(4, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const input: []const f32 = &.{ 0.1, 0.2, 0.3, 0.4 };
    const target: []const f32 = &.{ 0.5 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Compute gradients many times
    for (0..100) |_| {
        _ = try net.computeGradients(target, loss_fn);
    }

    // Check that network still works
    var output: [1]f32 = undefined;
    _ = try net.forward(input, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: Optimizer step" {
    const allocator = testing.allocator;

    for (0..100) |_| {
        var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

        const lyr = try layer.Dense.init(allocator, 4, 8, .relu, backend.Backend{ .cpu = {} });

        try sgd.init(allocator, &lyr);

        // Run many steps
        for (0..100) |_| {
            @memset(lyr.grad_weights, 0.1);
            @memset(lyr.grad_bias, 0.1);
            sgd.step(&lyr, 0.01);
        }

        lyr.deinit();
        sgd.deinit(allocator);
    }
}

test "memory leak: Complex training loop" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    _ = try net.addDense(4, 8, .relu);
    _ = try net.addDense(8, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0, 0.0, 0.0 },
        &.{ 0.5, 0.5, 0.5, 0.5 },
        &.{ 1.0, 1.0, 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 0.5 },
        &.{ 1.0 },
    };

    const opt = optimizer.Optimizer{ .adam = optimizer.Adam{} };
    const loss_fn = loss.Loss{ .mse = {} };

    // Full training loop
    try net.initOptimizer(&opt);
    try net.train(training_data, training_targets, 50, 0.01, loss_fn);
    net.deinitOptimizer(&opt);

    // Verify network still works
    var output: [1]f32 = undefined;
    _ = try net.forward(training_data[0], &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: Multiple optimizers" {
    const allocator = testing.allocator;

    // Create networks with different optimizers
    var networks: [10]*network.Network = undefined;
    var optimizers: [10]optimizer.Optimizer = undefined;

    for (0..10) |i| {
        networks[i] = try network.Network.init(allocator, backend.Backend{ .cpu = {} });

        _ = try networks[i].addDense(2, 4, .relu);
        _ = try networks[i].addDense(4, 1, .linear);

        // Different optimizers for different networks
        optimizers[i] = switch (i % 3) {
            0 => optimizer.Optimizer{ .sgd = optimizer.Sgd{ .momentum = 0.0 } },
            1 => optimizer.Optimizer{ .sgd = optimizer.Sgd{ .momentum = 0.9 } },
            2 => optimizer.Optimizer{ .adam = optimizer.Adam{} },
        };

        try networks[i].initOptimizer(&optimizers[i]);
    }

    // Use them
    var input: [2]f32 = .{ 0.5, 0.5 };
    var output: [1]f32 = undefined;
    for (networks) |net| {
        _ = try net.forward(&input, &output);
        try testing.expect(std.math.isFinite(output[0]));
    }

    // Clean up
    for (networks, optimizers) |net, opt| {
        net.deinitOptimizer(&opt);
        net.deinit();
    }
}

test "memory leak: Network copy scenario" {
    const allocator = testing.allocator;

    const net1 = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net1.deinit();

    _ = try net1.addDense(2, 4, .relu);
    _ = try net1.addDense(4, 1, .linear);

    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };
    try net1.train(training_data, training_targets, 10, 0.01, loss_fn);

    // Create second network
    const net2 = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net2.deinit();

    _ = try net2.addDense(2, 4, .relu);
    _ = try net2.addDense(4, 1, .linear);

    // Both networks should work independently
    var output1: [1]f32 = undefined;
    var output2: [1]f32 = undefined;

    _ = try net1.forward(&.{ 0.5, 0.5 }, &output1);
    _ = try net2.forward(&.{ 0.5, 0.5 }, &output2);

    try testing.expect(std.math.isFinite(output1[0]));
    try testing.expect(std.math.isFinite(output2[0]));
}

test "memory leak: Nested allocations" {
    const allocator = testing.allocator;

    // Test allocation/deallocation nesting
    for (0..10) |_| {
        const net1 = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
        defer net1.deinit();

        _ = try net1.addDense(2, 4, .relu);
        _ = try net1.addDense(4, 1, .linear);

        for (0..5) |_| {
            const net2 = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
            defer net2.deinit();

            _ = try net2.addDense(2, 4, .relu);
            _ = try net2.addDense(4, 1, .linear);

            var input: [2]f32 = .{ 0.5, 0.5 };
            var output: [1]f32 = undefined;
            _ = try net2.forward(&input, &output);
        }

        // Outer network should still work
        var input: [2]f32 = .{ 0.5, 0.5 };
        var output: [1]f32 = undefined;
        _ = try net1.forward(&input, &output);
        try testing.expect(std.math.isFinite(output[0]));
    }
}

test "memory leak: Deep nesting" {
    const allocator = testing.allocator;

    // Create a deeply nested structure
    var level1 = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer level1.deinit();

    _ = try level1.addDense(2, 4, .relu);
    _ = try level1.addDense(4, 1, .linear);

    for (0..5) |_| {
        var level2 = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
        defer level2.deinit();

        _ = try level2.addDense(2, 4, .relu);
        _ = try level2.addDense(4, 1, .linear);

        for (0..3) |_| {
            var level3 = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
            defer level3.deinit();

            _ = try level3.addDense(2, 4, .relu);
            _ = try level3.addDense(4, 1, .linear);

            var input: [2]f32 = .{ 0.5, 0.5 };
            var output: [1]f32 = undefined;
            _ = try level3.forward(&input, &output);
        }
    }

    // Level 1 should still work
    var input: [2]f32 = .{ 0.5, 0.5 };
    var output: [1]f32 = undefined;
    _ = try level1.forward(&input, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: Many small networks" {
    const allocator = testing.allocator;

    // Create many small networks
    for (0..200) |_| {
        const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
        defer net.deinit();

        _ = try net.addDense(1, 2, .relu);
        _ = try net.addDense(2, 1, .linear);

        var input: [1]f32 = .{ 0.5 };
        var output: [1]f32 = undefined;
        _ = try net.forward(&input, &output);
    }
}

test "memory leak: Large layer training" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    // Large layer
    _ = try net.addDense(128, 256, .relu);
    _ = try net.addDense(256, 128, .relu);
    _ = try net.addDense(128, 1, .linear);

    const input: []const f32 = &.{ [128]f32{0.1} ** 128 };
    const target: []const f32 = &.{ 0.5 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Train
    for (0..10) |_| {
        _ = try net.trainStep(input, target, 0.01, loss_fn);
    }

    // Verify
    var output: [1]f32 = undefined;
    _ = try net.forward(input, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: Repeated layer creation" {
    const allocator = testing.allocator;

    for (0..500) |_| {
        const lyr = try layer.Dense.init(allocator, 4, 4, .relu, backend.Backend{ .cpu = {} });

        const input: []const f32 = &.{ 0.1, 0.2, 0.3, 0.4 };
        var output: [4]f32 = undefined;
        try lyr.forward(input, &output);

        lyr.deinit();
    }
}

test "memory leak: Activation function cycles" {
    const allocator = testing.allocator;

    _ = activation.Activation{ .relu = {} };
    _ = activation.Activation{ .sigmoid = {} };
    _ = activation.Activation{ .tanh = {} };

    // Cycle through activations
    for (0..100) |_| {
        const lyr1 = try layer.Dense.init(allocator, 4, 8, .relu, backend.Backend{ .cpu = {} });
        const lyr2 = try layer.Dense.init(allocator, 8, 4, .sigmoid, backend.Backend{ .cpu = {} });
        const lyr3 = try layer.Dense.init(allocator, 4, 2, .tanh, backend.Backend{ .cpu = {} });

        const input: []const f32 = &.{ 0.1, 0.2, 0.3, 0.4 };
        var output: [2]f32 = undefined;

        const temp1 = try lyr1.forward(input, &[_]f32{0} ** 8);
        const temp2 = try lyr2.forward(temp1, &[_]f32{0} ** 4);
        _ = try lyr3.forward(temp2, &output);

        lyr1.deinit();
        lyr2.deinit();
        lyr3.deinit();
    }
}

test "memory leak: Optimizer with momentum" {
    const allocator = testing.allocator;

    for (0..50) |_| {
        var sgd: optimizer.Sgd = .{ .momentum = 0.9 };

        const lyr = try layer.Dense.init(allocator, 10, 10, .relu, backend.Backend{ .cpu = {} });

        try sgd.init(allocator, &lyr);

        // Run many steps
        for (0..50) |_| {
            @memset(lyr.grad_weights, 0.01);
            @memset(lyr.grad_bias, 0.01);
            sgd.step(&lyr, 0.01);
        }

        lyr.deinit();
        sgd.deinit(allocator);
    }
}

test "memory leak: Adam state accumulation" {
    const allocator = testing.allocator;

    for (0..50) |_| {
        var adam: optimizer.Adam = .{};

        const lyr = try layer.Dense.init(allocator, 8, 8, .relu, backend.Backend{ .cpu = {} });

        try adam.init(allocator, &lyr);

        // Run many steps
        for (0..100) |_| {
            @memset(lyr.grad_weights, 0.1);
            @memset(lyr.grad_bias, 0.1);
            adam.step(&lyr, 0.001);
        }

        lyr.deinit();
        adam.deinit(allocator);
    }
}

test "memory leak: RMSprop state accumulation" {
    const allocator = testing.allocator;

    for (0..50) |_| {
        var rmsprop: optimizer.Rmsprop = .{};

        const lyr = try layer.Dense.init(allocator, 8, 8, .relu, backend.Backend{ .cpu = {} });

        try rmsprop.init(allocator, &lyr);

        // Run many steps
        for (0..100) |_| {
            @memset(lyr.grad_weights, 0.1);
            @memset(lyr.grad_bias, 0.1);
            rmsprop.step(&lyr, 0.001);
        }

        lyr.deinit();
        rmsprop.deinit(allocator);
    }
}

test "memory leak: Gradient accumulation" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    _ = try net.addDense(4, 8, .relu);
    _ = try net.addDense(8, 1, .linear);

    const input: []const f32 = &.{ 0.1, 0.2, 0.3, 0.4 };
    const target: []const f32 = &.{ 0.5 };
    const loss_fn = loss.Loss{ .mse = {} };

    // Accumulate gradients over many steps
    for (0..100) |_| {
        _ = try net.computeGradients(target, loss_fn);
    }

    // Network should still be valid
    var output: [1]f32 = undefined;
    _ = try net.forward(input, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: Mixed operations" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    _ = try net.addDense(2, 4, .relu);
    _ = try net.addDense(4, 1, .sigmoid);

    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Mixed forward and backward passes
    for (0..100) |_| {
        // Forward
        var output: [1]f32 = undefined;
        _ = try net.forward(&.{ 0.5, 0.5 }, &output);

        // Training step
        _ = try net.trainStep(training_data[0], training_targets[0], 0.01, loss_fn);
    }

    // Final forward pass
    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5, 0.5 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

test "memory leak: Complex network with multiple optimizers" {
    const allocator = testing.allocator;

    const net = try network.Network.init(allocator, backend.Backend{ .cpu = {} });
    defer net.deinit();

    _ = try net.addDense(2, 8, .relu);
    _ = try net.addDense(8, 8, .relu);
    _ = try net.addDense(8, 4, .relu);
    _ = try net.addDense(4, 1, .linear);

    const opt = optimizer.Optimizer{ .adam = optimizer.Adam{} };
    try net.initOptimizer(&opt);

    const training_data = &[_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 1.0, 1.0 },
    };
    const training_targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
    };

    const loss_fn = loss.Loss{ .mse = {} };

    // Full training
    try net.train(training_data, training_targets, 100, 0.001, loss_fn);

    net.deinitOptimizer(&opt);

    // Verify network works
    var output: [1]f32 = undefined;
    _ = try net.forward(&.{ 0.5, 0.5 }, &output);
    try testing.expect(std.math.isFinite(output[0]));
}

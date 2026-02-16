/// Helper utilities for testing ZigNeuron
/// Provides common test data, network builders, and assertion helpers

const std = @import("std");
const zn = @import("ZigNeuron");
const layer = zn.layer;
const activation = zn.activation;
const loss = zn.loss;
const network = zn.network;
const optimizer = zn.optimizer;
const backend_module = zn.backend;

/// Configuration for creating a network quickly
pub const LayerConfig = struct {
    /// Number of input neurons
    input_size: usize,
    /// Number of output neurons
    output_size: usize,
    /// Activation function to use
    activation: activation.ActivationType,
};

/// Create a network from a list of layer configurations
pub fn createNetwork(
    allocator: std.mem.Allocator,
    be: backend_module.Backend,
    layers: []const LayerConfig,
) !network.Network {
    var net = try network.Network.init(allocator, be);
    for (layers) |layer_config| {
        _ = try net.addDense(layer_config.input_size, layer_config.output_size, layer_config.activation);
    }
    return net;
}

/// Create XOR training data
pub fn getXorData(allocator: std.mem.Allocator) !struct {
    data: []const []const f32,
    targets: []const []const f32,
} {
    const data = &[_][]const f32{
        &.{ 0.0, 0.0 },
        &.{ 0.0, 1.0 },
        &.{ 1.0, 0.0 },
        &.{ 1.0, 1.0 },
    };
    const targets = &[_][]const f32{
        &.{ 0.0 },
        &.{ 1.0 },
        &.{ 1.0 },
        &.{ 0.0 },
    };
    return .{ .data = data, .targets = targets };
}

/// Check if two arrays are approximately equal
pub fn expectApproxEqual(
    actual: []const f32,
    expected: []const f32,
    tolerance: f32,
) !void {
    if (actual.len != expected.len) return error.ShapeMismatch;

    for (actual, expected) |a, e| {
        const diff = if (a > e) a - e else e - a;
        if (diff > tolerance) return error.ApproximationFailed;
    }
}

/// Assert that training converges by checking loss decreases
pub fn expectTrainingConverges(
    net: *network.Network,
    data: []const []const f32,
    targets: []const []const f32,
    epochs: usize,
    learning_rate: f32,
    loss_fn: loss.Loss,
) !void {
    var initial_loss: f32 = 0;
    var final_loss: f32 = 0;

    // Measure initial loss
    for (data, targets) |sample, target| {
        const sample_loss = try net.trainStep(sample, target, 0, loss_fn); // Just forward pass
        initial_loss += sample_loss;
    }
    initial_loss /= @as(f32, @floatFromInt(data.len));

    // Train
    try net.train(data, targets, epochs, learning_rate, loss_fn);

    // Measure final loss
    for (data, targets) |sample, target| {
        const sample_loss = try net.trainStep(sample, target, 0, loss_fn); // Just forward pass
        final_loss += sample_loss;
    }
    final_loss /= @as(f32, @floatFromInt(data.len));

    // Loss should decrease (network is learning)
    if (final_loss >= initial_loss) {
        return error.TrainingDidNotConverge;
    }
}

/// Create SGD optimizer
pub fn createSgdOptimizer(momentum: f32) optimizer.Optimizer {
    return optimizer.Optimizer{ .sgd = optimizer.Sgd{ .momentum = momentum } };
}

/// Create Adam optimizer
pub fn createAdamOptimizer() optimizer.Optimizer {
    return optimizer.Optimizer{ .adam = optimizer.Adam{} };
}

/// Create RMSprop optimizer
pub fn createRmspropOptimizer() optimizer.Optimizer {
    return optimizer.Optimizer{ .rmsprop = optimizer.Rmsprop{} };
}

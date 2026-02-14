/// Comprehensive Feedforward Neural Network (FNN) example for ZigNeuron
/// Demonstrates various use cases including regression, classification, and advanced training techniques
const std = @import("std");
const zn = @import("ZigNeuron");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Get default backend (GPU if available, CPU fallback)
    const backend = zn.backend.Backend.default();
    std.debug.print("Using backend: ", .{});
    switch (backend) {
        .gpu => |gpu| switch (gpu) {
            .metal => std.debug.print("Metal (Apple Silicon GPU)\n", .{}),
            .vulkan => std.debug.print("Vulkan (Cross-platform GPU)\n", .{}),
        },
        .cpu => std.debug.print("CPU (fallback)\n", .{}),
    }

    const separator = "==============================";
    std.debug.print("\n{s}\n", .{separator});
    std.debug.print("  Comprehensive FNN Example for ZigNeuron\n", .{});
    std.debug.print("{s}\n\n", .{separator});

    // Part 1: Regression - Sinewave Approximation
    try part1SinewaveRegression(allocator, backend);

    // Part 2: Binary Classification - Spiral-like Dataset
    try part2BinaryClassification(allocator, backend);

    // Part 3: Multi-class Classification - Iris-like
    try part3MulticlassClassification(allocator, backend);

    // Part 4: Advanced Topics
    try part4AdvancedTopics(allocator, backend);

    std.debug.print("\nAll examples completed successfully!\n", .{});
}

/// Part 1: Sinewave Regression
/// Architecture: 1 -> 16 -> 16 -> 1 (ReLU hidden, linear output)
/// Loss: MSE
/// Optimizer: SGD
fn part1SinewaveRegression(allocator: std.mem.Allocator, backend: zn.backend.Backend) !void {
    const separator = "--------------------------------------------------";
    std.debug.print("\n{s}\n", .{separator});
    std.debug.print("  PART 1: Sinewave Regression (1->16->16->1)\n", .{});
    std.debug.print("{s}\n\n", .{separator});

    // Generate training data: y = sin(2*pi*x) with some noise
    const num_samples: usize = 100;
    var data = try allocator.alloc([]const f32, num_samples);
    var targets = try allocator.alloc([]const f32, num_samples);
    defer {
        for (data) |d| allocator.free(d);
        for (targets) |t| allocator.free(t);
    }

    const pi: f32 = 3.14159265359;
    for (0..num_samples) |i| {
        const x: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(num_samples - 1));
        const y = std.math.sin(2 * pi * x);
        // Add small noise
        const noisy_y = y + std.math.sin(2 * pi * x) * 0.05;

        const sample = try allocator.alloc(f32, 1);
        sample[0] = x;

        const target = try allocator.alloc(f32, 1);
        target[0] = noisy_y;

        data[i] = sample;
        targets[i] = target;
    }

    // Create network: 1 -> 16 -> 16 -> 1
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();

    _ = try network.addDense(1, 16, .tanh);
    _ = try network.addDense(16, 16, .tanh);
    _ = try network.addDense(16, 1, .tanh);  // Tanh output for better gradient flow

    std.debug.print("Network architecture: 1 -> 16 -> 16 -> 1\n", .{});
    std.debug.print("Loss function: MSE\n", .{});
    std.debug.print("Optimizer: SGD\n\n", .{});

    // Training with SGD
    const epochs: usize = 1000;
    const learning_rate: f32 = 0.1;

    std.debug.print("Training for {} epochs with learning rate {}...\n\n", .{ epochs, learning_rate });

    const loss_fn = zn.loss.Loss{ .mse = {} };

    // Train using the built-in train method
    try network.train(data, targets, epochs, learning_rate, loss_fn);

    std.debug.print("\nFinal training results:\n", .{});

    // Test on some samples
    for (0..5) |i| {
        const sample = data[i * 20];
        var output: [1]f32 = undefined;
        _ = try network.forward(sample, &output);

        const expected = targets[i * 20][0];
        const predicted = output[0];
        const err_val = if (predicted > expected) predicted - expected else expected - predicted;

        std.debug.print("  Input: {d:.2} -> Predicted: {d:.4}, Expected: {d:.4}, Error: {d:.4}\n", .{
            sample[0],
            predicted,
            expected,
            err_val,
        });
    }

    // Demonstrate single sample inference
    std.debug.print("\nSingle sample inference test:\n", .{});
    const test_input: []const f32 = &.{ 0.5 };
    var test_output: [1]f32 = undefined;
    _ = try network.forward(test_input, &test_output);
    std.debug.print("  Input: 0.5 -> Output: {d:.4}\n", .{test_output[0]});

    // Demonstrate batch inference
    std.debug.print("\nBatch inference test (5 samples):\n", .{});
    const batch_inputs = [_][]const f32{
        &.{ 0.0 }, &.{ 0.25 }, &.{ 0.5 }, &.{ 0.75 }, &.{ 1.0 },
    };
    var batch_outputs: [5][1]f32 = undefined;

    for (batch_inputs, 0..) |input, i| {
        _ = try network.forward(input, &batch_outputs[i]);
        std.debug.print("  Sample {}: Input: {d:.1} -> Output: {d:.4}\n", .{ i, input[0], batch_outputs[i][0] });
    }
}

/// Part 2: Binary Classification - Linearly Separable Dataset
/// Architecture: 2 -> 32 -> 16 -> 1 (ReLU hidden, sigmoid output)
/// Loss: Binary Cross-Entropy
/// Demonstrates: Classification with probability output, accuracy calculation
fn part2BinaryClassification(allocator: std.mem.Allocator, backend: zn.backend.Backend) !void {
    const separator = "--------------------------------------------------";
    std.debug.print("\n{s}\n", .{separator});
    std.debug.print("  PART 2: Binary Classification - Linearly Separable (2->32->16->1)\n", .{});
    std.debug.print("{s}\n\n", .{separator});

    // Generate simple linearly separable dataset
    // Class 0: points where x + y < 0
    // Class 1: points where x + y >= 0
    const num_samples: usize = 200;

    var data = try allocator.alloc([]const f32, num_samples);
    var targets = try allocator.alloc([]const f32, num_samples);
    defer {
        for (data) |d| allocator.free(d);
        for (targets) |t| allocator.free(t);
    }

    // Compute mean and std for normalization
    var sum_x1: f32 = 0;
    var sum_x2: f32 = 0;
    for (0..num_samples) |i| {
        sum_x1 += std.math.sin(@as(f32, @floatFromInt(i)) / 10) * 2;
        sum_x2 += std.math.cos(@as(f32, @floatFromInt(i)) / 10) * 2;
    }
    const mean_x1 = sum_x1 / @as(f32, @floatFromInt(num_samples));
    const mean_x2 = sum_x2 / @as(f32, @floatFromInt(num_samples));

    // Simple normalization: center around 0
    for (0..num_samples) |i| {
        const sample = try allocator.alloc(f32, 2);
        const target = try allocator.alloc(f32, 1);

        // Generate random-like data using sine/cosine
        var x1 = std.math.sin(@as(f32, @floatFromInt(i)) / 10) * 2;
        var x2 = std.math.cos(@as(f32, @floatFromInt(i)) / 10) * 2;

        // Normalize
        x1 -= mean_x1;
        x2 -= mean_x2;

        // Scale to [-1, 1] range
        x1 *= 0.5;
        x2 *= 0.5;

        // Simple decision boundary: x1 + x2 > 0
        sample[0] = x1;
        sample[1] = x2;

        // Binary target
        target[0] = if (x1 + x2 >= 0) 1.0 else 0.0;

        data[i] = sample;
        targets[i] = target;
    }

    // Create network: 2 -> 32 -> 16 -> 1 with sigmoid output
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();

    _ = try network.addDense(2, 32, .relu);
    _ = try network.addDense(32, 16, .relu);
    _ = try network.addDense(16, 1, .sigmoid);

    std.debug.print("Network architecture: 2 -> 32 -> 16 -> 1 (sigmoid output)\n", .{});
    std.debug.print("Loss function: Binary Cross-Entropy\n", .{});
    std.debug.print("Optimizer: SGD with momentum (0.9)\n\n", .{});

    // Training with SGD and momentum
    const epochs: usize = 500;
    const learning_rate: f32 = 0.001;

    const loss_fn = zn.loss.Loss{ .binary_cross_entropy = {} };

    std.debug.print("Training for {} epochs...\n\n", .{epochs});

    // Use the built-in train method for simplicity
    try network.train(data, targets, epochs, learning_rate, loss_fn);

    // Final evaluation
    std.debug.print("\nFinal Results:\n", .{});

    var final_correct: usize = 0;
    for (data, targets) |sample, target| {
        var output: [1]f32 = undefined;
        _ = try network.forward(sample, &output);
        if ((output[0] > 0.5) == (target[0] > 0.5)) {
            final_correct += 1;
        }
    }

    const final_accuracy = @as(f32, @floatFromInt(final_correct)) / @as(f32, @floatFromInt(num_samples));
    std.debug.print("  Final Accuracy: {d:.2}% ({}/{} samples)\n", .{ final_accuracy * 100, final_correct, num_samples });

    // Show probability outputs
    std.debug.print("\nSample probability outputs:\n", .{});
    for (0..5) |i| {
        var output: [1]f32 = undefined;
        _ = try network.forward(data[i], &output);
        std.debug.print("  Sample {}: Class {} -> Probability: {d:.4}\n", .{ i, @as(usize, @intFromFloat(targets[i][0])), output[0] });
    }
}

/// Part 3: Multi-class Classification - Iris-like
/// Architecture: 4 -> 20 -> 10 -> 3 (ReLU hidden, softmax output)
/// Loss: Cross-Entropy
fn part3MulticlassClassification(allocator: std.mem.Allocator, backend: zn.backend.Backend) !void {
    const separator = "--------------------------------------------------";
    std.debug.print("\n{s}\n", .{separator});
    std.debug.print("  PART 3: Multi-class Classification - Iris-like (4->20->10->3)\n", .{});
    std.debug.print("{s}\n\n", .{separator});

    // Generate Iris-like dataset
    // 3 classes, 4 features each
    const num_samples_per_class: usize = 50;
    const total_samples = num_samples_per_class * 3;

    var data = try allocator.alloc([]const f32, total_samples);
    var targets = try allocator.alloc([]const f32, total_samples);
    defer {
        for (data) |d| allocator.free(d);
        for (targets) |t| allocator.free(t);
    }

    for (0..total_samples) |i| {
        const sample = try allocator.alloc(f32, 4);
        const target = try allocator.alloc(f32, 3);  // One-hot encoding

        const class_id = i / num_samples_per_class;
        const sample_in_class = i % num_samples_per_class;

        // Generate features based on class
        const base_features = [_]f32{
            5.0 + @as(f32, @floatFromInt(class_id)) * 1.0,  // sepal length
            3.0 + @as(f32, @floatFromInt(class_id)) * 0.5,  // sepal width
            3.5 + @as(f32, @floatFromInt(class_id)) * 1.2,  // petal length
            1.0 + @as(f32, @floatFromInt(class_id)) * 0.8,  // petal width
        };

        // Normalize features to [0, 1] range
        const normalized = [_]f32{
            (base_features[0] - 5.0) / 2.0,   // Normalize to ~[0, 1]
            (base_features[1] - 3.0) / 1.5,   // Normalize to ~[0, 1]
            (base_features[2] - 3.5) / 3.0,   // Normalize to ~[0, 1]
            (base_features[3] - 1.0) / 2.0,   // Normalize to ~[0, 1]
        };

        // Add noise
        const noise = std.math.sin(@as(f32, @floatFromInt(sample_in_class))) * 0.2;
        for (0..4) |j| {
            sample[j] = normalized[j] + noise * 0.05;
        }

        // One-hot encoding
        @memset(target, 0.0);
        target[class_id] = 1.0;

        data[i] = sample;
        targets[i] = target;
    }

    // Create network: 4 -> 20 -> 10 -> 3 with softmax output
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();

    _ = try network.addDense(4, 10, .relu);
    _ = try network.addDense(10, 5, .relu);
    _ = try network.addDense(5, 3, .relu);  // Use ReLU for logits, apply softmax only during inference

    std.debug.print("Network architecture: 4 -> 10 -> 5 -> 3 (logits output)\n", .{});
    std.debug.print("Loss function: Cross-Entropy\n\n", .{});

    const softmax = zn.activation.Activation{ .softmax = {} };

    // Training
    const epochs: usize = 500;
    const learning_rate: f32 = 0.005;

    const loss_fn = zn.loss.Loss{ .cross_entropy = {} };

    std.debug.print("Training for {} epochs...\n\n", .{epochs});

    // Train using manual loop to apply softmax for accuracy calculation
    for (0..epochs) |epoch| {
        var total_loss: f32 = 0;
        var correct: usize = 0;

        for (data, targets) |sample, target| {
            const sample_loss = try network.trainStep(sample, target, learning_rate, loss_fn);
            total_loss += sample_loss;

            // Calculate accuracy
            var output: [3]f32 = undefined;
            _ = try network.forward(sample, &output);

            // Apply softmax for probability interpretation
            var probs: [3]f32 = undefined;
            try softmax.softmaxForward(&output, &probs);

            // Find predicted class (argmax)
            var predicted_class: usize = 0;
            var max_prob = probs[0];
            for (probs, 0..) |p, j| {
                if (p > max_prob) {
                    max_prob = p;
                    predicted_class = j;
                }
            }

            // Find true class
            var true_class: usize = 0;
            for (target, 0..) |t, j| {
                if (t > 0.5) {
                    true_class = j;
                    break;
                }
            }

            if (predicted_class == true_class) {
                correct += 1;
            }
        }

        const avg_loss = total_loss / @as(f32, @floatFromInt(data.len));
        const accuracy = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(data.len));

        if (epoch % 100 == 0) {
            std.debug.print("Epoch {}: Loss = {d:.4}, Accuracy = {d:.2}%\n", .{ epoch, avg_loss, accuracy * 100 });
        }
    }

    // Final evaluation with softmax probabilities
    std.debug.print("\nFinal Results with Softmax Probabilities:\n", .{});

    for (0..5) |i| {
        var output: [3]f32 = undefined;
        _ = try network.forward(data[i], &output);

        // Apply softmax for probabilities
        var probs: [3]f32 = undefined;
        try softmax.softmaxForward(&output, &probs);

        // Find predicted class
        var predicted_class: usize = 0;
        var max_prob = probs[0];
        for (probs, 0..) |p, j| {
            if (p > max_prob) {
                max_prob = p;
                predicted_class = j;
            }
        }

        // Find true class
        var true_class: usize = 0;
        for (targets[i], 0..) |t, j| {
            if (t > 0.5) {
                true_class = j;
                break;
            }
        }

        std.debug.print("  Sample {}: True class = {}\n", .{ i, true_class });
        std.debug.print("    Probabilities: [{d:.4}, {d:.4}, {d:.4}]\n", .{ probs[0], probs[1], probs[2] });
        std.debug.print("    Predicted class: {} (confidence: {d:.2}%)\n", .{ predicted_class, max_prob * 100 });

        if (predicted_class == true_class) {
            std.debug.print("    Result: CORRECT\n", .{});
        } else {
            std.debug.print("    Result: WRONG\n", .{});
        }
    }
}

/// Part 4: Advanced Topics
/// - Learning rate scheduling
/// - Early stopping
/// - Training/validation split
fn part4AdvancedTopics(allocator: std.mem.Allocator, backend: zn.backend.Backend) !void {
    const separator = "--------------------------------------------------";
    std.debug.print("\n{s}\n", .{separator});
    std.debug.print("  PART 4: Advanced Training Topics\n", .{});
    std.debug.print("{s}\n\n", .{separator});

    // Generate simple linearly separable data
    const num_samples: usize = 200;
    const train_ratio: f32 = 0.8;
    const num_train = @as(usize, @intFromFloat(@as(f32, @floatFromInt(num_samples)) * train_ratio));
    const num_val = num_samples - num_train;

    var all_data = try allocator.alloc([]const f32, num_samples);
    var all_targets = try allocator.alloc([]const f32, num_samples);
    defer {
        for (all_data) |d| allocator.free(d);
        for (all_targets) |t| allocator.free(t);
    }

    for (0..num_samples) |i| {
        const sample = try allocator.alloc(f32, 2);
        const target = try allocator.alloc(f32, 1);

        const x1: f32 = std.math.sin(@as(f32, @floatFromInt(i)) / 20);
        const x2: f32 = std.math.cos(@as(f32, @floatFromInt(i)) / 20);

        // Simple decision boundary: x1 + x2 > 0
        sample[0] = x1;
        sample[1] = x2;
        target[0] = if (x1 + x2 > 0) 1.0 else 0.0;

        all_data[i] = sample;
        all_targets[i] = target;
    }

    // Split into training and validation sets
    const training_data = all_data[0..num_train];
    const training_targets = all_targets[0..num_train];
    const validation_data = all_data[num_train..];
    const validation_targets = all_targets[num_train..];

    std.debug.print("Training set: {} samples\n", .{num_train});
    std.debug.print("Validation set: {} samples\n", .{num_val});
    std.debug.print("\n", .{});

    // Create network
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();

    _ = try network.addDense(2, 16, .relu);
    _ = try network.addDense(16, 8, .relu);
    _ = try network.addDense(8, 1, .sigmoid);

    std.debug.print("Network architecture: 2 -> 16 -> 8 -> 1\n", .{});
    std.debug.print("Loss function: Binary Cross-Entropy\n\n", .{});

    // Learning rate scheduling parameters
    const base_lr: f32 = 0.05;
    const min_lr: f32 = 0.001;
    const lr_decay: f32 = 0.98;
    const max_epochs: usize = 500;

    // Early stopping parameters
    const patience: usize = 30;
    var best_val_loss: f32 = std.math.inf(f32);
    var epochs_without_improvement: usize = 0;

    const loss_fn = zn.loss.Loss{ .binary_cross_entropy = {} };

    std.debug.print("Training with Learning Rate Scheduling and Early Stopping\n", .{});
    std.debug.print("Initial LR: {d}, Min LR: {d}, Decay: {d}\n", .{ base_lr, min_lr, lr_decay });
    std.debug.print("Early stopping patience: {d} epochs\n\n", .{patience});

    for (0..max_epochs) |epoch| {
        // Learning rate scheduling (exponential decay)
        const decayed_lr = base_lr * std.math.pow(f32, lr_decay, @as(f32, @floatFromInt(epoch)));
        const current_lr = if (decayed_lr > min_lr) decayed_lr else min_lr;

        // ===== Training Phase =====
        var train_correct: usize = 0;
        var train_loss: f32 = 0;

        for (training_data, training_targets) |sample, target| {
            train_loss += try network.trainStep(sample, target, current_lr, loss_fn);

            var output: [1]f32 = undefined;
            _ = try network.forward(sample, &output);

            if ((output[0] > 0.5) == (target[0] > 0.5)) {
                train_correct += 1;
            }
        }

        const train_accuracy = @as(f32, @floatFromInt(train_correct)) / @as(f32, @floatFromInt(num_train));
        const avg_train_loss = train_loss / @as(f32, @floatFromInt(num_train));

        // ===== Validation Phase =====
        var val_correct: usize = 0;
        var val_loss: f32 = 0;

        for (validation_data, validation_targets) |sample, target| {
            var output: [1]f32 = undefined;
            _ = try network.forward(sample, &output);

            val_loss += try loss_fn.forward(&output, target);

            if ((output[0] > 0.5) == (target[0] > 0.5)) {
                val_correct += 1;
            }
        }

        const val_accuracy = @as(f32, @floatFromInt(val_correct)) / @as(f32, @floatFromInt(num_val));
        const avg_val_loss = val_loss / @as(f32, @floatFromInt(num_val));

        // Print progress
        if (epoch % 50 == 0) {
            std.debug.print("Epoch {}: LR={d:.4}, Train Loss={d:.4}, Train Acc={d:.2}%, Val Loss={d:.4}, Val Acc={d:.2}%\n", .{
                epoch,
                current_lr,
                avg_train_loss,
                train_accuracy * 100,
                avg_val_loss,
                val_accuracy * 100,
            });
        }

        // Early stopping check
        if (avg_val_loss < best_val_loss) {
            best_val_loss = avg_val_loss;
            epochs_without_improvement = 0;
            // Note: In a real scenario, you'd save the best model here
        } else {
            epochs_without_improvement += 1;
        }

        if (epochs_without_improvement >= patience) {
            std.debug.print("\nEarly stopping triggered at epoch {}!\n", .{epoch});
            std.debug.print("Best validation loss: {d:.4}\n", .{best_val_loss});
            break;
        }
    }

    // Final evaluation
    const separator30 = "------------------------------";
    std.debug.print("\n{s}\n", .{separator30});
    std.debug.print("Final Results\n", .{});
    std.debug.print("{s}\n\n", .{separator30});

    // Training set evaluation
    var train_correct_final: usize = 0;
    for (training_data, training_targets) |sample, target| {
        var output: [1]f32 = undefined;
        _ = try network.forward(sample, &output);
        if ((output[0] > 0.5) == (target[0] > 0.5)) {
            train_correct_final += 1;
        }
    }
    const train_final_acc = @as(f32, @floatFromInt(train_correct_final)) / @as(f32, @floatFromInt(num_train));

    // Validation set evaluation
    var val_correct_final: usize = 0;
    for (validation_data, validation_targets) |sample, target| {
        var output: [1]f32 = undefined;
        _ = try network.forward(sample, &output);
        if ((output[0] > 0.5) == (target[0] > 0.5)) {
            val_correct_final += 1;
        }
    }
    const val_final_acc = @as(f32, @floatFromInt(val_correct_final)) / @as(f32, @floatFromInt(num_val));

    std.debug.print("Training Accuracy: {d:.2}%\n", .{train_final_acc * 100});
    std.debug.print("Validation Accuracy: {d:.2}%\n", .{val_final_acc * 100});

    // Check for overfitting
    const acc_diff = train_final_acc - val_final_acc;
    if (acc_diff > 0.1) {
        std.debug.print("\nNote: Potential overfitting detected (train-val gap: {d:.2}%)\n", .{acc_diff * 100});
    } else {
        std.debug.print("\nGood generalization (train-val gap: {d:.2}%)\n", .{acc_diff * 100});
    }
}

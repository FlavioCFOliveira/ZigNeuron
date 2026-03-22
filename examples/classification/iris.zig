/// Iris Flower Classification Example
///
/// Dataset: Fisher's Iris Dataset (150 samples, 3 classes, 4 features)
/// Classes: Setosa, Versicolor, Virginica
/// Features: sepal length, sepal width, petal length, petal width
///
/// Reference: Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems.
///   Annals of Eugenics, 7(2), 179-188.
const std = @import("std");
const zn = @import("ZigNeuron");

// Iris dataset (150 samples, 4 features)
// Features: [sepal_length, sepal_width, petal_length, petal_width]
const iris_data = [150][4]f32{
    // Class 0: Setosa (samples 0-49)
    .{ 5.1, 3.5, 1.4, 0.2 }, .{ 4.9, 3.0, 1.4, 0.2 }, .{ 4.7, 3.2, 1.3, 0.2 },
    .{ 4.6, 3.1, 1.5, 0.2 }, .{ 5.0, 3.6, 1.4, 0.2 }, .{ 5.4, 3.9, 1.7, 0.4 },
    .{ 4.6, 3.4, 1.4, 0.3 }, .{ 5.0, 3.4, 1.5, 0.2 }, .{ 4.4, 2.9, 1.4, 0.2 },
    .{ 4.9, 3.1, 1.5, 0.1 }, .{ 5.4, 3.7, 1.5, 0.2 }, .{ 4.8, 3.4, 1.6, 0.2 },
    .{ 4.8, 3.0, 1.4, 0.1 }, .{ 4.3, 3.0, 1.1, 0.1 }, .{ 5.8, 4.0, 1.2, 0.2 },
    .{ 5.7, 4.4, 1.5, 0.4 }, .{ 5.4, 3.9, 1.3, 0.4 }, .{ 5.1, 3.5, 1.4, 0.3 },
    .{ 5.7, 3.8, 1.7, 0.3 }, .{ 5.1, 3.8, 1.5, 0.3 }, .{ 5.4, 3.4, 1.7, 0.2 },
    .{ 5.1, 3.7, 1.5, 0.4 }, .{ 4.6, 3.6, 1.0, 0.2 }, .{ 5.1, 3.3, 1.7, 0.5 },
    .{ 4.8, 3.4, 1.9, 0.2 }, .{ 5.0, 3.0, 1.6, 0.2 }, .{ 5.0, 3.4, 1.6, 0.4 },
    .{ 5.2, 3.5, 1.5, 0.2 }, .{ 5.2, 3.4, 1.4, 0.2 }, .{ 4.7, 3.2, 1.6, 0.2 },
    .{ 4.8, 3.1, 1.6, 0.2 }, .{ 5.4, 3.4, 1.5, 0.4 }, .{ 5.2, 4.1, 1.5, 0.1 },
    .{ 5.5, 4.2, 1.4, 0.2 }, .{ 4.9, 3.1, 1.5, 0.2 }, .{ 5.0, 3.2, 1.2, 0.2 },
    .{ 5.5, 3.5, 1.3, 0.2 }, .{ 4.9, 3.6, 1.4, 0.1 }, .{ 4.4, 3.0, 1.3, 0.2 },
    .{ 5.1, 3.4, 1.5, 0.2 }, .{ 5.0, 3.5, 1.3, 0.3 }, .{ 4.5, 2.3, 1.3, 0.3 },
    .{ 4.4, 3.2, 1.3, 0.2 }, .{ 5.0, 3.5, 1.6, 0.6 }, .{ 5.1, 3.8, 1.9, 0.4 },
    .{ 4.8, 3.0, 1.4, 0.3 }, .{ 5.1, 3.8, 1.6, 0.2 }, .{ 4.6, 3.2, 1.4, 0.2 },
    .{ 5.3, 3.7, 1.5, 0.2 }, .{ 5.0, 3.3, 1.4, 0.2 },
    // Class 1: Versicolor (samples 50-99)
    .{ 7.0, 3.2, 4.7, 1.4 }, .{ 6.4, 3.2, 4.5, 1.5 }, .{ 6.9, 3.1, 4.9, 1.5 },
    .{ 5.5, 2.3, 4.0, 1.3 }, .{ 6.5, 2.8, 4.6, 1.5 }, .{ 5.7, 2.8, 4.5, 1.3 },
    .{ 6.3, 3.3, 4.7, 1.6 }, .{ 4.9, 2.4, 3.3, 1.0 }, .{ 6.6, 2.9, 4.6, 1.3 },
    .{ 5.2, 2.7, 3.9, 1.4 }, .{ 5.0, 2.0, 3.5, 1.0 }, .{ 5.9, 3.0, 4.2, 1.5 },
    .{ 6.0, 2.2, 4.0, 1.0 }, .{ 6.1, 2.9, 4.7, 1.4 }, .{ 5.6, 2.9, 3.6, 1.3 },
    .{ 6.7, 3.1, 4.4, 1.4 }, .{ 5.6, 3.0, 4.5, 1.5 }, .{ 5.8, 2.7, 4.1, 1.0 },
    .{ 6.2, 2.2, 4.5, 1.5 }, .{ 5.6, 2.5, 3.9, 1.1 }, .{ 5.9, 3.2, 4.8, 1.8 },
    .{ 6.1, 2.8, 4.0, 1.3 }, .{ 6.3, 2.5, 4.9, 1.5 }, .{ 6.1, 2.8, 4.7, 1.2 },
    .{ 6.4, 2.9, 4.3, 1.3 }, .{ 6.6, 3.0, 4.4, 1.4 }, .{ 6.8, 2.8, 4.8, 1.4 },
    .{ 6.7, 3.0, 5.0, 1.7 }, .{ 6.0, 2.9, 4.5, 1.5 }, .{ 5.7, 2.6, 3.5, 1.0 },
    .{ 5.5, 2.4, 3.8, 1.1 }, .{ 5.5, 2.4, 3.7, 1.0 }, .{ 5.8, 2.7, 3.9, 1.2 },
    .{ 6.0, 2.7, 5.1, 1.6 }, .{ 5.4, 3.0, 4.5, 1.5 }, .{ 6.0, 3.4, 4.5, 1.6 },
    .{ 6.7, 3.1, 4.7, 1.5 }, .{ 6.3, 2.3, 4.4, 1.3 }, .{ 5.6, 3.0, 4.1, 1.3 },
    .{ 5.5, 2.5, 4.0, 1.3 }, .{ 5.5, 2.6, 4.4, 1.2 }, .{ 6.1, 3.0, 4.6, 1.4 },
    .{ 5.8, 2.6, 4.0, 1.2 }, .{ 5.0, 2.3, 3.3, 1.0 }, .{ 5.6, 2.7, 4.2, 1.3 },
    .{ 5.7, 3.0, 4.2, 1.2 }, .{ 5.7, 2.9, 4.2, 1.3 }, .{ 6.2, 2.9, 4.3, 1.3 },
    .{ 5.1, 2.5, 3.0, 1.1 }, .{ 5.7, 2.8, 4.1, 1.3 },
    // Class 2: Virginica (samples 100-149)
    .{ 6.3, 3.3, 6.0, 2.5 }, .{ 5.8, 2.7, 5.1, 1.9 }, .{ 7.1, 3.0, 5.9, 2.1 },
    .{ 6.3, 2.9, 5.6, 1.8 }, .{ 6.5, 3.0, 5.8, 2.2 }, .{ 7.6, 3.0, 6.6, 2.1 },
    .{ 4.9, 2.5, 4.5, 1.7 }, .{ 7.3, 2.9, 6.3, 1.8 }, .{ 6.7, 2.5, 5.8, 1.8 },
    .{ 7.2, 3.6, 6.1, 2.5 }, .{ 6.5, 3.2, 5.1, 2.0 }, .{ 6.4, 2.7, 5.3, 1.9 },
    .{ 6.8, 3.0, 5.5, 2.1 }, .{ 5.7, 2.5, 5.0, 2.0 }, .{ 5.8, 2.8, 5.1, 2.4 },
    .{ 6.4, 3.2, 5.3, 2.3 }, .{ 6.5, 3.0, 5.5, 1.8 }, .{ 7.7, 3.8, 6.7, 2.2 },
    .{ 7.7, 2.6, 6.9, 2.3 }, .{ 6.0, 2.2, 5.0, 1.5 }, .{ 6.9, 3.2, 5.7, 2.3 },
    .{ 5.6, 2.8, 4.9, 2.0 }, .{ 7.7, 2.8, 6.7, 2.0 }, .{ 6.3, 2.7, 4.9, 1.8 },
    .{ 6.7, 3.3, 5.7, 2.1 }, .{ 7.2, 3.2, 6.0, 1.8 }, .{ 6.2, 2.8, 4.8, 1.8 },
    .{ 6.1, 3.0, 4.9, 1.8 }, .{ 6.4, 2.8, 5.6, 2.1 }, .{ 7.2, 3.0, 5.8, 1.6 },
    .{ 7.4, 2.8, 6.1, 1.9 }, .{ 7.9, 3.8, 6.4, 2.0 }, .{ 6.4, 2.8, 5.6, 2.2 },
    .{ 6.3, 2.8, 5.1, 1.5 }, .{ 6.1, 2.6, 5.6, 1.4 }, .{ 7.7, 3.0, 6.1, 2.3 },
    .{ 6.3, 3.4, 5.6, 2.4 }, .{ 6.4, 3.1, 5.5, 1.8 }, .{ 6.0, 3.0, 4.8, 1.8 },
    .{ 6.9, 3.1, 5.4, 2.1 }, .{ 6.7, 3.1, 5.6, 2.4 }, .{ 6.9, 3.1, 5.1, 2.3 },
    .{ 5.8, 2.7, 5.1, 1.9 }, .{ 6.8, 3.2, 5.9, 2.3 }, .{ 6.7, 3.3, 5.7, 2.5 },
    .{ 6.7, 3.0, 5.2, 2.3 }, .{ 6.3, 2.5, 5.0, 1.9 }, .{ 6.5, 3.0, 5.2, 2.0 },
    .{ 6.2, 3.4, 5.4, 2.3 }, .{ 5.9, 3.0, 5.1, 1.8 },
};

// One-hot encoded labels (3 classes)
const iris_labels = [150][3]f32{
    // Class 0: Setosa (50 samples)
    .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 },
    .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 },
    .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 },
    .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 },
    .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 },
    .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 },
    .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 },
    .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 },
    .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 },
    .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 0, 0 },
    // Class 1: Versicolor (50 samples)
    .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 },
    .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 },
    .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 },
    .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 },
    .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 },
    .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 },
    .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 },
    .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 },
    .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 },
    .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 }, .{ 0, 1, 0 },
    // Class 2: Virginica (50 samples)
    .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 },
    .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 },
    .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 },
    .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 },
    .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 },
    .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 },
    .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 },
    .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 },
    .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 },
    .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 }, .{ 0, 0, 1 },
};

/// Normalize features to zero mean and unit variance
fn normalizeFeatures(data: []const [4]f32, normalized: [][4]f32) void {
    // Calculate mean for each feature
    var mean: [4]f32 = .{ 0, 0, 0, 0 };
    for (data) |sample| {
        for (sample, 0..) |value, i| {
            mean[i] += value;
        }
    }
    for (&mean) |*m| m.* /= @as(f32, @floatFromInt(data.len));

    // Calculate std for each feature
    var std_dev: [4]f32 = .{ 0, 0, 0, 0 };
    for (data) |sample| {
        for (sample, 0..) |value, i| {
            const diff = value - mean[i];
            std_dev[i] += diff * diff;
        }
    }
    for (&std_dev) |*s| {
        s.* = @sqrt(s.* / @as(f32, @floatFromInt(data.len)));
        if (s.* < 1e-8) s.* = 1.0; // Prevent division by zero
    }

    // Normalize
    for (data, normalized) |sample, *norm| {
        for (sample, 0..) |value, i| {
            norm.*[i] = (value - mean[i]) / std_dev[i];
        }
    }
}

/// Shuffle dataset
fn shuffleDataset(_allocator: std.mem.Allocator, data: []const [4]f32, _labels: []const [3]f32, indices: []usize) !void {
    _ = _allocator;
    _ = _labels;
    const io = std.Io.Threaded.global_single_threaded.io();
    const timestamp = std.Io.Clock.now(.real, io);
    var prng = std.Random.DefaultPrng.init(@intCast(timestamp.nanoseconds));
    const random = prng.random();

    for (0..data.len) |i| {
        indices[i] = i;
    }

    // Fisher-Yates shuffle
    var i = data.len;
    while (i > 1) {
        i -= 1;
        const j = random.int(usize) % (i + 1);
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    std.debug.print("\n=== Iris Flower Classification ===\n", .{});
    std.debug.print("Dataset: 150 samples, 4 features, 3 classes\n", .{});
    std.debug.print("Classes: Setosa, Versicolor, Virginica\n\n", .{});

    // Normalize features
    var normalized_data: [150][4]f32 = undefined;
    normalizeFeatures(&iris_data, &normalized_data);

    // Shuffle dataset
    var indices: [150]usize = undefined;
    try shuffleDataset(allocator, &normalized_data, &iris_labels, &indices);

    // Split: 80% train (120), 20% test (30)
    const train_size = 120;
    const test_size = 30;

    // Prepare training data
    var train_data: [120][]const f32 = undefined;
    var train_labels: [120][]const f32 = undefined;
    for (0..train_size) |i| {
        train_data[i] = &normalized_data[indices[i]];
        train_labels[i] = &iris_labels[indices[i]];
    }

    // Prepare test data
    var test_data: [30][]const f32 = undefined;
    var test_labels: [30][]const f32 = undefined;
    for (0..test_size) |i| {
        test_data[i] = &normalized_data[indices[train_size + i]];
        test_labels[i] = &iris_labels[indices[train_size + i]];
    }

    // Create backend and network
    const backend = try zn.backend.Backend.init(allocator);
    var net = try zn.network.Network.init(allocator, backend);
    defer net.deinit();

    // Build network: 4 -> 16 -> 8 -> 3
    // Input: 4 features
    // Hidden: 16 neurons with ReLU
    // Hidden: 8 neurons with ReLU
    // Output: 3 neurons with softmax (for 3 classes)
    _ = try net.addDense(4, 16, .relu);
    _ = try net.addDense(16, 8, .relu);
    _ = try net.addDense(8, 3, .linear); // Linear + cross_entropy_logits = softmax cross-entropy

    // Training
    std.debug.print("Training...\n", .{});
    const epochs = 200;
    const learning_rate = 0.01;
    const loss_fn = zn.loss.Loss{ .cross_entropy_logits = {} };

    try net.train(&train_data,&train_labels, epochs, learning_rate, loss_fn, null, null);

    // Evaluation with metrics module
    std.debug.print("\n=== Evaluation ===\n", .{});

    // Collect predictions and targets for metrics
    var outputs: [30][3]f32 = undefined;
    var output_slices: [30][]const f32 = undefined;
    for (test_data, 0..) |input, i| {
        _ = try net.forward(input, &outputs[i]);
        output_slices[i] = &outputs[i];
    }

    // Compute metrics
    var metrics = try zn.metrics.evaluateClassification(
        allocator,
        &output_slices,
        &test_labels,
        3, // num_classes
    );
    defer metrics.deinit(allocator);

    // Print metrics
    metrics.print();
    zn.metrics.printPerClassMetrics(metrics.confusion_matrix);

    const class_names = [3][]const u8{ "Setosa", "Versicolor", "Virginica" };
    std.debug.print("\nPer-class Accuracy:\n", .{});
    for (0..3) |c| {
        var class_total: usize = 0;
        var class_correct: usize = 0;
        for (metrics.confusion_matrix[c], 0..) |val, i| {
            class_total += val;
            if (i == c) class_correct = val;
        }
        const class_acc = if (class_total > 0)
            @as(f32, @floatFromInt(class_correct)) / @as(f32, @floatFromInt(class_total)) * 100
        else
            0;
        std.debug.print("  {s}: {d:.1}% ({}/{} correct)\n", .{ class_names[c], class_acc, class_correct, class_total });
    }
}

/// MNIST Handwritten Digit Classification Example
///
/// Dataset: MNIST (70,000 samples, 28x28 grayscale images, 10 classes)
/// Classes: Digits 0-9
///
/// This example uses a subset of the MNIST dataset for demonstration.
/// For the full dataset, download from: https://yann.lecun.com/exdb/mnist/
///
/// Reference: LeCun, Y., et al. (1998). Gradient-based learning applied to
///   document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
const std = @import("std");
const zn = @import("ZigNeuron");

// Small MNIST subset for demonstration (100 samples, first 10 of each digit)
// In practice, you would load the full dataset from binary files
const mnist_subset = [100][784]f32{
    // Digit 0 (samples 0-9)
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    // Digit 1 (samples 10-19)
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    .{0} ** 784,
    // ... (placeholders for other digits)
};

// One-hot labels for subset
const mnist_labels = [100][10]f32{
    // Digit 0
    .{1} ++ .{0} ** 9, .{1} ++ .{0} ** 9, .{1} ++ .{0} ** 9,
    .{1} ++ .{0} ** 9, .{1} ++ .{0} ** 9, .{1} ++ .{0} ** 9,
    .{1} ++ .{0} ** 9, .{1} ++ .{0} ** 9, .{1} ++ .{0} ** 9,
    .{1} ++ .{0} ** 9,
    // Digit 1
    .{0, 1} ++ .{0} ** 8, .{0, 1} ++ .{0} ** 8, .{0, 1} ++ .{0} ** 8,
    .{0, 1} ++ .{0} ** 8, .{0, 1} ++ .{0} ** 8, .{0, 1} ++ .{0} ** 8,
    .{0, 1} ++ .{0} ** 8, .{0, 1} ++ .{0} ** 8, .{0, 1} ++ .{0} ** 8,
    .{0, 1} ++ .{0} ** 8,
    // ... (placeholders for other digits)
} ++ .{.{0} ** 10} ** 80; // Remaining samples

/// Load MNIST dataset from binary files
/// Download from: https://yann.lecun.com/exdb/mnist/
/// Files: train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz
///        t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz
fn loadMnistDataset(allocator: std.mem.Allocator, image_path: []const u8, label_path: []const u8) !struct {
    images: [][784]f32,
    labels: [][10]f32,
} {
    _ = allocator;
    _ = image_path;
    _ = label_path;
    // TODO: Implement MNIST binary file loading
    // See: http://yann.lecun.com/exdb/mnist/
    @panic("MNIST dataset loading not implemented. Please download the dataset from https://yann.lecun.com/exdb/mnist/");
}

/// Normalize pixel values to [0, 1]
fn normalizeImages(images: [][784]f32) void {
    for (images) |*img| {
        for (img) |*pixel| {
            pixel.* /= 255.0;
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    std.debug.print("\n=== MNIST Digit Classification ===\n", .{});
    std.debug.print("Dataset: 28x28 grayscale images, 10 classes (digits 0-9)\n", .{});
    std.debug.print("Note: Using demo mode with synthetic data.\n", .{});
    std.debug.print("For real results, download MNIST from https://yann.lecun.com/exdb/mnist/\n\n", .{});

    // In a real implementation, you would:
    // 1. Download MNIST dataset
    // 2. Load with loadMnistDataset()
    // 3. Normalize with normalizeImages()
    // 4. Split into train/test sets
    // 5. Train the network

    // For now, just show the network architecture that would be used
    std.debug.print("Recommended network architecture for MNIST:\n", .{});
    std.debug.print("  Input: 784 (28x28)\n", .{});
    std.debug.print("  Dense: 784 -> 128 (ReLU)\n", .{});
    std.debug.print("  Dropout: 0.2\n", .{});
    std.debug.print("  Dense: 128 -> 64 (ReLU)\n", .{});
    std.debug.print("  Dense: 64 -> 10 (Linear + Softmax Cross-Entropy)\n", .{});
    std.debug.print("\nExpected accuracy: ~97-98% with proper training\n", .{});

    // Create backend and network
    const backend = try zn.backend.Backend.init(allocator);
    var net = try zn.network.Network.init(allocator, backend);
    defer net.deinit();

    // Build network
    _ = try net.addDense(784, 128, .relu);
    _ = try net.addDropout(128, 0.2);
    _ = try net.addDense(128, 64, .relu);
    _ = try net.addDense(64, 10, .linear);

    std.debug.print("\nNetwork created successfully!\n", .{});
    std.debug.print("To train with real data, implement loadMnistDataset() above.\n", .{});
}

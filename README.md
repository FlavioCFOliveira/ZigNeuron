# ZigNeuron

A high-performance neural network library written in Zig with GPU acceleration support.

## Features

✨ **GPU Acceleration**
- Native Metal support for Apple Silicon
- Automatic CPU fallback
- Optimized thresholds for maximum GPU utilization

🚀 **Performance Optimizations**
- Reduced GPU activation threshold (64 elements)
- Cache-optimized matrix multiplication with loop tiling
- Pre-allocated work buffers to minimize allocation overhead
- Batch processing infrastructure

🎯 **Modern ML Features**
- Multiple activation functions (ReLU, Sigmoid, Tanh, Linear, Softmax)
- Loss functions: MSE, Binary Cross-Entropy, **Cross-Entropy with Logits**
- He/Xavier weight initialization
- Gradient clipping for training stability

📊 **Examples**
- XOR problem solving
- Sinewave regression
- Binary classification
- **Multi-class classification (Iris-style dataset)**
- Advanced training with learning rate scheduling and early stopping

## Quick Start

### Building

```bash
zig build
```

### Running Examples

```bash
# Run all comprehensive examples
zig build -Dexamples fnn

# Run specific example
zig build -Dexamples
./zig-out/bin/fnn_comprehensive
```

### Testing

```bash
zig build test
```

## GPU Acceleration

ZigNeuron automatically detects and uses the best available backend:

1. **Metal** (macOS with Apple Silicon) - Highest priority
2. **CPU** (Fallback)

GPU is activated for operations with:
- Matrix multiplication: ≥512 elements
- Activations/Loss: ≥64 elements

See [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md) for detailed information.

## Example Usage

```zig
const std = @import("std");
const zn = @import("ZigNeuron");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    // Auto-select best backend (Metal > CPU)
    const backend = zn.backend.Backend.default();
    const network = try zn.network.Network.init(allocator, backend);
    defer network.deinit();
    
    // Build network: 4 -> 20 -> 10 -> 3
    _ = try network.addDense(4, 20, .relu);
    _ = try network.addDense(20, 10, .relu);
    _ = try network.addDense(10, 3, .linear);  // Linear for logits
    
    // Train with cross-entropy (numerically stable)
    const loss_fn = zn.loss.Loss{ .cross_entropy_logits = {} };
    try network.train(data, targets, 1000, 0.005, loss_fn);
}
```

## Performance

**Apple Silicon M-series** (Metal GPU):
- Multi-class classification: ~10.5 seconds for 1000 epochs
- Accuracy: 96% on Iris-style dataset
- GPU utilization optimized with reduced thresholds

**CPU Fallback**:
- Cache-optimized with loop tiling
- Expected ~35-50% slower than GPU

## Documentation

- [GPU Optimizations Guide](GPU_OPTIMIZATIONS.md) - Detailed GPU acceleration docs
- [Benchmark Results](BenchmarkTests.md) - Performance comparisons
- [Unit Tests](UnitTests.md) - Test coverage details

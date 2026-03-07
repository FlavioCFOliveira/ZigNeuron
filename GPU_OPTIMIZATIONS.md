# GPU Optimizations in ZigNeuron

This document describes the GPU optimization strategies used in ZigNeuron for high-performance neural network training and inference.

## Architecture Overview

ZigNeuron uses a tiered backend architecture with automatic device selection:

1. **Metal** (Apple Silicon) - Primary GPU backend on macOS
2. **CPU** (Fallback) - Optimized SIMD implementation, always available

### Platform Support

| Platform | GPU Backend | CPU Fallback |
|----------|-------------|--------------|
| macOS (Apple Silicon) | Metal | Yes |
| macOS (Intel) | CPU only | Yes |
| Linux | CPU only | Yes |
| Windows | CPU only | Yes |

Metal is the exclusive GPU backend, optimized specifically for Apple Silicon's unified memory architecture. CPU fallback uses SIMD vectorization (NEON on ARM, SSE/AVX on x86) for competitive performance on smaller workloads.

## Metal Optimizations

### Unified Memory Architecture

On Apple Silicon, ZigNeuron uses **Shared Storage Mode** for Metal buffers:
- No CPU-GPU data transfers required
- Zero-copy memory access
- Automatic cache coherency

```zig
// Buffer allocation in shared mode
var buffer = try ctx.allocBuffer(byte_length, .StorageModeShared);
```

### Command Batching

Multiple GPU operations are batched into single command buffers:
- Reduces kernel launch overhead
- Better GPU utilization
- Automatic synchronization at batch end

```zig
try backend.beginCommandBatch();
// ... multiple operations ...
try backend.endCommandBatch();
```

### Threadgroup Optimization

Threadgroup sizes are optimized for Apple GPU SIMD width (32 threads):
- Uses multiples of 32 for occupancy
- Typical configuration: 8x4x1 = 32 threads

```zig
// Optimal threadgroup dispatch
encoder.dispatchThreads(
    MTLSize.make(width, height, depth),
    MTLSize.make(8, 4, 1)  // 32 threads
);
```

### Buffer Pooling

Metal buffers are pooled to reduce allocation overhead:
- Power-of-2 bucketing
- Lazy deallocation
- Reuse across operations

## CPU Fallback Optimizations

### SIMD Vectorization

CPU operations use architecture-specific SIMD:
- **Apple Silicon (ARM64)**: NEON (4 floats per vector)
- **x86-64**: SSE/AVX (4-8 floats per vector)
- Applies to: ReLU, Sigmoid, Tanh activations, element-wise operations, gradient computations

```zig
const Vec4 = @Vector(4, f32);
// Process 4 elements at once
```

### Cache-Friendly Matrix Multiplication

Blocked matrix multiplication for cache efficiency:
- 32x32 blocks for L1 cache
- Accumulate in registers
- Loop tiling optimization

```zig
const block_size = 32;
// Blocked matmul implementation
```

## Future CUDA Support

### Roadmap

CUDA support is planned for Linux and Windows platforms to enable GPU acceleration on NVIDIA hardware:

| Phase | Target | Status |
|-------|--------|--------|
| 1 | CUDA kernel development (matmul, activations) | Planned |
| 2 | cuDNN integration for optimized primitives | Planned |
| 3 | Unified backend API (Metal/CUDA/CPU) | Planned |
| 4 | Multi-GPU support | Planned |

### Implementation Strategy

The CUDA backend will follow the same design principles as Metal:
- **Unified memory** on supported hardware (Pascal+)
- **Stream-based execution** for asynchronous operations
- **Kernel fusion** to minimize launch overhead
- **Tensor Core utilization** for matrix operations on Volta+

### API Compatibility

When CUDA support is added, the existing API will remain unchanged:

```zig
// Current API - will automatically use Metal on macOS, CUDA on Linux/Windows when available
var backend = try Backend.init(.auto);

// Explicit selection will support .cuda in the future
var backend = try Backend.init(.cuda);  // Future API
```

## Performance Tips

### When to Use GPU

| Scenario | Recommendation |
|----------|----------------|
| Small networks (< 64 neurons) | CPU may be faster |
| Large networks (> 256 neurons) | Use GPU |
| Batch size = 1 | CPU likely faster |
| Batch size > 16 | Use GPU |
| Recurrent layers | GPU recommended |
| Conv1D/Conv2D layers | GPU recommended |

### Memory Layout

- Use contiguous arrays when possible
- Align data to 16-byte boundaries
- Batch operations to amortize overhead

### Synchronization

- Minimize CPU-GPU synchronization points
- Use `beginCommandBatch()` / `endCommandBatch()`
- Let Metal handle implicit synchronization

## Profiling

Use the built-in performance benchmarks:

```bash
zig build test_performance
./zig-out/bin/test_performance
```

## Further Reading

- [Metal Performance Shaders Documentation](https://developer.apple.com/documentation/metalperformanceshaders)
- [Apple Silicon Memory Architecture](https://developer.apple.com/documentation/apple-silicon)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/) (future)

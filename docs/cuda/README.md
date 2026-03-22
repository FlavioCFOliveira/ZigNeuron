# CUDA Backend Documentation

Complete documentation for the ZigNeuron CUDA backend, providing GPU acceleration on NVIDIA hardware.

## Documentation Structure

| Document | Purpose |
|----------|---------|
| [API Reference](./api.md) | Complete API reference for all public functions |
| [User Guide](./user-guide.md) | Tutorials and usage examples |
| [Developer Guide](./developer-guide.md) | Internal implementation details |
| [Best Practices](./best-practices.md) | Recommended patterns and optimizations |
| [Troubleshooting](./troubleshooting.md) | Common issues and solutions |

## Quick Links

### For Users
- [Quick Start Guide](./user-guide.md#quick-start)
- [Basic Operations](./user-guide.md#basic-operations)
- [Training Workflows](./user-guide.md#training-workflows)
- [Troubleshooting](./troubleshooting.md)

### For Developers
- [Architecture Overview](./developer-guide.md#architecture-overview)
- [Adding New Operations](./developer-guide.md#adding-new-operations)
- [Kernel Design](./developer-guide.md#kernel-design)
- [Security Considerations](./developer-guide.md#security-considerations)

### Reference
- [CudaBackend API](./api.md#cudabackend)
- [Error Types](./api.md#error-types)
- [Configuration Constants](./api.md#configuration-constants)

## Overview

The CUDA backend provides:

- **Dynamic Driver Loading** - No static CUDA dependencies
- **Automatic Kernel Loading** - PTX and NVRTC runtime compilation
- **Memory Pooling** - Efficient buffer reuse with power-of-2 buckets
- **Async Execution** - Non-blocking operations with streams
- **Tensor Core Support** - FP16 acceleration for sm_70+
- **Comprehensive Operations** - MatMul, activations, loss functions, optimizers, normalization

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x86_64 | Supported | Primary development platform |
| Windows x86_64 | Supported | Visual C++ redistributables required |
| macOS | Not Supported | Use Metal backend instead |

## Supported GPU Architectures

| Architecture | Compute Capability | Status |
|--------------|-------------------|--------|
| Pascal | 6.0, 6.1 | Supported |
| Volta | 7.0 | Supported |
| Turing | 7.5 | Supported |
| Ampere | 8.0, 8.6 | **Recommended** |
| Ada Lovelace | 8.9 | **Recommended** |
| Hopper | 9.0 | Experimental |

## Quick Example

```zig
const std = @import("std");
const cuda = @import("ZigNeuron").cuda;

pub fn main() !void {
    // Check if CUDA is available
    if (!cuda.CudaBackend.isAvailable()) {
        std.log.info("CUDA not available");
        return;
    }

    // Initialize backend
    var backend = try cuda.CudaBackend.init(std.heap.page_allocator);
    defer backend.deinit();

    // Perform matrix multiplication
    const M: usize = 128;
    const N: usize = 64;
    const K: usize = 256;

    var A: [M * K]f32 = undefined;
    var B: [K * N]f32 = undefined;
    var C: [M * N]f32 = undefined;

    // Initialize A and B...

    try backend.matMul(
        &A, &B, &C,
        M, N, K,
        false, false, false,
    );

    std.log.info("Computation complete");
}
```

## Building with CUDA Support

```bash
# Build the library
zig build

# Run tests (auto-detects CUDA)
zig build test

# Build with explicit CUDA support
zig build -Dcuda=true
```

## Additional Resources

- [CUDA Implementation Guide](../CUDA_IMPLEMENTATION_GUIDE.md) - Kernel development guide
- [CUDA Integration Guide](../CUDA_INTEGRATION.md) - Integration overview
- [CUDA Implementation Summary](../CUDA_IMPLEMENTATION_SUMMARY.md) - Kernel summary
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

## Contributing

When contributing to the CUDA backend:

1. Follow the existing code structure
2. Add tests for new functionality
3. Update relevant documentation
4. Ensure compatibility with minimum sm_60
5. Consider both PTX and NVRTC paths

See the [Developer Guide](./developer-guide.md) for detailed contribution guidelines.

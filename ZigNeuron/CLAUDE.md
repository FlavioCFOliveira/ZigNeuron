# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

- **Build the library**: `zig build`
- **Run tests**: `zig build test`
- **Run examples**: `zig build -Dexamples=true run-examples`
- **Install to prefix**: `zig build -p <path>`

## Architecture

ZigNeuron is a neural network library built in Zig. The project uses a modular structure:

- **src/main.zig**: Main library source file containing core types and functions
- **build.zig**: Build configuration using Zig's build system with Module-based approach

### Component Structure

The library is organized into logical components:
- **Layers** - Dense layer with weights, bias, and activation function
- **Activations** - ReLU, sigmoid, tanh activation functions with derivatives
- **Loss Functions** - MSE (implemented), cross-entropy, binary cross-entropy (TODO)
- **Network** - High-level network composition with backpropagation training
- **Backend** - GPU/CPU execution layer with Metal support for Apple Silicon (GPU priority)

## Project Structure

- `src/` - Source files
  - `main.zig` - Library root with exports for all modules
  - `backend.zig` - GPU/CPU execution backend (GPU priority)
  - `layer.zig` - Dense layer with weights, bias, and activation
  - `activation.zig` - Activation functions (ReLU, Sigmoid, Tanh)
  - `loss.zig` - Loss functions (MSE implemented, cross-entropy planned)
  - `network.zig` - Network composition with backpropagation training
  - `metal/` - Metal GPU backend files (stub structure in place)
- `build.zig` - Build definition
- `examples/xor.zig` - XOR neural network example demonstrating training
- `zig-out/` - Build output directory (created on build)

## Testing Requirements

**Unit tests are mandatory** for every component. Each piece of the neural network must have comprehensive test coverage:

### Unit Tests
Unit tests verify correctness of individual components:

- **Layers** - Test forward pass, backward pass, different input shapes, edge cases
- **Activations** - Test forward pass, derivative, numerical stability
- **Loss Functions** - Test forward pass, gradient computation, edge cases
- **Optimizers** - Test parameter updates, learning rate behavior, edge cases
- **Network** - Test end-to-end training, gradient flow, convergence

When adding new components, write tests first to define expected behavior. The test suite should cover:
- Correctness (mathematical accuracy)
- Edge cases (empty inputs, very large values, zero gradients)
- Shape compatibility for different input dimensions

### Performance Tests and Benchmarks

**Performance tests are mandatory** to ensure the library meets speed and memory efficiency requirements:

- **Microbenchmarks** - Benchmark individual functions (forward/backward passes, activation computations)
- **Integration benchmarks** - Benchmark complete layers and networks with realistic sizes
- **Memory benchmarks** - Track allocation counts and peak memory usage
- **GPU benchmarks** - Measure GPU utilization and kernel execution times on Apple Silicon

### Memory Benchmark Tests

**Memory benchmark tests are mandatory** to verify resource efficiency:

- **Allocation tracking** - Count allocations per operation to detect unnecessary allocations
- **Peak memory usage** - Measure maximum memory consumption during operations
- **Memory leaks** - Verify no memory leaks in long-running operations
- **Pre-allocation efficiency** - Test buffer reuse and caching strategies
- **Gradient memory** - Track memory used by gradients vs. weights

Run benchmarks:
- Before and after performance-critical changes
- As part of the CI/CD pipeline
- To establish baselines and detect regressions

## API Design Principles

The library interface must be as simple as possible for human usage:

- **Minimal surface area** - Expose only what users need to know
- **Intuitive defaults** - sensible defaults for all parameters
- **Clear naming** - Use descriptive, easy-to-understand names
- **Simple error handling** - Clear error messages and minimal error cases
- **Avoid unnecessary abstraction** - Don't create abstractions that don't provide value

When adding new features, ask: "Does this make the library easier to use?"

## Performance Requirements

ZigNeuron has two non-negotiable requirements:

1. **Performance** - Execute as fast as possible. Use the fastest algorithms available, even if they are more complex.
2. **Resource Efficiency** - Use as few system resources as possible. Minimize memory allocation and avoid unnecessary allocations.

These requirements often conflict. Prioritize performance, but always consider resource usage and provide mechanisms to control it.

## GPU and CPU Targeting

**GPU (Metal) execution is the PRIORITY for ALL neural network operations. CPU is only used as a fallback.**

### Execution Priority Rules

1. **Training** - Must use GPU (Metal) first. CPU fallback only if no GPU available.
2. **Inference** - Must use GPU (Metal) first. CPU fallback only if no GPU available.
3. **Gradient computation** - Must use GPU (Metal) first.
4. **Forward pass** - Must use GPU (Metal) first.

### Implementation Requirements

- All training and inference operations should be implemented in Metal shaders first
- CPU implementations should be marked with `// TODO: GPU implementation` comments
- GPU code should use Metal compute shaders for:
  - Matrix multiplication (matmul)
  - Activation functions (ReLU, Sigmoid, Tanh)
  - Loss functions
  - Gradient computation
- Fallback CPU implementations must maintain numerical accuracy

### Apple Silicon Optimization

The library should leverage Apple Silicon capabilities:
- Use Metal Performance Shaders (MPS) where available
- Use shared memory architecture (no CPU-GPU data transfer)
- Use M-series GPU compute capabilities (SIMD, parallel execution)
- Optimize for memory bandwidth between CPU and GPU shared memory

When adding new features, prioritize:
1. **GPU/Metal implementation first** - Always implement on GPU first
2. **CPU fallback** - Implement CPU fallback only after GPU is working
3. **Performance** - GPU implementation must be faster than CPU
4. **Apple Silicon** - Optimize for M-series chips (M1, M2, M3, etc.)

The library should degrade gracefully:
- If Metal device unavailable → use CPU
- If specific kernel unavailable → use fallback CPU implementation

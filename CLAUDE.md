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
- **Backend** - GPU/CPU execution layer with automatic device detection (Metal > Vulkan > CPU)

## Project Structure

- `src/` - Source files
  - `main.zig` - Library root with exports for all modules
  - `backend.zig` - GPU/CPU execution backend with automatic device detection
  - `layer.zig` - Dense layer with weights, bias, and activation
  - `activation.zig` - Activation functions (ReLU, Sigmoid, Tanh)
  - `loss.zig` - Loss functions (MSE implemented, cross-entropy planned)
  - `network.zig` - Network composition with backpropagation training
- `build.zig` - Build definition
- `examples/xor.zig` - XOR neural network example demonstrating training
- `zig-out/` - Build output directory (created on build)

## Test Evidence Files

Evidence from test runs is stored in the repository root:

| File | Purpose |
|------|---------|
| `UnitTests.md` | Evidence from unit test runs |
| `BenchmarkTests.md` | Evidence from performance benchmark runs |
| `MemoryTests.md` | Evidence from memory profiling runs |

These files document actual test results, including timestamps, environment info, and measured values for historical comparison.

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Dense Layer | Done | Forward pass implemented |
| Activations | Done | ReLU, Sigmoid, Tanh with derivatives |
| Loss Functions | Partial | MSE only |
| Backpropagation | Done | Training works with MSE loss |
| GPU Backend | Stub | Metal/Vulkan stubs ready |
| Vulkan Support | TODO | Needs implementation |
| Optimizers | TODO | SGD, Adam, RMSprop needed |
| Unit Tests | TODO | Test coverage needed |
| Performance Benchmarks | TODO | Benchmark suite needed |

## Testing Requirements

**Unit tests are strongly encouraged** for every component. Each piece of the neural network should have test coverage:

### Unit Tests (Encouraged)
Unit tests verify correctness of individual components:
- **Layers** - Test forward pass, different input shapes
- **Activations** - Test forward pass, derivative values
- **Loss Functions** - Test forward pass, gradient computation
- **Network** - Test end-to-end training convergence

### Performance Tests and Benchmarks (Planned)

**Performance tests are encouraged** to ensure the library meets speed requirements:
- **Microbenchmarks** - Benchmark individual functions (forward/backward passes)
- **Integration benchmarks** - Benchmark complete layers and networks

### Memory Tests (Planned)

**Memory tests are encouraged** to verify resource efficiency:
- **Allocation tracking** - Track allocations per operation
- **Peak memory usage** - Measure memory consumption

### Test Evidence Documentation

**All test executions must document their evidence.** For each test run, the results should be recorded in dedicated documentation files:

| Test Type | Evidence File | Content |
|-----------|--------------|---------|
| Unit Tests | `UnitTests.md` | Test results, pass/fail status, execution time |
| Performance Benchmarks | `BenchmarkTests.md` | Benchmark results, comparison baseline, profiling data |
| Memory Tests | `MemoryTests.md` | Allocation counts, peak memory, leak detection |

When running tests:
1. Execute the test suite
2. Record all evidence in the appropriate documentation file
3. Include: date/time, environment info, test parameters, results
4. Use markdown formatting for readability (tables, code blocks)

When adding new features:
1. Write tests first to define expected behavior
2. Run tests and document results
3. Update the relevant evidence file before merging

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

**GPU execution is the PRIORITY for ALL neural network operations. CPU is only used as a last resort fallback.**

### Device Detection and Priority

The library automatically detects available hardware and selects the best execution backend:

1. **Metal (Apple Silicon)** - Uses Metal compute shaders on M-series chips (M1, M2, M3, etc.)
2. **Vulkan** - Uses Vulkan compute shaders for cross-platform GPU support
3. **CPU** - Falls back to CPU computation if no GPU available

### Execution Priority Rules

1. **Training** - Must use GPU (Metal or Vulkan) first. CPU fallback only if no GPU available.
2. **Inference** - Must use GPU (Metal or Vulkan) first. CPU fallback only if no GPU available.
3. **Gradient computation** - Must use GPU (Metal or Vulkan) first.
4. **Forward pass** - Must use GPU (Metal or Vulkan) first.

### Implementation Requirements

- GPU implementations should be prioritized for:
  - Matrix multiplication (matmul)
  - Activation functions (ReLU, Sigmoid, Tanh)
  - Loss functions
  - Gradient computation
- CPU implementations should be marked with `// TODO: GPU implementation` comments
- Fallback CPU implementations must maintain numerical accuracy

### Platform Support

The library supports multiple platforms:

| Platform | GPU Backend | CPU Fallback |
|----------|-------------|--------------|
| macOS (Apple Silicon) | Metal | Yes |
| Linux | Vulkan | Yes |
| Windows | Vulkan | Yes |
| Other | Vulkan or CPU | Yes |

### Apple Silicon Optimization

The library should leverage Apple Silicon capabilities:
- Use Metal Performance Shaders (MPS) where available
- Use shared memory architecture (no CPU-GPU data transfer)
- Use M-series GPU compute capabilities (SIMD, parallel execution)
- Optimize for memory bandwidth between CPU and GPU shared memory

### Vulkan Cross-Platform Support

For non-Apple platforms (Linux, Windows):
- Use Vulkan compute shaders
- Support SPIR-V shader compilation
- Detect Vulkan support at runtime
- Fall back to CPU if Vulkan unavailable

When adding new features, prioritize:
1. **GPU/Metal implementation first** - Always implement on GPU first
2. **CPU fallback** - Implement CPU fallback only after GPU is working
3. **Performance** - GPU implementation must be faster than CPU
4. **Apple Silicon** - Optimize for M-series chips (M1, M2, M3, etc.)

The library should degrade gracefully:
- If Metal device unavailable → try Vulkan
- If Vulkan unavailable → use CPU
- If specific kernel unavailable → use fallback CPU implementation

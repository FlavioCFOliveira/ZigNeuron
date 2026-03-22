# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

- **Build the library**: `zig build`
- **Run tests**: `zig build test`
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
- **Backend** - GPU/CPU execution layer with automatic device detection (Metal > CUDA > CPU)

## CUDA Backend Documentation

Comprehensive CUDA backend documentation is available in `docs/cuda/`:

| Document | Purpose |
|----------|---------|
| `docs/cuda/README.md` | Documentation index and quick start |
| `docs/cuda/api.md` | Complete API reference for all CUDA functions |
| `docs/cuda/user-guide.md` | Tutorials and usage examples |
| `docs/cuda/developer-guide.md` | Internal implementation details |
| `docs/cuda/best-practices.md` | Performance optimization recommendations |
| `docs/cuda/troubleshooting.md` | Common issues and solutions |

### CUDA Quick Example

```zig
const std = @import("std");
const cuda = @import("ZigNeuron").cuda;

pub fn main() !void {
    // Check if CUDA is available
    if (!cuda.CudaBackend.isAvailable()) {
        std.log.info("CUDA not available, using CPU fallback");
        return;
    }

    // Initialize CUDA backend
    var backend = try cuda.CudaBackend.init(std.heap.page_allocator);
    defer backend.deinit();

    // Allocate and upload data
    var input: [100]f32 = undefined;
    var output: [100]f32 = undefined;

    var d_buffer = try backend.allocBuffer(100 * @sizeOf(f32));
    defer backend.freeBuffer(d_buffer);

    try backend.upload(d_buffer.ptr, &input);
    try backend.reluForward(&input, &output);
}
```

## Project Structure

- `src/` - Source files
  - `main.zig` - Library root with exports for all modules
  - `backend.zig` - GPU/CPU execution backend with automatic device detection (Metal/CUDA/CPU)
  - `layer.zig` - Dense layer with weights, bias, and activation
  - `activation.zig` - Activation functions (ReLU, Sigmoid, Tanh)
  - `loss.zig` - Loss functions (MSE, Cross-Entropy, KL-Divergence)
  - `network.zig` - Network composition with backpropagation training
  - `cuda.zig` - CUDA backend implementation
  - `cuda_context.zig` - CUDA context and resource management
  - `cuda_driver.zig` - CUDA Driver API bindings
  - `cuda_kernels.zig` - CUDA kernel PTX and source code
  - `metal_context.zig` - Metal backend implementation
- `docs/cuda/` - CUDA backend documentation
  - `README.md` - Documentation index
  - `api.md` - API reference
  - `user-guide.md` - User guide
  - `developer-guide.md` - Developer guide
  - `best-practices.md` - Best practices
  - `troubleshooting.md` - Troubleshooting guide
- `build.zig` - Build definition
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
| Dense Layer | Done | Forward and backward pass implemented |
| Activations | Done | ReLU, Sigmoid, Tanh with correct derivatives |
| Loss Functions | Done | MSE, Cross-Entropy, KL-Divergence |
| Backpropagation | Done | Training works on CPU, Metal, and CUDA |
| Metal Backend | Done | Metal compute shaders on Apple Silicon |
| CUDA Backend | Done | CUDA kernels for Linux/Windows NVIDIA GPUs |
| Optimizers | Partial | SGD, Adam, RMSprop basic structs |
| Unit Tests | Done | Coverage for FNN, RNN, VAE, CNN |
| Performance Benchmarks | Done | Benchmark suite available |
| Security Audit | In Progress | Phase 1-2 corrections being applied |

## Security Fixes Applied

### Phase 1 - Critical Security Fixes

| Vulnerability | File | Status | Description |
|--------------|------|--------|-------------|
| VULN-001: Use-After-Free | `src/metal_context.zig:65-68` | ✅ Fixed | Shader source freed after library creation |
| VULN-002: Integer Overflow | `src/tensor.zig:14-21` | ✅ Fixed | Tensor dimension validation |
| VULN-003: Race Condition | `src/backend.zig` | ⚠️ Documented | Thread-safety limitation documented |
| VULN-004: Double-Free | `src/cuda_context.zig` | ✅ Fixed | Buffer lifecycle validation |

### Phase 2 - CUDA Backend Security Fixes (2026-03-20)

| Vulnerability | File | Status | Description |
|--------------|------|--------|-------------|
| CRIT-003: Stream Race Condition | `src/cuda_context.zig` | ✅ Fixed | Thread-safe `ThreadSafeStream` wrapper with mutex protection |
| CRIT-002: Use-After-Free | `src/cuda_driver.zig:820-824` | ✅ Fixed | Reference-counted `CudaDriverRef` for safe driver access |

## Performance Fixes Applied

### Phase 2 - Critical Performance Optimizations

| Fix | File | Status | Description | Impact |
|-----|------|--------|-------------|--------|
| F2.1: Attention Allocation | `src/backend.zig:2809`, `src/layer.zig:978` | ✅ Fixed | Pre-allocated scores buffer | 2-3x seq2seq speedup |
| F2.2: LayerNorm Welford | `src/backend.zig:2495` | ✅ Fixed | Single-pass variance calculation | 2x memory reduction |
| F2.3: SIMD Normalization | `src/optimization.zig` | ✅ Fixed | Vectorized batch/layer norm | 4x CPU speedup |

**Note:** Phase 2 optimizations focus on hot paths identified in profiling.

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

## Multi-Agent Specialist Consensus

All interaction processes—including analysis, development, testing, documentation, and research—must integrate the perspectives of all available specialized agents. No task or development phase is considered complete until consensus is reached among all involved specialists (agents).

### Status Reporting and Specialist Perspectives

Whenever a Status Report (PDS - Ponto de Situação) is requested, specialized agents must be consulted to assess the project state from their respective domains. The resulting output should be a highly professional technical report that clearly outlines each agent's perspective on the current progress, challenges, and next steps.

### Specialized Agent Registry

To ensure high-quality implementations and architectural integrity, the following specialized agents must be involved in their respective domains:

| Agent | Responsibility | Key Interactions |
|-------|----------------|------------------|
| **Neural Net Architect** | Design and implementation of core neural network components (layers, activations, backprop). | Consults *Technical Researcher* for formulas; *Zig Performance Architect* for implementation. |
| **Zig Performance Architect** | Optimization for speed and resource efficiency. Ensures zero-allocation patterns and SIMD usage. | Validates *Neural Net Architect* designs; optimize *Security Architect* audited code. |
| **Metal Performance Expert** | High-performance GPU acceleration for Apple Silicon. MSL kernels, pipelines, and M-series memory optimization. | Optimizes *Backend* for macOS; collaborates with *Zig Performance Architect* for unified memory. |
| **CUDA Performance Optimizer** | High-performance GPU acceleration for NVIDIA. CUDA kernels, shared memory, and Tensor Core utilization. | Optimizes *Backend* for Linux/Windows; consults *Security Architect* for kernel safety. |
| **Security Architect** | Rigorous audit of low-level code: memory management, GPU kernels, and concurrency. | Reviews all *Zig Performance Architect* optimizations and *Backend* changes. |
| **Technical Researcher** | Mathematical derivations, GPU API documentation (Metal/CUDA), and state-of-the-art research. | Provides foundations for *Neural Net Architect* and *ML Architecture Expert*. |
| **ML Architecture Expert** | Designing reference examples, benchmarking scenarios, and validating library usability. | Uses *ML Dataset Fetcher* for data; validates *Neural Net Architect* outputs. |
| **ML Dataset Fetcher** | Identifying, sourcing, and organizing datasets for training and validation examples. | Supports *ML Architecture Expert* with data pipelines. |
| **Roadmap Coordinator** | **EXCLUSIVE** coordination of roadmap, sprints and tasks via Groadmap CLI (`rmp` commands). | Coordinates task workflows; delegates implementation to other specialists. |
| **Task Creator** | Creates structured tasks and sprints for the roadmap; delegates persistence to Roadmap Coordinator. | Prepares task data; delegates to Roadmap Coordinator for CLI execution. |

### Interaction Workflow

1.  **Requirement/Research**: *Technical Researcher* provides the theoretical basis.
2.  **Architecture**: *Neural Net Architect* designs the component logic.
3.  **GPU Optimization**: *Metal Performance Expert* (Apple) or *CUDA Performance Optimizer* (NVIDIA) implements and optimizes kernels.
4.  **CPU Optimization**: *Zig Performance Architect* refines the fallback implementation and CPU-GPU data orchestration.
5.  **Security/Audit**: *Security Architect* performs a final review of low-level and memory-sensitive code.
6.  **Validation**: *ML Architecture Expert* implements an example to verify end-to-end functionality.

No significant change to core logic or backend should be merged without the explicit consensus of the *Neural Net Architect*, *Zig Performance Architect*, and relevant *GPU Expert*.

## Roadmap, Sprint and Task Management

All project planning, task coordination, and sprint management **MUST** be done exclusively through the **roadmap-coordinator** and **task-creator** skills using the Groadmap CLI (`rmp`).

### Task Creation Workflow

1. **Task Creator Skill** (`task-creator`): Use when creating new tasks or sprints
   - Collects all required data (title, type, priority, description, technical requirements, acceptance criteria)
   - Presents structured task for user confirmation
   - Delegates persistence to roadmap-coordinator

2. **Roadmap Coordinator Skill** (`roadmap-coordinator`): Use for all task coordination
   - Retrieves tasks via CLI (`rmp task next`)
   - Manages state transitions (`rmp task stat`)
   - Delegates implementation to appropriate specialists
   - Generates execution reports

### CLI Commands Reference

```bash
# Task management
rmp task next [N]              # Get next tasks
rmp task get -r <roadmap> <id> # Get task details
rmp task stat -r <roadmap> <id> <BACKLOG|SPRINT|DOING|TESTING|COMPLETED>

# Sprint management
rmp sprint list -r <roadmap>
rmp sprint start|close|reopen -r <roadmap> <id>
```

### State Machine

Tasks follow this workflow: `BACKLOG → SPRINT → DOING → TESTING → COMPLETED`

### Important Rules

- **ALWAYS** use `roadmap-coordinator` for task/sprint coordination
- **NEVER** create tasks manually outside the CLI workflow
- **MANDATORY** state transitions via `rmp task stat`
- Task ordering commands (`reorder`, `move-to`, `swap`) affect position only, not status

ZigNeuron has two non-negotiable requirements:

1. **Performance** - Execute as fast as possible. Use the fastest algorithms available, even if they are more complex.
2. **Resource Efficiency** - Use as few system resources as possible. Minimize memory allocation and avoid unnecessary allocations.

These requirements often conflict. Prioritize performance, but always consider resource usage and provide mechanisms to control it.

## GPU and CPU Targeting

**GPU execution is the PRIORITY for ALL neural network operations. CPU is only used as a last resort fallback.**

### Device Detection and Priority

The library automatically detects available hardware and selects the best execution backend:

1. **Metal (Apple Silicon)** - Uses Metal compute shaders on M-series chips (M1, M2, M3, etc.)
2. **CUDA (NVIDIA)** - Uses CUDA kernels on NVIDIA GPUs
3. **CPU** - Falls back to CPU computation if no GPU available

### Execution Priority Rules

1. **Training** - Must use GPU first (Metal/CUDA). CPU fallback only if no GPU available.
2. **Inference** - Must use GPU first (Metal/CUDA). CPU fallback only if no GPU available.
3. **Gradient computation** - Must use GPU first.
4. **Forward pass** - Must use GPU first.

### Implementation Requirements

- GPU implementations should be prioritized for:
  - Matrix multiplication (matmul) - Use Metal Performance Shaders or CUDA
  - Activation functions (ReLU, Sigmoid, Tanh) - Custom compute kernels
  - Loss functions - Custom compute kernels
  - Gradient computation - GPU-accelerated backward passes
- CPU implementations should be marked with `// TODO: GPU implementation` comments
- Fallback CPU implementations must maintain numerical accuracy

### CUDA-Specific Requirements

When implementing CUDA operations:
- Use dynamic driver loading (no static CUDA dependencies)
- Implement overflow protection for size calculations (`std.math.mul`)
- Validate PTX before loading (security requirement)
- Use memory pooling for buffer reuse
- Support both PTX loading and NVRTC runtime compilation
- Add kernel launch parameter validation
- Document compute capability requirements
- Add CUDA-specific error handling

### Platform Support

The library supports multiple platforms:

| Platform | GPU Backend | CPU Fallback |
|----------|-------------|--------------|
| macOS (Apple Silicon) | Metal | Yes |
| Linux (NVIDIA) | CUDA | Yes |
| Windows (NVIDIA) | CUDA | Yes |
| Linux/Windows (no GPU) | CPU | Yes |
| Other | CPU | Yes |

### Apple Silicon Optimization

The library should leverage Apple Silicon capabilities:
- Use Metal Performance Shaders (MPS) where available
- Use shared memory architecture (no CPU-GPU data transfer)
- Use M-series GPU compute capabilities (SIMD, parallel execution)
- Optimize for memory bandwidth between CPU and GPU shared memory

### NVIDIA CUDA Optimization

The CUDA backend leverages NVIDIA GPU capabilities:
- Dynamic driver loading (libcuda.so / nvcuda.dll)
- Memory pooling for efficient buffer reuse
- Tensor Cores for FP16 acceleration (sm_70+)
- Shared memory tiling for matrix operations
- Async execution with CUDA streams
- NVRTC runtime compilation for kernel flexibility

### Cross-Platform GPU Support

The library automatically detects available hardware and selects the optimal backend:

1. **macOS (Apple Silicon)** → Metal
2. **Linux/Windows (NVIDIA)** → CUDA
3. **All platforms** → CPU (fallback)

When adding new features, prioritize:
1. **GPU implementation first** - Metal for macOS, CUDA for Linux/Windows
2. **CPU fallback** - Implement after GPU version is working
3. **Performance** - GPU implementation must be faster than CPU
4. **Platform optimization** - Metal for M-series, CUDA for NVIDIA

The library degrades gracefully:
- If Metal/CUDA unavailable → use CPU
- If specific kernel unavailable → use fallback CPU implementation
- If out of memory → return error for caller to handle

## Documentation Language Policy

**All documentation, comments, and commits must be written in English.**

| Content Type | Language |
|--------------|----------|
| Code comments | English |
| Markdown documentation (all .md files) | English |
| Commit messages | English |
| Pull request descriptions | English |

**Reason**: English is the standard language for open-source software projects and ensures consistency, better international collaboration, and easier discoverability.

**Example commit message format**:
```
Fix bug in loss function

- Updated BCE gradient formula to (p-t) for correct sigmoid gradient
- Added log-sum-exp trick for numerical stability
- Added gradient and weight clipping to prevent explosion

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

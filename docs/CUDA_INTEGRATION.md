# CUDA Integration Guide for ZigNeuron

This document describes the CUDA backend implementation for ZigNeuron, providing GPU acceleration on NVIDIA hardware for Linux and Windows platforms.

## Architecture Overview

The CUDA backend follows the same architectural patterns as the Metal backend:

```
┌─────────────────────────────────────────────────────────────┐
│                    ZigNeuron Library                          │
├─────────────────────────────────────────────────────────────┤
│                    backend.zig                                │
│         (Backend dispatch: Metal > CUDA > CPU)                │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Metal      │  │    CUDA      │  │     CPU      │      │
│  │  (macOS)     │  │ (Linux/Win)  │  │  (Fallback)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                   cuda.zig                                    │
├─────────────────────────────────────────────────────────────┤
│              cuda_context.zig  cuda_driver.zig              │
│    (Context management)         (Driver API bindings)       │
├─────────────────────────────────────────────────────────────┤
│                    libcuda.so / nvcuda.dll                   │
│                   (NVIDIA CUDA Driver)                        │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
src/
├── cuda.zig           # Main CUDA backend implementation
├── cuda_driver.zig    # CUDA Driver API bindings with dynamic loading
├── cuda_context.zig   # CUDA context and resource management
├── backend.zig        # Backend dispatch (updated for CUDA)
└── main.zig           # Library exports

shaders/cuda/
├── kernels.cu         # CUDA kernel source code
└── kernels.ptx        # Compiled PTX (generated at build time)
```

## Key Features

### 1. Dynamic Driver Loading

The CUDA backend uses dynamic loading of the NVIDIA driver (`libcuda.so` on Linux, `nvcuda.dll` on Windows) rather than static linking. This provides:

- **Graceful degradation**: No CUDA runtime dependency if hardware unavailable
- **Version independence**: Works with any CUDA driver version
- **Distribution**: No need to ship CUDA libraries

### 2. Memory Management

- **Buffer pooling**: Automatic reuse of device buffers via power-of-2 buckets
- **Unified memory**: Uses CUDA managed memory when available (Pascal+)
- **Zero-copy**: Avoids unnecessary host-device transfers

### 3. Async Execution

- **Stream-based**: All operations use CUDA streams for async execution
- **Event synchronization**: Events for timing and dependencies
- **Non-blocking**: CPU can continue while GPU processes

### 4. Kernel Optimization

- **PTX loading**: Pre-compiled PTX kernels loaded at runtime
- **Occupancy calculator**: Automatic thread block size selection
- **SM targeting**: Optimized for sm_60+ (Pascal and newer)

## Usage

### Building with CUDA Support

```bash
# Compile with CUDA support
zig build -Dcuda=true

# Compile CUDA kernels
zig build cuda

# Run tests with CUDA
zig build test -Dcuda=true
```

### Basic Example

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

    // Allocate device buffers
    var d_a = try backend.allocBuffer(1024 * @sizeOf(f32));
    defer backend.freeBuffer(d_a);
    var d_b = try backend.allocBuffer(1024 * @sizeOf(f32));
    defer backend.freeBuffer(d_b);

    // Upload data
    const a = try std.heap.page_allocator.alloc(f32, 1024);
    defer std.heap.page_allocator.free(a);
    // ... fill a ...
    try backend.upload(d_a.ptr, a);

    // Run kernel
    try backend.reluForward(a, output);

    // Download results
    try backend.download(output, d_b.ptr);
}
```

### Matrix Multiplication

```zig
// C = A * B where A: MxK, B: KxN, C: MxN
var A: [M * K]f32 = undefined;
var B: [K * N]f32 = undefined;
var C: [M * N]f32 = undefined;

// Fill A and B...

try backend.matMul(
    &A, &B, &C,
    M, N, K,
    false,  // transpose_a
    false,  // transpose_b
    false,  // accumulate
);
```

## API Reference

### CudaBackend

```zig
pub const CudaBackend = struct {
    /// Initialize CUDA backend with the best available device
    pub fn init(allocator: std.mem.Allocator) !CudaBackend;

    /// Cleanup CUDA resources
    pub fn deinit(self: *CudaBackend) void;

    /// Check if CUDA is available on this system
    pub fn isAvailable() bool;

    /// Get number of CUDA devices
    pub fn getDeviceCount() i32;

    /// Synchronize the CUDA stream
    pub fn synchronize(self: *CudaBackend) !void;

    // Memory operations
    pub fn allocBuffer(self: *CudaBackend, size: usize) !DeviceBuffer;
    pub fn freeBuffer(self: *CudaBackend, buffer: DeviceBuffer) void;
    pub fn upload(self: *CudaBackend, dst: CUdeviceptr, src: []const f32) !void;
    pub fn download(self: *CudaBackend, dst: []f32, src: CUdeviceptr) !void;

    // Matrix operations
    pub fn matMul(self: *CudaBackend, a: []const f32, b: []const f32, c: []f32,
                  m: usize, n: usize, k: usize,
                  transpose_a: bool, transpose_b: bool, accumulate: bool) !void;
    pub fn matMulBatch(self: *CudaBackend, a: []const f32, b: []const f32, c: []f32,
                       batch_size: usize, n: usize, k: usize, accumulate: bool) !void;

    // Element-wise operations
    pub fn elementWiseOp(self: *CudaBackend, op: ElementWiseOp,
                         a: []const f32, b: []const f32, c: []f32) !void;
    pub fn scale(self: *CudaBackend, a: []const f32, scalar: f32, c: []f32) !void;
    pub fn mapOp(self: *CudaBackend, op: MapOp, input: []const f32, output: []f32) !void;

    // Activations
    pub fn reluForward(self: *CudaBackend, input: []const f32, output: []f32) !void;
    pub fn reluBackward(self: *CudaBackend, output: []const f32,
                        grad_output: []const f32, grad_input: []f32) !void;
    pub fn sigmoidForward(self: *CudaBackend, input: []const f32, output: []f32) !void;
    pub fn sigmoidBackward(self: *CudaBackend, output: []const f32,
                           grad_output: []const f32, grad_input: []f32) !void;
    pub fn tanhForward(self: *CudaBackend, input: []const f32, output: []f32) !void;
    pub fn tanhBackward(self: *CudaBackend, output: []const f32,
                        grad_output: []const f32, grad_input: []f32) !void;
    pub fn softmaxForward(self: *CudaBackend, input: []const f32, output: []f32,
                          batch_size: usize, features: usize) !void;

    // Loss functions
    pub fn mseBackward(self: *CudaBackend, output: []const f32,
                        target: []const f32, grad_output: []f32) !void;
    pub fn crossEntropyBackward(self: *CudaBackend, output: []const f32,
                                 target: []const f32, grad_output: []f32) !void;

    // Optimizers
    pub fn sgdUpdate(self: *CudaBackend, weights: []f32, gradients: []const f32,
                     learning_rate: f32, weight_decay: f32) !void;
    pub fn adamUpdate(self: *CudaBackend, weights: []f32, gradients: []const f32,
                      m: []f32, v: []f32, learning_rate: f32, beta1: f32,
                      beta2: f32, epsilon: f32, t: u32) !void;
};
```

### CudaContext

Lower-level context management:

```zig
pub const CudaContext = struct {
    /// Device properties
    pub const DeviceProperties = struct {
        compute_capability_major: i32,
        compute_capability_minor: i32,
        total_memory: usize,
        multiprocessor_count: i32,
        max_threads_per_block: i32,
        warp_size: i32,
        hasTensorCores() bool,
        hasUnifiedMemory() bool,
    };

    /// Initialize context with best available device
    pub fn init(allocator: std.mem.Allocator, driver: *CudaDriver) !*CudaContext;

    /// Cleanup context
    pub fn deinit(self: *CudaContext) void;

    /// Synchronize the CUDA stream
    pub fn synchronize(self: *CudaContext) !void;

    /// Get/return buffers from pool
    pub fn getBuffer(self: *CudaContext, size: usize) !DeviceBuffer;
    pub fn returnBuffer(self: *CudaContext, buffer: DeviceBuffer) void;

    /// Load and launch kernels
    pub fn loadKernel(self: *CudaContext, name: []const u8, ptx_code: []const u8) !void;
    pub fn launchKernel(self: *CudaContext, kernel_name: []const u8,
                        grid_dim: [3]u32, block_dim: [3]u32,
                        shared_mem_bytes: u32, args: []const ?*anyopaque) !void;
};
```

## Implementation Details

### Error Handling

CUDA operations return `CUresult` which is converted to Zig errors:

```zig
try cuda_driver.checkCuda(result);
```

Available errors:
- `CudaOutOfMemory`
- `CudaInvalidValue`
- `CudaInvalidDevice`
- `CudaInvalidContext`
- `CudaNotInitialized`
- `CudaNoDevice`
- `CudaDeviceUnavailable`

### Memory Pool Configuration

Buffer pooling uses power-of-2 buckets from 4 bytes to 128MB:

```zig
pub const MEMORY_POOL_BUCKETS: usize = 32;
pub const MIN_POOL_SIZE: usize = 4;
pub const MAX_POOL_SIZE: usize = 128 * 1024 * 1024;
```

### Thread Configuration

Default configurations:

```zig
pub const DEFAULT_BLOCK_SIZE: c_uint = 256;
pub const DEFAULT_THREADS_PER_BLOCK: c_uint = 256;
```

Kernels automatically calculate grid dimensions based on problem size.

### PTX Generation

Kernels are written in CUDA-C and compiled to PTX:

```bash
nvcc -ptx -arch=sm_60 -O3 -lineinfo -o kernels.ptx kernels.cu
```

The PTX is then embedded or loaded at runtime.

## Performance Considerations

### 1. Memory Transfers

Minimize host-device transfers by:
- Keeping data on GPU between operations
- Using pinned host memory for async transfers
- Using managed memory for automatic migration

### 2. Kernel Fusion

Combine operations where possible:
- `matMulBiasReLU` instead of separate calls
- Fused optimizer updates
- Fused activations

### 3. Stream Usage

For independent operations:
```zig
// Upload data asynchronously
try backend.uploadAsync(d_a.ptr, a);

// Launch kernel
try context.launchKernel(...);

// Download results asynchronously
try backend.downloadAsync(c, d_c.ptr);

// Synchronize at end
try backend.synchronize();
```

### 4. Occupancy

The backend uses the CUDA occupancy calculator to select optimal block sizes:

```zig
if (driver.occupancyMaxPotentialBlockSize) |occupancy_fn| {
    _ = occupancy_fn(&min_grid_size, &max_threads_per_block,
                    function, null, 0, 0);
}
```

## Hardware Requirements

### Minimum Requirements

- NVIDIA GPU with Compute Capability 6.0+ (Pascal)
- CUDA Driver 11.0+
- 4GB+ VRAM (8GB+ recommended)

### Recommended for Training

- NVIDIA RTX 3090/4090 or A100
- 16GB+ VRAM
- Compute Capability 8.0+ (Ampere)
- Tensor Cores for FP16/BF16

### Compute Capability Support

| GPU Generation | Compute Capability | Status |
|----------------|---------------------|--------|
| Maxwell (GM20x) | 5.2 | Limited |
| Pascal (GP10x) | 6.0, 6.1 | Supported |
| Volta (GV100) | 7.0 | Supported |
| Turing (TU10x) | 7.5 | Supported |
| Ampere (GA10x) | 8.0, 8.6 | Recommended |
| Ada Lovelace (AD10x) | 8.9 | Recommended |
| Hopper (GH100) | 9.0 | Experimental |

## Troubleshooting

### "CUDA driver not found"

Install NVIDIA drivers:
```bash
# Ubuntu
sudo apt install nvidia-driver-535

# Verify
nvidia-smi
```

### "Out of memory"

- Reduce batch size
- Use gradient checkpointing
- Enable memory pooling
- Use mixed precision (FP16)

### "Invalid PTX"

- Ensure PTX is compiled for sm_60+
- Check nvcc version matches driver
- Recompile kernels: `zig build cuda`

### Performance issues

- Check GPU utilization with `nvidia-smi`
- Verify async execution is working
- Profile with Nsight Systems
- Ensure memory transfers are minimized

## Future Enhancements

### Planned

- [ ] cuBLAS integration for optimized GEMM
- [ ] cuDNN integration for convolutions
- [ ] NCCL integration for multi-GPU
- [ ] Tensor Core FP16/BF16 support
- [ ] Graph API for kernel fusion
- [ ] CUDA streams for async execution

### Under Consideration

- [ ] MPS (Multi-Process Service) support
- [ ] Unified memory on-demand migration
- [ ] P2P GPU transfers
- [ ] CUDA Graphs for repeated execution

## References

- [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

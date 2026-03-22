# CUDA Backend API Reference

Complete API reference for the ZigNeuron CUDA backend.

## Table of Contents

- [Overview](#overview)
- [CudaBackend](#cudabackend)
- [CudaContext](#cudacontext)
- [DeviceBuffer](#devicebuffer)
- [CUDA Driver](#cuda-driver)
- [Error Types](#error-types)
- [Kernel Management](#kernel-management)
- [Memory Operations](#memory-operations)
- [Matrix Operations](#matrix-operations)
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Normalization](#normalization)

## Overview

The CUDA backend provides GPU acceleration for neural network operations on NVIDIA hardware. It supports:

- Dynamic driver loading (no static CUDA dependencies)
- Automatic kernel loading from PTX or NVRTC compilation
- Memory pooling for efficient buffer reuse
- Async execution with streams
- Tensor Core support (sm_70+)

## CudaBackend

The main interface for CUDA operations.

### Initialization

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
};
```

**Example:**
```zig
const std = @import("std");
const cuda = @import("ZigNeuron").cuda;

pub fn main() !void {
    // Check availability first
    if (!cuda.CudaBackend.isAvailable()) {
        std.log.info("CUDA not available");
        return;
    }

    // Initialize backend
    var backend = try cuda.CudaBackend.init(std.heap.page_allocator);
    defer backend.deinit();

    // Use backend...
    try backend.synchronize();
}
```

### Memory Operations

```zig
/// Allocate device buffer
pub fn allocBuffer(self: *CudaBackend, size: usize) !DeviceBuffer;

/// Free device buffer
pub fn freeBuffer(self: *CudaBackend, buffer: DeviceBuffer) void;

/// Upload data to device (synchronous)
pub fn upload(self: *CudaBackend, dst: CUdeviceptr, src: []const f32) !void;

/// Upload data asynchronously
pub fn uploadAsync(self: *CudaBackend, dst: CUdeviceptr, src: []const f32) !void;

/// Download data from device (synchronous)
pub fn download(self: *CudaBackend, dst: []f32, src: CUdeviceptr) !void;

/// Download data asynchronously
pub fn downloadAsync(self: *CudaBackend, dst: []f32, src: CUdeviceptr) !void;
```

**Example:**
```zig
// Allocate device buffer
var d_buffer = try backend.allocBuffer(1024 * @sizeOf(f32));
defer backend.freeBuffer(d_buffer);

// Upload data
var host_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
try backend.upload(d_buffer.ptr, &host_data);

// Download results
var result: [4]f32 = undefined;
try backend.download(&result, d_buffer.ptr);
```

### Matrix Operations

```zig
/// Matrix multiplication: C = A * B + (accumulate ? C : 0)
/// Dimensions: A: MxK, B: KxN, C: MxN
pub fn matMul(
    self: *CudaBackend,
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
    transpose_a: bool,
    transpose_b: bool,
    accumulate: bool,
) !void;

/// Batched matrix multiplication
/// Each batch: A: 1xK, B: KxN, C: 1xN
pub fn matMulBatch(
    self: *CudaBackend,
    a: []const f32,
    b: []const f32,
    c: []f32,
    batch_size: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) !void;

/// Tensor Core matrix multiplication (FP16)
/// Requires: sm_70+, dimensions multiple of 16
pub fn matMulTensorCore(
    self: *CudaBackend,
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) !void;
```

**Example:**
```zig
// Standard matrix multiplication
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
    false,  // transpose_a
    false,  // transpose_b
    false,  // accumulate
);
```

### Element-wise Operations

```zig
pub const ElementWiseOp = enum {
    add,
    sub,
    mul,
    div,
};

/// Element-wise operation: c = a op b
pub fn elementWiseOp(
    self: *CudaBackend,
    op: ElementWiseOp,
    a: []const f32,
    b: []const f32,
    c: []f32,
) !void;

/// Scalar multiplication: c = a * scalar
pub fn scale(self: *CudaBackend, a: []const f32, scalar: f32, c: []f32) !void;

/// Fill buffer with constant value
pub fn fill(self: *CudaBackend, data: []f32, value: f32) !void;
```

**Example:**
```zig
var a: [100]f32 = undefined;
var b: [100]f32 = undefined;
var c: [100]f32 = undefined;

// c = a + b
try backend.elementWiseOp(.add, &a, &b, &c);

// c = a * 2.0
try backend.scale(&a, 2.0, &c);
```

### Map Operations

```zig
pub const MapOp = enum {
    exp,
    log,
    sqrt,
    abs,
    square,
    inv,
};

/// Apply unary function element-wise
pub fn mapOp(
    self: *CudaBackend,
    op: MapOp,
    input: []const f32,
    output: []f32,
) !void;
```

## Activation Functions

```zig
/// ReLU forward: output = max(0, input)
pub fn reluForward(self: *CudaBackend, input: []const f32, output: []f32) !void;

/// ReLU backward
pub fn reluBackward(
    self: *CudaBackend,
    output: []const f32,
    grad_output: []const f32,
    grad_input: []f32,
) !void;

/// Sigmoid forward: output = 1 / (1 + exp(-input))
pub fn sigmoidForward(self: *CudaBackend, input: []const f32, output: []f32) !void;

/// Sigmoid backward
pub fn sigmoidBackward(
    self: *CudaBackend,
    output: []const f32,
    grad_output: []const f32,
    grad_input: []f32,
) !void;

/// Tanh forward
pub fn tanhForward(self: *CudaBackend, input: []const f32, output: []f32) !void;

/// Tanh backward
pub fn tanhBackward(
    self: *CudaBackend,
    output: []const f32,
    grad_output: []const f32,
    grad_input: []f32,
) !void;

/// Softmax forward (per-row)
pub fn softmaxForward(
    self: *CudaBackend,
    input: []const f32,
    output: []f32,
    batch_size: usize,
    features: usize,
) !void;
```

**Example:**
```zig
var input: [100]f32 = undefined;
var output: [100]f32 = undefined;

// Apply ReLU
try backend.reluForward(&input, &output);

// Softmax for 10 samples, 10 features each
try backend.softmaxForward(&input, &output, 10, 10);
```

## Loss Functions

```zig
/// MSE loss backward: grad = 2 * (output - target) / n
pub fn mseBackward(
    self: *CudaBackend,
    output: []const f32,
    target: []const f32,
    grad_output: []f32,
) !void;

/// Cross-entropy loss backward
pub fn crossEntropyBackward(
    self: *CudaBackend,
    output: []const f32,
    target: []const f32,
    grad_output: []f32,
) !void;

/// Binary cross-entropy backward
pub fn binaryCrossEntropyBackward(
    self: *CudaBackend,
    output: []const f32,
    target: []const f32,
    grad_output: []f32,
    epsilon: f32,
) !void;

/// KL divergence backward
pub fn klDivergenceBackward(
    self: *CudaBackend,
    output: []const f32,
    grad_output: []f32,
    epsilon: f32,
) !void;
```

## Optimizers

```zig
/// SGD update with weight decay
/// weights = weights - lr * (grad + wd * weights)
pub fn sgdUpdate(
    self: *CudaBackend,
    weights: []f32,
    gradients: []const f32,
    learning_rate: f32,
    weight_decay: f32,
) !void;

/// Adam update
pub fn adamUpdate(
    self: *CudaBackend,
    weights: []f32,
    gradients: []const f32,
    m: []f32,           // First moment
    v: []f32,           // Second moment
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: u32,             // Timestep
) !void;

/// RMSprop update
pub fn rmspropUpdate(
    self: *CudaBackend,
    weights: []f32,
    gradients: []const f32,
    g_avg: []f32,       // Moving average of squared gradients
    lr: f32,
    rho: f32,
    eps: f32,
) !void;
```

**Example:**
```zig
var weights: [1000]f32 = undefined;
var gradients: [1000]f32 = undefined;

// SGD update
try backend.sgdUpdate(
    &weights,
    &gradients,
    0.01,   // learning_rate
    0.0001, // weight_decay
);
```

## Normalization

```zig
/// Layer normalization forward
pub fn layerNormForward(
    self: *CudaBackend,
    input: []const f32,
    output: []f32,
    gamma: []const f32,
    beta: []const f32,
    batch_size: usize,
    feature_size: usize,
    epsilon: f32,
) !void;

/// Batch normalization forward (training)
pub fn batchNormForwardTraining(
    self: *CudaBackend,
    input: []const f32,
    output: []f32,
    gamma: []const f32,
    beta: []const f32,
    running_mean: []f32,
    running_var: []f32,
    batch_size: usize,
    num_features: usize,
    momentum: f32,
    epsilon: f32,
) !void;

/// Batch normalization forward (inference)
pub fn batchNormForwardInference(
    self: *CudaBackend,
    input: []const f32,
    output: []f32,
    gamma: []const f32,
    beta: []const f32,
    running_mean: []const f32,
    running_var: []const f32,
    batch_size: usize,
    num_features: usize,
    epsilon: f32,
) !void;

/// Add bias to output
pub fn addBias(
    self: *CudaBackend,
    output: []f32,
    bias: []const f32,
    batch_size: usize,
    bias_size: usize,
) !void;
```

## CudaContext

Low-level context management for advanced use cases.

```zig
pub const CudaContext = struct {
    /// Device properties
    pub const DeviceProperties = struct {
        compute_capability_major: i32,
        compute_capability_minor: i32,
        total_memory: usize,
        multiprocessor_count: i32,
        max_threads_per_block: i32,
        max_block_dim_x: i32,
        max_block_dim_y: i32,
        max_block_dim_z: i32,
        max_grid_dim_x: i32,
        max_grid_dim_y: i32,
        max_grid_dim_z: i32,
        max_shared_memory_per_block: i32,
        warp_size: i32,
        memory_clock_rate: i32,
        global_memory_bus_width: i32,
        l2_cache_size: i32,
        max_threads_per_multiprocessor: i32,
        unified_addressing: i32,
        managed_memory: i32,
        concurrent_managed_access: i32,
        name: [256]u8,

        /// Check if device has Tensor Cores (sm_70+)
        pub fn hasTensorCores(self: DeviceProperties) bool;

        /// Check if device supports unified memory
        pub fn hasUnifiedMemory(self: DeviceProperties) bool;
    };

    /// Initialize context
    pub fn init(allocator: std.mem.Allocator) !*CudaContext;

    /// Cleanup context
    pub fn deinit(self: *CudaContext) void;

    /// Synchronize stream
    pub fn synchronize(self: *CudaContext) !void;

    /// Device properties
    pub fn getDeviceProperties(self: *CudaContext) DeviceProperties;
};
```

## DeviceBuffer

A handle to device memory.

```zig
pub const DeviceBuffer = struct {
    ptr: CUdeviceptr,       // Device pointer
    size: usize,            // Buffer size in bytes
    context: *CudaContext,  // Owning context

    /// Return buffer to pool
    pub fn deinit(self: *DeviceBuffer) void;
};
```

## CUDA Driver

Low-level driver API bindings.

### CudaDriver

```zig
pub const CudaDriver = struct {
    /// Initialize driver
    pub fn init(allocator: std.mem.Allocator) !CudaDriver;

    /// Cleanup driver
    pub fn deinit(self: *CudaDriver) void;

    /// Check if initialized
    pub fn isAvailable(self: *const CudaDriver) bool;

    /// Get error string
    pub fn getErrorString(self: *const CudaDriver, result: CUresult) []const u8;

    /// Get error name
    pub fn getErrorName(self: *const CudaDriver, result: CUresult) []const u8;
};
```

### CudaDriverRef

Thread-safe reference-counted driver access.

```zig
pub const CudaDriverRef = struct {
    /// Acquire reference to global driver
    pub fn acquire(allocator: std.mem.Allocator) !CudaDriverRef;

    /// Release reference
    pub fn release(self: CudaDriverRef) void;

    /// Check if reference is valid
    pub fn isValid(self: CudaDriverRef) bool;
};
```

## Error Types

```zig
pub const CudaError = error{
    CudaDriverNotFound,        // CUDA driver library not found
    CudaFunctionNotFound,      // Required function not in driver
    CudaInitFailed,            // Driver initialization failed
    CudaOutOfMemory,           // Device out of memory
    CudaInvalidValue,          // Invalid parameter value
    CudaInvalidDevice,         // Invalid device index
    CudaInvalidContext,        // Context not valid
    CudaInvalidHandle,         // Invalid resource handle
    CudaNotInitialized,        // CUDA not initialized
    CudaDeinitialized,         // CUDA already shut down
    CudaNoDevice,              // No CUDA-capable device
    CudaDeviceUnavailable,     // Device currently unavailable
    CudaUnknownError,          // Unclassified CUDA error
    UnsupportedPlatform,       // Platform not supported (e.g., macOS)
    NvrtcNotAvailable,         // NVRTC not available
    NvrtcProgramCreationFailed, // Failed to create NVRTC program
    NvrtcCompilationFailed,    // NVRTC compilation error
    InvalidPtx,               // Invalid PTX code
    UnsupportedPtxVersion,    // PTX version not supported
    NoBinaryForGpu,           // No binary for this GPU architecture
    ContextNotInitialized,    // CUDA context not initialized
    StreamNotInitialized,     // CUDA stream not initialized
    KernelNotFound,           // Requested kernel not loaded
    TooManyKernelArguments,   // Exceeded MAX_KERNEL_ARGS
    NullKernelArgument,       // Null pointer in kernel arguments
    InvalidGridDimension,     // Grid dimension exceeds limit
    InvalidBlockDimension,    // Block dimension exceeds limit
    TooManyThreadsPerBlock,   // Thread count exceeds limit
    SharedMemoryTooLarge,     // Shared memory exceeds limit
};
```

## Kernel Management

```zig
/// Load kernel from PTX code
pub fn loadKernel(
    self: *CudaContext,
    name: []const u8,
    ptx_code: []const u8,
) !void;

/// Compile and load kernel from CUDA C source
pub fn compileAndLoadKernel(
    self: *CudaContext,
    name: []const u8,
    source: []const u8,
) !void;

/// Check if kernel is loaded
pub fn hasKernel(self: *const CudaContext, name: []const u8) bool;

/// Launch kernel
pub fn launchKernel(
    self: *CudaContext,
    kernel_name: []const u8,
    grid_dim: [3]u32,       // Grid dimensions (x, y, z)
    block_dim: [3]u32,      // Block dimensions (x, y, z)
    shared_mem_bytes: u32,  // Shared memory size
    args: []const ?*anyopaque, // Kernel arguments
) !void;
```

**Example:**
```zig
// Load PTX kernel
const ptx_code = @embedFile("kernels/my_kernel.ptx");
try context.loadKernel("my_kernel", ptx_code);

// Or compile from source
try context.compileAndLoadKernel("my_kernel", cuda_source);

// Launch kernel
var arg1: f32 = 1.0;
var arg2: i32 = 100;
const args = [_]?*anyopaque{
    @ptrCast(&arg1),
    @ptrCast(&arg2),
};

try context.launchKernel(
    "my_kernel",
    .{ 10, 1, 1 },    // grid
    .{ 256, 1, 1 },   // block
    0,                // shared memory
    &args,
);
```

## Configuration Constants

```zig
/// Maximum number of kernel arguments
pub const MAX_KERNEL_ARGS: usize = 16;

/// Maximum shared memory per block (48KB)
pub const MAX_SHARED_MEMORY_PER_BLOCK: u32 = 48 * 1024;

/// Maximum grid dimension
pub const MAX_GRID_DIM: u32 = 65535;

/// Maximum threads per block
pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

/// Default block size for element-wise operations
pub const DEFAULT_BLOCK_SIZE: c_uint = 256;

/// Default threads per block
pub const DEFAULT_THREADS_PER_BLOCK: c_uint = 256;

/// Tile size for shared memory tiled matrix multiplication
pub const TILE_SIZE: u32 = 32;

/// Memory pool buckets (powers of 2 from 4 bytes to 128MB)
pub const MEMORY_POOL_BUCKETS: usize = 32;
pub const MIN_POOL_SIZE: usize = 4;
pub const MAX_POOL_SIZE: usize = 128 * 1024 * 1024;
```

## Type Mappings

| CUDA C Type | Zig Type | Description |
|-------------|----------|-------------|
| `float` | `f32` | 32-bit float |
| `int` | `c_int` | Signed integer |
| `unsigned int` | `c_uint` | Unsigned integer |
| `size_t` | `usize` | Size type |
| `CUdeviceptr` | `u64` | Device pointer |
| `CUresult` | `enum(c_int)` | CUDA result codes |
| `CUstream` | `*opaque{}` | CUDA stream |
| `CUcontext` | `*opaque{}` | CUDA context |
| `CUmodule` | `*opaque{}` | CUDA module |
| `CUfunction` | `*opaque{}` | CUDA function |

## See Also

- [User Guide](./user-guide.md) - Usage examples and tutorials
- [Developer Guide](./developer-guide.md) - Implementation details
- [CUDA Integration Guide](../CUDA_INTEGRATION.md) - Integration overview
- [CUDA Implementation Guide](../CUDA_IMPLEMENTATION_GUIDE.md) - Kernel development

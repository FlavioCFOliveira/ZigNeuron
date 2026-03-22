# CUDA Backend Developer Guide

Internal implementation details for developers working on the ZigNeuron CUDA backend.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [File Structure](#file-structure)
- [Driver Loading](#driver-loading)
- [Context Management](#context-management)
- [Memory Management](#memory-management)
- [Kernel Loading](#kernel-loading)
- [Kernel Launch](#kernel-launch)
- [Security Considerations](#security-considerations)
- [Performance Implementation](#performance-implementation)
- [Adding New Operations](#adding-new-operations)
- [Testing](#testing)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                          │
│              (User code, neural networks)                    │
├─────────────────────────────────────────────────────────────┤
│                    CudaBackend (cuda.zig)                    │
│  - High-level API (matMul, reluForward, etc.)                │
│  - Buffer management                                         │
│  - Operation dispatch                                        │
├─────────────────────────────────────────────────────────────┤
│                    CudaContext (cuda_context.zig)            │
│  - Context lifecycle                                         │
│  - Stream management                                         │
│  - Kernel cache                                              │
│  - Memory pool                                               │
├─────────────────────────────────────────────────────────────┤
│                    CudaDriver (cuda_driver.zig)                │
│  - Dynamic library loading                                   │
│  - Function pointer management                               │
│  - Error handling                                            │
│  - Reference counting                                        │
├─────────────────────────────────────────────────────────────┤
│                    CUDA Driver Library                        │
│              (libcuda.so / nvcuda.dll)                      │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
src/
├── cuda.zig                  # Main CUDA backend API
├── cuda_context.zig          # Context and resource management
├── cuda_driver.zig           # CUDA driver API bindings
├── cuda_kernels.zig        # Kernel PTX and source code
└── cuda_tensor_core.zig      # Tensor Core operations (optional)
```

## Driver Loading

### Dynamic Loading Pattern

The CUDA backend uses dynamic loading to avoid static dependencies:

```zig
// From cuda_driver.zig
const lib_name = switch (@import("builtin").os.tag) {
    .linux => "libcuda.so",
    .windows => "nvcuda.dll",
    else => return error.UnsupportedPlatform,
};

// Open library
const lib = std.DynLib.open(lib_name) catch |err| {
    return error.CudaDriverNotFound;
};

// Load function pointers
try driver.loadFunction("cuInit", &driver.cuInit);
try driver.loadFunction("cuDeviceGetCount", &driver.deviceGetCount);
// ... more functions
```

### Reference Counting

The `CudaDriverRef` provides thread-safe reference counting:

```zig
pub const CudaDriverRef = struct {
    driver: *CudaDriver,

    pub fn acquire(allocator: std.mem.Allocator) !CudaDriverRef {
        // Check if already initialized (fast path)
        if (is_driver_initialized.load(.acquire)) {
            const prev_count = driver_ref_count.fetchAdd(1, .acquire);
            if (prev_count > 0) {
                return CudaDriverRef{ .driver = &global_driver.? };
            }
        }

        // Slow path: initialize under mutex
        // ... initialization code ...
    }

    pub fn release(self: CudaDriverRef) void {
        const prev_count = driver_ref_count.fetchSub(1, .acq_rel);
        if (prev_count == 1) {
            // Last reference, deinitialize
            self.driver.deinit();
            global_driver = null;
            is_driver_initialized.store(false, .release);
        }
    }
};
```

**Key Points:**
- Fast path is lock-free using atomic operations
- Slow path uses mutex for initialization
- Reference count ensures safe cleanup
- Prevents use-after-free (CRIT-002 fix)

## Context Management

### Context Structure

```zig
pub const CudaContext = struct {
    allocator: std.mem.Allocator,
    driver: cuda_driver.CudaDriverRef,

    // CUDA handles
    device: CUdevice,
    context: ?*CUcontext,
    stream: ThreadSafeStream,  // Thread-safe wrapper

    // Device properties
    device_props: DeviceProperties,

    // Resource management
    kernels: std.StringHashMap(Kernel),
    buffer_pools: [MEMORY_POOL_BUCKETS]std.ArrayListUnmanaged(DeviceBuffer),
    temp_buffers: std.ArrayListUnmanaged(*DeviceBuffer),
    modules: std.ArrayListUnmanaged(*CUmodule),
};
```

### Thread-Safe Stream

```zig
const ThreadSafeStream = struct {
    stream: ?*CUstream,
    mutex: std.atomic.Mutex,

    pub fn get(self: *ThreadSafeStream) ?*CUstream {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        return self.stream;
    }

    pub fn getOrError(self: *ThreadSafeStream) !*CUstream {
        // Same pattern, returns error if not initialized
    }

    pub fn take(self: *ThreadSafeStream) ?*CUstream {
        // Atomically get and clear (for cleanup)
    }
};
```

### Device Selection

```zig
fn selectBestDevice(driver_ref: *CudaDriver, count: c_int) !CUdevice {
    var best_device: CUdevice = 0;
    var best_score: i32 = -1;

    var i: c_int = 0;
    while (i < count) : (i += 1) {
        var device: CUdevice = 0;
        try cuda_driver.checkCuda(driver_ref.deviceGet.?(&device, i));

        const props = try queryDeviceProperties(driver_ref, device);

        // Score: compute capability * 100 + multiprocessors
        const score = props.computeCapability() * 100 + props.multiprocessor_count;

        if (score > best_score) {
            best_score = score;
            best_device = device;
        }
    }

    return best_device;
}
```

## Memory Management

### Buffer Pool

The memory pool uses power-of-2 buckets:

```zig
pub const MEMORY_POOL_BUCKETS: usize = 32;  // 4 bytes to 128MB
pub const MIN_POOL_SIZE: usize = 4;
pub const MAX_POOL_SIZE: usize = 128 * 1024 * 1024;

pub fn getBuffer(self: *CudaContext, size: usize) !DeviceBuffer {
    const pool_idx = getPoolIndex(size);

    // Try to get from pool
    if (pool_idx < MEMORY_POOL_BUCKETS) {
        if (self.buffer_pools[pool_idx].items.len > 0) {
            return self.buffer_pools[pool_idx].pop().?;
        }
    }

    // Allocate new buffer
    const aligned_size = getPooledSize(size);
    var ptr: CUdeviceptr = 0;
    try cuda_driver.checkCuda(self.driver.driver.memAlloc.?(&ptr, aligned_size));

    return DeviceBuffer{
        .ptr = ptr,
        .size = aligned_size,
        .pool_index = pool_idx,
    };
}

pub fn returnBuffer(self: *CudaContext, buffer: DeviceBuffer) void {
    // Validate buffer not already freed (VULN-004 fix)
    if (buffer.ptr == 0) {
        std.log.warn("Attempting to return already-freed buffer to pool");
        return;
    }

    if (buffer.pool_index) |idx| {
        if (idx < MEMORY_POOL_BUCKETS) {
            self.buffer_pools[idx].append(self.allocator, buffer) catch {
                // Pool full, free directly
                var buf = buffer;
                buf.deinit(self.driver.driver);
            };
            return;
        }
    }
    // Not poolable, free directly
    var buf = buffer;
    buf.deinit(self.driver.driver);
}
```

### Memory Transfer

```zig
pub fn upload(self: *CudaContext, dst: CUdeviceptr, src: []const u8) !void {
    try cuda_driver.checkCuda(self.driver.driver.memcpyHtoD.?(
        dst,
        src.ptr,
        src.len,
    ));
}

pub fn uploadAsync(self: *CudaContext, dst: CUdeviceptr, src: []const u8) !void {
    const strm = self.stream.getOrError() catch |err| return err;
    try cuda_driver.checkCuda(self.driver.driver.memcpyHtoDAsync.?(
        dst,
        src.ptr,
        src.len,
        strm,
    ));
}
```

## Kernel Loading

### PTX Loading

```zig
pub fn loadKernel(
    self: *CudaContext,
    name: []const u8,
    ptx_code: []const u8,
) !void {
    // Check if already loaded
    if (self.kernels.contains(name)) {
        return;
    }

    // SECURITY: Validate PTX before loading
    validateAndLogPtx(ptx_code, name) catch |err| {
        return error.InvalidPtx;
    };

    // Ensure null-terminated
    const ptx_z = try self.allocator.dupeZ(u8, ptx_code);
    defer self.allocator.free(ptx_z);

    // Load module
    var module: *CUmodule = undefined;
    const result = self.driver.driver.moduleLoadData.?(
        &module,
        ptx_z.ptr,
    );

    if (result != .SUCCESS) {
        // Handle specific errors
        switch (result) {
            .ERROR_INVALID_PTX => return error.InvalidPtx,
            .ERROR_UNSUPPORTED_PTX_VERSION => return error.UnsupportedPtxVersion,
            .ERROR_NO_BINARY_FOR_GPU => return error.NoBinaryForGpu,
            else => return error.CudaUnknownError,
        }
    }

    // Get function
    var function: *CUfunction = undefined;
    const name_z = try self.allocator.dupeZ(u8, name);
    defer self.allocator.free(name_z);

    try cuda_driver.checkCuda(self.driver.driver.moduleGetFunction.?(
        &function,
        module,
        name_z,
    ));

    // Get occupancy info
    var min_grid_size: c_int = 0;
    var max_threads_per_block: c_int = 0;
    if (self.driver.driver.occupancyMaxPotentialBlockSize) |occupancy_fn| {
        _ = occupancy_fn(
            &min_grid_size,
            &max_threads_per_block,
            function,
            null,
            0,
            0,
        );
    }

    // Store in cache
    const kernel = Kernel{
        .function = function,
        .module = module,
        .max_threads_per_block = max_threads_per_block,
        .min_grid_size = min_grid_size,
    };

    const name_copy = try self.allocator.dupe(u8, name);
    try self.kernels.put(name_copy, kernel);
}
```

### NVRTC Compilation

```zig
pub fn compileAndLoadKernel(
    self: *CudaContext,
    name: []const u8,
    source: []const u8,
) !void {
    // Check if already loaded
    if (self.kernels.contains(name)) {
        return;
    }

    if (!self.driver.driver.is_nvrtc_available) {
        return error.NvrtcNotAvailable;
    }

    // Compile to PTX
    const ptx = try self.compileKernel(source, name);
    defer self.allocator.free(ptx);

    // Load compiled PTX
    try self.loadKernel(name, ptx);
}

fn compileKernel(
    self: *CudaContext,
    source: []const u8,
    name: []const u8,
) ![]u8 {
    var program: *nvrtcProgram = undefined;

    // Create program
    const src_z = try self.allocator.dupeZ(u8, source);
    defer self.allocator.free(src_z);

    const name_z = try self.allocator.dupeZ(u8, name);
    defer self.allocator.free(name_z);

    const create_result = self.driver.driver.nvrtcCreateProgram.?(
        &program,
        src_z.ptr,
        name_z.ptr,
        0, null, null,
    );
    if (!create_result.isSuccess()) return error.NvrtcProgramCreationFailed;
    defer _ = self.driver.driver.nvrtcDestroyProgram.?(&program);

    // Compile
    const compile_result = self.driver.driver.nvrtcCompileProgram.?(
        program,
        0, null,  // No options (default PTX for compatibility)
    );

    // Get log
    var log_size: usize = 0;
    _ = self.driver.driver.nvrtcGetProgramLogSize.?(program, &log_size);
    if (log_size > 1) {
        const log_buf = try self.allocator.alloc(u8, log_size);
        defer self.allocator.free(log_buf);
        _ = self.driver.driver.nvrtcGetProgramLog.?(program, log_buf.ptr);
        if (!compile_result.isSuccess()) {
            std.log.err("NVRTC compilation failed: {s}", .{log_buf});
        }
    }

    if (!compile_result.isSuccess()) {
        return error.NvrtcCompilationFailed;
    }

    // Get PTX
    var ptx_size: usize = 0;
    _ = self.driver.driver.nvrtcGetPTXSize.?(program, &ptx_size);

    const ptx = try self.allocator.alloc(u8, ptx_size);
    _ = self.driver.driver.nvrtcGetPTX.?(program, ptx.ptr);

    return ptx;
}
```

## Kernel Launch

### Launch with Validation

```zig
pub fn launchKernel(
    self: *CudaContext,
    kernel_name: []const u8,
    grid_dim: [3]u32,
    block_dim: [3]u32,
    shared_mem_bytes: u32,
    args: []const ?*anyopaque,
) !void {
    const strm = self.stream.getOrError() catch |err| return err;
    const kernel = try self.getKernel(kernel_name);

    // SECURITY: Validate parameters (HIGH-005 fix)

    // Validate argument count
    if (args.len > MAX_KERNEL_ARGS) {
        return error.TooManyKernelArguments;
    }

    // Validate all arguments non-null
    for (args, 0..) |arg, i| {
        if (arg == null) {
            std.log.err("Kernel '{s}' argument {d} is null", .{ kernel_name, i });
            return error.NullKernelArgument;
        }
    }

    // Validate grid dimensions
    if (grid_dim[0] > self.device_props.max_grid_dim_x or
        grid_dim[0] > MAX_GRID_DIM) {
        return error.InvalidGridDimension;
    }
    // ... validate other dimensions

    // Validate block dimensions
    const total_threads = @as(u64, block_dim[0]) *
                          @as(u64, block_dim[1]) *
                          @as(u64, block_dim[2]);
    if (total_threads > self.device_props.max_threads_per_block or
        total_threads > MAX_THREADS_PER_BLOCK) {
        return error.TooManyThreadsPerBlock;
    }

    // Validate shared memory
    if (shared_mem_bytes > self.device_props.max_shared_memory_per_block) {
        return error.SharedMemoryTooLarge;
    }

    // Prepare and launch
    var kernel_params: [MAX_KERNEL_ARGS]?*anyopaque = undefined;
    @memcpy(kernel_params[0..args.len], args);

    try cuda_driver.checkCuda(self.driver.driver.launchKernel.?(
        kernel.function,
        grid_dim[0], grid_dim[1], grid_dim[2],
        block_dim[0], block_dim[1], block_dim[2],
        shared_mem_bytes,
        strm,
        &kernel_params,
        null,
    ));
}
```

### Configuration Helpers

```zig
/// Element-wise operations configuration
pub fn getElementWiseConfig(
    _: *const CudaContext,
    total_elements: usize,
) struct { grid: u32, block: u32 } {
    const block = DEFAULT_BLOCK_SIZE;  // 256
    const grid = @as(u32, @intCast((total_elements + block - 1) / block));
    return .{ .grid = grid, .block = block };
}

/// Tiled matrix multiplication configuration
pub fn getTiledMatMulConfig(
    _: *const CudaContext,
    m: usize,
    n: usize,
) struct { grid_x: u32, grid_y: u32, block_x: u32, block_y: u32, shared_mem_bytes: u32 } {
    const block_x = TILE_SIZE;  // 32
    const block_y = TILE_SIZE;  // 32
    const grid_x = @as(u32, @intCast((n + block_x - 1) / block_x));
    const grid_y = @as(u32, @intCast((m + block_y - 1) / block_y));
    const shared_mem_bytes = 2 * TILE_SIZE * TILE_SIZE * @sizeOf(f32);
    return .{
        .grid_x = grid_x,
        .grid_y = grid_y,
        .block_x = block_x,
        .block_y = block_y,
        .shared_mem_bytes = shared_mem_bytes,
    };
}
```

## Security Considerations

### PTX Validation

```zig
pub const PtxValidationError = error{
    PtxTooShort,
    InvalidPtxHeader,
    InvalidPtxVersion,
    InvalidPtxTarget,
    SuspiciousPtxPattern,
    PtxContainsNullBytes,
    PtxTooLarge,
};

pub fn validatePtx(ptx: []const u8) PtxValidationError!void {
    // Check minimum length
    if (ptx.len < 30) {
        return PtxValidationError.PtxTooShort;
    }

    // Check maximum size (32MB)
    if (ptx.len > MAX_PTX_SIZE) {
        return PtxValidationError.PtxTooLarge;
    }

    // Check for null bytes (except terminator)
    const content_to_check = if (ptx[ptx.len - 1] == 0)
        ptx[0 .. ptx.len - 1]
    else
        ptx;
    if (std.mem.indexOfScalar(u8, content_to_check, 0) != null) {
        return PtxValidationError.PtxContainsNullBytes;
    }

    // Validate PTX header
    if (!std.mem.startsWith(u8, ptx, ".version")) {
        // Check after whitespace/comments
        // ...
    }

    // Check for suspicious patterns
    const SUSPICIOUS_PATTERNS = &[_][]const u8{
        "// malicious",
        "/* exploit",
        "<script",
        "javascript:",
        "eval(",
        "execve",
        "system(",
    };
    for (SUSPICIOUS_PATTERNS) |pattern| {
        if (std.mem.indexOf(u8, ptx, pattern) != null) {
            return PtxValidationError.SuspiciousPtxPattern;
        }
    }
}
```

### Integer Overflow Protection

```zig
// Use std.math.mul with overflow checking
const size = try std.math.mul(usize,
    try std.math.mul(usize, m, k),
    @sizeOf(f32));

// Instead of: const size = m * k * @sizeOf(f32);  // May overflow!
```

### Buffer Lifecycle Validation

```zig
pub fn freeBuffer(self: *CudaContext, buffer: *DeviceBuffer) void {
    // SECURITY: Validate buffer state before freeing (VULN-004)
    if (buffer.ptr == 0) {
        std.log.warn("Attempting to free already-freed buffer");
        return;
    }
    buffer.deinit(self.driver.driver);
}
```

## Performance Implementation

### Vectorized Kernels

```zig
/// Check if vectorized kernels can be used
fn canUseVectorized(n: usize, ptr1: *const anyopaque, ptr2: *const anyopaque) bool {
    const VECTORIZATION_THRESHOLD: usize = 1024;
    if (n < VECTORIZATION_THRESHOLD) return false;

    const addr1 = @intFromPtr(ptr1);
    const addr2 = @intFromPtr(ptr2);
    return (addr1 % 16 == 0) and (addr2 % 16 == 0);
}

// Usage in reluForward:
const use_vec4 = canUseVectorized(input.len, input.ptr, output.ptr);
if (use_vec4 and self.context.hasKernel("relu_forward_vec4")) {
    try self.activationForwardVec4("relu_forward_vec4", input, output);
} else {
    try self.activationForward("relu_forward", input, output);
}
```

### Tensor Core Selection

```zig
fn shouldUseTensorCores(self: *CudaBackend, m: usize, n: usize, k: usize) bool {
    // Check device capability
    if (!self.context.device_props.hasTensorCores()) {
        return false;
    }

    // Check dimensions compatible
    if (m % 16 != 0 or n % 16 != 0 or k % 16 != 0) {
        return false;
    }

    // Check matrix large enough
    return m >= 64 and n >= 64 and k >= 64;
}
```

### Tiled Matrix Multiplication

```zig
// Configuration for shared memory tiling
const TILE_SIZE: u32 = 32;
const config = self.context.getTiledMatMulConfig(m, n);

try self.context.launchKernel(
    "matmul_tiled",
    .{ config.grid_x, config.grid_y, 1 },
    .{ config.block_x, config.block_y, 1 },
    config.shared_mem_bytes,  // 8KB for 2 tiles
    &args,
);
```

## Adding New Operations

### Step 1: Add Kernel Source

In `cuda_kernels.zig`:

```zig
/// My operation kernel CUDA C source
pub const MY_OPERATION_SOURCE =
    \\extern "C" __global__ void my_operation(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        output[idx] = my_function(input[idx]);
    \\    }
    \\}
;

/// PTX (optional - for fallback)
pub const MY_OPERATION_PTX =
    \\.version 7.5
    \\.target sm_80
    \\.address_size 64
    \\    // ... PTX code ...
;
```

### Step 2: Register Kernel

In `cuda.zig` `loadBuiltinKernels`:

```zig
const kernels = [_]KernelDef{
    // ... existing kernels ...
    .{ .name = "my_operation", .source = cuda_kernels.MY_OPERATION_SOURCE, .ptx = cuda_kernels.MY_OPERATION_PTX },
};
```

### Step 3: Add Backend Method

```zig
pub fn myOperation(
    self: *CudaBackend,
    input: []const f32,
    output: []f32,
) !void {
    const size = try std.math.mul(usize, input.len, @sizeOf(f32));

    var d_input = try self.context.getBuffer(size);
    defer self.context.returnBuffer(d_input);
    var d_output = try self.context.getBuffer(size);
    defer self.context.returnBuffer(d_output);

    try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

    var len_u32: u32 = @intCast(input.len);
    const args = [_]?*anyopaque{
        @ptrCast(&d_input.ptr),
        @ptrCast(&d_output.ptr),
        @ptrCast(&len_u32),
    };

    const config = self.context.getElementWiseConfig(input.len);

    try self.context.launchKernel(
        "my_operation",
        .{ config.grid, 1, 1 },
        .{ config.block, 1, 1 },
        0,
        &args,
    );

    try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
}
```

### Step 4: Add to Backend Dispatch

In `backend.zig`:

```zig
pub const Backend = union(enum) {
    // ... existing backends ...
    cuda: *cuda.CudaBackend,

    pub fn myOperation(self: Backend, input: []const f32, output: []f32) !void {
        switch (self) {
            .cuda => |backend| try backend.myOperation(input, output),
            .cpu => try cpuMyOperation(input, output),
            // ...
        }
    }
};
```

## Testing

### Unit Tests

```zig
test "CUDA ReLU forward" {
    if (!cuda.CudaBackend.isAvailable()) {
        return error.SkipZigTest;
    }

    var backend = try cuda.CudaBackend.init(std.testing.allocator);
    defer backend.deinit();

    var input = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    var output: [4]f32 = undefined;
    var expected = [_]f32{ 0.0, 0.0, 1.0, 2.0 };

    try backend.reluForward(&input, &output);

    try std.testing.expectEqualSlices(f32, &expected, &output);
}
```

### Validation Tests

```zig
test "CUDA vs CPU numerical accuracy" {
    if (!cuda.CudaBackend.isAvailable()) {
        return error.SkipZigTest;
    }

    var backend = try cuda.CudaBackend.init(std.testing.allocator);
    defer backend.deinit();

    // Generate random input
    var input: [1000]f32 = undefined;
    var rng = std.rand.DefaultPrng.init(42);
    for (&input) |*val| {
        val.* = rng.random().float(f32) * 2.0 - 1.0;
    }

    // CPU computation
    var cpu_output: [1000]f32 = undefined;
    for (input, 0..) |val, i| {
        cpu_output[i] = @max(0.0, val);
    }

    // GPU computation
    var gpu_output: [1000]f32 = undefined;
    try backend.reluForward(&input, &gpu_output);

    // Compare with tolerance
    for (cpu_output, gpu_output, 0..) |cpu, gpu, i| {
        const diff = @abs(cpu - gpu);
        try std.testing.expect(diff < 1e-5);
    }
}
```

### Performance Tests

```zig
test "CUDA matmul performance" {
    if (!cuda.CudaBackend.isAvailable()) {
        return error.SkipZigTest;
    }

    var backend = try cuda.CudaBackend.init(std.testing.allocator);
    defer backend.deinit();

    const N: usize = 1024;
    var A = try allocator.alloc(f32, N * N);
    var B = try allocator.alloc(f32, N * N);
    var C = try allocator.alloc(f32, N * N);
    defer allocator.free(A);
    defer allocator.free(B);
    defer allocator.free(C);

    // Initialize...

    var timer = try std.time.Timer.start();
    try backend.matMul(A, B, C, N, N, N, false, false, false);
    const elapsed = timer.read();

    const gflops = (2.0 * @as(f64, N) * N * N) / @as(f64, elapsed) * 1e9;
    std.log.info("MatMul {d}x{d}x{d}: {d:.2} GFLOPS", .{ N, N, N, gflops });
}
```

## References

- [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [API Reference](./api.md) - Public API documentation
- [User Guide](./user-guide.md) - Usage examples

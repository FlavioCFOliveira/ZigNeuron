# CUDA Backend Troubleshooting Guide

Common issues and solutions for the ZigNeuron CUDA backend.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Numerical Issues](#numerical-issues)
- [Debugging Tips](#debugging-tips)
- [Known Limitations](#known-limitations)

## Installation Issues

### "CUDA driver not found"

**Symptom:**
```
error: CudaDriverNotFound
```

**Causes:**
1. NVIDIA drivers not installed
2. Library not in search path
3. Wrong architecture (e.g., macOS)

**Solutions:**

**Linux:**
```bash
# Check if driver is installed
nvidia-smi

# If not installed, install drivers
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535

# Fedora
sudo dnf install akmod-nvidia

# Verify after reboot
nvidia-smi
```

**Windows:**
1. Download drivers from [NVIDIA website](https://www.nvidia.com/drivers)
2. Install with "Custom (Advanced)" option
3. Select "Perform clean installation"

**Library Path:**
```bash
# Add to LD_LIBRARY_PATH (Linux)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Add to PATH (Windows)
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin;%PATH%
```

### "NVRTC not available"

**Symptom:**
```
warning: NVRTC library not found. Falling back to pre-compiled PTX.
```

**Cause:** NVRTC (runtime compilation) library not found.

**Solution:**
```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Or set path explicitly
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Runtime Errors

### "Out of memory"

**Symptom:**
```
error: CudaOutOfMemory
```

**Causes:**
1. Batch size too large
2. Model too large for GPU
3. Memory leaks
4. Memory fragmentation

**Solutions:**

**1. Reduce Batch Size:**
```zig
// Instead of batch_size = 256
try trainModel(batch_size: 64);  // Try smaller
```

**2. Check Memory Usage:**
```bash
# Monitor in real-time
watch -n 0.5 nvidia-smi
```

**3. Process in Chunks:**
```zig
fn processLargeArray(
    backend: *cuda.CudaBackend,
    data: []const f32,
) !void {
    const chunk_size: usize = 10000;
    var offset: usize = 0;

    while (offset < data.len) : (offset += chunk_size) {
        const end = @min(offset + chunk_size, data.len);
        const chunk = data[offset..end];

        // Process chunk
        try backend.processChunk(chunk);

        // Explicitly sync and free resources
        try backend.synchronize();
    }
}
```

**4. Clear Memory Pools:**
```zig
// Force return all pooled buffers
for (&context.buffer_pools) |*pool| {
    while (pool.items.len > 0) {
        var buf = pool.pop().?;
        buf.deinit(context.driver.driver);
    }
}
```

### "Invalid PTX"

**Symptom:**
```
error: InvalidPtx
error: UnsupportedPtxVersion
error: NoBinaryForGpu
```

**Causes:**
1. PTX version incompatible with driver
2. GPU architecture not supported
3. Corrupted PTX code

**Solutions:**

**1. Use NVRTC Runtime Compilation:**
```zig
// Instead of loading embedded PTX
try context.compileAndLoadKernel("my_kernel", cuda_source);

// This generates PTX for your specific GPU
```

**2. Check GPU Capability:**
```zig
const props = backend.context.device_props;
std.log.info("Compute Capability: {d}.{d}", .{
    props.compute_capability_major,
    props.compute_capability_minor,
});

// Verify PTX target matches
try backend.validatePtx(ptx_code);
```

**3. Update NVIDIA Driver:**
```bash
# Check current version
nvidia-smi | grep "Driver Version"

# Update to latest
sudo apt update
sudo apt install nvidia-driver-XXX
```

### "Invalid kernel arguments"

**Symptom:**
```
error: NullKernelArgument
error: TooManyKernelArguments
```

**Causes:**
1. Null pointer passed to kernel
2. Too many arguments (max 16)
3. Wrong argument order

**Solutions:**

**1. Check Arguments:**
```zig
// Ensure all buffers are valid
if (buffer.ptr == 0) {
    return error.InvalidBuffer;
}

// Check argument count
if (args.len > cuda.MAX_KERNEL_ARGS) {
    return error.TooManyArguments;
}
```

**2. Use Structured Arguments:**
```zig
// Instead of many individual args, use struct
const KernelArgs = extern struct {
    input: CUdeviceptr,
    output: CUdeviceptr,
    size: u32,
};

var args = KernelArgs{
    .input = d_input.ptr,
    .output = d_output.ptr,
    .size = @intCast(size),
};

try context.launchKernel(
    "my_kernel",
    .{ grid, 1, 1 },
    .{ block, 1, 1 },
    0,
    &[_]?*anyopaque{@ptrCast(&args)},
);
```

### "Invalid grid/block dimensions"

**Symptom:**
```
error: InvalidGridDimension
error: TooManyThreadsPerBlock
error: SharedMemoryTooLarge
```

**Causes:**
1. Grid dimensions exceed device limits
2. Block size too large
3. Too much shared memory requested

**Solutions:**

**Check Device Limits:**
```zig
const props = backend.context.device_props;
std.log.info("Max threads per block: {d}", .{props.max_threads_per_block});
std.log.info("Max grid dimensions: {d} x {d} x {d}", .{
    props.max_grid_dim_x,
    props.max_grid_dim_y,
    props.max_grid_dim_z,
});
std.log.info("Max shared memory: {d} bytes", .{props.max_shared_memory_per_block});
```

**Calculate Safe Configuration:**
```zig
fn calculateSafeConfig(
    backend: *cuda.CudaBackend,
    total_elements: usize,
) !struct { grid: u32, block: u32 } {
    const props = backend.context.device_props;

    // Start with default block size
    var block: u32 = 256;

    // Reduce if needed
    if (block > props.max_threads_per_block) {
        block = @intCast(props.max_threads_per_block);
    }

    // Calculate grid
    var grid = @as(u32, @intCast((total_elements + block - 1) / block));

    // Clamp to max grid
    if (grid > props.max_grid_dim_x) {
        grid = @intCast(props.max_grid_dim_x);
        std.log.warn("Grid size clamped to {d}", .{grid});
    }

    return .{ .grid = grid, .block = block };
}
```

## Performance Issues

### Low GPU Utilization

**Symptom:** GPU utilization < 50% in `nvidia-smi`

**Causes:**
1. Kernel launch overhead
2. CPU-GPU synchronization too frequent
3. Small problem sizes
4. Memory-bound operations

**Solutions:**

**1. Batch Operations:**
```zig
// Instead of many small operations
for (0..100) |_| {
    try backend.reluForward(small_input, small_output);
}

// Do one large operation
try backend.reluForward(large_input, large_output);
```

**2. Use Async Operations:**
```zig
// Upload while GPU is working
try backend.uploadAsync(d_input.ptr, next_batch);
try backend.reluForward(current_input, current_output);
```

**3. Increase Batch Size:**
```zig
// Larger batches = more parallel work
const batch_size = 256;  // Instead of 32
```

**4. Profile with Nsight:**
```bash
# Capture GPU trace
nsys profile -o trace ./your_app

# Analyze in Nsight Systems
nsys-ui trace.nsys-rep
```

### Slow Memory Transfers

**Symptom:** CPU-GPU transfers taking significant time

**Solutions:**

**1. Use Pinned Memory:**
```zig
// Allocate pinned host memory for faster transfers
var pinned = try cuda.allocPinned(size);
defer cuda.freePinned(pinned);
```

**2. Minimize Transfers:**
```zig
// BAD: Transfer each layer
for (layers) |layer| {
    try backend.upload(d_layer.ptr, layer.weights);
    // ... compute ...
    try backend.download(result, d_output.ptr);
}

// GOOD: Keep data on GPU
// Upload once at start
// Download once at end
```

**3. Async Transfers:**
```zig
try backend.uploadAsync(d_input.ptr, host_input);
// Do other work while transfer happens
// ...
try backend.synchronize();
```

### Tensor Cores Not Used

**Symptom:** No performance improvement on sm_70+

**Causes:**
1. Dimensions not aligned to 16
2. Matrices too small
3. Tensor Cores not enabled

**Solutions:**

**1. Check Alignment:**
```zig
// Must be multiples of 16
const M: usize = 256;  // Good
const N: usize = 257;  // Bad - will not use Tensor Cores

// Pad dimensions
const M_aligned = (M + 15) / 16 * 16;
```

**2. Check Capability:**
```zig
if (!backend.context.device_props.hasTensorCores()) {
    std.log.info("Tensor Cores not available");
}
```

**3. Verify Usage:**
```zig
// Check if Tensor Core path is taken
if (backend.shouldUseTensorCores(M, N, K)) {
    std.log.info("Using Tensor Cores");
    try backend.matMulTensorCore(A, B, C, M, N, K, false);
} else {
    std.log.info("Using standard CUDA cores");
    try backend.matMul(A, B, C, M, N, K, false, false, false);
}
```

## Numerical Issues

### NaN or Inf Results

**Symptom:**
```
warning: NaN detected in output
warning: Inf detected in output
```

**Causes:**
1. Division by zero
2. Exp overflow
3. Gradient explosion

**Solutions:**

**1. Add Epsilon for Stability:**
```zig
// Before division
const eps = 1e-8;
if (@abs(divisor) < eps) {
    divisor = eps;
}
```

**2. Clip Gradients:**
```zig
// Gradient clipping
const max_grad: f32 = 5.0;
for (&gradients) |*g| {
    g.* = std.math.clamp(g.*, -max_grad, max_grad);
}
```

**3. Check Inputs:**
```zig
// Validate before GPU operation
for (input) |val| {
    if (std.math.isNan(val) or std.math.isInf(val)) {
        return error.InvalidInput;
    }
}
```

### Numerical Mismatch with CPU

**Symptom:** GPU results differ from CPU implementation

**Causes:**
1. Different floating-point precision
2. Race conditions in reduction
3. Associativity differences

**Solutions:**

**1. Use Tolerant Comparison:**
```zig
const TOLERANCE: f32 = 1e-4;

for (cpu_result, gpu_result) |cpu, gpu| {
    const diff = @abs(cpu - gpu);
    const relative_diff = diff / @abs(cpu);

    if (diff > TOLERANCE and relative_diff > TOLERANCE) {
        std.log.err("Mismatch: CPU={d}, GPU={d}", .{cpu, gpu});
    }
}
```

**2. Check Reduction Order:**
```zig
// Summation order affects result
// GPU and CPU may accumulate differently
// Use Kahan summation for consistency
```

**3. Verify FP16 vs FP32:**
```zig
// Check if Tensor Cores using FP16
// Compare with FP32 reference
```

## Debugging Tips

### Enable Debug Logging

```zig
// Set log level
try std.log.setLevel(.debug);

// Enable CUDA debug output
export CU_LOG_LEVEL=3;
```

### Check Kernel Launch

```zig
// Log kernel launches
std.log.debug("Launching kernel: {s}", .{kernel_name});
std.log.debug("  Grid: {} x {} x {}", .{grid_dim});
std.log.debug("  Block: {} x {} x {}", .{block_dim});
std.log.debug("  Shared mem: {} bytes", .{shared_mem});
```

### Validate Memory Access

```zig
// Check bounds before kernel launch
if (buffer_size < required_size) {
    std.log.err("Buffer too small: {d} < {d}", .{
        buffer_size, required_size
    });
    return error.BufferTooSmall;
}
```

### Use Compute Sanitizer

```bash
# Check for memory errors
compute-sanitizer --tool memcheck ./your_app

# Check for race conditions
compute-sanitizer --tool racecheck ./your_app

# Profile API usage
compute-sanitizer --tool initcheck ./your_app
```

### Capture Debug Info

```zig
// Dump intermediate results
fn debugDump(
    name: []const u8,
    data: []const f32,
) void {
    std.log.debug("{s}: ", .{name});
    for (data[0..@min(10, data.len)]) |val| {
        std.log.debug("  {d:.6}", .{val});
    }
    if (data.len > 10) {
        std.log.debug("  ... ({d} more)", .{data.len - 10});
    }
}
```

### Kernel Debugging

```zig
// Add printf in kernel (for debugging)
const DEBUG_KERNEL_SOURCE =
    \\extern "C" __global__ void debug_kernel(
    \\    const float* input, float* output, int n) {
    \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (idx < n) {
    \\        float val = input[idx];
    \\        if (idx < 5) printf("Thread %d: input=%f\\n", idx, val);
    \\        output[idx] = val;
    \\    }
    \\}
;
```

## Known Limitations

### Current Limitations

1. **Single Stream:** Currently uses default stream only
   - Workaround: Create multiple contexts for parallel streams

2. **No Unified Memory:** Explicit copies required
   - Workaround: Use `upload`/`download` explicitly

3. **FP16 Limited:** Tensor Cores require manual conversion
   - Workaround: Use `matMulTensorCore` with FP32 inputs

4. **No CUDA Graphs:** Each kernel launch has overhead
   - Workaround: Batch operations where possible

5. **Linux/Windows Only:** macOS not supported
   - Workaround: Use Metal backend on macOS

### Platform-Specific Issues

**Linux:**
- May require LD_LIBRARY_PATH setup
- Kernel driver must match user libraries

**Windows:**
- WSL not officially supported
- May need Visual C++ redistributables
- Long paths can cause issues with NVRTC

### Version Compatibility

| CUDA Driver | NVRTC | PTX | Status |
|-------------|-------|-----|--------|
| 11.0+ | 11.0+ | 7.0+ | Supported |
| 12.0+ | 12.0+ | 8.0+ | Supported |
| 10.x | 10.x | 6.x | Limited |

## Getting Help

### Diagnostic Information

When reporting issues, include:

```zig
// Run this to collect system info
pub fn collectDiagnostics() void {
    std.log.info("=== CUDA Diagnostics ===");

    // Zig version
    std.log.info("Zig version: {s}", .{builtin.zig_version_string});

    // OS
    std.log.info("OS: {s}", .{@tagName(builtin.os.tag)});

    // CUDA availability
    std.log.info("CUDA available: {}", .{cuda.CudaBackend.isAvailable()});

    if (cuda.CudaBackend.isAvailable()) {
        // Device info
        const count = cuda.CudaBackend.getDeviceCount();
        std.log.info("Device count: {d}", .{count});

        // Get properties for first device
        var backend = cuda.CudaBackend.init(std.heap.page_allocator) catch return;
        defer backend.deinit();

        const props = backend.context.device_props;
        std.log.info("Device name: {s}", .{props.name});
        std.log.info("Compute capability: {d}.{d}", .{
            props.compute_capability_major,
            props.compute_capability_minor,
        });
        std.log.info("Total memory: {d} MB", .{
            props.total_memory / (1024 * 1024),
        });
    }
}
```

### Resources

- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [ZigNeuron Issues](https://github.com/FlavioCFOliveira/ZigNeuron/issues)

## See Also

- [User Guide](./user-guide.md) - Usage tutorials
- [API Reference](./api.md) - Complete API
- [Developer Guide](./developer-guide.md) - Implementation details

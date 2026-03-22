---
name: CUDA_ERROR_INVALID_PTX Research
description: Comprehensive research on CUDA ERROR_INVALID_PTX (code 218) causes and solutions
type: reference
---

# CUDA ERROR_INVALID_PTX (Code 218) Research Report

## Executive Summary

ERROR_INVALID_PTX (error code 218) indicates that **PTX JIT compilation failed**. This is distinct from ERROR_UNSUPPORTED_PTX_VERSION (code 222). The error typically occurs due to:
1. Syntax errors in PTX code
2. Architecture incompatibility
3. Driver/Toolkit version mismatches
4. Invalid PTX instructions for target device

## Error Code Definitions

From `src/cuda_driver.zig`:
```zig
ERROR_INVALID_PTX = 218,           // PTX JIT compilation failed
ERROR_UNSUPPORTED_PTX_VERSION = 222,  // PTX version not supported
```

**Critical Distinction:**
- Code 218 = PTX syntax/compilation error
- Code 222 = PTX version too new for driver

## Common Causes

### 1. PTX Syntax Errors
- Missing `.version` or `.target` directives
- Invalid register types (e.g., using `.b64` on sm_50 where only `.u64` is valid)
- Missing null-termination of PTX string
- Label naming issues (e.g., `$str1` prefix can cause parsing errors)
- Unclosed braces or malformed kernel entry points

### 2. Architecture Incompatibility
- PTX compiled for newer architecture (e.g., sm_80) running on older GPU (sm_50)
- Using instructions not supported by target compute capability
- Shared memory constraints exceeded

### 3. Version Mismatch
| CUDA Version | PTX ISA Version | Minimum Driver |
|--------------|-----------------|----------------|
| 12.0 | 8.0 | 525.60.13 |
| 11.8 | 7.8 | 520.61.05 |
| 11.0 | 7.0 | 450.36.06 |
| 10.0 | 6.3 | 410.48 |
| 9.0 | 6.0 | 384.81 |

### 4. Current ZigNeuron PTX Analysis

Current PTX header in `src/cuda_kernels.zig`:
```ptx
.version 6.0
.target sm_50
.address_size 64
```

**Assessment:**
- `.version 6.0` corresponds to CUDA 9.0 (PTX ISA 6.0)
- `.target sm_50` targets Maxwell architecture (compute capability 5.0)
- Should be compatible with driver 535.288.01 (CUDA 12.2)

## PTX Validation Methods

### Method 1: Extract and Validate PTX File
```bash
# Extract PTX to file
cat > /tmp/test.ptx << 'EOF'
.version 6.0
.target sm_50
.address_size 64

.visible .entry add_bias(
    .param .u64 output,
    .param .u64 bias,
    .param .u32 batch_size,
    .param .u32 bias_size
) {
    // kernel body
    ret;
}
EOF

# Validate with ptxas (if available)
ptxas -arch=sm_50 /tmp/test.ptx -o /tmp/test.cubin 2>&1
```

### Method 2: Runtime Validation with Enhanced Logging
```zig
const result = driver.moduleLoadData.?(&module, ptx.ptr);
if (result.isError()) {
    const error_name = driver.getErrorName(result);
    const error_string = driver.getErrorString(result);
    std.log.err("moduleLoadData failed: {s} ({s})", .{ error_name, error_string });

    // Additional diagnostics
    if (result == .ERROR_INVALID_PTX) {
        std.log.err("PTX validation failed - check syntax and compatibility", .{});
    }
}
```

### Method 3: NVRTC Compilation Alternative
Instead of embedded PTX, compile CUDA C++ at runtime:
```zig
// Use NVRTC to compile CUDA source to PTX dynamically
// This catches errors at compile time with detailed messages
const nvrtc = @import("cuda_nvrtc.zig");
```

## Debugging Steps for ZigNeuron

### Step 1: Verify PTX String Integrity
```zig
// Add debug logging before loading
std.log.debug("PTX length: {d}", .{ptx.len});
std.log.debug("PTX first 100 chars: {s}", .{ptx[0..@min(100, ptx.len)]});
std.log.debug("PTX last 10 chars: {s}", .{ptx[ptx.len-10..]});

// Ensure null termination
const null_terminated = std.mem.concat(allocator, u8, &[_][]const u8{ ptx, &[_]u8{0} }) catch return error.OutOfMemory;
defer allocator.free(null_terminated);
```

### Step 2: Check Target Architecture Compatibility
```zig
// Query device compute capability
var major: c_int = undefined;
var minor: c_int = undefined;
try driver.deviceGetAttribute(&major, CUdevice_attribute.COMPUTE_CAPABILITY_MAJOR, device);
try driver.deviceGetAttribute(&minor, CUdevice_attribute.COMPUTE_CAPABILITY_MINOR, device);
const compute_capability = @as(f32, @floatFromInt(major)) + @as(f32, @floatFromInt(minor)) / 10;

std.log.info("GPU compute capability: {d}.{d}", .{ major, minor });

// Verify PTX target matches or is older
// sm_50 PTX can run on sm_52, sm_60, etc. (forward compatible)
// sm_52 PTX cannot run on sm_50 (not backward compatible)
```

### Step 3: Test with Minimal PTX
```zig
const MINIMAL_PTX =
    \\.version 6.0
    \\.target sm_50
    \\.address_size 64
    \\
    \\.visible .entry test_kernel() {
    \\    ret;
    \\}
;

// Try loading minimal PTX first
const result = driver.moduleLoadData.?(&module, MINIMAL_PTX.ptr);
if (result.isError()) {
    std.log.err("Even minimal PTX failed - driver/PTX version issue", .{});
}
```

## Recommended Solutions

### Solution 1: Regenerate PTX with Correct Version
If the embedded PTX is corrupted or has syntax issues:
```bash
# Compile from CUDA source with explicit architecture
nvcc -ptx -arch=sm_50 -o output.ptx input.cu

# Or for multiple architectures
nvcc -gencode arch=compute_50,code=sm_50 \
       -gencode arch=compute_60,code=sm_60 \
       -ptx -o output.ptx input.cu
```

### Solution 2: Use NVRTC Instead of Embedded PTX
**Advantages:**
- Compile CUDA C++ at runtime with detailed error messages
- Automatic architecture detection
- No embedded strings to maintain

**Disadvantages:**
- Requires CUDA toolkit at runtime
- Slightly slower first-time compilation

### Solution 3: PTX Compiler API
Use `libnvptxcompiler` for validation before loading:
```c
nvPTXCompilerHandle compiler;
nvPTXCompilerCreate(&compiler, ptx_len, ptx_code);

const char* options[] = { "--gpu-name=sm_50" };
nvPTXCompilerResult compile_result = nvPTXCompilerCompile(compiler, 1, options);

if (compile_result != NVPTXCOMPILE_SUCCESS) {
    size_t error_size;
    nvPTXCompilerGetErrorLogSize(compiler, &error_size);
    char* error_log = malloc(error_size);
    nvPTXCompilerGetErrorLog(compiler, error_log);
    printf("PTX compilation error: %s\n", error_log);
}
```

### Solution 4: Fallback to Lower PTX Version
If ERROR_INVALID_PTX persists, try lowering the PTX version:
```zig
// Try PTX 5.0 (CUDA 8.0) for maximum compatibility
pub const PTX_HEADER =
    \\.version 5.0
    \\.target sm_50
    \\.address_size 64
    \\
;
```

## ZigNeuron-Specific Recommendations

1. **Add PTX Validation Layer**: Before `cuModuleLoadData`, validate:
   - String is null-terminated
   - `.version` directive exists and is supported
   - `.target` is compatible with current GPU

2. **Implement NVRTC Fallback**: If embedded PTX fails, try NVRTC compilation

3. **Add Debug Mode**: In debug builds, dump PTX to file for manual inspection

4. **Error Recovery**: Distinguish between:
   - ERROR_INVALID_PTX (syntax error)
   - ERROR_UNSUPPORTED_PTX_VERSION (version mismatch)
   - ERROR_NO_BINARY_FOR_GPU (architecture mismatch)

## References

- [NVIDIA CUDA Driver API Documentation - CUresult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [OpenMM Issue #3474 - CUDA_ERROR_INVALID_PTX](https://github.com/openmm/openmm/issues/3474)
- [PTX Compiler API Documentation](https://docs.nvidia.com/cuda/ptx-compiler-api/)
- [NVRTC Documentation](https://docs.nvidia.com/cuda/nvrtc/)

## Related Memory Files

- `../cuda-performance-optimizer/MEMORY.md` - CUDA implementation details
- `../neural-net-architect/MEMORY.md` - Neural network architecture notes

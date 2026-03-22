---
name: CUDA PTX Version Compatibility Research
description: Comprehensive research on CUDA PTX version compatibility, NVRTC configuration options, and solutions for ERROR_UNSUPPORTED_PTX_VERSION
type: reference
---

# CUDA PTX Version Compatibility Research

**Date:** 2026-03-22
**Researcher:** Technical Research Specialist
**Context:** ZigNeuron CUDA Backend PTX Loading Issues

## Executive Summary

The ERROR_UNSUPPORTED_PTX_VERSION (code 222) and ERROR_INVALID_PTX (code 218) errors occur because:
- CUDA Toolkit 12.6 generates **PTX ISA 8.5**
- Driver 535 (bundled with CUDA 12.2) only supports up to **PTX ISA 8.2** natively
- PTX is NOT backward compatible - newer PTX cannot be loaded by older drivers

## PTX Version Compatibility Matrix

| PTX ISA Version | CUDA Toolkit | Minimum Driver | Driver 535 Support |
|-----------------|--------------|----------------|-------------------|
| PTX 7.0 | CUDA 9.0+ | 385+ | ✅ Native |
| PTX 7.2 | CUDA 9.2+ | 396+ | ✅ Native |
| PTX 7.5 | CUDA 10.0+ | 410+ | ✅ Native |
| PTX 8.0 | CUDA 11.0+ | 450+ | ✅ Native |
| PTX 8.2 | CUDA 12.2 | 535+ | ✅ Native |
| **PTX 8.5** | **CUDA 12.5+** | **545+** | ❌ **UNSUPPORTED** |
| PTX 8.6 | CUDA 12.6 | 550+ | ❌ UNSUPPORTED |

**Key Finding:** Driver 535 supports up to PTX 8.2. PTX 8.5 requires Driver 545+.

## NVRTC Architecture Options

### Virtual vs Real Architecture

| Option | Type | Output | Forward Compatible | CUBIN Available |
|--------|------|--------|-------------------|---------------|
| `--gpu-architecture=compute_52` | Virtual | PTX only | ✅ Yes | ❌ No |
| `--gpu-architecture=sm_52` | Real | CUBIN + PTX | ❌ No* | ✅ Yes |

*CUBIN is architecture-specific, but PTX fallback enables forward compatibility if included

### Recommended Architecture Targets for Compatibility

| Target GPU | Recommended Flag | Compatibility |
|------------|------------------|---------------|
| Maxwell (GTX 9xx) | `compute_52` or `sm_52` | Widest support |
| Pascal (GTX 10xx) | `compute_61` | Good support |
| Volta (V100) | `compute_70` | CUDA 9.0+ |
| Turing (RTX 20xx) | `compute_75` | CUDA 10.0+ |
| Ampere (A100/RTX 30xx) | `compute_80` | CUDA 11.0+ |
| Ada (RTX 40xx) | `compute_89` | CUDA 11.8+ |
| Hopper (H100) | `compute_90` | CUDA 12.0+ |

## Solution Options

### Option 1: Configure NVRTC for Lower PTX Version (RECOMMENDED)

Configure NVRTC to generate PTX 8.0 or lower by targeting a virtual architecture:

```cpp
// Option A: Target compute_80 for maximum compatibility with modern GPUs
const char* opts[] = {"--gpu-architecture=compute_80"};
nvrtcCompileProgram(prog, 1, opts);

// Option B: Target compute_52 for widest GPU support (Maxwell+)
const char* opts[] = {"--gpu-architecture=compute_52"};
nvrtcCompileProgram(prog, 1, opts);
```

**Zig Implementation:**
```zig
const opts = [_][*c]const u8{"--gpu-architecture=compute_80"};
const result = nvrtcCompileProgram(prog, opts.len, &opts);
```

### Option 2: Use CUBIN Instead of PTX

Compile directly to binary machine code for the target architecture:

```cpp
// Target specific GPU architecture (e.g., sm_80 for Ampere)
const char* opts[] = {"--gpu-architecture=sm_80"};
nvrtcCompileProgram(prog, 1, opts);

// Get CUBIN instead of PTX
size_t cubin_size;
nvrtcGetCUBINSize(prog, &cubin_size);
char* cubin = new char[cubin_size];
nvrtcGetCUBIN(prog, cubin);
```

**Advantages:**
- No JIT compilation overhead at runtime
- Bypasses PTX version compatibility issues
- Faster module loading

**Disadvantages:**
- Architecture-specific (must know target GPU at compile time)
- No forward compatibility with newer GPUs
- Must compile for multiple architectures to support different GPUs

### Option 3: Upgrade CUDA Driver

Install driver version 545 or higher to natively support PTX 8.5:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-545

# Or download from NVIDIA
# https://www.nvidia.com/drivers
```

### Option 4: Use Forward Compatibility Package

If staying on Driver 535, install the forward compatibility package:

```bash
# Ubuntu 22.04 example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-12-5_555.42.02-1_amd64.deb
sudo dpkg -i cuda-compat-12-5_555.42.02-1_amd64.deb

# Set library path
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/compat:$LD_LIBRARY_PATH
```

## NVRTC Compilation Best Practices

### 1. Detect GPU Architecture at Runtime

```cpp
// Query GPU compute capability
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device_id);
int major = prop.major;
int minor = prop.minor;

// Construct architecture string
char arch_str[16];
snprintf(arch_str, sizeof(arch_str), "sm_%d%d", major, minor);
const char* opts[] = {arch_str};
```

### 2. Generate Both PTX and CUBIN

```cpp
// For maximum compatibility and performance:
// 1. Generate CUBIN for known target architecture
// 2. Fall back to PTX for other architectures

const char* opts[] = {"--gpu-architecture=sm_80"};
nvrtcCompileProgram(prog, 1, opts);

// Get CUBIN first
size_t cubin_size;
nvrtcGetCUBINSize(prog, &cubin_size);
if (cubin_size > 0) {
    // Use CUBIN for this specific architecture
    char* cubin = new char[cubin_size];
    nvrtcGetCUBIN(prog, cubin);
} else {
    // Fall back to PTX
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    char* ptx = new char[ptx_size];
    nvrtcGetPTX(prog, ptx);
}
```

### 3. Compile-Time Architecture Selection

```zig
// Zig compile-time architecture selection
const gpu_arch = comptime blk: {
    if (@hasDecl(@import("builtin"), "cuda_arch")) {
        break :blk @import("builtin").cuda_arch;
    } else {
        break :blk "compute_80"; // Default to Ampere virtual arch
    }
};

const opts = [_][*c]const u8{gpu_arch};
```

### 4. Caching Strategy

```cpp
// Cache compiled kernels to avoid recompilation
struct KernelCache {
    std::string source_hash;
    std::vector<char> ptx_or_cubin;
    std::string arch;
};

// Check cache before compilation
auto it = kernel_cache.find(hash);
if (it != kernel_cache.end()) {
    // Use cached version
    return it->second;
}

// Otherwise compile and cache
nvrtcCompileProgram(prog, ...);
kernel_cache[hash] = compiled_result;
```

## Recommended Solution for ZigNeuron

Given the project's requirements for broad compatibility:

1. **Primary Approach:** Configure NVRTC to target `compute_80` (virtual architecture)
   - Supports all Ampere and newer GPUs
   - Compatible with Driver 450+ (CUDA 11.0+)
   - Provides forward compatibility via JIT compilation

2. **Secondary Approach:** Detect GPU at runtime and compile for specific `sm_XX`
   - Use `nvrtcGetCUBIN` for known architectures
   - Fall back to PTX for unknown/unsupported architectures

3. **Implementation Pattern:**
   ```zig
   pub fn compileKernel(source: []const u8, arch: []const u8) ![]const u8 {
       const prog = nvrtcCreateProgram(...);

       // Use virtual architecture for compatibility
       const arch_opt = std.fmt.allocPrint(allocator, "--gpu-architecture={s}", .{arch});
       defer allocator.free(arch_opt);

       const opts = [_][*c]const u8{arch_opt.ptr};
       const result = nvrtcCompileProgram(prog, 1, &opts);

       if (result != NVRTC_SUCCESS) {
           // Handle compilation error
       }

       var ptx_size: usize = undefined;
       try nvrtcGetPTXSize(prog, &ptx_size);

       const ptx = try allocator.alloc(u8, ptx_size);
       try nvrtcGetPTX(prog, ptx.ptr);

       return ptx;
   }
   ```

## References

- [NVRTC Documentation](https://docs.nvidia.com/cuda/nvrtc/index.html)
- [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/pdf/CUDA_Compatibility.pdf)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA Driver Compatibility Matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)
- [Stack Overflow: NVRTC CUBIN Generation](https://stackoverflow.com/questions/69782228/when-should-nvrtc-compilation-produce-a-cubin)
- [Stack Overflow: PTX Version Error](https://stackoverflow.com/questions/79441777/cumoduleloaddataex-returns-cuda-error-unsupported-ptx-version)

## Action Items

1. [ ] Update `src/cuda_kernels.zig` to pass `--gpu-architecture=compute_80` to NVRTC
2. [ ] Add runtime GPU architecture detection for CUBIN compilation
3. [ ] Implement kernel caching to avoid repeated compilations
4. [ ] Add documentation about driver requirements to CUDA backend docs
5. [ ] Consider implementing multi-architecture support (PTX + multiple CUBINs)

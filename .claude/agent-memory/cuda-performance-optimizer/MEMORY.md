# CUDA Performance Optimizer - Agent Memory

## ZigNeuron CUDA Implementation - Key Findings

### Architecture Support
- Target: NVIDIA Turing (7.5) through Hopper (9.0)
- Minimum compute capability: 7.5 (Turing)
- Supports FP16, BF16, TF32 tensor cores where available

### Memory Optimization Patterns

#### Vectorized Loads (float4)
- Process 4 floats per thread for memory-bound kernels
- Increases effective memory bandwidth by ~4x
- Applies to: activations, optimizers, element-wise ops

#### Shared Memory Tiling
- MatMul: 128x128 output tile with 8-wide K dimension
- Transpose: 32x32 tiles with padding (33 columns) to avoid bank conflicts
- Occupancy target: ~16KB SMEM per block allows 3 blocks/SM

#### Coalesced Access Patterns
- Thread i accesses element i (contiguous)
- Use 2D/3D thread blocks that map directly to memory layout
- Avoid strided access patterns (e.g., A[i * 32])

### Warp-Level Primitives

#### When to Use
- Reductions across threads in a warp (softmax, layernorm, batchnorm)
- Fast intra-warp communication without SMEM
- Prefer over shared memory for warps ≤ 32 elements

#### Key Patterns
```cuda
// Sum reduction
float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
```

### Kernel Fusion Opportunities

#### High-Value Fusions
1. MatMul + Bias + ReLU (common in FFN layers)
2. Dropout (random gen + mask + scale in one kernel)
3. Softmax (max + exp + sum + normalize in one kernel)
4. LayerNorm (mean + var + normalize + scale in one kernel)

#### Avoid Fusing
- Operations with different memory/compute ratios
- Kernels requiring global synchronization between steps

### Thread Configuration Guidelines

| Operation | Block Config | Occupancy |
|-----------|--------------|-----------|
| Element-wise | 256 threads, 1D | High (>50%) |
| MatMul | 128 threads, 2D (16x8) | Medium (3 blocks/SM) |
| Softmax | 128 threads, 1D | Medium |
| LayerNorm | 256 threads, 1D | Medium |
| Conv2D | 256 threads, 2D (16x16) | Medium |

### cuBLAS Integration Thresholds

#### Use cuBLAS When
- Matrix dimensions > 256 for all M, N, K
- Batch size > 32 for batched operations
- Need maximum throughput for large matrices

#### Use Custom Kernels When
- Matrix dimensions < 256 (launch overhead dominates)
- Memory bandwidth bound (element-wise)
- Custom memory access patterns (strided, blocked)

### PTX Generation for Zig Integration

#### Recommended Workflow
1. Compile .cu to .ptx with `nvcc -ptx`
2. Embed PTX in Zig using `@embedFile`
3. Load at runtime with `cuModuleLoadData`
4. Get kernels with `cuModuleGetFunction`

#### Architecture Flags
```
-gencode arch=compute_75,code=sm_75  # Turing
-gencode arch=compute_80,code=sm_80  # Ampere
-gencode arch=compute_86,code=sm_86  # Ampere RTX
-gencode arch=compute_89,code=sm_89  # Ada
-gencode arch=compute_90,code=sm_90  # Hopper
```

### Random Number Generation

#### Philox2x32 for Deterministic RNG
- Counter-based RNG suitable for GPU
- Deterministic across different runs
- Fast enough for dropout

### Validation Checklist
- [ ] Numerical correctness vs CPU reference
- [ ] Coalesced memory access patterns
- [ ] Shared memory bank conflict avoidance
- [ ] Proper thread divergence minimization
- [ ] Occupancy analysis (Nsight Compute)
- [ ] Memory bandwidth utilization (>80% target)

## File Locations
- CUDA kernels: `/data/dev/github.com/FlavioCFOliveira/ZigNeuron/kernels/`
- Documentation: `/data/dev/github.com/FlavioCFOliveira/ZigNeuron/docs/CUDA_IMPLEMENTATION_*.md`

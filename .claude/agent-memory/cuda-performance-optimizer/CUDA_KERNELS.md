# CUDA Kernels for ZigNeuron

## Overview

Created comprehensive CUDA kernel implementations for ZigNeuron neural network library.
Date: 2026-03-07

## Kernel Files Created

| File | Operations | Status |
|------|------------|--------|
| `common.h` | Shared utilities, warp primitives, math functions | Complete |
| `matmul.cu` | Tiled matmul, batched, transpose variants | Complete |
| `activation.cu` | ReLU, Sigmoid, Tanh, GELU, Softmax | Complete |
| `optimizer.cu` | SGD, Adam, RMSprop updates | Complete |
| `loss.cu` | MSE, Cross-Entropy, BCE, KL divergence | Complete |
| `convolution.cu` | Conv1D/2D forward and backward | Complete |
| `normalization.cu` | LayerNorm, BatchNorm | Complete |
| `attention.cu` | Scaled dot-product attention | Complete |
| `dropout.cu` | Dropout, VAE sampling | Complete |
| `recurrent.cu` | LSTM, GRU, RNN forward/backward | Complete |
| `auxiliary.cu` | Element-wise ops, map functions | Complete |

## Key Optimization Strategies

### 1. Memory Coalescing
- Vectorized float4 loads for 4x memory throughput
- Contiguous thread access patterns
- Shared memory for transpose operations

### 2. Warp-Level Primitives
- `__shfl_sync` for intra-warp reduction
- `__shfl_down_sync` for tree-based reductions
- Warp divergence minimization

### 3. Shared Memory Tiling
- 128x128 tiles for matrix multiplication
- Bank conflict avoidance with padding (+1)
- Double buffering potential for async copy (future)

### 4. Occupancy Targets
- 128-256 threads per block (4-8 warps)
- 50%+ occupancy for memory-bound kernels
- Register pressure monitoring

## Recommended Block Sizes by Operation

| Operation | Block Size | Threads | Occupancy |
|-----------|-----------|---------|-----------|
| MatMul (tiled) | 128x1 | 128 | ~75% |
| Activation | 256x1 | 256 | ~50% |
| Softmax | 128x1 | 128 | ~75% |
| LayerNorm | 256x1 | 256 | ~50% |
| Conv1D | 256x1 | 256 | ~50% |
| Conv2D | 16x16 | 256 | ~50% |
| Optimizer | 256x1 | 256 | ~50% |
| Attention | 128x1 | 128 | ~75% |

## cuBLAS Integration Points

Use cuBLAS for:
- Large matmul (M,N,K > 256)
- Batched operations
- Transpose variants

Use custom kernels for:
- Small matmul (overhead matters)
- Element-wise operations
- Fused operations

## Tensor Core Considerations

- WMMA API available for SM 7.0+
- FP16/BF16/TF32 support
- Best for: Large GEMM operations
- Not yet implemented (future optimization)

## Known Limitations

1. Conv2D backward uses atomics (could be optimized with kernel split)
2. Attention is not Flash Attention (memory bound for long sequences)
3. No CUDA Graph support yet
4. No multi-stream execution yet

## Next Steps

1. Integrate with Zig backend (cuda.zig)
2. Add PTX runtime compilation
3. Profile on target hardware
4. Implement cuBLAS fallbacks
5. Add Tensor Core kernels for Ampere+

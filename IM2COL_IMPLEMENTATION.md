# im2col Conv2D Optimization Implementation Plan

## Overview
This document outlines the implementation of im2col-based Conv2D optimization for 10-20x speedup by converting convolution to matrix multiplication.

## Algorithm

### im2col Transformation
Converts image patches to columns for GEMM-based convolution:
- Input: [batch, channels, height, width] (NCHW)
- Output: [output_height * output_width, kernel_height * kernel_width * channels]

For each output position (oh, ow):
  For each kernel position (kh, kw):
    For each input channel (c):
      row = oh * output_width + ow
      col = (kh * kernel_width + kw) * channels + c
      ih = oh * stride + kh - pad
      iw = ow * stride + kw - pad
      if (ih, iw) is within bounds:
        output[row, col] = input[c, ih, iw]
      else:
        output[row, col] = 0 (padding)

### GEMM-based Convolution
```
result = GEMM(im2col_output, weights_reshaped)
```
- Weights reshaped from [out_channels, in_channels, kH, kW] to [out_channels, in_channels*kH*kW]
- Result shape: [batch][oh*ow][out_channels]

### Backward Pass
1. gradients_w = GEMM(output_grad_reshaped, im2col_input)
2. gradients_input = GEMM(weight^T, output_grad) then col2im

## Implementation Files

1. `src/backend.zig` - im2col function and Conv2D optimization
2. `src/cuda.zig` - GPU kernels for im2col/col2im
3. `src/cuda_kernels.zig` - CUDA kernel source code

## Performance Expectations
- CPU: 5-10x speedup via SIMD-accelerated GEMM
- GPU: 10-20x speedup via cuBLAS/optimized kernels

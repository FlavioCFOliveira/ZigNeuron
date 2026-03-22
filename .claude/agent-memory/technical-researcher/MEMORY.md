# Technical Research Specialist Memory

## Key Research Findings
- **Seq2seq**: Requires `forward` functions to accept optional initial states. [Link](research_advanced_layers.md)
- **VAE**: Formula for KL Divergence (Gaussian) and Reparameterization Trick documented. [Link](research_advanced_layers.md)
- **Attention**: Needs Batched MatMul (4D) and Transpose/Permute kernels.
- **Conv1D**: Direct sliding window with dilation is preferred over im2col for 1D.

## Backend Gaps
- `random_normal` sampling on GPU.
- `LayerNorm` implementation.
- `exp`, `log`, `pow` element-wise primitives.
- `conv1d` specialized kernels.

## Performance Optimization
- Prioritize direct 1D conv kernels for Apple Silicon to maximize memory bandwidth.
- Use Philox/XORWOW for efficient parallel random number generation in VAEs.

## CUDA Error Research
- **ERROR_INVALID_PTX (code 218)**: Comprehensive research on causes and solutions documented. [Link](CUDA_ERROR_INVALID_PTX_RESEARCH.md)
- **PTX Version Compatibility**: PTX 6.0 (CUDA 9.0) is compatible with driver 535+ (CUDA 12.2).
- **Architecture Compatibility**: sm_50 PTX is forward-compatible (runs on sm_52, sm_60, etc.) but not backward-compatible.

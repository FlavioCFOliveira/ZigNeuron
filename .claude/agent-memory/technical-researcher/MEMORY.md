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

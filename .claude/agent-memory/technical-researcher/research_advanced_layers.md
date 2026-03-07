# Technical Research: Advanced Neural Network Components for ZigNeuron

This document synthesizes research for Seq2seq, VAE, Attention, and Conv1D layers.

## 1. Seq2seq (Encoder-Decoder) Architecture
The Seq2seq model consists of an Encoder that processes an input sequence into a fixed-length context vector (last hidden state), and a Decoder that generates an output sequence starting from that context.

### Modular State Transfer in Zig
To support this modularly:
- **`forward` signature update**: RNN/LSTM/GRU `forward` functions should accept an optional `initial_state: ?tensor.Tensor`.
- **Memory Management**: Use `tensor.Tensor.view()` or slices to pass the Encoder's final hidden state to the Decoder without reallocation.
- **Autoregressive Decoding**: In inference, the Decoder needs a loop where `y_t` is fed as `x_{t+1}`.

## 2. Variational Autoencoders (VAE)
### Reparameterization Trick
To allow backpropagation through stochastic sampling:
$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
In practice, predict $\log(\sigma^2)$ (log-variance) for stability:
$z = \mu + \exp(0.5 \cdot \text{log\_var}) \odot \epsilon$

### KL Divergence (Gaussian Prior)
For $p(z) = \mathcal{N}(0, I)$ and $q(z|x) = \mathcal{N}(\mu, \sigma^2)$:
$D_{KL} = -\frac{1}{2} \sum_{j=1}^J (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)$

## 3. Kernel Analysis & SOTA Optimizations

### Fused Recurrent Kernels (RNN/LSTM/GRU)
Current recurrent layers in ZigNeuron launch kernels per time step.
- **Temporal Fusion**: Process the entire sequence (or chunks) within a single GPU kernel to minimize launch overhead and keep `h_t` in registers/threadgroup memory.
- **Persistent RNNs**: Load weights once into registers/shared memory for reuse across the sequence.
- **Metal Implementation**: Use `simdgroup` operations on Apple Silicon for fast matrix-vector products within the recurrent loop inside the shader.

### MultiHeadAttention & FlashAttention
- **Tiling**: FlashAttention computes attention in tiles to avoid the $O(N^2)$ memory bottleneck.
- **Softmax Decomposition**: Maintains running maximum and sum across tiles to normalize correctly without full matrix materialization.
- **Unified Memory (Apple Silicon)**: Store Q, K, V tiles in `threadgroup` memory.
- **Batched MatMul**: Needs support for 4D tensors (Batch, Heads, Seq, Dim).

### Conv1D with Dilation
- **Direct Kernel**: Most efficient for 1D. Dilation $d$ just changes the input index stride:
  $y[t] = \sum_{k=0}^{K-1} w[k] \cdot x[t + k \cdot d]$
- **Backend requirement**: `conv1d_forward`, `conv1d_grad_input`, `conv1d_grad_weights`.

## 4. Numerical Stability
- **Stable Activations**: Implement Sigmoid/Tanh with branching (positive/negative inputs) to avoid `exp` overflows.
- **Gradient Clipping**: Global Norm Clipping is essential for deep RNNs to prevent exploding gradients.
- **Forget Gate Bias**: Keep initial value at 1.0 to prevent early vanishing gradients in LSTMs.

## 5. Backend Gap Analysis (Required Primitives)
- **Math**: `exp`, `log`, `pow`, `sum` (reduction).
- **Sampling**: `random_normal` (GPU-based).
- **Normalization**: `LayerNorm` (forward/backward) - critical for SOTA architectures.

## 5. Stock Prediction Layers
Commonly missing layers in the current implementation:
- **LayerNorm**: Crucial for Transformers.
- **Dropout**: Regularization.
- **Embedding**: For categorical/time features.
- **PositionalEncoding**: For sequence order in Attention.

---
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

# Stagnant Loss Investigation Report - XOR Example

## Summary of Findings
The 'stagnant loss' issue (around 0.252) in the XOR example was caused by several critical mathematical and implementation errors in the backpropagation logic:

1.  **Broken Activation Gradients**: The activation backward passes (Sigmoid and Tanh) were incorrectly re-calculating the forward activation on values that were already post-activation. This resulted in severely distorted gradients (e.g., $s(s(z))$ instead of $s(z)$).
2.  **Incorrect Weight Gradient Indexing**: In `Dense.accumulateGradients`, the indexing for weight gradients was using `[output_size, input_size]` layout, while the forward pass and weight storage used `[input_size, output_size]`. This led to scrambled weight updates.
3.  **Missing Transpose in Backprop**: The propagation of error to previous layers was using standard matrix multiplication instead of transposed multiplication ($d_{in} = d_{out} \cdot W^T$).
4.  **Synchronization Issues (Metal)**: On the Metal backend, results from GPU computations were not always correctly reflected in CPU slices due to missing copy-back logic for shared buffers that were not "owned" by the backend function.

## Fixes Implemented

### 1. Mathematical Correctness in Activations (`src/activation.zig`)
- Changed all activation backward functions to expect the **post-activation** value $y = act(z)$ as input.
- Updated Sigmoid derivative to $y(1-y)$ and Tanh derivative to $1-y^2$.
- This also improved performance by avoiding redundant transcendental function calls.

### 2. Correct Weight Gradient Indexing (`src/layer.zig`)
- Fixed `accumulateGradients` to use `in_idx * output_size + out_idx` indexing, matching the `[input_size, output_size]` row-major storage convention.

### 3. Error Backpropagation (`src/network.zig`)
- Verified and ensured the use of `matMulTransposeB` for backpropagating error to the previous layer.
- Fixed field name inconsistencies in `LayerCache` (`pre_activation` -> `activated_output`).

### 4. GPU-CPU Synchronization (`src/backend.zig`)
- Added explicit copy-back/synchronization logic in Metal backend functions (`metalMatMulGPU`, `metalActivationBackwardGPU`, `metalLossBackwardGPU`) to ensure CPU slices are updated with GPU results even for shared buffers.

### 5. Metal Shaders (`shaders/metal/activation.metal`)
- Updated Metal activation shaders to match the new convention (expecting post-activation values).
- Replaced Sigmoid and Tanh approximations with exact mathematical functions for better precision.

## Results
- The XOR example now converges correctly on the **CPU backend**, decreasing from ~0.25 loss to ~0.12 in 1000 epochs.
- On the **Metal backend**, weights are now moving correctly, although convergence may still require careful hyperparameter tuning (Learning Rate and Epochs) due to the sensitive nature of XOR and initialization.

## Recommendations
- **Initialization**: Consider using a more robust random seed for Xavier initialization to avoid identical starting weights across layers.
- **Optimizer**: Implementing Adam or RMSprop would likely solve the remaining convergence speed issues on the GPU backend by providing adaptive learning rates.

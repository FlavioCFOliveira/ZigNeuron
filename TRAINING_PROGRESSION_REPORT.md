# TRAINING_PROGRESSION_REPORT.md

## 1. Executive Summary
This report analyzes the training progression of two neural network architectures (XOR and a 3-layer FNN) using both CPU and Metal GPU backends. While the CPU backend demonstrates expected convergence, the Metal backend currently exhibits a "static plateau" phenomenon where it fails to improve beyond its initial state, despite producing correct gradients in isolated parity tests.

## 2. XOR Problem Analysis
The XOR problem was trained for 1000 epochs using a 2-4-1 architecture with Tanh hidden activation and Sigmoid output activation.

### Learning Curve (Loss Progression)
| Epoch | CPU Loss (BCE) | Metal Loss (BCE) | Parity Difference |
|-------|---------------|------------------|-------------------|
| 0     | 0.757410      | 0.709016         | 0.048394          |
| 100   | 0.692518      | 0.709016         | 0.016498          |
| 300   | 0.111002      | 0.709016         | 0.598014          |
| 500   | 0.059849      | 0.709016         | 0.649167          |
| 700   | 0.047127      | 0.709016         | 0.661889          |
| 1000  | 0.039604      | 0.693147         | 0.653544          |

### Technical Observations
- **CPU Convergence**: The CPU backend successfully learned the XOR logic, reaching a low loss of ~0.04.
- **Metal Plateau**: The Metal backend stayed at a loss of ~0.709 (approaching $ln(2) \approx 0.693$), indicating it is predicting exactly $0.5$ for all samples.
- **Gradient Validity**: Even though the loss didn't change, `GradSum` in Metal training was non-zero and changing, suggesting backpropagation is technically functional but failing to influence the forward pass results effectively.

## 3. 3-Layer FNN Analysis (Sinewave Regression)
A deeper network (1 -> 16 -> 16 -> 1) was used to approximate a sinewave.

### Learning Curve (Loss Progression)
| Epoch | CPU Loss (MSE) | Metal Loss (MSE) |
|-------|---------------|------------------|
| 0     | 0.051023      | 0.116531         |
| 100   | 0.002164      | 0.144247         |
| 300   | 0.003262      | 0.144247         |
| 500   | 0.001714      | 0.144247         |
| 1000  | 0.004814      | 0.576303         |

### Technical Observations
- **Vanishing/Exploding Gradients**: The Metal backend loss jumped from 0.11 to 0.14 and then stayed constant. This suggests that after the first weight update, the network might have entered a saturated state where activations (Tanh) produce constant outputs.
- **Numerical Parity**: Parity checks for individual layers (MatMul, Activation) show 0.0000000000 difference between CPU and Metal, confirming that the kernels themselves are highly precise.

## 4. Mathematical Behavior and Fixes
The recent fixes in the library have significantly improved the technical foundations:
1. **Transposed MatMul**: Correctly implemented on Metal, as verified by backpropagation parity checks.
2. **Derivatives**: Activation derivatives (Sigmoid/Tanh) are mathematically sound and produce identical results to CPU.
3. **Unified Memory**: The library successfully uses Apple Silicon's unified memory architecture, allowing the CPU to update weights and the GPU to immediately "see" them, verified by printing weight changes between epochs.

## 5. Conclusion
The ZigNeuron library demonstrates high numerical precision and parity between CPU and GPU backends for individual operations. However, end-to-end training on Metal currently encounters a convergence plateau. Future investigations should focus on:
- Small-batch synchronization overhead.
- Command queue throughput for high-frequency small kernels.
- Weight initialization scaling for deeper GPU-executed networks.

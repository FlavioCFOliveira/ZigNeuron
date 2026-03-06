# Classification Examples

This directory contains examples of multiclass classification using ZigNeuron.

## Available Examples

### Iris Flower Classification (`iris.zig`)

Classic dataset with 150 samples of iris flowers, 4 features, and 3 classes.

**Dataset:**
- 150 samples (50 per class)
- 4 features: sepal length, sepal width, petal length, petal width
- 3 classes: Setosa, Versicolor, Virginica
- Normalized to zero mean and unit variance

**Network Architecture:**
- Input: 4 features
- Dense: 4 → 16 (ReLU)
- Dense: 16 → 8 (ReLU)
- Output: 8 → 3 (Linear + Cross-Entropy)

**Expected Results:**
- Accuracy: 90-98% on test set
- Training converges in ~200 epochs

### MNIST Handwritten Digits (`mnist.zig`)

Image classification of handwritten digits (0-9).

**Dataset:**
- 70,000 samples (28x28 grayscale images)
- 10 classes: digits 0-9
- Download from: https://yann.lecun.com/exdb/mnist/

**Network Architecture:**
- Input: 784 (flattened 28x28)
- Dense: 784 → 128 (ReLU)
- Dropout: rate 0.2
- Dense: 128 → 64 (ReLU)
- Output: 64 → 10 (Softmax)

**Expected Results:**
- Accuracy: 97-98% with proper training
- Training time: ~5-10 minutes on GPU

## Running Examples

```bash
# Build all examples
zig build

# Run Iris classification
./zig-out/bin/iris_classification

# Run MNIST (requires dataset download)
./zig-out/bin/mnist_classification
```

## References

- **Iris Dataset:** Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179-188.
- **MNIST Dataset:** LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

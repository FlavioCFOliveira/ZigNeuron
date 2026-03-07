# Zig Performance Architect Memory

## Performance Bottlenecks & Optimization Patterns

### Matrix Multiplication (MatMul)
- **Problem**: Standard i-j-k loop order causes poor cache performance (cache misses on B and C).
- **Solution**: Use **i-k-j loop order** combined with **tiling/blocking**.
- **Result**: CONTIGUOUS memory access on the innermost loop significantly boosts CPU performance.
- **Pattern**:
  ```zig
  for (ii..i_end) |i| {
      for (kk..k_end) |p| {
          const a_val = a[i * k + p];
          const b_row = b[p * n ..];
          const c_row = c[i * n ..];
          for (jj..j_end) |j| {
              c_row[j] += a_val * b_row[j];
          }
      }
  }
  ```

### SIMD & Auto-Vectorization
- **Problem**: Index-based loops (`0..len`) can be harder for the compiler to vectorize.
- **Solution**: Use **multi-array iteration** (`for (a, b, c) |av, bv, *cv|`).
- **Result**: Provides clearer hints to the Zig compiler for SIMD optimization. Always prefer `for (src, *dest) |s, d| d.* = s;` over `for (0..len) |i| dest[i] = src[i];`.

### Zero-Allocation Engineering
- **Rule**: NO heap allocations in `forward`, `backward`, or `trainStep`.
- **Pattern**: Move `allocator.alloc` calls to the `init` phase. Add workspace buffers (as slices or `tensor.Tensor`) to the layer/network structs.
- **Buffers**: Use `tensor.Tensor` to ensure compatibility with both CPU slices and GPU (`MTLBuffer`) resources.
- **Reuse**: Reuse buffers across different steps if the size allows.

### Activation Function Derivatives
- **Softmax**: Use the $O(N)$ formula for backpropagation: `grad_input_j = softmax_j * (grad_output_j - sum(grad_output_i * softmax_i))`. This avoids the $O(N^2)$ Jacobian matrix.
- **Consistency**: Activation `backward` methods should take the **activated output** `y` (result of `forward`) rather than the input `x` to avoid redundant computations.

### Recurrent Layers
- **Safety**: Always check `input.len / input_size` against `max_seq_len`.
- **Flexibility**: Support both many-to-many (full sequence output) and many-to-one (last state output) by checking the provided `output` buffer size.

## Project Structure (ZigNeuron)
- `src/backend.zig`: Core mathematical operations. High-priority for optimization.
- `src/network.zig`: Orchestrates training. Must manage global workspace buffers.
- `src/layer.zig`: Layer implementations. Each layer should own its gradient and workspace buffers.
- `src/test_all.zig`: Centralized test runner using `std.testing.refAllDecls` to avoid redundant/outdated test code in the root runner.

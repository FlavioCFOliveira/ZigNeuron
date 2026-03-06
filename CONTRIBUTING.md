# Contributing to ZigNeuron

Thank you for your interest in contributing to ZigNeuron! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## How to Contribute

### Reporting Issues

When reporting bugs, please include:
- Zig version (`zig version`)
- Operating system and architecture
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Error messages (full stack trace if available)

### Suggesting Features

- Check if the feature is already requested
- Describe the use case clearly
- Explain why it would be valuable
- Consider implementation complexity

### Pull Requests

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/ZigNeuron.git
   cd ZigNeuron
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure `zig build test` passes

4. **Commit**
   ```bash
   git add .
   git commit -m "feat: Add your feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### Formatting

- Use 4 spaces for indentation
- Max line length: 100 characters
- Trailing commas in multi-line literals
- Opening brace on same line

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Functions | camelCase | `forwardPass` |
| Types | PascalCase | `DenseLayer` |
| Variables | snake_case | `learning_rate` |
| Constants | SCREAMING_SNAKE_CASE | `MAX_EPOCHS` |

### Documentation

- All public functions must have doc comments
- Use `///` for function documentation
- Include usage examples where helpful
- Document parameters and return values

```zig
/// Compute forward pass through the layer
///
/// Parameters:
///   - input: Input tensor of shape [batch_size, input_size]
///   - output: Pre-allocated output buffer
///
/// Returns: Error if dimensions mismatch
pub fn forward(self: *Dense, input: []const f32, output: []f32) !void {
```

## Testing

### Running Tests

```bash
# All tests
zig build test

# Specific test
zig test src/layer.zig

# With verbose output
zig build test --verbose
```

### Test Requirements

- Unit tests for all public functions
- Integration tests for layer combinations
- Numerical gradient checking for backward passes
- Performance benchmarks for optimizations

### Test Evidence

When adding features, update the evidence files:
- `UnitTests.md` - Unit test results
- `BenchmarkTests.md` - Performance comparisons
- `MemoryTests.md` - Memory profiling

## Architecture Guidelines

### Adding New Layers

1. Create struct in `src/layer.zig`
2. Implement `forward()` and `backward()`
3. Add to `Layer` union
4. Implement `getWeights()` and `getBias()`
5. Add unit tests
6. Create example in `examples/`

### Adding New Activations

1. Add to `Activation` enum in `src/activation.zig`
2. Implement `forward()` and `backward()`
3. Add Metal kernel if applicable
4. Update weight initialization in `Dense.init()`

### Adding New Loss Functions

1. Add to `Loss` union in `src/loss.zig`
2. Implement `forward()` and `backward()`
3. Document use cases in comments

## Performance Considerations

- **Zero-allocation**: Avoid allocations in hot paths
- **SIMD**: Use `@Vector` for CPU operations
- **Batching**: Group GPU operations
- **Memory**: Reuse buffers when possible

## Security

- Check for integer overflow in tensor dimensions
- Validate all inputs (non-zero sizes, valid ranges)
- Use safe arithmetic operations (`std.math.add`, etc.)
- Never use `.?` without null checks

## Questions?

Feel free to open an issue for:
- Clarification on architecture decisions
- Help with implementation
- Discussion of new features

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

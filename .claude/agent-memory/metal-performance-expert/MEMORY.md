# Metal Performance Expert Memory

## Best Practices & Patterns

### Command Buffer Management
- **Lifecycle**: When NOT using command batching, ensure that the `MTLCommandBuffer` used to create the encoder is the same one that is `commit()`'d. 
- **Bug Alert**: Avoid calling a helper like `getCommandBuffer()` multiple times within an operation if it returns a new buffer each time when no batch is active.
- **Unified Memory**: On Apple Silicon, `MTLStorageModeShared` is coherent between CPU and GPU, but `waitUntilCompleted()` or appropriate barriers are still needed to ensure visibility after GPU writes.

### Kernel Optimization
- **Threadgroup Size**: For simple operations on M-series chips, a threadgroup size of 256 or 512 is generally good. 
- **Indexing**: Always check bounds `if (gid < size)` in kernels.

## Debugging Insights
- **Constant Output/Flat Loss**: Often caused by zero gradients or failed weight updates. Check the command buffer commit logic and kernel indexing.
- **Hangs**: Can occur if a command buffer is never committed but `waitUntilCompleted()` is called on a different (empty) command buffer.

## File Paths
- `src/backend.zig`: Core logic for dispatching Metal kernels.
- `src/metal_context.zig`: Manages persistent Metal resources (device, queue, pipelines).
- `shaders/metal/`: MSL source files.
- `src/metal.zig`: Objective-C interop and Metal API bindings.

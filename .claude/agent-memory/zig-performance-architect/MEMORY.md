# Performance Audit Findings - ZigNeuron

## Critical Performance Issues

### 1. No SIMD in CPU Operations
- **Location:** `src/backend.zig` - all cpu* functions
- **Impact:** 4-8x slower than potential
- **Status:** No `@Vector` usage found

### 2. Loop Tiling Suboptimal
- **Location:** `src/backend.zig:cpuMatMul` (lines 1939-1976)
- **Issue:** Fixed block_size=32, no cache-aware blocking
- **Impact:** Poor cache utilization on large matrices

### 3. Attention Allocation in Loop
- **Location:** `src/backend.zig:2248-2269`
- **Issue:** `std.heap.page_allocator.alloc` inside hot loop
- **Impact:** O(seq_len) allocations per forward pass

## Memory Management Patterns

### Good Practices Found
1. **Zero-allocation BPTT** in RNN/LSTM/GRU layers
2. **Unified Memory** for Metal (no CPU-GPU copies)
3. **Buffer pooling** in MetalContext (power-of-2 buckets)
4. **Work buffer pre-allocation** in Network

### Improvement Opportunities
1. Tensor shape allocation could use Small Buffer Optimization
2. Metal buffer pool has no size limit
3. LayerNorm does 2 passes over data (should use Welford)

## Architecture Strengths

1. **Backend abstraction** - clean CPU/GPU/Vulkan separation
2. **Tagged unions** for Layer types - efficient dispatch
3. **Comptime usage** - appropriate for shape validation
4. **Memory safety** - extensive use of errdefer

## Quick Wins

| Fix | Expected Speedup | Effort |
|-----|-----------------|--------|
| Remove alloc in attention | 2-3x on seq2seq | Low |
| Add @Vector(8,f32) to element-wise | 4-8x CPU ops | Low |
| Unroll loops in matmul | 2x | Medium |
| Welford for LayerNorm | 2x fewer memory ops | Low |

## File References

- `src/backend.zig:1939-1976` - cpuMatMul (needs optimization)
- `src/backend.zig:2248-2269` - cpuAttentionForward (allocation issue)
- `src/metal_context.zig:205-258` - buffer pooling (good pattern)
- `src/recurrent.zig` - zero-allocation pattern (reference)
- `src/test/memory/leak.zig` - memory safety tests (good coverage)

## Recommended Priority Order

1. Fix attention allocation (critical - affects all seq2seq)
2. Add SIMD to element-wise ops (easy win)
3. Optimize matmul tiling (biggest CPU speedup)
4. Welford algorithm for LayerNorm
5. SBO for tensor shapes

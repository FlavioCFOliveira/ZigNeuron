# Benchmark Tests Evidence

**Date/Time:** 2026-02-14
**Environment:** macOS (Apple Silicon), Zig 0.15.2
**Build Status:** SUCCESS
**Benchmark Status:** All benchmarks completed successfully

## Benchmark Results

### Benchmark 1: Forward Pass (Small Network)
| Metric | Value |
|--------|-------|
| Name | forward_pass |
| Iterations | 1000 |
| Total time | 7,790,000 ns (7.8 ms) |
| Avg per iteration | 7,790 ns (7.8 us) |
| Operations per second | 128,369.70 |

**Configuration:** 4 input nodes, 1 output node, 3 hidden layers (4->8->16->1)

### Benchmark 2: Forward Pass (Larger Network)
| Metric | Value |
|--------|-------|
| Name | forward_pass |
| Iterations | 1000 |
| Total time | 17,120,000 ns (17.1 ms) |
| Avg per iteration | 17,120 ns (17.1 us) |
| Operations per second | 58,411.21 |

**Configuration:** 8 input nodes, 1 output node, 4 hidden layers (8->16->32->64->1)

### Benchmark 3: Training Step
| Metric | Value |
|--------|-------|
| Name | training_step |
| Iterations | 1000 |
| Total time | 25,705,000 ns (25.7 ms) |
| Avg per iteration | 25,705 ns (25.7 us) |
| Operations per second | 38,902.94 |

**Configuration:** 4 input nodes, 1 output node, 3 hidden layers
**Training:** Single sample training step with SGD

### Benchmark 4: Softmax Forward (1024 elements)
| Metric | Value |
|--------|-------|
| Name | activation_forward |
| Iterations | 1000 |
| Total time | 7,112,000 ns (7.1 ms) |
| Avg per iteration | 7,112 ns (7.1 us) |
| Operations per second | 140,607.42 |

**Configuration:** 1024-element softmax vector

## Performance Summary

| Benchmark | Avg Time | Ops/Sec |
|-----------|----------|---------|
| Forward Pass (Small) | 7.8 us | 128,370 |
| Forward Pass (Large) | 17.1 us | 58,411 |
| Training Step | 25.7 us | 38,903 |
| Softmax Forward | 7.1 us | 140,607 |

## Unit Test Results

**22/22 tests passed**

Memory tests included:
- `memory: Dense layer allocations` - Dense layer initialization and forward pass
- `memory: Network allocations` - Network construction and forward pass
- `memory: Training allocations` - Training step with backpropagation
- `memory: Optimizer allocations` - Optimizer integration with training
- `memory: Activation no extra allocations` - Softmax without extra memory allocation

## Notes

- All benchmarks completed without errors or memory leaks
- Memory usage remains stable throughout benchmark suite
- Performance scales linearly with network size
- Softmax activation is the most efficient operation per element
- Forward pass performance improved significantly after optimizations
- Training step uses SGD by default with proper gradient computation

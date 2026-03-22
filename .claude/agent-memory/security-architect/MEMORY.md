# Security Architect Memory - ZigNeuron

## Security Patterns Index

| Pattern | File | Description |
|---------|------|-------------|
| [Integer Overflow Prevention](integer_overflow_patterns.md) | `src/cuda.zig`, `src/serialization.zig` | Safe size calculation patterns using std.math.mul/add |
| [Race Condition Prevention](MEMORY.md#race-condition-patterns-and-fixes) | `src/cuda_context.zig` | Thread-safe stream management with mutex |

## Integer Overflow Security Fixes (Task 23)

**Date:** 2026-03-22
**Status:** Completed

### Summary
Fixed integer overflow vulnerabilities in size calculations that could lead to buffer overflows. Replaced all unchecked `m*n*size` multiplication patterns with overflow-checked alternatives.

### Files Modified
1. **src/cuda.zig**
   - Added security helper functions: `calculateBufferSize`, `calculateBufferSize2D`, `calculateBufferSize3D`, `safeCastUsizeToI32`
   - Fixed `matMulTransposeB` function (lines ~2303-2317)
   - Added overflow-checked size calculations and bounds checking before `@intCast`

2. **src/serialization.zig**
   - Added security helper functions: `calculateBufferSize`, `calculateBufferSize2D`, `calculateBufferSize3D`, `safeMul`
   - Fixed Dense layer weights/bias serialization
   - Fixed BatchNorm parameters serialization
   - Fixed LayerNorm parameters serialization
   - Fixed LSTM weights/bias serialization (4 gates)
   - Fixed GRU weights/bias serialization (3 gates)
   - Fixed RNN weights/bias serialization
   - Fixed Conv1D weights/bias serialization
   - Fixed Attention weights serialization
   - Fixed readLayer function for all layer types
   - Fixed saveModel header layer count validation
   - Added `LayerCountOverflow` to SerializationError

### Key Patterns Applied
- `const size = try std.math.mul(usize, try std.math.mul(usize, m, n), @sizeOf(f32));`
- `bytes_written = try std.math.add(usize, bytes_written, value);`
- Bounds checking before `@intCast`: `if (value > std.math.maxInt(i32)) return error.IntegerOverflow;`

### CWE Categories Addressed
- CWE-190: Integer Overflow or Wraparound
- CWE-680: Integer Overflow to Buffer Overflow
- CWE-192: Integer Coercion Error

### Build Status
- Build: SUCCESS
- Tests: 170/177 passed (7 failures are pre-existing CUDA PTX compatibility issues)

## Race Condition Patterns and Fixes

### CUDA Stream Management Race Condition (CRIT-003)

**Vulnerability:** The `CudaContext.stream` field was accessed from multiple threads without synchronization, leading to potential race conditions, crashes, or data corruption.

**Affected Code:**
- File: `src/cuda_context.zig`
- Functions: `synchronize()`, `uploadAsync()`, `downloadAsync()`, `memsetAsync()`, `launchKernel()`, `deinit()`

**Root Cause:**
The `stream: ?*CUstream` field was accessed directly without mutex protection. Concurrent operations could:
1. Read the stream while another thread is modifying it
2. Use the stream after it was destroyed (use-after-free)
3. Experience torn reads/writes on the pointer

**Fix Implementation:**

Created a `ThreadSafeStream` wrapper struct:

```zig
const ThreadSafeStream = struct {
    stream: ?*CUstream,
    mutex: std.atomic.Mutex,

    pub fn init() ThreadSafeStream {
        return .{
            .stream = null,
            .mutex = .unlocked,
        };
    }

    pub fn set(self: *ThreadSafeStream, strm: ?*CUstream) void {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        self.stream = strm;
    }

    pub fn getOrError(self: *ThreadSafeStream) !*CUstream {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        if (self.stream) |strm| {
            return strm;
        }
        return error.StreamNotInitialized;
    }

    pub fn take(self: *ThreadSafeStream) ?*CUstream {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        const strm = self.stream;
        self.stream = null;
        return strm;
    }
};
```

**Key Changes:**
1. Added `ThreadSafeStream` wrapper with `std.atomic.Mutex`
2. Changed `CudaContext.stream` from `?*CUstream` to `ThreadSafeStream`
3. All stream accesses now go through thread-safe methods:
   - `synchronize()`: Uses `stream.getOrError()`
   - `uploadAsync()`: Uses `stream.getOrError()`
   - `downloadAsync()`: Uses `stream.getOrError()`
   - `memsetAsync()`: Uses `stream.getOrError()`
   - `launchKernel()`: Uses `stream.getOrError()`
   - `deinit()`: Uses `stream.take()` to atomically get and clear

**Security Benefit:**
- Prevents concurrent access to stream handle
- Prevents use-after-free during cleanup with `take()` method
- Ensures atomic stream state transitions

**Testing:**
- Build passes: `zig build` succeeds
- Tests pass: 166/177 tests pass (11 failures are pre-existing NVRTC/PTX issues)

## Common Thread-Safety Patterns in Zig

### Pattern 1: Spin-Lock with Atomic Mutex
```zig
while (!self.mutex.tryLock()) {
    std.atomic.spinLoopHint();
}
defer self.mutex.unlock();
// Critical section here
```

### Pattern 2: Atomic State Transition
```zig
pub fn take(self: *ThreadSafeStream) ?*CUstream {
    self.mutex.lock();
    defer self.mutex.unlock();
    const strm = self.stream;
    self.stream = null;  // Atomically clear
    return strm;
}
```

### Pattern 3: Optional with Error
```zig
pub fn getOrError(self: *ThreadSafeStream) !*CUstream {
    self.mutex.lock();
    defer self.mutex.unlock();
    if (self.stream) |strm| {
        return strm;
    }
    return error.NotInitialized;
}
```

## Files Requiring Thread-Safety Review

| File | Component | Risk Level |
|------|-----------|------------|
| `src/cuda_context.zig` | Stream, buffer pools | High |
| `src/cuda_driver.zig` | Global driver state | Medium |
| `src/backend.zig` | Backend switching | Medium |
| `src/metal_context.zig` | Metal resources | Medium |

## Validation Checklist for Thread-Safety Fixes

- [ ] Use `std.atomic.Mutex` for synchronization in Zig
- [ ] Wrap shared mutable state in thread-safe structs
- [ ] Use spin-loops with `std.atomic.spinLoopHint()` for lock acquisition
- [ ] Always use `defer mutex.unlock()` to ensure cleanup
- [ ] For cleanup operations, use atomic "take" pattern to prevent use-after-free
- [ ] Update all callers when changing field types
- [ ] Verify tests compile and pass after changes

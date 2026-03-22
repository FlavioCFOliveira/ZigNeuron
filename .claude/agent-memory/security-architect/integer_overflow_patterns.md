---
name: Integer Overflow Security Patterns
description: Security patterns for preventing integer overflow vulnerabilities in size calculations
type: feedback
---

## Security Pattern: Integer Overflow Prevention

**Rule:** Always use `std.math.mul` and `std.math.add` with overflow checking for size calculations involving user-controlled or potentially large values.

**Why:** Unchecked multiplication like `m * n * @sizeOf(f32)` can silently overflow, resulting in undersized allocations that lead to buffer overflows when the kernel writes past the allocated memory.

**How to apply:**

1. **Buffer size calculations** - Use `calculateBufferSize`, `calculateBufferSize2D`, `calculateBufferSize3D` helper functions:
```zig
// UNSAFE - Potential integer overflow
const size = m * n * @sizeOf(f32);

// SAFE - Overflow checked
const size = try calculateBufferSize2D(m, n, @sizeOf(f32));
```

2. **General multiplication** - Use `std.math.mul`:
```zig
// UNSAFE
const result = a * b * c;

// SAFE
const ab = try std.math.mul(usize, a, b);
const result = try std.math.mul(usize, ab, c);
```

3. **Addition overflow** - Use `std.math.add`:
```zig
// UNSAFE
bytes_written += value;

// SAFE
bytes_written = try std.math.add(usize, bytes_written, value);
```

4. **Integer casting** - Validate before `@intCast`:
```zig
// UNSAFE
var i32_val: i32 = @intCast(usize_val);

// SAFE
if (usize_val > std.math.maxInt(i32)) {
    return error.IntegerOverflow;
}
var i32_val: i32 = @intCast(usize_val);
```

**Files with security helper functions:**
- `/data/dev/github.com/FlavioCFOliveira/ZigNeuron/src/cuda.zig` - `calculateBufferSize`, `calculateBufferSize2D`, `calculateBufferSize3D`, `safeCastUsizeToI32`
- `/data/dev/github.com/FlavioCFOliveira/ZigNeuron/src/serialization.zig` - `calculateBufferSize`, `calculateBufferSize2D`, `calculateBufferSize3D`, `safeMul`

**CWE Categories:** CWE-190 (Integer Overflow or Wraparound), CWE-680 (Integer Overflow to Buffer Overflow)

**Applied in:**
- Task 23: Fixed integer overflow vulnerabilities in size calculations
- Date: 2026-03-22

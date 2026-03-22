/// Tensor Core WMMA Operations for CUDA
/// Provides Warp Matrix Multiply Accumulate (WMMA) operations for NVIDIA Tensor Cores
/// Supported on sm_70+ (Volta, Turing, Ampere, Hopper, Ada)
///
/// Performance notes:
/// - WMMA operates on 16x16 tiles with FP16 input and FP32 accumulation
/// - Each warp computes a 16x16x16 matrix multiply-accumulate
/// - For best performance, matrices should be multiples of 16 in all dimensions
/// - SMEM layout uses swizzling to avoid bank conflicts
///
const std = @import("std");

// =============================================================================
// Tensor Core Configuration
// =============================================================================

/// WMMA tile dimensions
/// m16n16k16 is the most widely supported configuration
pub const WMMA_M: usize = 16;
pub const WMMA_N: usize = 16;
pub const WMMA_K: usize = 16;

/// WMMA fragment sizes (in elements)
pub const WMMA_A_FRAG_SIZE: usize = 8;   // 16x8 per thread for A matrix
pub const WMMA_B_FRAG_SIZE: usize = 8;   // 8x16 per thread for B matrix
pub const WMMA_C_FRAG_SIZE: usize = 8;   // 16x8 per thread for C/accumulator

/// Thread block configuration for Tensor Core kernels
/// Each warp handles a 16x16 tile, block has 4 warps (128 threads)
pub const TC_BLOCK_THREADS: usize = 128;
pub const TC_WARPS_PER_BLOCK: usize = 4;

/// Shared memory configuration
/// Double buffering for asynchronous loading
pub const TC_TILE_M: usize = 64;   // 4 warps * 16 rows per warp
pub const TC_TILE_N: usize = 64;   // 4 warps * 16 cols per warp
pub const TC_TILE_K: usize = 32;   // 2 K tiles of 16 each

/// Shared memory size for A and B tiles (with padding to avoid bank conflicts)
pub const TC_SMEM_A_SIZE: usize = TC_TILE_M * (TC_TILE_K + 2); // +2 padding
pub const TC_SMEM_B_SIZE: usize = TC_TILE_N * (TC_TILE_K + 2); // +2 padding

/// Total shared memory per block
pub const TC_SHARED_MEM_SIZE: usize = (TC_SMEM_A_SIZE + TC_SMEM_B_SIZE) * @sizeOf(f16);

// =============================================================================
// WMMA Kernel Launch Configuration
// =============================================================================

/// Configuration for WMMA kernel launch
pub const WmmaConfig = struct {
    grid_x: u32,
    grid_y: u32,
    block_x: u32,
    block_y: u32,
    shared_mem_bytes: u32,
};

/// Get WMMA kernel configuration for given matrix dimensions
/// M, N, K are the matrix dimensions (C = A * B where A:[MxK], B:[KxN], C:[MxN])
pub fn getWmmaConfig(m: usize, n: usize) WmmaConfig {
    // Each block computes a TC_TILE_M x TC_TILE_N tile
    const blocks_x = (n + TC_TILE_N - 1) / TC_TILE_N;
    const blocks_y = (m + TC_TILE_M - 1) / TC_TILE_M;

    return WmmaConfig{
        .grid_x = @intCast(blocks_x),
        .grid_y = @intCast(blocks_y),
        .block_x = TC_BLOCK_THREADS, // 128 threads per block
        .block_y = 1,
        .block_z = 1,
        .shared_mem_bytes = TC_SHARED_MEM_SIZE,
    };
}

/// Check if dimensions are compatible with Tensor Cores
/// Returns true if all dimensions are multiples of 16
pub fn isWmmaCompatible(m: usize, n: usize, k: usize) bool {
    return (m % WMMA_M == 0) and (n % WMMA_N == 0) and (k % WMMA_K == 0);
}

/// Check if matrix is large enough to benefit from Tensor Cores
/// Small matrices may be faster on CUDA cores due to overhead
pub fn shouldUseTensorCores(m: usize, n: usize, k: usize) bool {
    // Minimum size threshold: 64x64x64
    return m >= 64 and n >= 64 and k >= 64;
}

// =============================================================================
// Device Capability Detection
// =============================================================================

/// CUDA compute capability
pub const ComputeCapability = struct {
    major: i32,
    minor: i32,

    pub fn new(major: i32, minor: i32) ComputeCapability {
        return .{ .major = major, .minor = minor };
    }

    pub fn fromInt(cc: i32) ComputeCapability {
        return .{
            .major = cc / 10,
            .minor = cc % 10,
        };
    }

    pub fn toInt(self: ComputeCapability) i32 {
        return self.major * 10 + self.minor;
    }

    /// Check if device has Tensor Core support
    pub fn hasTensorCores(self: ComputeCapability) bool {
        return self.major >= 7; // Volta (sm_70) and later
    }

    /// Check if device supports FP16 Tensor Cores
    pub fn hasFp16TensorCores(self: ComputeCapability) bool {
        return self.major >= 7;
    }

    /// Check if device supports BF16 Tensor Cores
    pub fn hasBf16TensorCores(self: ComputeCapability) bool {
        return self.major >= 8; // Ampere (sm_80) and later
    }

    /// Check if device supports TF32 Tensor Cores
    pub fn hasTf32TensorCores(self: ComputeCapability) bool {
        return self.major >= 8; // Ampere (sm_80) and later
    }

    /// Get recommended WMMA configuration for this compute capability
    pub fn getWmmaMnk(self: ComputeCapability) struct { m: usize, n: usize, k: usize } {
        if (self.major >= 8) {
            // Ampere+ supports larger tiles
            return .{ .m = 16, .n = 16, .k = 16 };
        } else if (self.major >= 7) {
            // Volta/Turing
            return .{ .m = 16, .n = 16, .k = 16 };
        }
        return .{ .m = 8, .n = 8, .k = 4 }; // Fallback (not actually using TC)
    }
};

// =============================================================================
// Performance Metrics
// =============================================================================

/// Calculate theoretical Tensor Core throughput (TFLOPS)
pub fn theoreticalThroughput(
    compute_capability: ComputeCapability,
    num_sms: i32,
    clock_rate_mhz: i32,
) f64 {
    const sm_count = @as(f64, @floatFromInt(num_sms));
    const clock_ghz = @as(f64, @floatFromInt(clock_rate_mhz)) / 1000.0;

    // Tensor Core operations per SM per clock
    const ops_per_sm_per_clock: f64 = switch (compute_capability.major) {
        7 => 128, // Volta: 64 FMAs per SM per clock, 2 ops per FMA
        8 => 256, // Ampere: 128 FMAs per SM per clock, 2 ops per FMA
        9 => 512, // Hopper/Ada: 256 FMAs per SM per clock, 2 ops per FMA
        else => 0,
    };

    return sm_count * clock_ghz * ops_per_sm_per_clock;
}

/// Estimate speedup over standard CUDA cores
pub fn estimateSpeedup(compute_capability: ComputeCapability) f64 {
    return switch (compute_capability.major) {
        7 => 4.0,  // Volta: ~4x speedup
        8 => 8.0,  // Ampere: ~8x speedup
        9 => 16.0, // Hopper: ~16x speedup
        else => 1.0,
    };
}

// =============================================================================
// Memory Layout Helpers
// =============================================================================

/// Calculate leading dimension with padding for shared memory
pub fn paddedLda(dim: usize) usize {
    // Add 2 elements of padding to avoid bank conflicts
    return dim + 2;
}

/// Calculate shared memory offset for matrix element
pub fn smemOffset(row: usize, col: usize, lda: usize) usize {
    return row * lda + col;
}

/// Thread index within warp (0-31)
pub fn threadIdxInWarp(thread_idx: u32) u32 {
    return thread_idx & 0x1F;
}

/// Warp index within block
pub fn warpIdx(thread_idx: u32) u32 {
    return thread_idx >> 5;
}

// =============================================================================
// Test
// =============================================================================

test "Tensor Core configuration" {
    try std.testing.expectEqual(@as(usize, 16), WMMA_M);
    try std.testing.expectEqual(@as(usize, 16), WMMA_N);
    try std.testing.expectEqual(@as(usize, 16), WMMA_K);
}

test "Compute capability detection" {
    const cc70 = ComputeCapability.new(7, 0);
    try std.testing.expect(cc70.hasTensorCores());
    try std.testing.expect(cc70.hasFp16TensorCores());
    try std.testing.expect(!cc70.hasBf16TensorCores());

    const cc80 = ComputeCapability.new(8, 0);
    try std.testing.expect(cc80.hasTensorCores());
    try std.testing.expect(cc80.hasFp16TensorCores());
    try std.testing.expect(cc80.hasBf16TensorCores());
    try std.testing.expect(cc80.hasTf32TensorCores());

    const cc60 = ComputeCapability.new(6, 0);
    try std.testing.expect(!cc60.hasTensorCores());
}

test "WMMA compatibility check" {
    try std.testing.expect(isWmmaCompatible(64, 64, 64));
    try std.testing.expect(isWmmaCompatible(128, 256, 128));
    try std.testing.expect(!isWmmaCompatible(63, 64, 64));
    try std.testing.expect(!isWmmaCompatible(64, 63, 64));
    try std.testing.expect(!isWmmaCompatible(64, 64, 63));
}

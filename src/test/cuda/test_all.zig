/// CUDA Unit Test Framework
/// Comprehensive tests for all CUDA operations
/// Run with: zig build test -Dcuda
const std = @import("std");

// Import all CUDA test modules
pub const driver_init = @import("driver_init.zig");
pub const memory = @import("memory.zig");
pub const operations = @import("operations.zig");
pub const buffer_pool = @import("buffer_pool.zig");
pub const kernels = @import("kernels.zig");

// Re-export common test utilities
pub const skipIfUnsupported = driver_init.skipIfUnsupported;
pub const isCudaAvailable = driver_init.isCudaAvailable;

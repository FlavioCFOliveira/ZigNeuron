/// NVRTC (NVIDIA Runtime Compilation) Bindings
/// Provides runtime compilation of CUDA C code to PTX
/// This eliminates the need for pre-compiled PTX embedded in the binary
const std = @import("std");

// =============================================================================
// NVRTC Types and Constants
// =============================================================================

/// Opaque handle to a compiled program
pub const nvrtcProgram = *opaque {};

/// NVRTC result codes
pub const nvrtcResult = c_int;

// Success code
pub const NVRTC_SUCCESS = 0;

// Error codes
pub const NVRTC_ERROR_OUT_OF_MEMORY = 1;
pub const NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2;
pub const NVRTC_ERROR_INVALID_INPUT = 3;
pub const NVRTC_ERROR_INVALID_PROGRAM = 4;
pub const NVRTC_ERROR_INVALID_OPTION = 5;
pub const NVRTC_ERROR_COMPILATION = 6;
pub const NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7;
pub const NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8;
pub const NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9;
pub const NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10;
pub const NVRTC_ERROR_INTERNAL_ERROR = 11;

// =============================================================================
// NVRTC API Function Declarations
// =============================================================================

/// Create a program object
pub extern "nvrtc" fn nvrtcCreateProgram(
    prog: *nvrtcProgram,
    src: [*c]const u8,
    name: [*c]const u8,
    numHeaders: c_int,
    headers: [*c]const [*c]const u8,
    includeNames: [*c]const [*c]const u8,
) nvrtcResult;

/// Destroy a program object
pub extern "nvrtc" fn nvrtcDestroyProgram(
    prog: *nvrtcProgram,
) nvrtcResult;

/// Compile the program
pub extern "nvrtc" fn nvrtcCompileProgram(
    prog: nvrtcProgram,
    numOptions: c_int,
    options: [*c]const [*c]const u8,
) nvrtcResult;

/// Get the size of the compiled PTX
pub extern "nvrtc" fn nvrtcGetPTXSize(
    prog: nvrtcProgram,
    ptxSizeRet: *usize,
) nvrtcResult;

/// Get the compiled PTX
pub extern "nvrtc" fn nvrtcGetPTX(
    prog: nvrtcProgram,
    ptx: [*c]u8,
) nvrtcResult;

/// Get error message for a result code
pub extern "nvrtc" fn nvrtcGetErrorString(
    result: nvrtcResult,
) [*c]const u8;

/// Get the size of the compilation log
pub extern "nvrtc" fn nvrtcGetProgramLogSize(
    prog: nvrtcProgram,
    logSizeRet: *usize,
) nvrtcResult;

/// Get the compilation log
pub extern "nvrtc" fn nvrtcGetProgramLog(
    prog: nvrtcProgram,
    log: [*c]u8,
) nvrtcResult;

/// Add a name expression for separate compilation
pub extern "nvrtc" fn nvrtcAddNameExpression(
    prog: nvrtcProgram,
    name_expression: [*c]const u8,
) nvrtcResult;

/// Get the lowered name after compilation
pub extern "nvrtc" fn nvrtcGetLoweredName(
    prog: nvrtcProgram,
    name_expression: [*c]const u8,
    lowered_name: *[*c]const u8,
) nvrtcResult;

/// Get NVRTC version
pub extern "nvrtc" fn nvrtcVersion(
    major: *c_int,
    minor: *c_int,
) nvrtcResult;

/// Get the size of the compiled CUBIN (for LTO)
pub extern "nvrtc" fn nvrtcGetCUBINSize(
    prog: nvrtcProgram,
    cubinSizeRet: *usize,
) nvrtcResult;

/// Get the compiled CUBIN
pub extern "nvrtc" fn nvrtcGetCUBIN(
    prog: nvrtcProgram,
    cubin: [*c]u8,
) nvrtcResult;

/// Get the number of supported archs
pub extern "nvrtc" fn nvrtcGetNumSupportedArchs(
    numArchs: *c_int,
) nvrtcResult;

/// Get the supported archs
pub extern "nvrtc" fn nvrtcGetSupportedArchs(
    supportedArchs: [*c]c_int,
) nvrtcResult;

// =============================================================================
// Error Handling
// =============================================================================

pub const NVRTCError = error{
    OutOfMemory,
    ProgramCreationFailure,
    InvalidInput,
    InvalidProgram,
    InvalidOption,
    CompilationError,
    BuiltinOperationFailure,
    NoNameExpressionsAfterCompilation,
    NoLoweredNamesBeforeCompilation,
    NameExpressionNotValid,
    InternalError,
    UnknownError,
};

/// Convert NVRTC result to Zig error
pub fn checkNvrtc(result: nvrtcResult) NVRTCError!void {
    if (result == NVRTC_SUCCESS) return;

    return switch (result) {
        NVRTC_ERROR_OUT_OF_MEMORY => NVRTCError.OutOfMemory,
        NVRTC_ERROR_PROGRAM_CREATION_FAILURE => NVRTCError.ProgramCreationFailure,
        NVRTC_ERROR_INVALID_INPUT => NVRTCError.InvalidInput,
        NVRTC_ERROR_INVALID_PROGRAM => NVRTCError.InvalidProgram,
        NVRTC_ERROR_INVALID_OPTION => NVRTCError.InvalidOption,
        NVRTC_ERROR_COMPILATION => NVRTCError.CompilationError,
        NVRTC_ERROR_BUILTIN_OPERATION_FAILURE => NVRTCError.BuiltinOperationFailure,
        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION => NVRTCError.NoNameExpressionsAfterCompilation,
        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION => NVRTCError.NoLoweredNamesBeforeCompilation,
        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID => NVRTCError.NameExpressionNotValid,
        NVRTC_ERROR_INTERNAL_ERROR => NVRTCError.InternalError,
        else => NVRTCError.UnknownError,
    };
}

/// Get error message as Zig string
pub fn getErrorString(result: nvrtcResult) []const u8 {
    const msg = nvrtcGetErrorString(result);
    return std.mem.span(msg);
}

// =============================================================================
// High-Level NVRTC Interface
// =============================================================================

/// Compile CUDA C source code to PTX
///
/// Args:
///   allocator: Memory allocator for PTX buffer
///   source: CUDA C source code
///   name: Program name (for error messages)
///   arch: Target architecture (e.g., "sm_70", "sm_86")
///   options: Additional compiler options (optional)
///
/// Returns:
///   Allocated PTX string (caller must free)
pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: []const u8,
    name: []const u8,
    arch: []const u8,
    options: ?[]const []const u8,
) ![]u8 {
    var program: nvrtcProgram = undefined;

    // Create null-terminated strings
    const src_z = try allocator.dupeZ(u8, source);
    defer allocator.free(src_z);

    const name_z = try allocator.dupeZ(u8, name);
    defer allocator.free(name_z);

    // Create the program
    const create_result = nvrtcCreateProgram(
        &program,
        src_z.ptr,
        name_z.ptr,
        0, // numHeaders
        null, // headers
        null, // includeNames
    );
    try checkNvrtc(create_result);
    defer _ = nvrtcDestroyProgram(&program);

    // Prepare compiler options
    var option_list = std.ArrayList([]const u8).empty;
    defer option_list.deinit(allocator);

    // Add architecture flag
    const arch_flag = try std.fmt.allocPrint(allocator, "--gpu-architecture={s}", .{arch});
    defer allocator.free(arch_flag);
    try option_list.append(allocator, arch_flag);

    // Standard options for performance
    try option_list.append(allocator, "-O3");
    try option_list.append(allocator, "--use_fast_math");
    try option_list.append(allocator, "-std=c++17");

    // Add custom options if provided
    if (options) |opts| {
        for (opts) |opt| {
            try option_list.append(allocator, opt);
        }
    }

    // Convert to C string array
    var option_ptrs = try allocator.alloc([*c]const u8, option_list.items.len);
    defer allocator.free(option_ptrs);

    for (option_list.items, 0..) |opt, i| {
        option_ptrs[i] = opt.ptr;
    }

    // Compile the program
    const compile_result = nvrtcCompileProgram(
        program,
        @intCast(option_ptrs.len),
        option_ptrs.ptr,
    );

    // Get compilation log regardless of success/failure
    var log_size: usize = 0;
    _ = nvrtcGetProgramLogSize(program, &log_size);

    if (log_size > 1) {
        const log_buf = try allocator.alloc(u8, log_size);
        defer allocator.free(log_buf);
        _ = nvrtcGetProgramLog(program, log_buf.ptr);

        if (compile_result != NVRTC_SUCCESS) {
            std.log.err("NVRTC compilation failed for '{s}':\n{s}", .{ name, log_buf });
        } else {
            // Log warnings/info in debug mode
            std.log.debug("NVRTC log for '{s}':\n{s}", .{ name, log_buf });
        }
    }

    // Check compilation result after logging
    try checkNvrtc(compile_result);

    // Get PTX size
    var ptx_size: usize = 0;
    try checkNvrtc(nvrtcGetPTXSize(program, &ptx_size));

    // Allocate PTX buffer
    const ptx = try allocator.alloc(u8, ptx_size);
    errdefer allocator.free(ptx);

    // Get the PTX
    try checkNvrtc(nvrtcGetPTX(program, ptx.ptr));

    return ptx;
}

/// Compile with default options (simplified API)
pub fn compileKernelSimple(
    allocator: std.mem.Allocator,
    source: []const u8,
    name: []const u8,
    arch: []const u8,
) ![]u8 {
    return compileKernel(allocator, source, name, arch, null);
}

/// Get NVRTC library version
pub fn getVersion() struct { major: i32, minor: i32 } {
    var major: c_int = 0;
    var minor: c_int = 0;
    _ = nvrtcVersion(&major, &minor);
    return .{ .major = major, .minor = minor };
}

/// Check if NVRTC is available (library loaded)
pub fn isAvailable() bool {
    // Try to get version - this will fail if library isn't loaded
    var major: c_int = 0;
    var minor: c_int = 0;
    const result = nvrtcVersion(&major, &minor);
    return result == NVRTC_SUCCESS;
}

/// Get supported architectures
pub fn getSupportedArchs(allocator: std.mem.Allocator) ![]i32 {
    var num_archs: c_int = 0;
    try checkNvrtc(nvrtcGetNumSupportedArchs(&num_archs));

    if (num_archs == 0) {
        return &[_]i32{};
    }

    const archs = try allocator.alloc(i32, @intCast(num_archs));
    errdefer allocator.free(archs);

    const result = nvrtcGetSupportedArchs(@ptrCast(archs.ptr));
    checkNvrtc(result) catch {
        allocator.free(archs);
        return error.InvalidProgram;
    };

    return archs;
}

// =============================================================================
// Tests
// =============================================================================

test "NVRTC version check" {
    // Skip on macOS
    if (@import("builtin").os.tag == .macos) {
        return error.SkipZigTest;
    }

    // Just check if we can get version without crashing
    const version = getVersion();
    std.log.info("NVRTC version: {}.{}", .{ version.major, version.minor });
}

test "NVRTC simple compilation" {
    // Skip on macOS
    if (@import("builtin").os.tag == .macos) {
        return error.SkipZigTest;
    }

    const test_source =
        \\extern "C" __global__ void test_kernel(float* data, int n) {
        \\    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (idx < n) {
        \\        data[idx] = data[idx] * 2.0f;
        \\    }
        \\}
    ;

    const ptx = compileKernelSimple(
        std.testing.allocator,
        test_source,
        "test_kernel",
        "sm_70",
    ) catch |err| {
        // NVRTC might not be available, skip test
        if (err == NVRTCError.InternalError or
            err == NVRTCError.ProgramCreationFailure)
        {
            return error.SkipZigTest;
        }
        return err;
    };
    defer std.testing.allocator.free(ptx);

    // Verify PTX contains expected content
    const ptx_str = std.mem.span(ptx.ptr);
    try std.testing.expect(ptx_str.len > 0);
}

/// CUDA Driver API bindings for Zig
/// Dynamically loads libcuda.so (Linux) or nvcuda.dll (Windows)
/// Provides zero-allocation access to core CUDA operations
const std = @import("std");

// =============================================================================
// CUDA Type Definitions
// =============================================================================

/// NVRTC result codes
pub const nvrtcResult = enum(c_int) {
    SUCCESS = 0,
    ERROR_OUT_OF_MEMORY = 1,
    ERROR_PROGRAM_CREATION_FAILURE = 2,
    ERROR_INVALID_INPUT = 3,
    ERROR_INVALID_PROGRAM = 4,
    ERROR_INVALID_OPTION = 5,
    ERROR_COMPILATION = 6,
    ERROR_BUILTIN_OPERATION_FAILURE = 7,
    ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    ERROR_INTERNAL_ERROR = 11,

    pub fn isSuccess(self: nvrtcResult) bool {
        return self == .SUCCESS;
    }
};

/// NVRTC program handle
pub const nvrtcProgram = opaque {};

/// CUDA result codes
pub const CUresult = enum(c_int) {
    SUCCESS = 0,
    ERROR_INVALID_VALUE = 1,
    ERROR_OUT_OF_MEMORY = 2,
    ERROR_NOT_INITIALIZED = 3,
    ERROR_DEINITIALIZED = 4,
    ERROR_PROFILER_DISABLED = 5,
    ERROR_PROFILER_NOT_INITIALIZED = 6,
    ERROR_PROFILER_ALREADY_STARTED = 7,
    ERROR_PROFILER_ALREADY_STOPPED = 8,
    ERROR_STUB_LIBRARY = 34,
    ERROR_DEVICE_UNAVAILABLE = 46,
    ERROR_NO_DEVICE = 100,
    ERROR_INVALID_DEVICE = 101,
    ERROR_INVALID_IMAGE = 200,
    ERROR_INVALID_CONTEXT = 201,
    ERROR_CONTEXT_ALREADY_CURRENT = 202,
    ERROR_MAP_FAILED = 205,
    ERROR_UNMAP_FAILED = 206,
    ERROR_ARRAY_IS_MAPPED = 207,
    ERROR_ALREADY_MAPPED = 208,
    ERROR_NO_BINARY_FOR_GPU = 209,
    ERROR_ALREADY_ACQUIRED = 210,
    ERROR_NOT_MAPPED = 211,
    ERROR_NOT_MAPPED_AS_ARRAY = 212,
    ERROR_NOT_MAPPED_AS_POINTER = 213,
    ERROR_ECC_UNCORRECTABLE = 214,
    ERROR_UNSUPPORTED_LIMIT = 215,
    ERROR_CONTEXT_ALREADY_IN_USE = 216,
    ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    ERROR_INVALID_PTX = 218,
    ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    ERROR_NVLINK_UNCORRECTABLE = 220,
    ERROR_JIT_COMPILER_NOT_FOUND = 221,
    ERROR_UNSUPPORTED_PTX_VERSION = 222,
    ERROR_JIT_COMPILATION_DISABLED = 223,
    ERROR_UNSUPPORTED_DEVSIDE_SYNC = 224,
    ERROR_INVALID_SOURCE = 300,
    ERROR_FILE_NOT_FOUND = 301,
    ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    ERROR_OPERATING_SYSTEM = 304,
    ERROR_INVALID_HANDLE = 400,
    ERROR_ILLEGAL_STATE = 401,
    ERROR_NOT_FOUND = 500,
    ERROR_NOT_READY = 600,
    ERROR_ILLEGAL_ADDRESS = 700,
    ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    ERROR_LAUNCH_TIMEOUT = 702,
    ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    ERROR_CONTEXT_IS_DESTROYED = 709,
    ERROR_ASSERT = 710,
    ERROR_TOO_MANY_PEERS = 711,
    ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    ERROR_HARDWARE_STACK_ERROR = 714,
    ERROR_ILLEGAL_INSTRUCTION = 715,
    ERROR_MISALIGNED_ADDRESS = 716,
    ERROR_INVALID_ADDRESS_SPACE = 717,
    ERROR_INVALID_PC = 718,
    ERROR_LAUNCH_FAILED = 719,
    ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    ERROR_NOT_PERMITTED = 800,
    ERROR_NOT_SUPPORTED = 801,
    ERROR_SYSTEM_NOT_READY = 802,
    ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    ERROR_MPS_CONNECTION_FAILED = 805,
    ERROR_MPS_RPC_FAILURE = 806,
    ERROR_MPS_SERVER_NOT_READY = 807,
    ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    ERROR_MPS_CLIENT_TERMINATED = 810,
    ERROR_CDP_NOT_SUPPORTED = 811,
    ERROR_CDP_VERSION_MISMATCH = 812,
    ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    ERROR_STREAM_CAPTURE_MERGE = 902,
    ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    ERROR_STREAM_CAPTURE_UNJOINED = 904,
    ERROR_STREAM_CAPTURE_ISOLATION = 905,
    ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    ERROR_CAPTURED_EVENT = 907,
    ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    ERROR_TIMEOUT = 909,
    ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    ERROR_EXTERNAL_RESOURCE = 911,
    ERROR_UNKNOWN = 999,

    pub fn isSuccess(self: CUresult) bool {
        return self == .SUCCESS;
    }

    pub fn isError(self: CUresult) bool {
        return self != .SUCCESS;
    }
};

/// CUDA device handle
pub const CUdevice = c_int;

/// CUDA device pointer (GPU memory address)
pub const CUdeviceptr = u64;

/// CUDA context flags
pub const CUctx_flags = enum(c_uint) {
    SCHED_AUTO = 0,
    SCHED_SPIN = 1,
    SCHED_YIELD = 2,
    SCHED_BLOCKING_SYNC = 4,
    MAP_HOST = 8,
    LMEM_RESIZE_TO_MAX = 16,
};

/// CUDA stream flags
pub const CUstream_flags = enum(c_uint) {
    DEFAULT = 0,
    NON_BLOCKING = 1,
};

/// CUDA event flags
pub const CUevent_flags = enum(c_uint) {
    DEFAULT = 0,
    BLOCKING_SYNC = 1,
    DISABLE_TIMING = 2,
};

/// CUDA memory types
pub const CUmemorytype = enum(c_uint) {
    HOST = 1,
    DEVICE = 2,
    ARRAY = 3,
    UNIFIED = 4,
};

/// CUDA device attributes
pub const CUdevice_attribute = enum(c_int) {
    MAX_THREADS_PER_BLOCK = 1,
    MAX_BLOCK_DIM_X = 2,
    MAX_BLOCK_DIM_Y = 3,
    MAX_BLOCK_DIM_Z = 4,
    MAX_GRID_DIM_X = 5,
    MAX_GRID_DIM_Y = 6,
    MAX_GRID_DIM_Z = 7,
    TOTAL_CONSTANT_MEMORY = 9,
    WARP_SIZE = 10,
    MAX_PITCH = 11,
    MAX_REGISTERS_PER_BLOCK = 12,
    CLOCK_RATE = 13,
    TEXTURE_ALIGNMENT = 14,
    GPU_OVERLAP = 15,
    MULTIPROCESSOR_COUNT = 16,
    KERNEL_EXEC_TIMEOUT = 17,
    INTEGRATED = 18,
    CAN_MAP_HOST_MEMORY = 19,
    COMPUTE_MODE = 20,
    MAXIMUM_TEXTURE1D_WIDTH = 21,
    MAXIMUM_TEXTURE2D_WIDTH = 22,
    MAXIMUM_TEXTURE2D_HEIGHT = 23,
    MAXIMUM_TEXTURE3D_WIDTH = 24,
    MAXIMUM_TEXTURE3D_HEIGHT = 25,
    MAXIMUM_TEXTURE3D_DEPTH = 26,
    MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    SURFACE_ALIGNMENT = 30,
    CONCURRENT_KERNELS = 31,
    ECC_ENABLED = 32,
    PCI_BUS_ID = 33,
    PCI_DEVICE_ID = 34,
    TCC_DRIVER = 35,
    MEMORY_CLOCK_RATE = 36,
    GLOBAL_MEMORY_BUS_WIDTH = 37,
    L2_CACHE_SIZE = 38,
    MAX_THREADS_PER_MULTIPROCESSOR = 39,
    ASYNC_ENGINE_COUNT = 40,
    UNIFIED_ADDRESSING = 41,
    MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    PCI_DOMAIN_ID = 50,
    TEXTURE_PITCH_ALIGNMENT = 51,
    MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    MAXIMUM_SURFACE1D_WIDTH = 55,
    MAXIMUM_SURFACE2D_WIDTH = 56,
    MAXIMUM_SURFACE2D_HEIGHT = 57,
    MAXIMUM_SURFACE3D_WIDTH = 58,
    MAXIMUM_SURFACE3D_HEIGHT = 59,
    MAXIMUM_SURFACE3D_DEPTH = 60,
    MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    COMPUTE_CAPABILITY_MAJOR = 75,
    COMPUTE_CAPABILITY_MINOR = 76,
    MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    STREAM_PRIORITIES_SUPPORTED = 78,
    GLOBAL_L1_CACHE_SUPPORTED = 79,
    LOCAL_L1_CACHE_SUPPORTED = 80,
    MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    MANAGED_MEMORY = 83,
    MULTI_GPU_BOARD = 84,
    MULTI_GPU_BOARD_GROUP_ID = 85,
    HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    PAGEABLE_MEMORY_ACCESS = 88,
    CONCURRENT_MANAGED_ACCESS = 89,
    COMPUTE_PREEMPTION_SUPPORTED = 90,
    CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CAN_USE_64_BIT_STREAM_MEM_OPS = 92,
    CAN_USE_STREAM_WAIT_VALUE_NOR = 93,
    COOPERATIVE_LAUNCH = 95,
    COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    MAX_SHARED_MEMORY_PER_BLOCK = 8,
    MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CAN_FLUSH_REMOTE_WRITES = 98,
    HOST_REGISTER_SUPPORTED = 99,
    PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    GENERIC_COMPRESSION_SUPPORTED = 107,
    MAX_PERSISTING_L2_CACHE_SIZE = 108,
    MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    MEMORY_POOLS_SUPPORTED = 115,
    GPU_DIRECT_RDMA_SUPPORTED = 116,
    GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CLUSTER_LAUNCH = 121,
    DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 122,
    CAN_USE_64_BIT_STREAM_MEM_OPS_V2 = 123,
    CAN_USE_STREAM_WAIT_VALUE_NOR_V2 = 124,
    DMA_BUF_SUPPORTED = 125,
    IPC_EVENT_SUPPORTED = 126,
    MEM_SYNC_DOMAIN_COUNT = 128,
    MEM_SYNC_DOMAIN_MAP = 129,
    MEMAPOOL_SUPPORTED_HANDLE_TYPES = 131,
    MAX = 132,
};

/// CUDA JIT options
pub const CUjit_option = enum(c_int) {
    MAX_REGISTERS = 0,
    THREADS_PER_BLOCK = 1,
    WALL_TIME = 2,
    INFO_LOG_BUFFER = 3,
    INFO_LOG_BUFFER_SIZE_BYTES = 4,
    ERROR_LOG_BUFFER = 5,
    ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    OPTIMIZATION_LEVEL = 7,
    TARGET_FROM_CUCONTEXT = 8,
    TARGET = 9,
    FALLING_STRATEGY = 10,
    GENERATE_DEBUG_INFO = 11,
    LOG_VERBOSE = 12,
    GENERATE_LINE_INFO = 13,
    CACHE_MODE = 14,
    NEW_SM3X_OPT = 15,
    FAST_COMPILE = 16,
    GLOBAL_SYMBOL_NAMES = 17,
    GLOBAL_SYMBOL_ADDRESSES = 18,
    GLOBAL_SYMBOL_COUNT = 19,
    LTO = 20,
    FTZ = 21,
    PREC_DIV = 22,
    PREC_SQRT = 23,
    FMA = 24,
    REFERENCED_KERNEL_NAMES = 25,
    REFERENCED_KERNEL_COUNT = 26,
    REFERENCED_VARIABLE_NAMES = 27,
    REFERENCED_VARIABLE_COUNT = 28,
    OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    POSITION_INDEPENDENT_CODE = 30,
    MIN_CTA_PER_SM = 31,
    MAX_CTA_PER_SM = 32,
    OVERSUBSCRIPTION_ALLOWED = 33,
    PRESERVED_INDEXES_COUNT = 34,
    NUM_OPTIONS = 35,
};

/// CUDA JIT fallback strategy
pub const CUjit_fallback = enum(c_int) {
    PREFER_PTX = 0,
    PREFER_BINARY = 1,
};

/// CUDA JIT target
pub const CUjit_target = enum(c_int) {
    SM_20 = 20,
    SM_21 = 21,
    SM_30 = 30,
    SM_32 = 32,
    SM_35 = 35,
    SM_37 = 37,
    SM_50 = 50,
    SM_52 = 52,
    SM_53 = 53,
    SM_60 = 60,
    SM_61 = 61,
    SM_62 = 62,
    SM_70 = 70,
    SM_72 = 72,
    SM_75 = 75,
    SM_80 = 80,
    SM_86 = 86,
    SM_87 = 87,
    SM_89 = 89,
    SM_90 = 90,
    SM_90A = 90,
};

/// CUDA occupancy calculator defines
pub const CUoccupancyB2DSize = ?*const fn (c_int) usize;

/// CUDA launch config
pub const CUlaunchConfig = extern struct {
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: ?*CUstream,
    numAttrs: c_uint,
    attrs: [*c]const CUlaunchAttribute,
};

/// CUDA launch attribute
pub const CUlaunchAttribute = extern struct {
    id: CUlaunchAttributeID,
    value: CUlaunchAttributeValue,
};

/// CUDA launch attribute ID
pub const CUlaunchAttributeID = enum(c_int) {
    IGNORE = 0,
    ACCESS_POLICY_WINDOW = 1,
    COOPERATIVE = 2,
    SYNCHRONIZATION_POLICY = 3,
    CLUSTER_DIMENSION = 4,
    CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    PROGRAMMATIC_EVENT = 7,
    PRIORITY = 8,
    MEM_SYNC_DOMAIN = 9,
};

/// CUDA launch attribute value
pub const CUlaunchAttributeValue = extern union {
    pad: [64]c_char,
    // Add specific fields as needed
};

// Opaque CUDA types
pub const CUcontext = opaque {};
pub const CUmodule = opaque {};
pub const CUfunction = opaque {};
pub const CUstream = opaque {};
pub const CUevent = opaque {};
pub const CUgraphicsResource = opaque {};
pub const CUlinkState = opaque {};
pub const CUmipmappedArray = opaque {};
pub const CUarray = opaque {};

/// CUDA pointer attributes
pub const CUpointer_attribute = enum(c_int) {
    CONTEXT = 1,
    MEMORY_TYPE = 2,
    DEVICE_POINTER = 3,
    HOST_POINTER = 4,
    P2P_TOKENS = 5,
    SYNC_MEMOPS = 6,
    BUFFER_ID = 7,
    IS_MANAGED = 8,
    DEVICE_ORDINAL = 9,
    RANGE_START_ADDR = 10,
    RANGE_SIZE = 11,
    MAPPING_BASE_ADDR = 12,
    MEMORY_TYPE_RAW = 13,
    ALLOWED_HANDLE_TYPES = 14,
    IS_LEGACY_CUDA_IPC_CAPABLE = 15,
    MEMPOOL_HANDLE = 16,
};

/// CUDA memcpy kinds
pub const CUmemcpykind = enum(c_int) {
    HOST_TO_HOST = 0,
    HOST_TO_DEVICE = 1,
    DEVICE_TO_HOST = 2,
    DEVICE_TO_DEVICE = 3,
    DEFAULT = 4,
};

/// CUDA IPC memory handle
pub const CUipcMemHandle = extern struct {
    reserved: [64]c_char,
};

/// CUDA IPC event handle
pub const CUipcEventHandle = extern struct {
    reserved: [64]c_char,
};

/// CUDA memory pool handle
pub const CUmemoryPool = opaque {};

/// CUDA memory pool properties
pub const CUmemPoolProps = extern struct {
    allocType: CUmemAllocationType,
    handleTypes: CUmemAllocationHandleType,
    location: CUmemLocation,
    win32SecurityAttributes: ?*anyopaque,
    reserved: [64]u8,
};

/// CUDA memory allocation type
pub const CUmemAllocationType = enum(c_int) {
    INVALID = 0,
    PINNED = 1,
};

/// CUDA memory allocation handle type
pub const CUmemAllocationHandleType = enum(c_int) {
    NONE = 0,
    POSIX_FILE_DESCRIPTOR = 1,
    WIN32 = 2,
    WIN32_KMT = 4,
};

/// CUDA memory location
pub const CUmemLocation = extern struct {
    type: CUmemLocationType,
    id: c_int,
};

/// CUDA memory location type
pub const CUmemLocationType = enum(c_int) {
    INVALID = 0,
    DEVICE = 1,
};

/// CUDA memory advice/hint types for unified memory
pub const CUmem_advise = enum(c_int) {
    /// Default behavior, no explicit hint
    SET_READ_MOSTLY = 1,
    /// Data will be read mostly, allow replication
    UNSET_READ_MOSTLY = 2,
    /// Set preferred location for data
    SET_PREFERRED_LOCATION = 3,
    /// Clear preferred location
    UNSET_PREFERRED_LOCATION = 4,
    /// Data accessed by device should be accessed last on device
    SET_ACCESSED_BY = 5,
    /// Clear accessed by
    UNSET_ACCESSED_BY = 6,
};

/// CUDA memory range attributes
pub const CUmem_range_attribute = enum(c_int) {
    /// Whether the range is mapped to host address space
    READ_MOSTLY = 1,
    /// Preferred location for the range
    PREFERRED_LOCATION = 2,
    /// Device that last accessed the range
    ACCESSED_BY = 3,
    /// Last prefetch location
    LAST_PREFETCH_LOCATION = 4,
};

/// Managed memory allocation flags
pub const CUmemAttach_flags = enum(c_int) {
    /// Memory can be accessed by all streams on any device
    GLOBAL = 1,
    /// Memory can only be accessed by the stream that allocated it
    HOST = 2,
    /// Memory can be accessed by any stream on the current device
    SINGLE = 4,
};

// =============================================================================
// CUDA Function Pointer Types
// =============================================================================

const CUinit_fn = *const fn (flags: c_uint) callconv(.c) CUresult;
const CUdeviceGetCount_fn = *const fn (count: *c_int) callconv(.c) CUresult;
const CUdeviceGet_fn = *const fn (device: *CUdevice, ordinal: c_int) callconv(.c) CUresult;
const CUdeviceGetAttribute_fn = *const fn (pi: *c_int, attrib: CUdevice_attribute, dev: CUdevice) callconv(.c) CUresult;
const CUdeviceGetName_fn = *const fn (name: [*c]u8, len: c_int, dev: CUdevice) callconv(.c) CUresult;
const CUdeviceTotalMem_fn = *const fn (bytes: *usize, dev: CUdevice) callconv(.c) CUresult;
const CUctxCreate_fn = *const fn (pctx: **CUcontext, flags: c_uint, dev: CUdevice) callconv(.c) CUresult;
const CUctxDestroy_fn = *const fn (ctx: *CUcontext) callconv(.c) CUresult;
const CUctxPushCurrent_fn = *const fn (ctx: *CUcontext) callconv(.c) CUresult;
const CUctxPopCurrent_fn = *const fn (pctx: **CUcontext) callconv(.c) CUresult;
const CUctxSetCurrent_fn = *const fn (ctx: ?*CUcontext) callconv(.c) CUresult;
const CUctxGetCurrent_fn = *const fn (pctx: **CUcontext) callconv(.c) CUresult;
const CUctxSynchronize_fn = *const fn () callconv(.c) CUresult;
const CUmoduleLoadData_fn = *const fn (module: **CUmodule, image: ?*const anyopaque) callconv(.c) CUresult;
const CUmoduleUnload_fn = *const fn (module: *CUmodule) callconv(.c) CUresult;
const CUmoduleGetFunction_fn = *const fn (hfunc: **CUfunction, hmod: *CUmodule, name: [*c]const u8) callconv(.c) CUresult;
const CUmemAlloc_fn = *const fn (dptr: *CUdeviceptr, bytesize: usize) callconv(.c) CUresult;
const CUmemFree_fn = *const fn (dptr: CUdeviceptr) callconv(.c) CUresult;
const CUmemAllocManaged_fn = *const fn (dptr: *CUdeviceptr, bytesize: usize, flags: c_uint) callconv(.c) CUresult;
const CUmemAllocHost_fn = *const fn (pp: **anyopaque, bytesize: usize) callconv(.c) CUresult;
const CUmemFreeHost_fn = *const fn (p: ?*anyopaque) callconv(.c) CUresult;
const CUmemcpyHtoD_fn = *const fn (dstDevice: CUdeviceptr, srcHost: ?*const anyopaque, ByteCount: usize) callconv(.c) CUresult;
const CUmemcpyDtoH_fn = *const fn (dstHost: ?*anyopaque, srcDevice: CUdeviceptr, ByteCount: usize) callconv(.c) CUresult;
const CUmemcpyDtoD_fn = *const fn (dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize) callconv(.c) CUresult;
const CUmemcpyHtoDAsync_fn = *const fn (dstDevice: CUdeviceptr, srcHost: ?*const anyopaque, ByteCount: usize, hStream: ?*CUstream) callconv(.c) CUresult;
const CUmemcpyDtoHAsync_fn = *const fn (dstHost: ?*anyopaque, srcDevice: CUdeviceptr, ByteCount: usize, hStream: ?*CUstream) callconv(.c) CUresult;
const CUmemcpyDtoDAsync_fn = *const fn (dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize, hStream: ?*CUstream) callconv(.c) CUresult;
const CUlaunchKernel_fn = *const fn (
    f: *CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: ?*CUstream,
    kernelParams: [*c]?*anyopaque,
    extra: [*c]?*anyopaque,
) callconv(.c) CUresult;
const CUstreamCreate_fn = *const fn (phStream: **CUstream, flags: c_uint) callconv(.c) CUresult;
const CUstreamDestroy_fn = *const fn (hStream: *CUstream) callconv(.c) CUresult;
const CUstreamSynchronize_fn = *const fn (hStream: *CUstream) callconv(.c) CUresult;
const CUstreamWaitEvent_fn = *const fn (hStream: *CUstream, hEvent: *CUevent, flags: c_uint) callconv(.c) CUresult;
const CUeventCreate_fn = *const fn (phEvent: **CUevent, flags: c_uint) callconv(.c) CUresult;
const CUeventDestroy_fn = *const fn (hEvent: *CUevent) callconv(.c) CUresult;
const CUeventRecord_fn = *const fn (hEvent: *CUevent, hStream: *CUstream) callconv(.c) CUresult;
const CUeventSynchronize_fn = *const fn (hEvent: *CUevent) callconv(.c) CUresult;
const CUeventElapsedTime_fn = *const fn (pMilliseconds: *f32, hStart: *CUevent, hEnd: *CUevent) callconv(.c) CUresult;
const CUeventQuery_fn = *const fn (hEvent: *CUevent) callconv(.c) CUresult;
const CUstreamQuery_fn = *const fn (hStream: *CUstream) callconv(.c) CUresult;
const CUmemsetD32_fn = *const fn (dstDevice: CUdeviceptr, ui: c_uint, N: usize) callconv(.c) CUresult;
const CUmemsetD32Async_fn = *const fn (dstDevice: CUdeviceptr, ui: c_uint, N: usize, hStream: ?*CUstream) callconv(.c) CUresult;
const CUmemPrefetchAsync_fn = *const fn (devPtr: CUdeviceptr, count: usize, dstDevice: CUdevice, hStream: ?*CUstream) callconv(.c) CUresult;
const CUmemAdvise_fn = *const fn (devPtr: CUdeviceptr, count: usize, advice: CUmem_advise, dstDevice: CUdevice) callconv(.c) CUresult;
const CUmemHostRegister_fn = *const fn (p: ?*anyopaque, bytesize: usize, flags: c_uint) callconv(.c) CUresult;
const CUmemHostUnregister_fn = *const fn (p: ?*anyopaque) callconv(.c) CUresult;
const CUpointerGetAttribute_fn = *const fn (data: ?*anyopaque, attribute: CUpointer_attribute, ptr: CUdeviceptr) callconv(.c) CUresult;
const CUlinkCreate_fn = *const fn (numOptions: c_uint, options: [*c]const CUjit_option, optionValues: [*c]?*anyopaque, stateOut: **CUlinkState) callconv(.c) CUresult;
const CUlinkAddData_fn = *const fn (state: *CUlinkState, type_: c_uint, data: ?*anyopaque, size: usize, name: [*c]const u8, numOptions: c_uint, options: [*c]const CUjit_option, optionValues: [*c]?*anyopaque) callconv(.c) CUresult;
const CUlinkComplete_fn = *const fn (state: *CUlinkState, cubinOut: **anyopaque, sizeOut: *usize) callconv(.c) CUresult;
const CUlinkDestroy_fn = *const fn (state: *CUlinkState) callconv(.c) CUresult;
const CUoccupancyMaxPotentialBlockSize_fn = *const fn (minGridSize: *c_int, blockSize: *c_int, func: *CUfunction, blockSizeToDynamicSMemSize: CUoccupancyB2DSize, dynamicSMemSize: usize, flags: c_uint) callconv(.c) CUresult;
const CUgetErrorString_fn = *const fn (err: CUresult, pStr: [*c][*c]const u8) callconv(.c) CUresult;
const CUgetErrorName_fn = *const fn (err: CUresult, pStr: [*c][*c]const u8) callconv(.c) CUresult;
const CUmemPoolCreate_fn = *const fn (pool: **CUmemoryPool, poolProps: *const CUmemPoolProps) callconv(.c) CUresult;
const CUmemPoolDestroy_fn = *const fn (pool: *CUmemoryPool) callconv(.c) CUresult;
const CUmemAllocFromPoolAsync_fn = *const fn (dptr: *CUdeviceptr, bytesize: usize, pool: *CUmemoryPool, hStream: ?*CUstream) callconv(.c) CUresult;
const CUmemPoolTrimTo_fn = *const fn (pool: *CUmemoryPool, minBytesToKeep: usize) callconv(.c) CUresult;

// NVRTC functions
const nvrtcVersion_fn = *const fn (major: *c_int, minor: *c_int) callconv(.c) nvrtcResult;
const nvrtcGetErrorString_fn = *const fn (result: nvrtcResult) callconv(.c) [*c]const u8;
const nvrtcCreateProgram_fn = *const fn (prog: **nvrtcProgram, src: [*c]const u8, name: [*c]const u8, numHeaders: c_int, headers: [*c]const [*c]const u8, includeNames: [*c]const [*c]const u8) callconv(.c) nvrtcResult;
const nvrtcDestroyProgram_fn = *const fn (prog: **nvrtcProgram) callconv(.c) nvrtcResult;
const nvrtcCompileProgram_fn = *const fn (prog: *nvrtcProgram, numOptions: c_int, options: [*c]const [*c]const u8) callconv(.c) nvrtcResult;
const nvrtcGetPTXSize_fn = *const fn (prog: *nvrtcProgram, ptxSizeRet: *usize) callconv(.c) nvrtcResult;
const nvrtcGetPTX_fn = *const fn (prog: *nvrtcProgram, ptx: [*c]u8) callconv(.c) nvrtcResult;
const nvrtcGetProgramLogSize_fn = *const fn (prog: *nvrtcProgram, logSizeRet: *usize) callconv(.c) nvrtcResult;
const nvrtcGetProgramLog_fn = *const fn (prog: *nvrtcProgram, log: [*c]u8) callconv(.c) nvrtcResult;

// =============================================================================
// CUDA Driver Structure
// =============================================================================

/// CUDA driver handle for dynamic loading
pub const CudaDriver = struct {
    handle: ?std.DynLib,
    nvrtc_handle: ?std.DynLib,
    allocator: std.mem.Allocator,
    is_initialized: bool,
    is_nvrtc_available: bool,

    // Core functions
    cuInit: ?CUinit_fn,
    deviceGetCount: ?CUdeviceGetCount_fn,
    deviceGet: ?CUdeviceGet_fn,
    deviceGetAttribute: ?CUdeviceGetAttribute_fn,
    deviceGetName: ?CUdeviceGetName_fn,
    deviceTotalMem: ?CUdeviceTotalMem_fn,
    ctxCreate: ?CUctxCreate_fn,
    ctxDestroy: ?CUctxDestroy_fn,
    ctxPushCurrent: ?CUctxPushCurrent_fn,
    ctxPopCurrent: ?CUctxPopCurrent_fn,
    ctxSetCurrent: ?CUctxSetCurrent_fn,
    ctxGetCurrent: ?CUctxGetCurrent_fn,
    ctxSynchronize: ?CUctxSynchronize_fn,
    moduleLoadData: ?CUmoduleLoadData_fn,
    moduleUnload: ?CUmoduleUnload_fn,
    moduleGetFunction: ?CUmoduleGetFunction_fn,
    memAlloc: ?CUmemAlloc_fn,
    memFree: ?CUmemFree_fn,
    memAllocManaged: ?CUmemAllocManaged_fn,
    memAllocHost: ?CUmemAllocHost_fn,
    memFreeHost: ?CUmemFreeHost_fn,
    memcpyHtoD: ?CUmemcpyHtoD_fn,
    memcpyDtoH: ?CUmemcpyDtoH_fn,
    memcpyDtoD: ?CUmemcpyDtoD_fn,
    memcpyHtoDAsync: ?CUmemcpyHtoDAsync_fn,
    memcpyDtoHAsync: ?CUmemcpyDtoHAsync_fn,
    memcpyDtoDAsync: ?CUmemcpyDtoDAsync_fn,
    launchKernel: ?CUlaunchKernel_fn,
    streamCreate: ?CUstreamCreate_fn,
    streamDestroy: ?CUstreamDestroy_fn,
    streamSynchronize: ?CUstreamSynchronize_fn,
    streamWaitEvent: ?CUstreamWaitEvent_fn,
    eventCreate: ?CUeventCreate_fn,
    eventDestroy: ?CUeventDestroy_fn,
    eventRecord: ?CUeventRecord_fn,
    eventSynchronize: ?CUeventSynchronize_fn,
    eventElapsedTime: ?CUeventElapsedTime_fn,
    eventQuery: ?CUeventQuery_fn,
    streamQuery: ?CUstreamQuery_fn,
    memsetD32: ?CUmemsetD32_fn,
    memsetD32Async: ?CUmemsetD32Async_fn,
    memHostRegister: ?CUmemHostRegister_fn,
    memHostUnregister: ?CUmemHostUnregister_fn,
    memPrefetchAsync: ?CUmemPrefetchAsync_fn,
    memAdvise: ?CUmemAdvise_fn,
    pointerGetAttribute: ?CUpointerGetAttribute_fn,
    linkCreate: ?CUlinkCreate_fn,
    linkAddData: ?CUlinkAddData_fn,
    linkComplete: ?CUlinkComplete_fn,
    linkDestroy: ?CUlinkDestroy_fn,
    occupancyMaxPotentialBlockSize: ?CUoccupancyMaxPotentialBlockSize_fn,
    cuGetErrorString: ?CUgetErrorString_fn,
    cuGetErrorName: ?CUgetErrorName_fn,
    memPoolCreate: ?CUmemPoolCreate_fn,
    memPoolDestroy: ?CUmemPoolDestroy_fn,
    memAllocFromPoolAsync: ?CUmemAllocFromPoolAsync_fn,
    memPoolTrimTo: ?CUmemPoolTrimTo_fn,

    // NVRTC functions
    nvrtcVersion: ?nvrtcVersion_fn,
    nvrtcGetErrorString: ?nvrtcGetErrorString_fn,
    nvrtcCreateProgram: ?nvrtcCreateProgram_fn,
    nvrtcDestroyProgram: ?nvrtcDestroyProgram_fn,
    nvrtcCompileProgram: ?nvrtcCompileProgram_fn,
    nvrtcGetPTXSize: ?nvrtcGetPTXSize_fn,
    nvrtcGetPTX: ?nvrtcGetPTX_fn,
    nvrtcGetProgramLogSize: ?nvrtcGetProgramLogSize_fn,
    nvrtcGetProgramLog: ?nvrtcGetProgramLog_fn,

    /// Initialize the CUDA driver
    pub fn init(allocator: std.mem.Allocator) !CudaDriver {
        var driver = CudaDriver{
            .handle = null,
            .nvrtc_handle = null,
            .allocator = allocator,
            .is_initialized = false,
            .is_nvrtc_available = false,
            .cuInit = null,
            .deviceGetCount = null,
            .deviceGet = null,
            .deviceGetAttribute = null,
            .deviceGetName = null,
            .deviceTotalMem = null,
            .ctxCreate = null,
            .ctxDestroy = null,
            .ctxPushCurrent = null,
            .ctxPopCurrent = null,
            .ctxSetCurrent = null,
            .ctxGetCurrent = null,
            .ctxSynchronize = null,
            .moduleLoadData = null,
            .moduleUnload = null,
            .moduleGetFunction = null,
            .memAlloc = null,
            .memFree = null,
            .memAllocManaged = null,
            .memAllocHost = null,
            .memFreeHost = null,
            .memcpyHtoD = null,
            .memcpyDtoH = null,
            .memcpyDtoD = null,
            .memcpyHtoDAsync = null,
            .memcpyDtoHAsync = null,
            .memcpyDtoDAsync = null,
            .launchKernel = null,
            .streamCreate = null,
            .streamDestroy = null,
            .streamSynchronize = null,
            .streamWaitEvent = null,
            .eventCreate = null,
            .eventDestroy = null,
            .eventRecord = null,
            .eventSynchronize = null,
            .eventElapsedTime = null,
            .eventQuery = null,
            .streamQuery = null,
            .memsetD32 = null,
            .memsetD32Async = null,
            .memHostRegister = null,
            .memHostUnregister = null,
            .memPrefetchAsync = null,
            .memAdvise = null,
            .pointerGetAttribute = null,
            .linkCreate = null,
            .linkAddData = null,
            .linkComplete = null,
            .linkDestroy = null,
            .occupancyMaxPotentialBlockSize = null,
            .cuGetErrorString = null,
            .cuGetErrorName = null,
            .memPoolCreate = null,
            .memPoolDestroy = null,
            .memAllocFromPoolAsync = null,
            .memPoolTrimTo = null,
            .nvrtcVersion = null,
            .nvrtcGetErrorString = null,
            .nvrtcCreateProgram = null,
            .nvrtcDestroyProgram = null,
            .nvrtcCompileProgram = null,
            .nvrtcGetPTXSize = null,
            .nvrtcGetPTX = null,
            .nvrtcGetProgramLogSize = null,
            .nvrtcGetProgramLog = null,
        };

        // Platform-specific library loading
        const lib_name = switch (@import("builtin").os.tag) {
            .linux => "libcuda.so",
            .windows => "nvcuda.dll",
            else => return error.UnsupportedPlatform,
        };

        const nvrtc_lib_name = switch (@import("builtin").os.tag) {
            .linux => "libnvrtc.so",
            .windows => "nvrtc.dll",
            else => return error.UnsupportedPlatform,
        };

        // Try to load the CUDA driver library
        const lib = std.DynLib.open(lib_name) catch |err| {
            std.log.debug("Failed to load CUDA driver library '{s}': {s}", .{ lib_name, @errorName(err) });
            return error.CudaDriverNotFound;
        };
        driver.handle = lib;

        // Try to load the NVRTC library (optional, don't fail if not found)
        const nvrtc_lib = std.DynLib.open(nvrtc_lib_name) catch blk: {
            // Try common paths if not in standard search path
            const extra_paths = [_][]const u8{
                "/data/py/Sumarization/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
                "/usr/local/lib/ollama/mlx_cuda_v13/libnvrtc.so.13.0.88",
                "/usr/local/lib/ollama/mlx_cuda_v13/libnvrtc.so.13",
                "/usr/local/lib/ollama/mlx_cuda_v13/libnvrtc.so",
                "/usr/local/cuda/lib64/libnvrtc.so",
                "/usr/local/cuda/lib/libnvrtc.so",
                "/usr/lib/x86_64-linux-gnu/libnvrtc.so",
                "/usr/lib/x86_64-linux-gnu/libnvrtc.so.12",
            };
            for (extra_paths) |path| {
                if (std.DynLib.open(path)) |lib_pkg| break :blk lib_pkg else |_| {}
            }
            break :blk error.NvrtcNotFound;
        };

        if (nvrtc_lib) |lib_pkg| {
            driver.nvrtc_handle = lib_pkg;
            driver.is_nvrtc_available = true;
        } else |_| {
            std.log.debug("NVRTC library not found. Falling back to pre-compiled PTX.", .{});
            driver.is_nvrtc_available = false;
        }

        // Load all function pointers
        try driver.loadFunction("cuInit", &driver.cuInit);
        try driver.loadFunction("cuDeviceGetCount", &driver.deviceGetCount);
        try driver.loadFunction("cuDeviceGet", &driver.deviceGet);
        try driver.loadFunction("cuDeviceGetAttribute", &driver.deviceGetAttribute);
        try driver.loadFunction("cuDeviceGetName", &driver.deviceGetName);
        try driver.loadFunction("cuDeviceTotalMem", &driver.deviceTotalMem);
        try driver.loadFunction("cuCtxCreate", &driver.ctxCreate);
        try driver.loadFunction("cuCtxDestroy", &driver.ctxDestroy);
        try driver.loadFunction("cuCtxPushCurrent", &driver.ctxPushCurrent);
        try driver.loadFunction("cuCtxPopCurrent", &driver.ctxPopCurrent);
        try driver.loadFunction("cuCtxSetCurrent", &driver.ctxSetCurrent);
        try driver.loadFunction("cuCtxGetCurrent", &driver.ctxGetCurrent);
        try driver.loadFunction("cuCtxSynchronize", &driver.ctxSynchronize);
        try driver.loadFunction("cuModuleLoadData", &driver.moduleLoadData);
        try driver.loadFunction("cuModuleUnload", &driver.moduleUnload);
        try driver.loadFunction("cuModuleGetFunction", &driver.moduleGetFunction);
        try driver.loadFunction("cuMemAlloc", &driver.memAlloc);
        try driver.loadFunction("cuMemFree", &driver.memFree);
        try driver.loadFunction("cuMemAllocManaged", &driver.memAllocManaged);
        try driver.loadFunction("cuMemAllocHost", &driver.memAllocHost);
        try driver.loadFunction("cuMemFreeHost", &driver.memFreeHost);
        try driver.loadFunction("cuMemcpyHtoD", &driver.memcpyHtoD);
        try driver.loadFunction("cuMemcpyDtoH", &driver.memcpyDtoH);
        try driver.loadFunction("cuMemcpyDtoD", &driver.memcpyDtoD);
        try driver.loadFunction("cuMemcpyHtoDAsync", &driver.memcpyHtoDAsync);
        try driver.loadFunction("cuMemcpyDtoHAsync", &driver.memcpyDtoHAsync);
        try driver.loadFunction("cuMemcpyDtoDAsync", &driver.memcpyDtoDAsync);
        try driver.loadFunction("cuLaunchKernel", &driver.launchKernel);
        try driver.loadFunction("cuStreamCreate", &driver.streamCreate);
        try driver.loadFunction("cuStreamDestroy", &driver.streamDestroy);
        try driver.loadFunction("cuStreamSynchronize", &driver.streamSynchronize);
        try driver.loadFunction("cuStreamWaitEvent", &driver.streamWaitEvent);
        try driver.loadFunction("cuEventCreate", &driver.eventCreate);
        try driver.loadFunction("cuEventDestroy", &driver.eventDestroy);
        try driver.loadFunction("cuEventRecord", &driver.eventRecord);
        try driver.loadFunction("cuEventSynchronize", &driver.eventSynchronize);
        try driver.loadFunction("cuEventElapsedTime", &driver.eventElapsedTime);
        try driver.loadFunction("cuEventQuery", &driver.eventQuery);
        try driver.loadFunction("cuStreamQuery", &driver.streamQuery);
        try driver.loadFunction("cuMemsetD32", &driver.memsetD32);
        try driver.loadFunction("cuMemsetD32Async", &driver.memsetD32Async);
        try driver.loadFunction("cuMemHostRegister", &driver.memHostRegister);
        try driver.loadFunction("cuMemHostUnregister", &driver.memHostUnregister);
        // Unified memory functions (may be null on older GPUs)
        driver.loadFunction("cuMemPrefetchAsync", &driver.memPrefetchAsync) catch { driver.memPrefetchAsync = null; };
        driver.loadFunction("cuMemAdvise", &driver.memAdvise) catch { driver.memAdvise = null; };
        try driver.loadFunction("cuPointerGetAttribute", &driver.pointerGetAttribute);
        try driver.loadFunction("cuLinkCreate", &driver.linkCreate);
        try driver.loadFunction("cuLinkAddData", &driver.linkAddData);
        try driver.loadFunction("cuLinkComplete", &driver.linkComplete);
        try driver.loadFunction("cuLinkDestroy", &driver.linkDestroy);
        try driver.loadFunction("cuOccupancyMaxPotentialBlockSize", &driver.occupancyMaxPotentialBlockSize);
        try driver.loadFunction("cuGetErrorString", &driver.cuGetErrorString);
        try driver.loadFunction("cuGetErrorName", &driver.cuGetErrorName);
        try driver.loadFunction("cuMemPoolCreate", &driver.memPoolCreate);
        try driver.loadFunction("cuMemPoolDestroy", &driver.memPoolDestroy);
        try driver.loadFunction("cuMemAllocFromPoolAsync", &driver.memAllocFromPoolAsync);
        try driver.loadFunction("cuMemPoolTrimTo", &driver.memPoolTrimTo);

        // Load NVRTC functions if available
        if (driver.is_nvrtc_available) {
            driver.loadNvrtcFunction("nvrtcVersion", &driver.nvrtcVersion) catch {
                driver.is_nvrtc_available = false;
            };
            driver.loadNvrtcFunction("nvrtcGetErrorString", &driver.nvrtcGetErrorString) catch {
                driver.is_nvrtc_available = false;
            };
            driver.loadNvrtcFunction("nvrtcCreateProgram", &driver.nvrtcCreateProgram) catch {
                driver.is_nvrtc_available = false;
            };
            driver.loadNvrtcFunction("nvrtcDestroyProgram", &driver.nvrtcDestroyProgram) catch {
                driver.is_nvrtc_available = false;
            };
            driver.loadNvrtcFunction("nvrtcCompileProgram", &driver.nvrtcCompileProgram) catch {
                driver.is_nvrtc_available = false;
            };
            driver.loadNvrtcFunction("nvrtcGetPTXSize", &driver.nvrtcGetPTXSize) catch {
                driver.is_nvrtc_available = false;
            };
            driver.loadNvrtcFunction("nvrtcGetPTX", &driver.nvrtcGetPTX) catch {
                driver.is_nvrtc_available = false;
            };
            driver.loadNvrtcFunction("nvrtcGetProgramLogSize", &driver.nvrtcGetProgramLogSize) catch {
                driver.is_nvrtc_available = false;
            };
            driver.loadNvrtcFunction("nvrtcGetProgramLog", &driver.nvrtcGetProgramLog) catch {
                driver.is_nvrtc_available = false;
            };
        }

        // Initialize the CUDA driver
        const result = driver.cuInit.?(0);
        if (result.isError()) {
            return error.CudaInitFailed;
        }

        driver.is_initialized = true;
        return driver;
    }

    fn loadFunction(self: *CudaDriver, name: [:0]const u8, ptr: anytype) !void {
        const func = self.handle.?.lookup(*const anyopaque, @ptrCast(name)) orelse {
            std.log.debug("Failed to find CUDA function: {s}", .{name});
            return error.CudaFunctionNotFound;
        };
        // Cast to the appropriate function pointer type
        ptr.* = @ptrCast(func);
    }

    fn loadNvrtcFunction(self: *CudaDriver, name: [:0]const u8, ptr: anytype) !void {
        if (self.nvrtc_handle) |h| {
            var h_mut = h;
            const func = h_mut.lookup(*const anyopaque, @ptrCast(name)) orelse {
                std.log.debug("Failed to find NVRTC function: {s}", .{name});
                return error.CudaFunctionNotFound;
            };
            ptr.* = @ptrCast(func);
        } else {
            return error.CudaDriverNotFound;
        }
    }

    /// Deinitialize the CUDA driver
    pub fn deinit(self: *CudaDriver) void {
        if (self.handle) |h| {
            var h_mut = h;
            h_mut.close();
            self.handle = null;
        }
        if (self.nvrtc_handle) |h| {
            var h_mut = h;
            h_mut.close();
            self.nvrtc_handle = null;
        }
        self.is_initialized = false;
        self.is_nvrtc_available = false;
    }

    /// Check if CUDA driver is available and initialized
    pub fn isAvailable(self: *const CudaDriver) bool {
        return self.is_initialized;
    }

    /// Get error string from CUDA result
    pub fn getErrorString(self: *const CudaDriver, result: CUresult) []const u8 {
        if (self.cuGetErrorString) |get_err_str| {
            var str: [*c]const u8 = undefined;
            _ = get_err_str(result, &str);
            if (str != null) {
                return std.mem.span(str);
            }
        }
        return "Unknown error";
    }

    /// Get error name from CUDA result
    pub fn getErrorName(self: *const CudaDriver, result: CUresult) []const u8 {
        if (self.cuGetErrorName) |get_err_name| {
            var str: [*c]const u8 = undefined;
            _ = get_err_name(result, &str);
            if (str != null) {
                return std.mem.span(str);
            }
        }
        return "Unknown error";
    }
};

// =============================================================================
// Global CUDA Driver Instance (Thread-Safe Reference Counted Singleton)
// SECURITY FIX: CRIT-002 - Use-after-free in global driver access
// =============================================================================

var global_driver: ?CudaDriver = null;
var driver_mutex: std.atomic.Mutex = .unlocked;
var driver_ref_count: std.atomic.Value(usize) = std.atomic.Value(usize).init(0);
var is_driver_initialized: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);

/// Thread-safe reference to the global CUDA driver
/// Automatically manages reference counting to prevent use-after-free
pub const CudaDriverRef = struct {
    driver: *CudaDriver,

    /// Acquire a reference to the global CUDA driver
    /// Thread-safe and ensures the driver stays alive until release() is called
    pub fn acquire(allocator: std.mem.Allocator) !CudaDriverRef {
        // First, try to increment ref count - if driver is already initialized
        // this is the fast path (lock-free)
        if (is_driver_initialized.load(.acquire)) {
            const prev_count = driver_ref_count.fetchAdd(1, .acquire);
            if (prev_count > 0) {
                // Driver is initialized and we have a reference
                return CudaDriverRef{ .driver = &global_driver.? };
            }
            // Ref count was 0, driver might be shutting down, decrement and take slow path
            _ = driver_ref_count.fetchSub(1, .release);
        }

        // Slow path: need to initialize or wait for initialization
        while (!driver_mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer driver_mutex.unlock();

        // Double-check after acquiring lock
        if (!is_driver_initialized.load(.acquire)) {
            // Initialize the driver
            global_driver = try CudaDriver.init(allocator);
            is_driver_initialized.store(true, .release);
        }

        // Increment ref count under mutex
        _ = driver_ref_count.fetchAdd(1, .release);

        return CudaDriverRef{ .driver = &global_driver.? };
    }

    /// Release this reference to the CUDA driver
    /// When the last reference is released, the driver is deinitialized
    pub fn release(self: CudaDriverRef) void {
        const prev_count = driver_ref_count.fetchSub(1, .acq_rel);

        if (prev_count == 1) {
            // We were the last reference, deinitialize the driver
            while (!driver_mutex.tryLock()) {
                std.atomic.spinLoopHint();
            }
            defer driver_mutex.unlock();

            // Double-check under mutex - another thread might have acquired
            if (driver_ref_count.load(.acquire) == 0 and is_driver_initialized.load(.acquire)) {
                self.driver.deinit();
                global_driver = null;
                is_driver_initialized.store(false, .release);
            }
        }
    }

    /// Check if the driver reference is still valid
    pub fn isValid(self: CudaDriverRef) bool {
        _ = self; // Parameter required for API consistency
        return is_driver_initialized.load(.acquire) and driver_ref_count.load(.acquire) > 0;
    }
};

/// Initialize the global CUDA driver (legacy API - use CudaDriverRef.acquire instead)
/// SECURITY: This is kept for backward compatibility but is inherently unsafe
/// because it doesn't provide lifetime management
pub fn initGlobalDriver(allocator: std.mem.Allocator) !void {
    _ = try CudaDriverRef.acquire(allocator);
}

/// Get the global CUDA driver instance (legacy API - unsafe)
/// SECURITY WARNING: Returns a raw pointer with no lifetime guarantees.
/// Use CudaDriverRef.acquire() instead for safe access.
pub fn getGlobalDriver() ?*CudaDriver {
    // Spin until we acquire the lock
    while (!driver_mutex.tryLock()) {
        std.atomic.spinLoopHint();
    }
    defer driver_mutex.unlock();
    return if (global_driver) |*d| d else null;
}

/// Deinitialize the global CUDA driver (legacy API)
/// SECURITY WARNING: This can cause use-after-free if other code still holds
/// references to the driver. Use CudaDriverRef.release() instead.
pub fn deinitGlobalDriver() void {
    // Spin until we acquire the lock
    while (!driver_mutex.tryLock()) {
        std.atomic.spinLoopHint();
    }
    defer driver_mutex.unlock();

    if (global_driver) |*d| {
        d.deinit();
        global_driver = null;
        is_driver_initialized.store(false, .release);
        driver_ref_count.store(0, .release);
    }
}

// =============================================================================
// Error Handling Helper
// =============================================================================

pub const CudaError = error{
    CudaDriverNotFound,
    CudaFunctionNotFound,
    CudaInitFailed,
    CudaOutOfMemory,
    CudaInvalidValue,
    CudaInvalidDevice,
    CudaInvalidContext,
    CudaInvalidHandle,
    CudaNotInitialized,
    CudaDeinitialized,
    CudaNoDevice,
    CudaDeviceUnavailable,
    CudaUnknownError,
    UnsupportedPlatform,
    NvrtcNotAvailable,
    NvrtcProgramCreationFailed,
    NvrtcCompilationFailed,
    InvalidPtx,
    UnsupportedPtxVersion,
    NoBinaryForGpu,
    ContextNotInitialized,
    StreamNotInitialized,
};

/// Convert CUDA result to Zig error
pub fn checkCuda(result: CUresult) CudaError!void {
    switch (result) {
        .SUCCESS => return,
        .ERROR_OUT_OF_MEMORY => return error.CudaOutOfMemory,
        .ERROR_INVALID_VALUE => return error.CudaInvalidValue,
        .ERROR_INVALID_DEVICE => return error.CudaInvalidDevice,
        .ERROR_INVALID_CONTEXT => return error.CudaInvalidContext,
        .ERROR_INVALID_HANDLE => return error.CudaInvalidHandle,
        .ERROR_NOT_INITIALIZED => return error.CudaNotInitialized,
        .ERROR_DEINITIALIZED => return error.CudaDeinitialized,
        .ERROR_NO_DEVICE => return error.CudaNoDevice,
        .ERROR_DEVICE_UNAVAILABLE => return error.CudaDeviceUnavailable,
        else => return error.CudaUnknownError,
    }
}

// =============================================================================
// Multi-GPU Device Enumeration
// =============================================================================

/// Device information structure for multi-GPU support
pub const CudaDeviceInfo = struct {
    device_id: c_int,
    name: [256]u8,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    total_memory: usize,
    multiprocessor_count: i32,
    max_threads_per_block: i32,
    warp_size: i32,
    memory_clock_rate: i32,
    global_memory_bus_width: i32,
    l2_cache_size: i32,
    max_threads_per_multiprocessor: i32,
    unified_addressing: i32,
    managed_memory: i32,
    concurrent_managed_access: i32,
    pci_bus_id: i32,
    pci_device_id: i32,

    /// Get compute capability as single number (e.g., 70 for 7.0)
    pub fn computeCapability(self: CudaDeviceInfo) i32 {
        return self.compute_capability_major * 10 + self.compute_capability_minor;
    }

    /// Check if device has Tensor Cores
    pub fn hasTensorCores(self: CudaDeviceInfo) bool {
        return self.compute_capability_major >= 7;
    }

    /// Check if device supports unified memory
    pub fn hasUnifiedMemory(self: CudaDeviceInfo) bool {
        return self.unified_addressing != 0 and self.managed_memory != 0;
    }

    /// Get device name as a slice
    pub fn getName(self: CudaDeviceInfo) []const u8 {
        return std.mem.sliceTo(&self.name, 0);
    }
};

/// Get the number of available CUDA devices
/// Returns 0 if CUDA is not available or no devices found
pub fn getDeviceCount() c_int {
    if (@import("builtin").os.tag == .macos) {
        return 0;
    }

    // Try to acquire driver reference temporarily
    var driver = CudaDriver.init(std.heap.page_allocator) catch {
        return 0;
    };
    defer driver.deinit();

    var count: c_int = 0;
    if (driver.deviceGetCount) |get_count| {
        const result = get_count(&count);
        if (result.isSuccess()) {
            return count;
        }
    }
    return 0;
}

/// Query device information for a specific device
/// device_id: Device index (0 to device_count - 1)
/// Returns error if device_id is invalid or CUDA is not available
pub fn queryDeviceInfo(device_id: c_int) !CudaDeviceInfo {
    if (@import("builtin").os.tag == .macos) {
        return error.CudaNotAvailable;
    }

    // Initialize driver temporarily
    var driver = CudaDriver.init(std.heap.page_allocator) catch |err| {
        return if (err == error.CudaDriverNotFound) error.CudaNotAvailable else err;
    };
    defer driver.deinit();

    // Get device count to validate device_id
    var device_count: c_int = 0;
    if (driver.deviceGetCount) |get_count| {
        const result = get_count(&device_count);
        if (!result.isSuccess() or device_id < 0 or device_id >= device_count) {
            return error.InvalidDevice;
        }
    } else {
        return error.CudaFunctionNotFound;
    }

    // Get device handle
    var device: CUdevice = 0;
    if (driver.deviceGet) |device_get| {
        try checkCuda(device_get(&device, device_id));
    } else {
        return error.CudaFunctionNotFound;
    }

    // Query device properties
    var info: CudaDeviceInfo = undefined;
    info.device_id = device_id;

    // Get device name
    if (driver.deviceGetName) |get_name| {
        var name_buf: [256]u8 = undefined;
        try checkCuda(get_name(&name_buf, name_buf.len, device));
        info.name = name_buf;
    } else {
        info.name = std.mem.zeroes([256]u8);
    }

    // Get compute capability
    if (driver.deviceGetAttribute) |get_attr| {
        try checkCuda(get_attr(&info.compute_capability_major, .COMPUTE_CAPABILITY_MAJOR, device));
        try checkCuda(get_attr(&info.compute_capability_minor, .COMPUTE_CAPABILITY_MINOR, device));
        try checkCuda(get_attr(&info.multiprocessor_count, .MULTIPROCESSOR_COUNT, device));
        try checkCuda(get_attr(&info.max_threads_per_block, .MAX_THREADS_PER_BLOCK, device));
        try checkCuda(get_attr(&info.warp_size, .WARP_SIZE, device));
        try checkCuda(get_attr(&info.memory_clock_rate, .MEMORY_CLOCK_RATE, device));
        try checkCuda(get_attr(&info.global_memory_bus_width, .GLOBAL_MEMORY_BUS_WIDTH, device));
        try checkCuda(get_attr(&info.l2_cache_size, .L2_CACHE_SIZE, device));
        try checkCuda(get_attr(&info.max_threads_per_multiprocessor, .MAX_THREADS_PER_MULTIPROCESSOR, device));
        try checkCuda(get_attr(&info.unified_addressing, .UNIFIED_ADDRESSING, device));
        try checkCuda(get_attr(&info.managed_memory, .MANAGED_MEMORY, device));
        try checkCuda(get_attr(&info.concurrent_managed_access, .CONCURRENT_MANAGED_ACCESS, device));
        try checkCuda(get_attr(&info.pci_bus_id, .PCI_BUS_ID, device));
        try checkCuda(get_attr(&info.pci_device_id, .PCI_DEVICE_ID, device));
    } else {
        return error.CudaFunctionNotFound;
    }

    // Get total memory
    if (driver.deviceTotalMem) |get_mem| {
        try checkCuda(get_mem(&info.total_memory, device));
    } else {
        info.total_memory = 0;
    }

    return info;
}

/// Enumerate all available CUDA devices
/// Returns an array of CudaDeviceInfo (caller must free with allocator)
pub fn enumerateDevices(allocator: std.mem.Allocator) ![]CudaDeviceInfo {
    const count = getDeviceCount();
    if (count == 0) {
        return allocator.alloc(CudaDeviceInfo, 0);
    }

    var devices = try allocator.alloc(CudaDeviceInfo, @intCast(count));
    errdefer allocator.free(devices);

    var i: c_int = 0;
    while (i < count) : (i += 1) {
        devices[@intCast(i)] = try queryDeviceInfo(i);
    }

    return devices;
}

/// Score a device for workload assignment
/// Higher score = better for computation
/// Considers: compute capability, memory size, multiprocessor count
pub fn scoreDeviceForWorkload(info: CudaDeviceInfo) i32 {
    var score: i32 = 0;

    // Compute capability is most important (features and performance)
    score += info.computeCapability() * 100;

    // Multiprocessor count indicates raw compute power
    score += info.multiprocessor_count * 10;

    // Memory size matters for large workloads (in GB)
    const memory_gb: i32 = @intCast(info.total_memory / (1024 * 1024 * 1024));
    score += memory_gb;

    return score;
}

/// Select the best device for general purpose computation
/// Returns the device index with the highest score
pub fn selectBestDeviceIndex() c_int {
    const count = getDeviceCount();
    if (count == 0) return -1;
    if (count == 1) return 0;

    var best_device: c_int = 0;
    var best_score: i32 = -1;

    var i: c_int = 0;
    while (i < count) : (i += 1) {
        const info = queryDeviceInfo(i) catch continue;
        const score = scoreDeviceForWorkload(info);
        if (score > best_score) {
            best_score = score;
            best_device = i;
        }
    }

    return best_device;
}

/// Get a list of devices sorted by suitability for workload
/// Most suitable devices come first
/// Returns indices of devices (caller must free result with allocator)
pub fn getDevicesBySuitability(allocator: std.mem.Allocator) ![]c_int {
    const count = getDeviceCount();
    if (count == 0) {
        return allocator.alloc(c_int, 0);
    }

    // Create array of (device_id, score) pairs
    const DeviceScore = struct {
        device_id: c_int,
        score: i32,
    };

    var scores = try allocator.alloc(DeviceScore, @intCast(count));
    defer allocator.free(scores);

    var i: c_int = 0;
    while (i < count) : (i += 1) {
        const info = queryDeviceInfo(i) catch continue;
        scores[@intCast(i)] = .{
            .device_id = i,
            .score = scoreDeviceForWorkload(info),
        };
    }

    // Sort by score (descending)
    std.mem.sort(DeviceScore, scores, {}, struct {
        fn lessThan(_: void, a: DeviceScore, b: DeviceScore) bool {
            return a.score > b.score; // Descending order
        }
    }.lessThan);

    // Extract device indices
    var result = try allocator.alloc(c_int, @intCast(count));
    for (scores, 0..) |s, idx| {
        result[idx] = s.device_id;
    }

    return result;
}

// =============================================================================
// Tests
// =============================================================================

test "CUDA driver availability" {
    // CUDA should not be available on macOS
    if (@import("builtin").os.tag == .macos) {
        // Skip test on macOS
        return error.SkipZigTest;
    }

    // Try to load the CUDA driver
    const driver = CudaDriver.init(std.testing.allocator) catch |err| {
        // It's OK if CUDA is not installed
        if (err == error.CudaDriverNotFound) {
            return;
        }
        return err;
    };
    defer driver.deinit();

    try std.testing.expect(driver.is_initialized);
}

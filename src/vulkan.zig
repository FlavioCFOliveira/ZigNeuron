/// Vulkan GPU backend implementation for ZigNeuron
/// Provides Vulkan compute shader support for cross-platform GPU execution
/// Note: This is a stub implementation - Vulkan runtime requires external libraries
const std = @import("std");

// Vulkan constants
pub const VK_API_VERSION_1_1 = 0x400001;
pub const VK_SUCCESS = 0;
pub const VK_ERROR_INITIALIZATION_FAILED = -3;
pub const VK_ERROR_DEVICE_LOST = -7;
pub const VK_ERROR_MEMORY_MAP_FAILED = -10;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x0001;
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x0002;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x0004;
pub const VK_SHARING_MODE_EXCLUSIVE = 0;
pub const VK_QUEUE_COMPUTE_BIT = 0x08;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x01;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x02;
pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x01;
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x01;
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 3;
pub const VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT = 0x01;
pub const VK_SHADER_STAGE_COMPUTE_BIT = 0x20;
pub const VK_NULL_HANDLE = 0;

// Maximum number of descriptors in pool
const MAX_DESCRIPTORS = 16;

// Vulkan opaque types (these would be the real Vulkan types with FFI)
// Vulkan opaque types (these would be the real Vulkan types with FFI)
pub const Instance = opaque {};
pub const PhysicalDevice = opaque {};
pub const Device = opaque {};
pub const Queue = opaque {};
pub const CommandPool = opaque {};
pub const CommandBuffer = opaque {};
pub const Buffer = opaque {};
pub const DeviceMemory = opaque {};
pub const DescriptorSetLayout = opaque {};
pub const DescriptorPool = opaque {};
pub const DescriptorSet = opaque {};
pub const PipelineLayout = opaque {};
pub const Pipeline = opaque {};
pub const ShaderModule = opaque {};

// Wrapper structs for Vulkan types (needed because opaques can't be in error unions)
pub const DescriptorSetLayoutWrapper = struct {
    pub fn deinit(self: *const DescriptorSetLayoutWrapper, device: anytype) void {
        _ = self;
        _ = device;
    }
};
pub const ShaderModuleWrapper = struct {
    pub fn deinit(self: *const ShaderModuleWrapper, device: anytype) void {
        _ = self;
        _ = device;
    }
};
pub const PipelineLayoutWrapper = struct {
    pub fn deinit(self: *const PipelineLayoutWrapper, device: anytype) void {
        _ = self;
        _ = device;
    }
};
pub const PipelineWrapper = struct {
    pub fn deinit(self: *const PipelineWrapper, device: anytype) void {
        _ = self;
        _ = device;
    }
};
pub const DescriptorPoolWrapper = struct {
    pub fn deinit(self: *const DescriptorPoolWrapper, device: anytype) void {
        _ = self;
        _ = device;
    }
};
pub const DescriptorSetWrapper = struct {
    pub fn deinit(self: *const DescriptorSetWrapper, device: anytype) void {
        _ = self;
        _ = device;
    }
};

// Activation type constants
pub const ActivationType = enum(u32) {
    relu = 0,
    sigmoid = 1,
    tanh = 2,
};

// Loss type constants
pub const LossType = enum(u32) {
    mse = 0,
    cross_entropy = 1,
    binary_cross_entropy = 2,
};

// Buffer wrapper (Zig wrapper around Buffer opaque)
pub const BufferWrapper = struct {
    size: usize,
    usage: u32,

    pub fn init(device: *anyopaque, size: usize, usage: u32) !BufferWrapper {
        _ = device;
        return BufferWrapper{ .size = size, .usage = usage };
    }

    pub fn map(self: *const BufferWrapper) !void {
        _ = self;
        return error.NotAvailable;
    }

    pub fn unmap(self: *const BufferWrapper) void {
        _ = self;
    }

    pub fn deinit(self: *const BufferWrapper, device: anytype) void {
        _ = self;
        _ = device;
    }

    pub fn writeData(self: *const BufferWrapper, data: []const u8) !void {
        _ = self;
        _ = data;
        return error.NotAvailable;
    }
};

// Command pool wrapper
pub const CommandPoolWrapper = struct {
    queue_family_index: u32,

    pub fn init(device: *anyopaque, queue_family_index: u32) !CommandPoolWrapper {
        _ = device;
        return CommandPoolWrapper{ .queue_family_index = queue_family_index };
    }

    pub fn deinit(self: *CommandPoolWrapper, device: anytype) void {
        _ = self;
        _ = device;
    }

    pub fn allocateBuffer(self: *CommandPoolWrapper, device: anytype) !CommandBufferWrapper {
        _ = self;
        _ = device;
        return CommandBufferWrapper{};
    }
};

// Command buffer wrapper
pub const CommandBufferWrapper = struct {
    pub fn begin(self: *CommandBufferWrapper, one_time_submit: bool) !void {
        _ = self;
        _ = one_time_submit;
    }

    pub fn end(self: *CommandBufferWrapper) !void {
        _ = self;
    }

    pub fn beginSingleTime(self: *CommandBufferWrapper, device: *anyopaque) !void {
        _ = self;
        _ = device;
    }

    pub fn endSingleTime(self: *CommandBufferWrapper, queue: *anyopaque) !void {
        _ = self;
        _ = queue;
    }

    pub fn free(self: *CommandBufferWrapper, device: anytype) void {
        _ = self;
        _ = device;
    }
};

// Vulkan device wrapper
pub const DeviceWrapper = struct {
    instance: *anyopaque,
    physical_device: *anyopaque,
    device: *anyopaque,
    queue_family_index: u32,
    queue: *anyopaque,

    // Cache for frequently used objects
    descriptor_pool: ?*anyopaque = null,
    descriptor_set_layout: ?*anyopaque = null,

    pub fn init() !DeviceWrapper {
        // Vulkan requires external libraries at runtime
        // Return error to indicate Vulkan is not available
        return error.VulkanNotAvailable;
    }

    pub fn createCommandPool(self: *const DeviceWrapper) !CommandPoolWrapper {
        return CommandPoolWrapper.init(self.device, self.queue_family_index);
    }

    pub fn createBuffer(self: *const DeviceWrapper, size: usize, usage: u32) !BufferWrapper {
        return BufferWrapper.init(self.device, size, usage);
    }

    pub fn createDescriptorSetLayout(self: *const DeviceWrapper, binding_count: u32) !DescriptorSetLayoutWrapper {
        _ = self;
        _ = binding_count;
        return DescriptorSetLayoutWrapper{};
    }

    pub fn createDescriptorPool(self: *const DeviceWrapper, max_sets: u32, binding_count: u32) !DescriptorPoolWrapper {
        _ = self;
        _ = max_sets;
        _ = binding_count;
        return DescriptorPoolWrapper{};
    }

    pub fn createShaderModule(self: *const DeviceWrapper, source: []const u8) !ShaderModuleWrapper {
        _ = self;
        _ = source;
        return ShaderModuleWrapper{};
    }

    pub fn createPipelineLayout(self: *const DeviceWrapper, descriptor_set_layout: DescriptorSetLayoutWrapper) !PipelineLayoutWrapper {
        _ = self;
        _ = descriptor_set_layout;
        return PipelineLayoutWrapper{};
    }

    pub fn createPipeline(
        self: *const DeviceWrapper,
        layout: PipelineLayoutWrapper,
        shader_module: ShaderModuleWrapper,
        entry_point: []const u8,
        work_group_size: [3]u32,
    ) !PipelineWrapper {
        _ = self;
        _ = layout;
        _ = shader_module;
        _ = entry_point;
        _ = work_group_size;
        return PipelineWrapper{};
    }

    pub fn deinit(self: *const DeviceWrapper) void {
        _ = self;
    }

    pub fn createDescriptorSet(
        self: *const DeviceWrapper,
        pool: *DescriptorPoolWrapper,
        layout: DescriptorSetLayoutWrapper,
        buffers: []const BufferWrapper,
    ) !DescriptorSetWrapper {
        _ = self;
        _ = pool;
        _ = layout;
        _ = buffers;
        return DescriptorSetWrapper{};
    }
};

// Shader modules - SPIR-V bytecode
// Generated from GLSL shaders using glslc

/// SPIR-V for matmul compute shader
/// Workgroup size: [16, 16, 1]
/// Input: A (m×k), B (k×n), Output: C (m×n)
pub const matmul_spv: []const u8 = @embedFile("shaders/matmul.comp.spv");

/// SPIR-V for activation forward (ReLU) compute shader
pub const activation_forward_spv: []const u8 = @embedFile("shaders/activation_forward.comp.spv");

/// SPIR-V for activation backward (ReLU derivative) compute shader
pub const activation_backward_spv: []const u8 = @embedFile("shaders/activation_backward.comp.spv");

/// SPIR-V for loss backward (MSE gradient) compute shader
pub const loss_backward_spv: []const u8 = @embedFile("shaders/loss_backward.comp.spv");

// Vulkan backend implementation
const MAX_WORKGROUP_SIZE = 256;

/// Execute matrix multiplication using Vulkan compute shaders
pub fn vulkanMatMul(
    device: *const DeviceWrapper,
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
) !void {
    // Validate inputs
    if (a.len < m * k) return error.BufferTooSmall;
    if (b.len < k * n) return error.BufferTooSmall;
    if (c.len < m * n) return error.BufferTooSmall;

    // For small matrices, CPU is often faster due to overhead
    const total_size = @as(usize, m) * n * k;
    if (total_size < 4096) {
        cpuMatMul(a, b, c, m, n, k);
        return;
    }

    // Use the device parameter to suppress unused warning
    _ = device;

    // Vulkan not available at runtime - return error
    return error.VulkanNotAvailable;
}

/// Execute activation forward using Vulkan
pub fn vulkanActivationForward(
    device: *const DeviceWrapper,
    activation: anytype,
    input: []f32,
    output: []f32,
) !void {
    if (input.len != output.len) return error.ShapeMismatch;

    // For small arrays, CPU is faster
    if (input.len < 256) {
        cpuActivationForward(activation, input, output);
        return;
    }

    _ = device;

    // Vulkan not available at runtime
    return error.VulkanNotAvailable;
}

/// Execute activation backward using Vulkan
pub fn vulkanActivationBackward(
    device: *const DeviceWrapper,
    activation: anytype,
    input: []const f32,
    grad_output: []const f32,
    grad_input: []f32,
) !void {
    if (input.len != grad_output.len or input.len != grad_input.len) {
        return error.ShapeMismatch;
    }

    if (input.len < 256) {
        cpuActivationBackward(activation, input, grad_output, grad_input);
        return;
    }

    _ = device;

    // Vulkan not available at runtime
    return error.VulkanNotAvailable;
}

/// Execute loss backward using Vulkan
pub fn vulkanLossBackward(
    device: *const DeviceWrapper,
    loss_fn: anytype,
    output: []const f32,
    target: []const f32,
    grad_output: []f32,
) !void {
    if (output.len != target.len or output.len != grad_output.len) {
        return error.ShapeMismatch;
    }

    if (output.len < 256) {
        cpuLossBackward(loss_fn, output, target, grad_output);
        return;
    }

    _ = device;

    // Vulkan not available at runtime
    return error.VulkanNotAvailable;
}

// CPU fallback implementations (public for use by benchmark code)
pub fn cpuMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
    // Standard matrix multiplication: C = A * B
    // A is m×k, B is k×n, C is m×n
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |p| {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

pub fn cpuActivationForward(act: anytype, input: []f32, output: []f32) void {
    for (0..input.len) |i| {
        output[i] = act.forward(input[i]);
    }
}

pub fn cpuActivationBackward(act: anytype, input: []const f32, grad_output: []const f32, grad_input: []f32) void {
    for (0..input.len) |i| {
        grad_input[i] = act.backward(input[i], grad_output[i]);
    }
}

pub fn cpuLossBackward(loss_fn: anytype, output: []const f32, target: []const f32, grad_output: []f32) void {
    // Handle cross_entropy_logits specially as it needs the whole vector at once
    if (loss_fn == .cross_entropy_logits) {
        // Gradient: softmax(logits) - target
        var max_logit: f32 = output[0];
        for (output[1..]) |o| {
            if (o > max_logit) max_logit = o;
        }
        
        var sum_exp: f32 = 0;
        for (output) |o| {
            sum_exp += std.math.exp(o - max_logit);
        }
        
        for (0..output.len) |i| {
            const prob = std.math.exp(output[i] - max_logit) / sum_exp;
            grad_output[i] = prob - target[i];
        }
        return;
    }
    
    // Element-wise losses
    for (0..output.len) |i| {
        switch (loss_fn) {
            .mse => |loss| {
                _ = loss;
                grad_output[i] = 2 * (output[i] - target[i]);
            },
            .cross_entropy => |loss| {
                _ = loss;
                const eps: f32 = 1e-8;
                var p = output[i];
                if (p < eps) p = eps;
                if (p > 1 - eps) p = 1 - eps;
                grad_output[i] = -target[i] / p;
            },
            .cross_entropy_logits => unreachable,  // Handled above
            .binary_cross_entropy => |loss| {
                _ = loss;
                const eps: f32 = 1e-8;
                var p = output[i];
                if (p < eps) p = eps;
                if (p > 1 - eps) p = 1 - eps;
                grad_output[i] = (p - target[i]) / (p * (1 - p));
            },
        }
    }
}

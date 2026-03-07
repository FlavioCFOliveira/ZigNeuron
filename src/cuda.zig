/// CUDA GPU backend implementation for ZigNeuron
/// Provides CUDA compute kernel support for NVIDIA GPUs on Linux/Windows
///
/// Architecture:
/// - CUDA driver API dynamic loading (libcuda.so / nvcuda.dll)
/// - Context and stream management
/// - Memory pooling for efficient buffer reuse
/// - Kernel loading from PTX or inline CUDA-C
/// - Async execution with streams
/// - Unified memory support where available
const std = @import("std");
const cuda_driver = @import("cuda_driver.zig");
const cuda_context = @import("cuda_context.zig");

const CUresult = cuda_driver.CUresult;
const CUdeviceptr = cuda_driver.CUdeviceptr;
const CudaDriver = cuda_driver.CudaDriver;
const CudaContext = cuda_context.CudaContext;

// Re-export CUDA types
pub const CudaError = cuda_driver.CudaError;
pub const CUdevice = cuda_driver.CUdevice;

// =============================================================================
// CudaBackend Structure
// =============================================================================

pub const CudaBackend = struct {
    allocator: std.mem.Allocator,
    driver: CudaDriver,
    context: *CudaContext,

    // Kernel PTX cache
    kernel_ptx: std.StringHashMap([]const u8),

    /// Initialize CUDA backend
    pub fn init(allocator: std.mem.Allocator) !CudaBackend {
        // Load CUDA driver
        var driver = try CudaDriver.init(allocator);
        errdefer driver.deinit();

        // Create CUDA context with best device
        const ctx = try CudaContext.init(allocator, &driver);
        errdefer ctx.deinit();

        var backend = CudaBackend{
            .allocator = allocator,
            .driver = driver,
            .context = ctx,
            .kernel_ptx = std.StringHashMap([]const u8).init(allocator),
        };

        // Load built-in kernels
        try backend.loadBuiltinKernels();

        return backend;
    }

    /// Cleanup CUDA backend
    pub fn deinit(self: *CudaBackend) void {
        // Free kernel PTX strings
        var ptx_iter = self.kernel_ptx.iterator();
        while (ptx_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.kernel_ptx.deinit();

        // Cleanup context
        self.context.deinit();

        // Cleanup driver
        self.driver.deinit();
    }

    /// Check if CUDA is available
    pub fn isAvailable() bool {
        if (@import("builtin").os.tag == .macos) {
            return false;
        }

        var driver = CudaDriver.init(std.heap.page_allocator) catch {
            return false;
        };
        defer driver.deinit();

        return driver.is_initialized;
    }

    /// Get number of CUDA devices
    pub fn getDeviceCount() i32 {
        if (@import("builtin").os.tag == .macos) {
            return 0;
        }

        var driver = CudaDriver.init(std.heap.page_allocator) catch {
            return 0;
        };
        defer driver.deinit();

        var count: c_int = 0;
        const result = driver.deviceGetCount.?(&count);
        if (result.isError()) {
            return 0;
        }
        return count;
    }

    /// Synchronize the CUDA stream
    pub fn synchronize(self: *CudaBackend) !void {
        try self.context.synchronize();
    }

    // =============================================================================
    // Memory Operations
    // =============================================================================

    /// Allocate device buffer
    pub fn allocBuffer(self: *CudaBackend, size: usize) !DeviceBuffer {
        const buf = try self.context.getBuffer(size);
        return DeviceBuffer{
            .ptr = buf.ptr,
            .size = buf.size,
            .context = self.context,
        };
    }

    /// Free device buffer
    pub fn freeBuffer(self: *CudaBackend, buffer: DeviceBuffer) void {
        var buf = CudaContext.DeviceBuffer{
            .ptr = buffer.ptr,
            .size = buffer.size,
            .pool_index = null,
        };
        self.context.returnBuffer(buf);
    }

    /// Upload data to device
    pub fn upload(self: *CudaBackend, dst: CUdeviceptr, src: []const f32) !void {
        try self.context.upload(dst, std.mem.sliceAsBytes(src));
    }

    /// Upload data asynchronously
    pub fn uploadAsync(self: *CudaBackend, dst: CUdeviceptr, src: []const f32) !void {
        try self.context.uploadAsync(dst, std.mem.sliceAsBytes(src));
    }

    /// Download data from device
    pub fn download(self: *CudaBackend, dst: []f32, src: CUdeviceptr) !void {
        try self.context.download(std.mem.sliceAsBytes(dst), src);
    }

    /// Download data asynchronously
    pub fn downloadAsync(self: *CudaBackend, dst: []f32, src: CUdeviceptr) !void {
        try self.context.downloadAsync(std.mem.sliceAsBytes(dst), src);
    }

    // =============================================================================
    // Core Neural Network Operations
    // =============================================================================

    /// Matrix multiplication: C = A * B + (accumulate ? C : 0)
    pub fn matMul(
        self: *CudaBackend,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: usize,
        n: usize,
        k: usize,
        transpose_a: bool,
        transpose_b: bool,
        accumulate: bool,
    ) !void {
        // Allocate device buffers
        const size_a = m * k * @sizeOf(f32);
        const size_b = k * n * @sizeOf(f32);
        const size_c = m * n * @sizeOf(f32);

        var d_a = try self.context.getBuffer(size_a);
        defer self.context.returnBuffer(d_a);
        var d_b = try self.context.getBuffer(size_b);
        defer self.context.returnBuffer(d_b);
        var d_c = try self.context.getBuffer(size_c);
        defer self.context.returnBuffer(d_c);

        // Upload data
        try self.context.upload(d_a.ptr, std.mem.sliceAsBytes(a));
        try self.context.upload(d_b.ptr, std.mem.sliceAsBytes(b));
        if (accumulate) {
            try self.context.upload(d_c.ptr, std.mem.sliceAsBytes(c));
        }

        // Determine kernel name based on transpose flags
        const kernel_name = if (transpose_a)
            "matmul_transpose_a"
        else if (transpose_b)
            "matmul_transpose_b"
        else
            "matmul";

        // Get config
        const config = self.context.getMatrixConfig(m, n);

        // Launch kernel
        const m_u32: u32 = @intCast(m);
        const n_u32: u32 = @intCast(n);
        const k_u32: u32 = @intCast(k);
        const acc_u32: u32 = @intFromBool(accumulate);

        const args = [_]?*anyopaque{
            @ptrCast(&d_a.ptr),
            @ptrCast(&d_b.ptr),
            @ptrCast(&d_c.ptr),
            @ptrCast(&m_u32),
            @ptrCast(&n_u32),
            @ptrCast(&k_u32),
            @ptrCast(&acc_u32),
        };

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid_x, config.grid_y, 1 },
            .{ config.block_x, config.block_y, 1 },
            0,
            &args,
        );

        // Download result
        try self.context.download(std.mem.sliceAsBytes(c), d_c.ptr);
    }

    /// Batched matrix multiplication
    pub fn matMulBatch(
        self: *CudaBackend,
        a: []const f32,
        b: []const f32,
        c: []f32,
        batch_size: usize,
        n: usize,
        k: usize,
        accumulate: bool,
    ) !void {
        const size_per_batch = n * k * @sizeOf(f32);
        const total_size_a = batch_size * k * @sizeOf(f32); // a is batch_size x k
        const total_size_b = batch_size * n * k * @sizeOf(f32);
        const total_size_c = batch_size * n * @sizeOf(f32);

        var d_a = try self.context.getBuffer(total_size_a);
        defer self.context.returnBuffer(d_a);
        var d_b = try self.context.getBuffer(total_size_b);
        defer self.context.returnBuffer(d_b);
        var d_c = try self.context.getBuffer(total_size_c);
        defer self.context.returnBuffer(d_c);

        try self.context.upload(d_a.ptr, std.mem.sliceAsBytes(a));
        try self.context.upload(d_b.ptr, std.mem.sliceAsBytes(b));
        if (accumulate) {
            try self.context.upload(d_c.ptr, std.mem.sliceAsBytes(c));
        }

        const bs_u32: u32 = @intCast(batch_size);
        const n_u32: u32 = @intCast(n);
        const k_u32: u32 = @intCast(k);
        const acc_u32: u32 = @intFromBool(accumulate);

        const args = [_]?*anyopaque{
            @ptrCast(&d_a.ptr),
            @ptrCast(&d_b.ptr),
            @ptrCast(&d_c.ptr),
            @ptrCast(&bs_u32),
            @ptrCast(&n_u32),
            @ptrCast(&k_u32),
            @ptrCast(&acc_u32),
        };

        const config = self.context.getElementWiseConfig(batch_size * n);

        try self.context.launchKernel(
            "matmul_batch",
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(c), d_c.ptr);
    }

    /// Element-wise operations
    pub fn elementWiseOp(
        self: *CudaBackend,
        op: ElementWiseOp,
        a: []const f32,
        b: []const f32,
        c: []f32,
    ) !void {
        const size = a.len * @sizeOf(f32);

        var d_a = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_a);
        var d_b = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_b);
        var d_c = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_c);

        try self.context.upload(d_a.ptr, std.mem.sliceAsBytes(a));
        try self.context.upload(d_b.ptr, std.mem.sliceAsBytes(b));

        const kernel_name = switch (op) {
            .add => "ew_add",
            .sub => "ew_sub",
            .mul => "ew_mul",
            .div => "ew_div",
        };

        const len_u32: u32 = @intCast(a.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_a.ptr),
            @ptrCast(&d_b.ptr),
            @ptrCast(&d_c.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(a.len);

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(c), d_c.ptr);
    }

    /// Scalar multiplication
    pub fn scale(self: *CudaBackend, a: []const f32, scalar: f32, c: []f32) !void {
        const size = a.len * @sizeOf(f32);

        var d_a = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_a);
        var d_c = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_c);

        try self.context.upload(d_a.ptr, std.mem.sliceAsBytes(a));

        const len_u32: u32 = @intCast(a.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_a.ptr),
            @ptrCast(&scalar),
            @ptrCast(&d_c.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(a.len);

        try self.context.launchKernel(
            "scale_buffer",
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(c), d_c.ptr);
    }

    /// Map operations (exp, log, sqrt, etc.)
    pub fn mapOp(
        self: *CudaBackend,
        op: MapOp,
        input: []const f32,
        output: []f32,
    ) !void {
        const size = input.len * @sizeOf(f32);

        var d_input = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

        const kernel_name = switch (op) {
            .exp => "map_exp",
            .log => "map_log",
            .sqrt => "map_sqrt",
            .abs => "map_abs",
            .square => "map_square",
            .inv => "map_inv",
        };

        const len_u32: u32 = @intCast(input.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(input.len);

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
    }

    // =============================================================================
    // Activation Functions
    // =============================================================================

    /// ReLU forward: output = max(0, input)
    pub fn reluForward(self: *CudaBackend, input: []const f32, output: []f32) !void {
        try self.activationForward("relu_forward", input, output);
    }

    /// ReLU backward
    pub fn reluBackward(self: *CudaBackend, output: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        try self.activationBackward("relu_backward", output, grad_output, grad_input);
    }

    /// Sigmoid forward: output = 1 / (1 + exp(-input))
    pub fn sigmoidForward(self: *CudaBackend, input: []const f32, output: []f32) !void {
        try self.activationForward("sigmoid_forward", input, output);
    }

    /// Sigmoid backward
    pub fn sigmoidBackward(self: *CudaBackend, output: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        try self.activationBackward("sigmoid_backward", output, grad_output, grad_input);
    }

    /// Tanh forward
    pub fn tanhForward(self: *CudaBackend, input: []const f32, output: []f32) !void {
        try self.activationForward("tanh_forward", input, output);
    }

    /// Tanh backward
    pub fn tanhBackward(self: *CudaBackend, output: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        try self.activationBackward("tanh_backward", output, grad_output, grad_input);
    }

    /// Softmax forward
    pub fn softmaxForward(self: *CudaBackend, input: []const f32, output: []f32, batch_size: usize, features: usize) !void {
        const size = input.len * @sizeOf(f32);

        var d_input = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

        const batch_u32: u32 = @intCast(batch_size);
        const feat_u32: u32 = @intCast(features);
        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&batch_u32),
            @ptrCast(&feat_u32),
        };

        const config = self.context.getElementWiseConfig(batch_size);

        try self.context.launchKernel(
            "softmax_forward",
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
    }

    fn activationForward(self: *CudaBackend, kernel_name: []const u8, input: []const f32, output: []f32) !void {
        const size = input.len * @sizeOf(f32);

        var d_input = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_input);
        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);

        try self.context.upload(d_input.ptr, std.mem.sliceAsBytes(input));

        const len_u32: u32 = @intCast(input.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_input.ptr),
            @ptrCast(&d_output.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(input.len);

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(output), d_output.ptr);
    }

    fn activationBackward(self: *CudaBackend, kernel_name: []const u8, output: []const f32, grad_output: []const f32, grad_input: []f32) !void {
        const size = output.len * @sizeOf(f32);

        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);
        var d_grad_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_grad_output);
        var d_grad_input = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_grad_input);

        try self.context.upload(d_output.ptr, std.mem.sliceAsBytes(output));
        try self.context.upload(d_grad_output.ptr, std.mem.sliceAsBytes(grad_output));

        const len_u32: u32 = @intCast(output.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_grad_output.ptr),
            @ptrCast(&d_grad_input.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(output.len);

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(grad_input), d_grad_input.ptr);
    }

    // =============================================================================
    // Loss Functions
    // =============================================================================

    /// MSE loss backward
    pub fn mseBackward(self: *CudaBackend, output: []const f32, target: []const f32, grad_output: []f32) !void {
        try self.lossBackward("mse_backward", output, target, grad_output);
    }

    /// Cross-entropy loss backward
    pub fn crossEntropyBackward(self: *CudaBackend, output: []const f32, target: []const f32, grad_output: []f32) !void {
        try self.lossBackward("cross_entropy_backward", output, target, grad_output);
    }

    fn lossBackward(self: *CudaBackend, kernel_name: []const u8, output: []const f32, target: []const f32, grad_output: []f32) !void {
        const size = output.len * @sizeOf(f32);

        var d_output = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_output);
        var d_target = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_target);
        var d_grad = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_grad);

        try self.context.upload(d_output.ptr, std.mem.sliceAsBytes(output));
        try self.context.upload(d_target.ptr, std.mem.sliceAsBytes(target));

        const len_u32: u32 = @intCast(output.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_output.ptr),
            @ptrCast(&d_target.ptr),
            @ptrCast(&d_grad.ptr),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(output.len);

        try self.context.launchKernel(
            kernel_name,
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(grad_output), d_grad.ptr);
    }

    // =============================================================================
    // Optimizers
    // =============================================================================

    /// SGD update
    pub fn sgdUpdate(
        self: *CudaBackend,
        weights: []f32,
        gradients: []const f32,
        learning_rate: f32,
        weight_decay: f32,
    ) !void {
        const size = weights.len * @sizeOf(f32);

        var d_weights = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_weights);
        var d_gradients = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_gradients);

        try self.context.upload(d_weights.ptr, std.mem.sliceAsBytes(weights));
        try self.context.upload(d_gradients.ptr, std.mem.sliceAsBytes(gradients));

        const len_u32: u32 = @intCast(weights.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_weights.ptr),
            @ptrCast(&d_gradients.ptr),
            @ptrCast(&learning_rate),
            @ptrCast(&weight_decay),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(weights.len);

        try self.context.launchKernel(
            "sgd_update",
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(weights), d_weights.ptr);
    }

    /// Adam update
    pub fn adamUpdate(
        self: *CudaBackend,
        weights: []f32,
        gradients: []const f32,
        m: []f32,
        v: []f32,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        t: u32,
    ) !void {
        const size = weights.len * @sizeOf(f32);

        var d_weights = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_weights);
        var d_gradients = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_gradients);
        var d_m = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_m);
        var d_v = try self.context.getBuffer(size);
        defer self.context.returnBuffer(d_v);

        try self.context.upload(d_weights.ptr, std.mem.sliceAsBytes(weights));
        try self.context.upload(d_gradients.ptr, std.mem.sliceAsBytes(gradients));
        try self.context.upload(d_m.ptr, std.mem.sliceAsBytes(m));
        try self.context.upload(d_v.ptr, std.mem.sliceAsBytes(v));

        const len_u32: u32 = @intCast(weights.len);
        const args = [_]?*anyopaque{
            @ptrCast(&d_weights.ptr),
            @ptrCast(&d_gradients.ptr),
            @ptrCast(&d_m.ptr),
            @ptrCast(&d_v.ptr),
            @ptrCast(&learning_rate),
            @ptrCast(&beta1),
            @ptrCast(&beta2),
            @ptrCast(&epsilon),
            @ptrCast(&t),
            @ptrCast(&len_u32),
        };

        const config = self.context.getElementWiseConfig(weights.len);

        try self.context.launchKernel(
            "adam_update",
            .{ config.grid, 1, 1 },
            .{ config.block, 1, 1 },
            0,
            &args,
        );

        try self.context.download(std.mem.sliceAsBytes(weights), d_weights.ptr);
        try self.context.download(std.mem.sliceAsBytes(m), d_m.ptr);
        try self.context.download(std.mem.sliceAsBytes(v), d_v.ptr);
    }

    // =============================================================================
    // Internal: Kernel Loading
    // =============================================================================

    fn loadBuiltinKernels(self: *CudaBackend) !void {
        // Load built-in PTX for essential kernels from pre-compiled files
        // Kernels should be compiled separately and loaded at runtime
        _ = self;
        // TODO: Load compiled PTX files from disk or embed them
    }

    /// Load a kernel from PTX file
    pub fn loadKernelFromFile(self: *CudaBackend, name: []const u8, path: []const u8) !void {
        const ptx = try std.fs.cwd().readFileAlloc(self.allocator, path, 10 * 1024 * 1024);
        errdefer self.allocator.free(ptx);

        try self.context.loadKernel(name, ptx);

        const name_copy = try self.allocator.dupe(u8, name);
        try self.kernel_ptx.put(name_copy, ptx);
    }

    /// Load a kernel from PTX string
    pub fn loadKernelFromString(self: *CudaBackend, name: []const u8, ptx: []const u8) !void {
        const ptx_copy = try self.allocator.dupe(u8, ptx);
        errdefer self.allocator.free(ptx_copy);

        try self.context.loadKernel(name, ptx_copy);

        const name_copy = try self.allocator.dupe(u8, name);
        try self.kernel_ptx.put(name_copy, ptx_copy);
    }
};

// =============================================================================
// Device Buffer Wrapper
// =============================================================================

pub const DeviceBuffer = struct {
    ptr: CUdeviceptr,
    size: usize,
    context: *CudaContext,

    pub fn deinit(self: *DeviceBuffer) void {
        var buf = CudaContext.DeviceBuffer{
            .ptr = self.ptr,
            .size = self.size,
            .pool_index = null,
        };
        self.context.returnBuffer(buf);
    }
};

// =============================================================================
// Enums
// =============================================================================

pub const ElementWiseOp = enum {
    add,
    sub,
    mul,
    div,
};

pub const MapOp = enum {
    exp,
    log,
    sqrt,
    abs,
    square,
    inv,
};

pub const ActivationType = enum {
    relu,
    sigmoid,
    tanh,
    softmax,
    linear,
    gelu,
};

// =============================================================================
// PTX Templates for Built-in Kernels
// =============================================================================

/// PTX header for sm_60+ (Pascal and newer)
pub const PTX_HEADER = ".version 7.0\\n.target sm_60\\n.address_size 64\\n";

/// Template for element-wise operations kernel
pub const EW_TEMPLATE = PTX_HEADER ++
    \\n    \\.visible .entry {s}(
    \\n    .param .u64 a,
    \\n    .param .u64 b,
    \\n    .param .u64 c,
    \\n    .param .u32 n
    \\n)
    \\n{{
    \\n    .reg .pred %p<2>;
    \\n    .reg .b32 %r<4>;
    \\n    .reg .b64 %rd<11>;
    \\n    .reg .f32 %f<4>;
    \\n
    \\n    ld.param.u64 %rd1, [a];
    \\n    ld.param.u64 %rd2, [b];
    \\n    ld.param.u64 %rd3, [c];
    \\n    ld.param.u32 %r1, [n];
    \\n
    \\n    mov.u32 %r2, %ctaid.x;
    \\n    mov.u32 %r3, %ntid.x;
    \\n    mov.u32 %r4, %tid.x;
    \\n    mad.lo.s32 %r2, %r2, %r3, %r4;
    \\n    cvt.s64.s32 %rd4, %r2;
    \\n    setp.ge.s32 %p1, %r2, %r1;
    \\n    @%p1 bra $L__exit;
    \\n
    \\n    shl.b64 %rd5, %rd4, 2;
    \\n    add.s64 %rd6, %rd1, %rd5;
    \\n    add.s64 %rd7, %rd2, %rd5;
    \\n    add.s64 %rd8, %rd3, %rd5;
    \\n
    \\n    ld.global.f32 %f1, [%rd6];
    \\n    ld.global.f32 %f2, [%rd7];
    \\n    {s} %f3, %f1, %f2;
    \\n    st.global.f32 [%rd8], %f3;
    \\n
    \\$L__exit:
    \\n    ret;
    \\n}};

// =============================================================================
// Tests
// =============================================================================

test "CUDA availability" {
    // CUDA should not be available on macOS
    if (@import("builtin").os.tag == .macos) {
        try std.testing.expect(!CudaBackend.isAvailable());
        return;
    }

    // On Linux/Windows, may or may not be available depending on hardware
    _ = CudaBackend.isAvailable();
}

test "CUDA backend initialization" {
    if (@import("builtin").os.tag == .macos) {
        return error.SkipZigTest;
    }

    var backend = CudaBackend.init(std.testing.allocator) catch |err| {
        if (err == cuda_driver.CudaError.CudaDriverNotFound or
            err == cuda_driver.CudaError.CudaInitFailed or
            err == error.NoCudaDevices)
        {
            return;
        }
        return err;
    };
    defer backend.deinit();
}

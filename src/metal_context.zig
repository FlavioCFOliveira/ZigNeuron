/// Metal Context for managing persistent resources
/// Optimized for Apple Silicon GPU performance
const std = @import("std");
const metal = @import("metal.zig");

pub const MetalContext = struct {
    allocator: std.mem.Allocator,
    device: metal.MTLDevice,
    command_queue: metal.MTLCommandQueue,
    pipelines: std.StringHashMap(metal.MTLComputePipelineState),
    library: *anyopaque,

    pub fn init(allocator: std.mem.Allocator) !*MetalContext {
        const self = try allocator.create(MetalContext);
        errdefer allocator.destroy(self);

        self.allocator = allocator;
        self.device = try metal.MTLDevice.create();
        errdefer self.device.release();

        self.command_queue = try self.device.newCommandQueue();
        errdefer self.command_queue.release();

        // Load Metal shaders from source (runtime compilation)
        const shader_paths = [_][]const u8{
            "shaders/metal/matmul.metal",
            "shaders/metal/activation.metal",
            "shaders/metal/loss.metal",
        };

        var sources: [shader_paths.len][]const u8 = undefined;
        defer {
            for (sources) |s| allocator.free(s);
        }

        for (shader_paths, 0..) |path, i| {
            sources[i] = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
        }

        const shader_source = try std.mem.concat(allocator, u8, &sources);
        defer allocator.free(shader_source);

        self.library = try self.device.newLibraryWithSource(shader_source);

        self.pipelines = std.StringHashMap(metal.MTLComputePipelineState).init(allocator);
        errdefer {
            var it = self.pipelines.valueIterator();
            while (it.next()) |p| p.release();
            self.pipelines.deinit();
        }

        // Pre-compile core kernels
        try self.registerPipeline("matmul");
        try self.registerPipeline("matmul_batch");
        try self.registerPipeline("matmul_tiled");
        try self.registerPipeline("matmul_transpose_a");
        try self.registerPipeline("matmul_transpose_b");
        try self.registerPipeline("matmul_batch_transpose_b");

        // Activation kernels
        try self.registerPipeline("relu_forward");
        try self.registerPipeline("relu_backward");
        try self.registerPipeline("sigmoid_forward");
        try self.registerPipeline("sigmoid_backward");
        try self.registerPipeline("tanh_forward");
        try self.registerPipeline("tanh_backward");
        try self.registerPipeline("softmax_forward");
        try self.registerPipeline("softmax_backward");
        try self.registerPipeline("linear_forward");
        try self.registerPipeline("linear_backward");

        // Loss kernels
        try self.registerPipeline("mse_backward");
        try self.registerPipeline("cross_entropy_backward");
        try self.registerPipeline("binary_cross_entropy_backward");

        return self;
    }

    pub fn deinit(self: *MetalContext) void {
        var it = self.pipelines.valueIterator();
        while (it.next()) |p| p.release();
        self.pipelines.deinit();

        metal.objc.release(self.library);
        self.command_queue.release();
        self.device.release();
        self.allocator.destroy(self);
    }

    fn registerPipeline(self: *MetalContext, name: []const u8) !void {
        const name_z = try self.allocator.dupeZ(u8, name);
        defer self.allocator.free(name_z);

        const function = try metal.getFunctionFromLibrary(self.library, name_z);
        const pipeline = try self.device.newComputePipelineStateWithFunction(function);
        try self.pipelines.put(name, pipeline);
    }

    pub fn getPipeline(self: *const MetalContext, name: []const u8) ?*const metal.MTLComputePipelineState {
        return self.pipelines.getPtr(name);
    }
};

/// CUDA Microbenchmark Suite (Phase 6)
const std = @import("std");
const cuda = @import("cuda.zig");

const BENCHMARK_ITERATIONS: usize = 100;
const WARMUP_ITERATIONS: usize = 10;

test "benchmark_cuda_memory_transfer" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch { return; };
    defer backend.deinit();

    const sizes = .{ 1024, 4096, 16384 };
    std.debug.print("\n=== Memory Transfer Benchmark ===\n", .{});

    inline for (sizes) |size| {
        const buffer_size = size * @sizeOf(f32);
        const d_buffer = try backend.allocBuffer(buffer_size);
        defer backend.freeBuffer(d_buffer);
        const h_buffer = try std.testing.allocator.alloc(f32, size);
        defer std.testing.allocator.free(h_buffer);
        for (h_buffer, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.001;

        for (0..BENCHMARK_ITERATIONS) |_| { try backend.upload(d_buffer.ptr, h_buffer); }
        for (0..BENCHMARK_ITERATIONS) |_| { try backend.download(h_buffer, d_buffer.ptr); }

        std.debug.print("Size {d}: Completed {d} H2D + {d} D2H transfers\n", .{ size, BENCHMARK_ITERATIONS, BENCHMARK_ITERATIONS });
    }
}

test "benchmark_cuda_activations" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch { return; };
    defer backend.deinit();

    const sizes = .{ 1024, 4096, 16384 };
    std.debug.print("\n=== Activation Function Benchmark ===\n", .{});

    inline for (sizes) |size| {
        const h_input = try std.testing.allocator.alloc(f32, size);
        defer std.testing.allocator.free(h_input);
        const h_output = try std.testing.allocator.alloc(f32, size);
        defer std.testing.allocator.free(h_output);
        for (h_input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) - 50.0;

        for (0..BENCHMARK_ITERATIONS) |_| { try backend.reluForward(h_input, h_output); }
        for (0..BENCHMARK_ITERATIONS) |_| { try backend.sigmoidForward(h_input, h_output); }
        for (0..BENCHMARK_ITERATIONS) |_| { try backend.tanhForward(h_input, h_output); }

        std.debug.print("Size {d}: Completed {d} iterations of each activation\n", .{ size, BENCHMARK_ITERATIONS });
    }
}

test "benchmark_cuda_softmax" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch { return; };
    defer backend.deinit();

    const configs = .{ .{ .batch = 32, .classes = 10 }, .{ .batch = 128, .classes = 1000 } };
    std.debug.print("\n=== Softmax Benchmark ===\n", .{});

    inline for (configs) |cfg| {
        const size = cfg.batch * cfg.classes;
        const h_input = try std.testing.allocator.alloc(f32, size);
        defer std.testing.allocator.free(h_input);
        const h_output = try std.testing.allocator.alloc(f32, size);
        defer std.testing.allocator.free(h_output);
        for (h_input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 10)) * 0.1;

        for (0..BENCHMARK_ITERATIONS) |_| { try backend.softmaxForward(h_input, h_output, cfg.batch, cfg.classes); }

        std.debug.print("Batch {d} x Classes {d}: Completed {d} iterations\n", .{ cfg.batch, cfg.classes, BENCHMARK_ITERATIONS });
    }
}

test "benchmark_cuda_sgd" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch { return; };
    defer backend.deinit();

    const sizes = .{ 1024, 4096, 16384 };
    std.debug.print("\n=== SGD Optimizer Benchmark ===\n", .{});

    inline for (sizes) |size| {
        const h_weights = try std.testing.allocator.alloc(f32, size);
        defer std.testing.allocator.free(h_weights);
        const h_gradients = try std.testing.allocator.alloc(f32, size);
        defer std.testing.allocator.free(h_gradients);
        for (h_weights, 0..) |*w, i| w.* = @as(f32, @floatFromInt(i)) * 0.01;
        for (h_gradients) |*g| g.* = 0.001;

        for (0..BENCHMARK_ITERATIONS) |_| { try backend.sgdUpdate(h_weights, h_gradients, 0.01, 0.0); }

        std.debug.print("Size {d}: Completed {d} SGD updates\n", .{ size, BENCHMARK_ITERATIONS });
    }
}

test "benchmark_cuda_adam" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch { return; };
    defer backend.deinit();

    const sizes = .{ 1024, 4096, 16384 };
    std.debug.print("\n=== Adam Optimizer Benchmark ===\n", .{});

    inline for (sizes) |size| {
        const h_weights = try std.testing.allocator.alloc(f32, size);
        defer std.testing.allocator.free(h_weights);
        const h_gradients = try std.testing.allocator.alloc(f32, size);
        defer std.testing.allocator.free(h_gradients);
        const h_m = try std.testing.allocator.alloc(f32, size);
        defer std.testing.allocator.free(h_m);
        const h_v = try std.testing.allocator.alloc(f32, size);
        defer std.testing.allocator.free(h_v);
        for (h_weights, 0..) |*w, i| w.* = @as(f32, @floatFromInt(i)) * 0.01;
        for (h_gradients) |*g| g.* = 0.001;
        @memset(h_m, 0);
        @memset(h_v, 0);

        for (0..BENCHMARK_ITERATIONS) |_| { try backend.adamUpdate(h_weights, h_gradients, h_m, h_v, 0.001, 0.9, 0.999, 1e-8, 1); }

        std.debug.print("Size {d}: Completed {d} Adam updates\n", .{ size, BENCHMARK_ITERATIONS });
    }
}

test "benchmark_cuda_matmul" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch { return; };
    defer backend.deinit();

    const configs = .{ .{ .m = 128, .n = 128, .k = 128 }, .{ .m = 512, .n = 512, .k = 512 } };
    std.debug.print("\n=== Matrix Multiplication Benchmark ===\n", .{});

    inline for (configs) |cfg| {
        const a_size = cfg.m * cfg.k;
        const b_size = cfg.k * cfg.n;
        const c_size = cfg.m * cfg.n;
        const h_a = try std.testing.allocator.alloc(f32, a_size);
        defer std.testing.allocator.free(h_a);
        const h_b = try std.testing.allocator.alloc(f32, b_size);
        defer std.testing.allocator.free(h_b);
        const h_c = try std.testing.allocator.alloc(f32, c_size);
        defer std.testing.allocator.free(h_c);
        for (h_a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) * 0.01;
        for (h_b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) * 0.01;

        for (0..WARMUP_ITERATIONS) |_| { try backend.matMul(h_a, h_b, h_c, cfg.m, cfg.n, cfg.k, false, false, false); }
        for (0..BENCHMARK_ITERATIONS) |_| { try backend.matMul(h_a, h_b, h_c, cfg.m, cfg.n, cfg.k, false, false, false); }

        std.debug.print("Matmul {d}x{d}x{d}: Completed {d} iterations\n", .{ cfg.m, cfg.n, cfg.k, BENCHMARK_ITERATIONS });
    }
}

test "cuda_performance_report" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        std.debug.print("\n=== CUDA Performance Report ===\nCUDA not available\n", .{});
        return;
    };
    defer backend.deinit();

    std.debug.print("\n=== CUDA Performance Report ===\n", .{});
    std.debug.print("Device: {s}\n", .{backend.context.device_props.name});
    std.debug.print("Compute Capability: {d}.{d}\n", .{ backend.context.device_props.compute_capability_major, backend.context.device_props.compute_capability_minor });
    std.debug.print("Total Memory: {d} MB\n", .{backend.context.device_props.total_memory / (1024 * 1024)});
    std.debug.print("SM Count: {d}\n", .{backend.context.device_props.multiprocessor_count});
}

test "benchmark_cuda_buffer_pool" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch { return; };
    defer backend.deinit();

    const sizes = .{ 1024, 4096, 16384 };
    std.debug.print("\n=== Buffer Pool Benchmark ===\n", .{});

    inline for (sizes) |size| {
        const buffer_size = size * @sizeOf(f32);
        for (0..BENCHMARK_ITERATIONS) |_| {
            const d_buffer = try backend.allocBuffer(buffer_size);
            backend.freeBuffer(d_buffer);
        }
        std.debug.print("Size {d}: Completed {d} alloc+free cycles\n", .{ size, BENCHMARK_ITERATIONS });
    }
}

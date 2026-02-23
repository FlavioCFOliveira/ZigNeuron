const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the main module for the library
    const lib_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .name = "ZigNeuron",
        .root_module = lib_module,
    });

    b.installArtifact(lib);

    // Create test module with test file
    const test_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("src/test_all.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("ZigNeuron", lib_module);

    // Test executable
    const test_exe = b.addTest(.{
        .root_module = test_module,
    });
    if (target.result.os.tag == .macos) {
        test_exe.linkLibC();
        test_exe.linkSystemLibrary("objc");
        test_exe.linkFramework("Metal");
        test_exe.linkFramework("Foundation");
        test_exe.linkFramework("QuartzCore");
    }

    const run_test_exe = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_test_exe.step);

    // Benchmark executable (CPU only)
    const enable_benchmarks = b.option(bool, "benchmarks", "Build benchmarks") orelse false;

    if (enable_benchmarks) {
        const benchmark_module = std.Build.Module.create(b, .{
            .root_source_file = b.path("src/benchmark_main.zig"),
            .target = target,
            .optimize = optimize,
        });
        benchmark_module.addImport("ZigNeuron", lib_module);

        const benchmark_exe = b.addExecutable(.{
            .name = "benchmarks",
            .root_module = benchmark_module,
        });
        if (target.result.os.tag == .macos) {
            benchmark_exe.linkLibC();
            benchmark_exe.linkSystemLibrary("objc");
            benchmark_exe.linkFramework("Metal");
            benchmark_exe.linkFramework("Foundation");
            benchmark_exe.linkFramework("QuartzCore");
        }

        b.installArtifact(benchmark_exe);

        const run_benchmarks = b.addRunArtifact(benchmark_exe);
        const benchmark_step = b.step("benchmarks", "Run benchmarks");
        benchmark_step.dependOn(&run_benchmarks.step);
    }

    // Vulkan vs CPU comparison benchmark
    const enable_benchmark_compare = b.option(bool, "benchmark-compare", "Build Vulkan vs CPU comparison benchmark") orelse false;

    if (enable_benchmark_compare) {
        const benchmark_module = std.Build.Module.create(b, .{
            .root_source_file = b.path("src/benchmark_compare.zig"),
            .target = target,
            .optimize = optimize,
        });
        benchmark_module.addImport("ZigNeuron", lib_module);

        const benchmark_exe = b.addExecutable(.{
            .name = "benchmark-compare",
            .root_module = benchmark_module,
        });
        if (target.result.os.tag == .macos) {
            benchmark_exe.linkLibC();
            benchmark_exe.linkSystemLibrary("objc");
            benchmark_exe.linkFramework("Metal");
            benchmark_exe.linkFramework("Foundation");
            benchmark_exe.linkFramework("QuartzCore");
        }

        b.installArtifact(benchmark_exe);

        const run_benchmarks = b.addRunArtifact(benchmark_exe);
        const benchmark_step = b.step("benchmark-compare", "Run Vulkan vs CPU comparison benchmark");
        benchmark_step.dependOn(&run_benchmarks.step);
    }

    // Vulkan shader compilation step
    const enable_vulkan = b.option(bool, "vulkan", "Build with Vulkan support") orelse true;

    if (enable_vulkan) {
        const compile_shaders_step = b.step("compile-shaders", "Compile Vulkan shaders to SPIR-V");

        // Define shader files and their output paths
        const shaders = [_]struct {
            name: []const u8,
            input: []const u8,
            output: []const u8,
        }{
            .{ .name = "matmul", .input = "shaders/matmul.comp", .output = "shaders/matmul.comp.spv" },
            .{ .name = "activation_forward", .input = "shaders/activation_forward.comp", .output = "shaders/activation_forward.comp.spv" },
            .{ .name = "activation_backward", .input = "shaders/activation_backward.comp", .output = "shaders/activation_backward.comp.spv" },
            .{ .name = "loss_backward", .input = "shaders/loss_backward.comp", .output = "shaders/loss_backward.comp.spv" },
        };

        for (shaders) |shader| {
            const compile_cmd = b.addSystemCommand(&.{
                "glslc",
                "-fshader-stage=compute",
                shader.input,
                "-o",
                shader.output,
            });

            compile_cmd.step.dependOn(b.getInstallStep());
            compile_shaders_step.dependOn(&compile_cmd.step);
        }

        // Add a step to run all Vulkan-related checks
        const vulkan_step = b.step("vulkan", "Check Vulkan compilation");
        vulkan_step.dependOn(compile_shaders_step);
    }

    // Metal shader compilation (macOS only)
    const enable_metal = b.option(bool, "metal", "Build with Metal support") orelse (target.result.os.tag == .macos);

    if (enable_metal) {
        const compile_metal_step = b.step("compile-metal", "Compile Metal shaders to metallib");

        // Define Metal shader files
        const metal_shaders = [_][]const u8{
            "shaders/metal/matmul.metal",
            "shaders/metal/activation.metal",
            "shaders/metal/loss.metal",
        };

        var air_files: std.ArrayListUnmanaged([]const u8) = .{};
        defer air_files.deinit(b.allocator);

        // Compile each shader to .air
        for (metal_shaders) |shader| {
            const shader_name = std.fs.path.basename(shader);
            const air_file = b.pathJoin(&.{ b.cache_root.path.?, b.fmt("{s}.air", .{shader_name}) });
            air_files.append(b.allocator, air_file) catch @panic("Out of memory");

            const compile_cmd = b.addSystemCommand(&.{
                "xcrun", "-sdk", "macosx", "metal",
                "-c", shader,
                "-o", air_file,
            });
            compile_cmd.step.dependOn(b.getInstallStep());
            compile_metal_step.dependOn(&compile_cmd.step);
        }

        // Link all .air files to .metallib
        const metallib_file = "shaders/metal/default.metallib";
        const link_cmd = b.addSystemCommand(&.{
            "xcrun", "-sdk", "macosx", "metallib",
            "-o", metallib_file,
        });
        link_cmd.addArgs(air_files.items);
        link_cmd.step.dependOn(compile_metal_step);

        // Add Metal compilation step
        const metal_step = b.step("metal", "Compile Metal shaders");
        metal_step.dependOn(&link_cmd.step);
    }

    // Performance test executable
    const perf_test_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("src/test_performance.zig"),
        .target = target,
        .optimize = optimize,
    });
    perf_test_module.addImport("ZigNeuron", lib_module);

    const perf_exe = b.addExecutable(.{
        .name = "test_performance",
        .root_module = perf_test_module,
    });

    // Link Objective-C runtime for Metal
    if (target.result.os.tag == .macos) {
        perf_exe.linkLibC();
        perf_exe.linkSystemLibrary("objc");
        perf_exe.linkFramework("Metal");
        perf_exe.linkFramework("Foundation");
        perf_exe.linkFramework("QuartzCore");
    }

    b.installArtifact(perf_exe);

    const run_perf = b.addRunArtifact(perf_exe);
    const perf_step = b.step("test-performance", "Run performance comparison tests");
    perf_step.dependOn(&run_perf.step);

    // Multi-backend test executable
    const all_backends_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("src/test_all_backends.zig"),
        .target = target,
        .optimize = optimize,
    });
    all_backends_module.addImport("ZigNeuron", lib_module);

    const all_backends_exe = b.addExecutable(.{
        .name = "test_all_backends",
        .root_module = all_backends_module,
    });
    if (target.result.os.tag == .macos) {
        all_backends_exe.linkLibC();
        all_backends_exe.linkSystemLibrary("objc");
        all_backends_exe.linkFramework("Metal");
        all_backends_exe.linkFramework("Foundation");
        all_backends_exe.linkFramework("QuartzCore");
    }

    b.installArtifact(all_backends_exe);

    const run_all_backends = b.addRunArtifact(all_backends_exe);
    const all_backends_step = b.step("test-backends", "Run comprehensive backend comparison tests");
    all_backends_step.dependOn(&run_all_backends.step);

    // Stock Prediction Example
    const stock_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("examples/stock_prediction/lstm.zig"),
        .target = target,
        .optimize = optimize,
    });
    stock_module.addImport("ZigNeuron", lib_module);

    const stock_exe = b.addExecutable(.{
        .name = "stock_lstm",
        .root_module = stock_module,
    });
    if (target.result.os.tag == .macos) {
        stock_exe.linkLibC();
        stock_exe.linkSystemLibrary("objc");
        stock_exe.linkFramework("Metal");
        stock_exe.linkFramework("Foundation");
        stock_exe.linkFramework("QuartzCore");
    }
    b.installArtifact(stock_exe);

    const run_stock = b.addRunArtifact(stock_exe);
    const stock_step = b.step("example-stock", "Run stock prediction LSTM example");
    stock_step.dependOn(&run_stock.step);

    // Attention Example
    const attention_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("examples/stock_prediction/attention_transformer.zig"),
        .target = target,
        .optimize = optimize,
    });
    attention_module.addImport("ZigNeuron", lib_module);

    const attention_exe = b.addExecutable(.{
        .name = "stock_attention",
        .root_module = attention_module,
    });
    if (target.result.os.tag == .macos) {
        attention_exe.linkLibC();
        attention_exe.linkSystemLibrary("objc");
        attention_exe.linkFramework("Metal");
        attention_exe.linkFramework("Foundation");
        attention_exe.linkFramework("QuartzCore");
    }
    b.installArtifact(attention_exe);

    const run_attention = b.addRunArtifact(attention_exe);
    const attention_step = b.step("example-attention", "Run stock prediction Attention example");
    attention_step.dependOn(&run_attention.step);

    // CNN Example
    const cnn_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("examples/stock_prediction/cnn_seq2seq.zig"),
        .target = target,
        .optimize = optimize,
    });
    cnn_module.addImport("ZigNeuron", lib_module);

    const cnn_exe = b.addExecutable(.{
        .name = "stock_cnn",
        .root_module = cnn_module,
    });
    if (target.result.os.tag == .macos) {
        cnn_exe.linkLibC();
        cnn_exe.linkSystemLibrary("objc");
        cnn_exe.linkFramework("Metal");
        cnn_exe.linkFramework("Foundation");
        cnn_exe.linkFramework("QuartzCore");
    }
    b.installArtifact(cnn_exe);

    const run_cnn = b.addRunArtifact(cnn_exe);
    const cnn_step = b.step("example-cnn", "Run stock prediction CNN example");
    cnn_step.dependOn(&run_cnn.step);
}

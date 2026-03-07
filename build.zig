const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const enable_cuda = b.option(bool, "cuda", "Build with CUDA support") orelse false;

    // Create the main module for the library
    const lib_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Add CUDA module if enabled
    if (enable_cuda) {
        const cuda_module = b.addModule("cuda_driver", .{
            .root_source_file = b.path("src/cuda_driver.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        lib_module.addImport("cuda_driver", cuda_module);

        const cuda_context_module = b.addModule("cuda_context", .{
            .root_source_file = b.path("src/cuda_context.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        cuda_context_module.addImport("cuda_driver", cuda_module);
        lib_module.addImport("cuda_context", cuda_context_module);
    }

    const lib = b.addLibrary(.{
        .name = "ZigNeuron",
        .root_module = lib_module,
    });

    b.installArtifact(lib);

    // =============================================================================
    // Tests
    // =============================================================================

    const test_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("src/test_all.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_module.addImport("ZigNeuron", lib_module);

    if (target.result.os.tag == .macos) {
        test_module.linkSystemLibrary("objc", .{});
        test_module.linkFramework("Metal", .{});
        test_module.linkFramework("Foundation", .{});
        test_module.linkFramework("QuartzCore", .{});
    }

    if (enable_cuda and target.result.os.tag != .macos) {
        // CUDA doesn't require explicit linking, but we need to ensure the driver API is available
        // The CUDA driver is loaded dynamically at runtime
        test_module.addImport("cuda_driver", b.addModule("cuda_driver", .{
            .root_source_file = b.path("src/cuda_driver.zig"),
        }));
    }

    const test_exe = b.addTest(.{
        .root_module = test_module,
    });

    const run_test_exe = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_test_exe.step);

    // =============================================================================
    // CUDA Support
    // =============================================================================

    if (enable_cuda and target.result.os.tag != .macos) {
        // CUDA kernel compilation step
        const compile_cuda_step = b.step("compile-cuda", "Compile CUDA kernels to PTX");

        const cuda_sources = [_][]const u8{
            "shaders/cuda/kernels.cu",
        };

        const cache_path = b.cache_root.path orelse ".zig-cache";
        for (cuda_sources) |source| {
            const basename = std.fs.path.basename(source);
            var ptx_file_buf: [256]u8 = undefined;
            const ptx_file = std.fmt.bufPrint(&ptx_file_buf, "{s}{c}{s}.ptx", .{ cache_path, std.fs.path.sep, basename }) catch continue;

            const compile_cmd = b.addSystemCommand(&.{
                "nvcc",
                "-ptx",
                "-arch=sm_60",  // Pascal and newer
                "-O3",
                "-lineinfo",
                "-o", ptx_file,
                source,
            });
            compile_cuda_step.dependOn(&compile_cmd.step);
        }

        // CUDA installation step - copy PTX files to output
        const cuda_install_step = b.step("cuda", "Compile and install CUDA kernels");
        cuda_install_step.dependOn(compile_cuda_step);
    }

    // =============================================================================
    // Metal Support (macOS only)
    // =============================================================================

    const enable_metal = b.option(bool, "metal", "Build with Metal support") orelse (target.result.os.tag == .macos);

    if (enable_metal and target.result.os.tag == .macos) {
        const compile_metal_step = b.step("compile-metal", "Compile Metal shaders to metallib");

        const metal_shaders = [_][]const u8{
            "shaders/metal/matmul.metal",
            "shaders/metal/activation.metal",
            "shaders/metal/loss.metal",
            "shaders/metal/fused.metal",
            "shaders/metal/attention.metal",
            "shaders/metal/recurrent.metal",
            "shaders/metal/convolution.metal",
            "shaders/metal/normalization.metal",
            "shaders/metal/optimizer.metal",
            "shaders/metal/auxiliary.metal",
        };

        var air_files: std.ArrayListUnmanaged([]const u8) = .{};
        defer air_files.deinit(b.allocator);

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

        const metallib_file = "shaders/metal/default.metallib";
        const link_cmd = b.addSystemCommand(&.{
            "xcrun", "-sdk", "macosx", "metallib",
            "-o", metallib_file,
        });
        link_cmd.addArgs(air_files.items);
        link_cmd.step.dependOn(compile_metal_step);

        const metal_step = b.step("metal", "Compile Metal shaders");
        metal_step.dependOn(&link_cmd.step);
    }

    // =============================================================================
    // Benchmarks
    // =============================================================================

    const enable_benchmarks = b.option(bool, "benchmarks", "Build benchmarks") orelse false;

    if (enable_benchmarks) {
        const benchmark_module = std.Build.Module.create(b, .{
            .root_source_file = b.path("src/benchmark_main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        benchmark_module.addImport("ZigNeuron", lib_module);

        if (target.result.os.tag == .macos) {
            benchmark_module.linkSystemLibrary("objc", .{});
            benchmark_module.linkFramework("Metal", .{});
            benchmark_module.linkFramework("Foundation", .{});
            benchmark_module.linkFramework("QuartzCore", .{});
        }

        const benchmark_exe = b.addExecutable(.{
            .name = "benchmarks",
            .root_module = benchmark_module,
        });

        b.installArtifact(benchmark_exe);

        const run_benchmarks = b.addRunArtifact(benchmark_exe);
        const benchmark_step = b.step("benchmarks", "Run benchmarks");
        benchmark_step.dependOn(&run_benchmarks.step);
    }

    // =============================================================================
    // Performance Tests
    // =============================================================================

    const perf_test_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("src/test_performance.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    perf_test_module.addImport("ZigNeuron", lib_module);

    if (target.result.os.tag == .macos) {
        perf_test_module.linkSystemLibrary("objc", .{});
        perf_test_module.linkFramework("Metal", .{});
        perf_test_module.linkFramework("Foundation", .{});
        perf_test_module.linkFramework("QuartzCore", .{});
    }

    const perf_exe = b.addExecutable(.{
        .name = "test_performance",
        .root_module = perf_test_module,
    });

    b.installArtifact(perf_exe);

    const run_perf = b.addRunArtifact(perf_exe);
    const perf_step = b.step("test-performance", "Run performance comparison tests");
    perf_step.dependOn(&run_perf.step);

    // =============================================================================
    // Examples
    // =============================================================================

    // Stock Prediction Examples
    addExample(b, lib_module, "stock_lstm", "examples/stock_prediction/lstm.zig", target, optimize);
    addExample(b, lib_module, "stock_attention", "examples/stock_prediction/attention_transformer.zig", target, optimize);
    addExample(b, lib_module, "stock_cnn", "examples/stock_prediction/cnn_seq2seq.zig", target, optimize);

    // Classification Examples
    addExample(b, lib_module, "iris_classification", "examples/classification/iris.zig", target, optimize);

    // Comprehensive Suite Examples
    const examples = [_][]const u8{
        "01_vanilla_rnn", "02_vanilla_bidirectional", "03_vanilla_twopath",
        "04_lstm", "05_lstm_bidirectional", "06_lstm_twopath",
        "07_gru", "08_gru_bidirectional", "09_gru_twopath",
        "10_lstm_seq2seq", "11_lstm_bidirectional_seq2seq", "12_lstm_seq2seq_vae",
        "13_gru_seq2seq", "14_gru_bidirectional_seq2seq", "15_gru_seq2seq_vae",
        "16_attention", "17_cnn_seq2seq", "18_dilated_cnn_seq2seq",
    };

    const suite_common_module = b.addModule("suite_common", .{
        .root_source_file = b.path("examples/comprehensive_suite/common.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    suite_common_module.addImport("ZigNeuron", lib_module);

    if (target.result.os.tag == .macos) {
        suite_common_module.linkSystemLibrary("objc", .{});
        suite_common_module.linkFramework("Metal", .{});
        suite_common_module.linkFramework("Foundation", .{});
        suite_common_module.linkFramework("QuartzCore", .{});
    }

    for (examples) |example_name| {
        const suite_module = b.addModule(example_name, .{
            .root_source_file = b.path(b.fmt("examples/comprehensive_suite/{s}.zig", .{example_name})),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        suite_module.addImport("ZigNeuron", lib_module);
        suite_module.addImport("common", suite_common_module);

        if (target.result.os.tag == .macos) {
            suite_module.linkSystemLibrary("objc", .{});
            suite_module.linkFramework("Metal", .{});
            suite_module.linkFramework("Foundation", .{});
            suite_module.linkFramework("QuartzCore", .{});
        }

        const suite_exe = b.addExecutable(.{
            .name = example_name,
            .root_module = suite_module,
        });
        b.installArtifact(suite_exe);

        const run_suite_exe = b.addRunArtifact(suite_exe);
        const suite_step = b.step(b.fmt("run-{s}", .{example_name}), b.fmt("Run comprehensive suite example: {s}", .{example_name}));
        suite_step.dependOn(&run_suite_exe.step);
    }
}

fn addExample(
    b: *std.Build,
    lib_module: *std.Build.Module,
    name: []const u8,
    path: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) void {
    const example_module = std.Build.Module.create(b, .{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    example_module.addImport("ZigNeuron", lib_module);

    if (target.result.os.tag == .macos) {
        example_module.linkSystemLibrary("objc", .{});
        example_module.linkFramework("Metal", .{});
        example_module.linkFramework("Foundation", .{});
        example_module.linkFramework("QuartzCore", .{});
    }

    const example_exe = b.addExecutable(.{
        .name = name,
        .root_module = example_module,
    });
    b.installArtifact(example_exe);

    const run_example = b.addRunArtifact(example_exe);
    const example_step = b.step(b.fmt("example-{s}", .{name}), b.fmt("Run example: {s}", .{name}));
    example_step.dependOn(&run_example.step);
}

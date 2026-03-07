const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the main module for the library
    const lib_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
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
        .link_libc = true,
    });
    test_module.addImport("ZigNeuron", lib_module);

    if (target.result.os.tag == .macos) {
        test_module.linkSystemLibrary("objc", .{});
        test_module.linkFramework("Metal", .{});
        test_module.linkFramework("Foundation", .{});
        test_module.linkFramework("QuartzCore", .{});
    }

    // Test executable
    const test_exe = b.addTest(.{
        .root_module = test_module,
    });

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

    // Metal shader compilation (macOS only)
    const enable_metal = b.option(bool, "metal", "Build with Metal support") orelse (target.result.os.tag == .macos);

    if (enable_metal) {
        const compile_metal_step = b.step("compile-metal", "Compile Metal shaders to metallib");

        // Define Metal shader files
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

    // Stock Prediction Example
    const stock_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("examples/stock_prediction/lstm.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    stock_module.addImport("ZigNeuron", lib_module);

    if (target.result.os.tag == .macos) {
        stock_module.linkSystemLibrary("objc", .{});
        stock_module.linkFramework("Metal", .{});
        stock_module.linkFramework("Foundation", .{});
        stock_module.linkFramework("QuartzCore", .{});
    }

    const stock_exe = b.addExecutable(.{
        .name = "stock_lstm",
        .root_module = stock_module,
    });
    b.installArtifact(stock_exe);

    const run_stock = b.addRunArtifact(stock_exe);
    const stock_step = b.step("example-stock", "Run stock prediction LSTM example");
    stock_step.dependOn(&run_stock.step);

    // Attention Example
    const attention_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("examples/stock_prediction/attention_transformer.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    attention_module.addImport("ZigNeuron", lib_module);

    if (target.result.os.tag == .macos) {
        attention_module.linkSystemLibrary("objc", .{});
        attention_module.linkFramework("Metal", .{});
        attention_module.linkFramework("Foundation", .{});
        attention_module.linkFramework("QuartzCore", .{});
    }

    const attention_exe = b.addExecutable(.{
        .name = "stock_attention",
        .root_module = attention_module,
    });
    b.installArtifact(attention_exe);

    const run_attention = b.addRunArtifact(attention_exe);
    const attention_step = b.step("example-attention", "Run stock prediction Attention example");
    attention_step.dependOn(&run_attention.step);

    // CNN Example
    const cnn_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("examples/stock_prediction/cnn_seq2seq.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    cnn_module.addImport("ZigNeuron", lib_module);

    if (target.result.os.tag == .macos) {
        cnn_module.linkSystemLibrary("objc", .{});
        cnn_module.linkFramework("Metal", .{});
        cnn_module.linkFramework("Foundation", .{});
        cnn_module.linkFramework("QuartzCore", .{});
    }

    const cnn_exe = b.addExecutable(.{
        .name = "stock_cnn",
        .root_module = cnn_module,
    });
    b.installArtifact(cnn_exe);

    const run_cnn = b.addRunArtifact(cnn_exe);
    const cnn_step = b.step("example-cnn", "Run stock prediction CNN example");
    cnn_step.dependOn(&run_cnn.step);

    // Classification Examples
    // Iris Classification
    const iris_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("examples/classification/iris.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    iris_module.addImport("ZigNeuron", lib_module);

    if (target.result.os.tag == .macos) {
        iris_module.linkSystemLibrary("objc", .{});
        iris_module.linkFramework("Metal", .{});
        iris_module.linkFramework("Foundation", .{});
        iris_module.linkFramework("QuartzCore", .{});
    }

    const iris_exe = b.addExecutable(.{
        .name = "iris_classification",
        .root_module = iris_module,
    });
    b.installArtifact(iris_exe);

    const run_iris = b.addRunArtifact(iris_exe);
    const iris_step = b.step("example-iris", "Run Iris flower classification example");
    iris_step.dependOn(&run_iris.step);

    // Comprehensive Suite
    const examples = [_][]const u8{
        "01_vanilla_rnn",
        "02_vanilla_bidirectional",
        "03_vanilla_twopath",
        "04_lstm",
        "05_lstm_bidirectional",
        "06_lstm_twopath",
        "07_gru",
        "08_gru_bidirectional",
        "09_gru_twopath",
        "10_lstm_seq2seq",
        "11_lstm_bidirectional_seq2seq",
        "12_lstm_seq2seq_vae",
        "13_gru_seq2seq",
        "14_gru_bidirectional_seq2seq",
        "15_gru_seq2seq_vae",
        "16_attention",
        "17_cnn_seq2seq",
        "18_dilated_cnn_seq2seq",
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

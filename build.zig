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

    // Classification Examples
    // Iris Classification
    const iris_module = std.Build.Module.create(b, .{
        .root_source_file = b.path("examples/classification/iris.zig"),
        .target = target,
        .optimize = optimize,
    });
    iris_module.addImport("ZigNeuron", lib_module);

    const iris_exe = b.addExecutable(.{
        .name = "iris_classification",
        .root_module = iris_module,
    });
    if (target.result.os.tag == .macos) {
        iris_exe.linkLibC();
        iris_exe.linkSystemLibrary("objc");
        iris_exe.linkFramework("Metal");
        iris_exe.linkFramework("Foundation");
        iris_exe.linkFramework("QuartzCore");
    }
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
    });
    suite_common_module.addImport("ZigNeuron", lib_module);

    for (examples) |example_name| {
        const suite_module = b.addModule(example_name, .{
            .root_source_file = b.path(b.fmt("examples/comprehensive_suite/{s}.zig", .{example_name})),
            .target = target,
            .optimize = optimize,
        });
        suite_module.addImport("ZigNeuron", lib_module);
        suite_module.addImport("common", suite_common_module);

        const suite_exe = b.addExecutable(.{
            .name = example_name,
            .root_module = suite_module,
        });

        if (target.result.os.tag == .macos) {
            suite_exe.linkLibC();
            suite_exe.linkSystemLibrary("objc");
            suite_exe.linkFramework("Metal");
            suite_exe.linkFramework("Foundation");
            suite_exe.linkFramework("QuartzCore");
        }
        b.installArtifact(suite_exe);

        const run_suite_exe = b.addRunArtifact(suite_exe);
        const suite_step = b.step(b.fmt("run-{s}", .{example_name}), b.fmt("Run comprehensive suite example: {s}", .{example_name}));
        suite_step.dependOn(&run_suite_exe.step);
    }
}

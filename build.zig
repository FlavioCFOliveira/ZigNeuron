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

    const run_test_exe = b.addRunArtifact(test_exe);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_test_exe.step);

    // Examples executable
    const enable_examples = b.option(bool, "examples", "Build examples") orelse false;

    if (enable_examples) {
        // XOR example using SGD
        const xor_module = std.Build.Module.create(b, .{
            .root_source_file = b.path("examples/xor.zig"),
            .target = target,
            .optimize = optimize,
        });
        xor_module.addImport("ZigNeuron", lib_module);

        const xor_exe = b.addExecutable(.{
            .name = "xor",
            .root_module = xor_module,
        });

        // XOR test example using custom gradient descent
        const xor_test_module = std.Build.Module.create(b, .{
            .root_source_file = b.path("examples/xor_test.zig"),
            .target = target,
            .optimize = optimize,
        });
        xor_test_module.addImport("ZigNeuron", lib_module);

        const xor_test_exe = b.addExecutable(.{
            .name = "xor_test",
            .root_module = xor_test_module,
        });

        // Comprehensive FNN example
        const fnn_module = std.Build.Module.create(b, .{
            .root_source_file = b.path("examples/fnn_comprehensive.zig"),
            .target = target,
            .optimize = optimize,
        });
        fnn_module.addImport("ZigNeuron", lib_module);

        const fnn_exe = b.addExecutable(.{
            .name = "fnn_comprehensive",
            .root_module = fnn_module,
        });

        b.installArtifact(xor_exe);
        b.installArtifact(xor_test_exe);
        b.installArtifact(fnn_exe);

        const run_cmd = b.addRunArtifact(xor_test_exe);
        run_cmd.step.dependOn(b.getInstallStep());

        const run_step = b.step("run-examples", "Run examples");
        run_step.dependOn(&run_cmd.step);

        // FNN example step
        const run_fnn = b.addRunArtifact(fnn_exe);
        const fnn_step = b.step("fnn", "Run comprehensive FNN example");
        fnn_step.dependOn(&run_fnn.step);
    }

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
}

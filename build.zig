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

        b.installArtifact(xor_exe);
        b.installArtifact(xor_test_exe);

        const run_cmd = b.addRunArtifact(xor_test_exe);
        run_cmd.step.dependOn(b.getInstallStep());

        const run_step = b.step("run-examples", "Run examples");
        run_step.dependOn(&run_cmd.step);
    }
}

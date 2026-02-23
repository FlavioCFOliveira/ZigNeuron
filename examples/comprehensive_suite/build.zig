const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_module = b.dependency("ZigNeuron", .{}).module("ZigNeuron");

    const common_module = b.addModule("common", .{
        .root_source_file = b.path("common.zig"),
        .imports = &.{
            .{ .name = "ZigNeuron", .module = lib_module },
        },
    });

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

    for (examples) |example_name| {
        const exe = b.addExecutable(.{
            .name = example_name,
            .root_source_file = b.path(b.fmt("{s}.zig", .{example_name})),
            .target = target,
            .optimize = optimize,
        });

        exe.root_module.addImport("ZigNeuron", lib_module);
        exe.root_module.addImport("common", common_module);

        if (target.result.os.tag == .macos) {
            exe.linkLibC();
            exe.linkSystemLibrary("objc");
            exe.linkFramework("Metal");
            exe.linkFramework("Foundation");
            exe.linkFramework("QuartzCore");
        }

        b.installArtifact(exe);
    }
}

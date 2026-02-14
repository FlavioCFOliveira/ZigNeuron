/// ZigNeuron - A neural network library written in Zig
pub const activation = @import("activation.zig");
pub const backend = @import("backend.zig");
pub const layer = @import("layer.zig");
pub const loss = @import("loss.zig");
pub const network = @import("network.zig");

pub const ZigNeuron = struct {};

test "basic test" {
    _ = ZigNeuron;
}

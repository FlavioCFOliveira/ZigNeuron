/// ZigNeuron - A neural network library written in Zig
pub const activation = @import("activation.zig");
pub const backend = @import("backend.zig");
pub const layer = @import("layer.zig");
pub const recurrent = @import("recurrent.zig");
pub const loss = @import("loss.zig");
pub const network = @import("network.zig");
pub const optimizer = @import("optimizer.zig");
pub const tensor = @import("tensor.zig");
pub const serialization = @import("serialization.zig");
pub const metrics = @import("metrics.zig");
pub const ZigNeuron = struct {};

test "basic test" {
    _ = ZigNeuron;
}

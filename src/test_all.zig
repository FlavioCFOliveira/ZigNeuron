/// Unit tests for ZigNeuron
const std = @import("std");

const zn = @import("ZigNeuron");
const activation = zn.activation;
const loss = zn.loss;
const layer = zn.layer;
const network = zn.network;
const backend = zn.backend;
const optimizer = zn.optimizer;

// Test modules from src/test/
const unit_activation = @import("test/unit/activation.zig");
const unit_loss = @import("test/unit/loss.zig");
const unit_layer = @import("test/unit/layer.zig");
const unit_optimizer = @import("test/unit/optimizer.zig");
const unit_backend = @import("test/unit/backend.zig");
const unit_recurrent = @import("test/unit/recurrent.zig");
const unit_recurrent_accumulation = @import("test/unit/recurrent_accumulation.zig");
const unit_vae = @import("test/unit/vae.zig");
const unit_phase4 = @import("test/unit/phase4.zig");
const unit_parity = @import("test/unit/parity.zig");

test {
    // Reference only known good unit tests
    std.testing.refAllDecls(unit_activation);
    std.testing.refAllDecls(unit_loss);
    std.testing.refAllDecls(unit_layer);
    std.testing.refAllDecls(unit_optimizer);
    std.testing.refAllDecls(unit_backend);
    std.testing.refAllDecls(unit_recurrent);
    std.testing.refAllDecls(unit_recurrent_accumulation);
    std.testing.refAllDecls(unit_vae);
    std.testing.refAllDecls(unit_phase4);
    std.testing.refAllDecls(unit_parity);
}


/// Model serialization - Save and load trained models
///
/// Format: .znn (ZigNeuron Native)
/// Binary format with header + layer data
///
/// File Structure:
/// - Magic: "ZNN\0" (4 bytes)
/// - Version: u32 (4 bytes)
/// - Layer Count: u32 (4 bytes)
/// - Layer Data: variable
///
const std = @import("std");
const layer = @import("layer.zig");
const activation = @import("activation.zig");
const tensor = @import("tensor.zig");
const backend = @import("backend.zig");
const recurrent = @import("recurrent.zig");
const metal = @import("metal.zig");

const MAGIC = "ZNN\x00";
const VERSION: u32 = 1;

/// Layer type identifiers for serialization
const LayerType = enum(u32) {
    dense = 1,
    rnn = 2,
    lstm = 3,
    gru = 4,
    sampling = 5,
    conv1d = 6,
    layer_norm = 7,
    batch_norm = 8,
    dropout = 9,
    attention = 10,
    bidirectional = 11,
    twopath = 12,
};

/// Header for the .znn file format
const FileHeader = packed struct {
    magic: [4]u8,
    version: u32,
    layer_count: u32,
};

/// Save a network to a file
/// Returns the number of bytes written
pub fn saveModel(layers: []const layer.Layer, path: []const u8) !usize {
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close();

    var writer = file.writer();
    var bytes_written: usize = 0;

    // Write header
    const header = FileHeader{
        .magic = MAGIC.*,
        .version = VERSION,
        .layer_count = @intCast(layers.len),
    };
    try writer.writeAll(std.mem.asBytes(&header));
    bytes_written += @sizeOf(FileHeader);

    // Write each layer
    for (layers) |l| {
        bytes_written += try writeLayer(writer, l);
    }

    return bytes_written;
}

/// Load a network from a file
/// Caller owns the returned layers and must deinit them
pub fn loadModel(allocator: std.mem.Allocator, path: []const u8, backend_inst: backend.Backend) !std.ArrayList(layer.Layer) {
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close();

    var reader = file.reader();

    // Read header
    var header: FileHeader = undefined;
    const header_bytes = try reader.readExact(std.mem.asBytes(&header), @sizeOf(FileHeader));
    _ = header_bytes;

    // Validate magic
    if (!std.mem.eql(u8, &header.magic, MAGIC)) {
        return error.InvalidMagic;
    }

    // Validate version
    if (header.version != VERSION) {
        return error.UnsupportedVersion;
    }

    // Read layers
    var layers = std.ArrayList(layer.Layer).init(allocator);
    errdefer {
        for (layers.items) |l| {
            l.deinit();
        }
        layers.deinit();
    }

    var i: u32 = 0;
    while (i < header.layer_count) : (i += 1) {
        const l = try readLayer(allocator, reader, backend_inst);
        try layers.append(l);
    }

    return layers;
}

/// Write a single layer to the output
fn writeLayer(writer: anytype, l: layer.Layer) !usize {
    var bytes_written: usize = 0;

    // Write layer type
    const layer_type = getLayerType(l);
    try writer.writeInt(u32, @intFromEnum(layer_type), .little);
    bytes_written += 4;

    // Write layer-specific data
    switch (l) {
        .dense => |d| {
            // Dense: input_size, output_size, activation, weights, bias
            try writer.writeInt(u32, @intCast(d.input_size), .little);
            try writer.writeInt(u32, @intCast(d.output_size), .little);
            try writer.writeInt(u32, @intFromEnum(d.act), .little);
            bytes_written += 12;

            // Write weights
            const weights_len = d.input_size * d.output_size;
            try writer.writeAll(std.mem.sliceAsBytes(d.weights.slice[0..weights_len]));
            bytes_written += weights_len * @sizeOf(f32);

            // Write bias
            try writer.writeAll(std.mem.sliceAsBytes(d.bias.slice[0..d.output_size]));
            bytes_written += d.output_size * @sizeOf(f32);
        },
        .batch_norm => |bn| {
            // BatchNorm: size, eps, momentum, gamma, beta, running_mean, running_var
            try writer.writeInt(u32, @intCast(bn.size), .little);
            try writer.writeAll(std.mem.asBytes(&bn.eps));
            try writer.writeAll(std.mem.asBytes(&bn.momentum));
            bytes_written += 4 + 8 + 8;

            // Write gamma, beta, running_mean, running_var
            try writer.writeAll(std.mem.sliceAsBytes(bn.gamma.slice[0..bn.size]));
            try writer.writeAll(std.mem.sliceAsBytes(bn.beta.slice[0..bn.size]));
            try writer.writeAll(std.mem.sliceAsBytes(bn.running_mean.slice[0..bn.size]));
            try writer.writeAll(std.mem.sliceAsBytes(bn.running_var.slice[0..bn.size]));
            bytes_written += bn.size * 4 * @sizeOf(f32);
        },
        .layer_norm => |ln| {
            // LayerNorm: size, eps, gamma, beta
            try writer.writeInt(u32, @intCast(ln.size), .little);
            try writer.writeAll(std.mem.asBytes(&ln.eps));
            bytes_written += 4 + 8;

            // Write gamma, beta
            try writer.writeAll(std.mem.sliceAsBytes(ln.gamma.slice[0..ln.size]));
            try writer.writeAll(std.mem.sliceAsBytes(ln.beta.slice[0..ln.size]));
            bytes_written += ln.size * 2 * @sizeOf(f32);
        },
        .dropout => |dr| {
            // Dropout: size, rate
            try writer.writeInt(u32, @intCast(dr.size), .little);
            try writer.writeAll(std.mem.asBytes(&dr.rate));
            bytes_written += 4 + 4;
        },
        .lstm => |lst| {
            // LSTM: input_size, hidden_size, max_seq_len, weights
            try writer.writeInt(u32, @intCast(lst.input_size), .little);
            try writer.writeInt(u32, @intCast(lst.hidden_size), .little);
            try writer.writeInt(u32, @intCast(lst.max_seq_len), .little);
            bytes_written += 12;

            // Calculate total weights size: 4 gates * (input + hidden) * hidden
            const weights_per_gate = (lst.input_size + lst.hidden_size) * lst.hidden_size;
            const total_weights = 4 * weights_per_gate;
            try writer.writeAll(std.mem.sliceAsBytes(lst.weights.slice[0..total_weights]));
            bytes_written += total_weights * @sizeOf(f32);

            // Biases: 4 * hidden_size
            const total_biases = 4 * lst.hidden_size;
            try writer.writeAll(std.mem.sliceAsBytes(lst.bias.slice[0..total_biases]));
            bytes_written += total_biases * @sizeOf(f32);
        },
        .gru => |g| {
            // GRU: input_size, hidden_size, max_seq_len, weights
            try writer.writeInt(u32, @intCast(g.input_size), .little);
            try writer.writeInt(u32, @intCast(g.hidden_size), .little);
            try writer.writeInt(u32, @intCast(g.max_seq_len), .little);
            bytes_written += 12;

            // Calculate total weights size: 3 gates * (input + hidden) * hidden
            const weights_per_gate = (g.input_size + g.hidden_size) * g.hidden_size;
            const total_weights = 3 * weights_per_gate;
            try writer.writeAll(std.mem.sliceAsBytes(g.weights.slice[0..total_weights]));
            bytes_written += total_weights * @sizeOf(f32);

            // Biases: 3 * hidden_size
            const total_biases = 3 * g.hidden_size;
            try writer.writeAll(std.mem.sliceAsBytes(g.bias.slice[0..total_biases]));
            bytes_written += total_biases * @sizeOf(f32);
        },
        .rnn => |r| {
            // RNN: input_size, hidden_size, activation, weights
            try writer.writeInt(u32, @intCast(r.input_size), .little);
            try writer.writeInt(u32, @intCast(r.hidden_size), .little);
            try writer.writeInt(u32, @intFromEnum(r.act), .little);
            bytes_written += 12;

            const weights_size = (r.input_size + r.hidden_size) * r.hidden_size;
            try writer.writeAll(std.mem.sliceAsBytes(r.weights.slice[0..weights_size]));
            bytes_written += weights_size * @sizeOf(f32);

            // Bias
            try writer.writeAll(std.mem.sliceAsBytes(r.bias.slice[0..r.hidden_size]));
            bytes_written += r.hidden_size * @sizeOf(f32);
        },
        .conv1d => |c| {
            // Conv1D: input_channels, output_channels, kernel_size, stride, weights
            try writer.writeInt(u32, @intCast(c.input_channels), .little);
            try writer.writeInt(u32, @intCast(c.output_channels), .little);
            try writer.writeInt(u32, @intCast(c.kernel_size), .little);
            try writer.writeInt(u32, @intCast(c.stride), .little);
            bytes_written += 16;

            const weights_size = c.input_channels * c.output_channels * c.kernel_size;
            try writer.writeAll(std.mem.sliceAsBytes(c.weights.slice[0..weights_size]));
            bytes_written += weights_size * @sizeOf(f32);

            // Bias
            try writer.writeAll(std.mem.sliceAsBytes(c.bias.slice[0..c.output_channels]));
            bytes_written += c.output_channels * @sizeOf(f32);
        },
        .attention => |a| {
            // Attention: d_model, num_heads, seq_len, weights
            try writer.writeInt(u32, @intCast(a.d_model), .little);
            try writer.writeInt(u32, @intCast(a.num_heads), .little);
            try writer.writeInt(u32, @intCast(a.seq_len), .little);
            bytes_written += 12;

            // W_q, W_k, W_v, W_o: each d_model * d_model
            const weights_size = a.d_model * a.d_model;
            try writer.writeAll(std.mem.sliceAsBytes(a.w_q.slice[0..weights_size]));
            try writer.writeAll(std.mem.sliceAsBytes(a.w_k.slice[0..weights_size]));
            try writer.writeAll(std.mem.sliceAsBytes(a.w_v.slice[0..weights_size]));
            try writer.writeAll(std.mem.sliceAsBytes(a.w_o.slice[0..weights_size]));
            bytes_written += 4 * weights_size * @sizeOf(f32);
        },
        .sampling => |s| {
            // Sampling: vocab_size, d_model, max_len, temperature
            try writer.writeInt(u32, @intCast(s.vocab_size), .little);
            try writer.writeInt(u32, @intCast(s.d_model), .little);
            try writer.writeInt(u32, @intCast(s.max_len), .little);
            try writer.writeAll(std.mem.asBytes(&s.temperature));
            bytes_written += 12 + 4;
        },
        .bidirectional => |b| {
            // Bidirectional: save underlying layer type and both directions
            try writer.writeInt(u32, @intFromEnum(getLayerType(.{ .rnn = b.fw_layer })), .little);
            bytes_written += 4;
            // Note: bidirectional wrapper doesn't store extra weights
            // The inner layers handle their own serialization
        },
        .twopath => |t| {
            // TwoPath: save both paths
            try writer.writeInt(u32, @intFromEnum(getLayerType(.{ .rnn = t.path1 })), .little);
            bytes_written += 4;
        },
    }

    return bytes_written;
}

/// Read a single layer from input
fn readLayer(allocator: std.mem.Allocator, reader: anytype, backend_inst: backend.Backend) !layer.Layer {
    // Read layer type
    const layer_type_int = try reader.readInt(u32, .little);
    const layer_type = std.meta.intToEnum(LayerType, layer_type_int) catch {
        return error.UnknownLayerType;
    };

    switch (layer_type) {
        .dense => {
            const input_size = try reader.readInt(u32, .little);
            const output_size = try reader.readInt(u32, .little);
            const act_int = try reader.readInt(u32, .little);
            const act = std.meta.intToEnum(activation.Activation, act_int) catch {
                return error.InvalidActivation;
            };

            const d = try layer.Dense.init(allocator, input_size, output_size, act, backend_inst);

            // Read weights
            const weights_len = input_size * output_size;
            const weights_bytes = try reader.readExact(std.mem.sliceAsBytes(d.weights.slice[0..weights_len]), weights_len * @sizeOf(f32));
            _ = weights_bytes;

            // Read bias
            const bias_bytes = try reader.readExact(std.mem.sliceAsBytes(d.bias.slice[0..output_size]), output_size * @sizeOf(f32));
            _ = bias_bytes;

            return layer.Layer{ .dense = d };
        },
        .batch_norm => {
            const size = try reader.readInt(u32, .little);
            var eps: f32 = undefined;
            var momentum: f32 = undefined;
            _ = try reader.readExact(std.mem.asBytes(&eps), @sizeOf(f32));
            _ = try reader.readExact(std.mem.asBytes(&momentum), @sizeOf(f32));

            const bn = try layer.BatchNorm.init(allocator, size, backend_inst);
            bn.eps = eps;
            bn.momentum = momentum;

            // Read gamma, beta, running_mean, running_var
            _ = try reader.readExact(std.mem.sliceAsBytes(bn.gamma.slice[0..size]), size * @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(bn.beta.slice[0..size]), size * @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(bn.running_mean.slice[0..size]), size * @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(bn.running_var.slice[0..size]), size * @sizeOf(f32));

            return layer.Layer{ .batch_norm = bn };
        },
        .layer_norm => {
            const size = try reader.readInt(u32, .little);
            var eps: f32 = undefined;
            _ = try reader.readExact(std.mem.asBytes(&eps), @sizeOf(f32));

            const ln = try layer.LayerNorm.init(allocator, size, backend_inst);
            ln.eps = eps;

            // Read gamma, beta
            _ = try reader.readExact(std.mem.sliceAsBytes(ln.gamma.slice[0..size]), size * @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(ln.beta.slice[0..size]), size * @sizeOf(f32));

            return layer.Layer{ .layer_norm = ln };
        },
        .dropout => {
            const size = try reader.readInt(u32, .little);
            var rate: f32 = undefined;
            _ = try reader.readExact(std.mem.asBytes(&rate), @sizeOf(f32));

            const dr = try layer.Dropout.init(allocator, size, rate, backend_inst);
            return layer.Layer{ .dropout = dr };
        },
        .lstm => {
            const input_size = try reader.readInt(u32, .little);
            const hidden_size = try reader.readInt(u32, .little);
            const max_seq_len = try reader.readInt(u32, .little);

            const l = try layer.LSTM.init(allocator, input_size, hidden_size, max_seq_len, backend_inst);

            // Read weights
            const weights_per_gate = (input_size + hidden_size) * hidden_size;
            const total_weights = 4 * weights_per_gate;
            _ = try reader.readExact(std.mem.sliceAsBytes(l.weights.slice[0..total_weights]), total_weights * @sizeOf(f32));

            // Read biases
            const total_biases = 4 * hidden_size;
            _ = try reader.readExact(std.mem.sliceAsBytes(l.bias.slice[0..total_biases]), total_biases * @sizeOf(f32));

            return layer.Layer{ .lstm = l };
        },
        .gru => {
            const input_size = try reader.readInt(u32, .little);
            const hidden_size = try reader.readInt(u32, .little);
            const max_seq_len = try reader.readInt(u32, .little);

            const g = try layer.GRU.init(allocator, input_size, hidden_size, max_seq_len, backend_inst);

            // Read weights
            const weights_per_gate = (input_size + hidden_size) * hidden_size;
            const total_weights = 3 * weights_per_gate;
            _ = try reader.readExact(std.mem.sliceAsBytes(g.weights.slice[0..total_weights]), total_weights * @sizeOf(f32));

            // Read biases
            const total_biases = 3 * hidden_size;
            _ = try reader.readExact(std.mem.sliceAsBytes(g.bias.slice[0..total_biases]), total_biases * @sizeOf(f32));

            return layer.Layer{ .gru = g };
        },
        .rnn => {
            const input_size = try reader.readInt(u32, .little);
            const hidden_size = try reader.readInt(u32, .little);
            const act_int = try reader.readInt(u32, .little);
            const act = std.meta.intToEnum(activation.Activation, act_int) catch {
                return error.InvalidActivation;
            };

            const r = try layer.VanillaRNN.init(allocator, input_size, hidden_size, act, backend_inst);

            // Read weights
            const weights_size = (input_size + hidden_size) * hidden_size;
            _ = try reader.readExact(std.mem.sliceAsBytes(r.weights.slice[0..weights_size]), weights_size * @sizeOf(f32));

            // Read bias
            _ = try reader.readExact(std.mem.sliceAsBytes(r.bias.slice[0..hidden_size]), hidden_size * @sizeOf(f32));

            return layer.Layer{ .rnn = r };
        },
        .conv1d => {
            const input_channels = try reader.readInt(u32, .little);
            const output_channels = try reader.readInt(u32, .little);
            const kernel_size = try reader.readInt(u32, .little);
            const stride = try reader.readInt(u32, .little);

            const c = try layer.Conv1D.init(allocator, input_channels, output_channels, kernel_size, stride, backend_inst);

            // Read weights
            const weights_size = input_channels * output_channels * kernel_size;
            _ = try reader.readExact(std.mem.sliceAsBytes(c.weights.slice[0..weights_size]), weights_size * @sizeOf(f32));

            // Read bias
            _ = try reader.readExact(std.mem.sliceAsBytes(c.bias.slice[0..output_channels]), output_channels * @sizeOf(f32));

            return layer.Layer{ .conv1d = c };
        },
        .attention => {
            const d_model = try reader.readInt(u32, .little);
            const num_heads = try reader.readInt(u32, .little);
            const seq_len = try reader.readInt(u32, .little);

            const a = try layer.Attention.init(allocator, d_model, num_heads, seq_len, backend_inst);

            // Read weights
            const weights_size = d_model * d_model;
            _ = try reader.readExact(std.mem.sliceAsBytes(a.w_q.slice[0..weights_size]), weights_size * @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(a.w_k.slice[0..weights_size]), weights_size * @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(a.w_v.slice[0..weights_size]), weights_size * @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(a.w_o.slice[0..weights_size]), weights_size * @sizeOf(f32));

            return layer.Layer{ .attention = a };
        },
        .sampling => {
            const vocab_size = try reader.readInt(u32, .little);
            const d_model = try reader.readInt(u32, .little);
            const max_len = try reader.readInt(u32, .little);
            var temperature: f32 = undefined;
            _ = try reader.readExact(std.mem.asBytes(&temperature), @sizeOf(f32));

            const s = try layer.SamplingLayer.init(allocator, vocab_size, d_model, max_len, temperature, backend_inst);
            return layer.Layer{ .sampling = s };
        },
        .bidirectional => {
            // For now, skip bidirectional layers in simple save/load
            // A full implementation would need to serialize the inner layer
            return error.UnsupportedLayer;
        },
        .twopath => {
            // For now, skip twopath layers in simple save/load
            return error.UnsupportedLayer;
        },
    }
}

/// Get the layer type enum for a given layer
fn getLayerType(l: layer.Layer) LayerType {
    return switch (l) {
        .dense => .dense,
        .rnn => .rnn,
        .lstm => .lstm,
        .gru => .gru,
        .sampling => .sampling,
        .conv1d => .conv1d,
        .layer_norm => .layer_norm,
        .batch_norm => .batch_norm,
        .dropout => .dropout,
        .attention => .attention,
        .bidirectional => .bidirectional,
        .twopath => .twopath,
    };
}

/// Read exact number of bytes from reader
fn readExact(reader: anytype, buffer: []u8, count: usize) !usize {
    const bytes_read = try reader.readAtLeast(buffer, count);
    if (bytes_read < count) {
        return error.ReadError;
    }
    return bytes_read;
}

// Error types
pub const SerializationError = error{
    InvalidMagic,
    UnsupportedVersion,
    UnknownLayerType,
    InvalidActivation,
    UnsupportedLayer,
    ReadError,
    WriteError,
};

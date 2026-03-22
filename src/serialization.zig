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

// =============================================================================
// Security Helper Functions
// =============================================================================

/// Safely calculate buffer size with overflow checking
/// Uses std.math.mul to detect overflow in multiplication chains
fn calculateBufferSize(num_elements: usize, element_size: usize) !usize {
    return std.math.mul(usize, num_elements, element_size);
}

/// Safely calculate buffer size for 2D arrays (rows * cols * element_size)
/// Returns error.Overflow if multiplication would overflow
fn calculateBufferSize2D(rows: usize, cols: usize, element_size: usize) !usize {
    const num_elements = try std.math.mul(usize, rows, cols);
    return std.math.mul(usize, num_elements, element_size);
}

/// Safely calculate buffer size for 3D arrays (d1 * d2 * d3 * element_size)
/// Returns error.Overflow if multiplication would overflow
fn calculateBufferSize3D(d1: usize, d2: usize, d3: usize, element_size: usize) !usize {
    const d1_d2 = try std.math.mul(usize, d1, d2);
    const num_elements = try std.math.mul(usize, d1_d2, d3);
    return std.math.mul(usize, num_elements, element_size);
}

/// Safely multiply two usize values with overflow checking
fn safeMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b);
}

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
    // SECURITY FIX: Validate layer count before @intCast
    if (layers.len > std.math.maxInt(u32)) {
        return error.LayerCountOverflow;
    }
    const header = FileHeader{
        .magic = MAGIC.*,
        .version = VERSION,
        .layer_count = @intCast(layers.len),
    };
    try writer.writeAll(std.mem.asBytes(&header));
    bytes_written += @sizeOf(FileHeader);

    // Write each layer
    for (layers) |l| {
        const layer_bytes = try writeLayer(writer, l);
        bytes_written = try std.math.add(usize, bytes_written, layer_bytes);
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
            // SECURITY FIX: Use overflow-checked size calculations
            const weights_len = try safeMul(d.input_size, d.output_size);
            const weights_bytes = try calculateBufferSize(weights_len, @sizeOf(f32));
            try writer.writeAll(std.mem.sliceAsBytes(d.weights.slice[0..weights_len]));
            bytes_written = try std.math.add(usize, bytes_written, weights_bytes);

            // Write bias
            const bias_bytes = try calculateBufferSize(d.output_size, @sizeOf(f32));
            try writer.writeAll(std.mem.sliceAsBytes(d.bias.slice[0..d.output_size]));
            bytes_written = try std.math.add(usize, bytes_written, bias_bytes);
        },
        .batch_norm => |bn| {
            // BatchNorm: size, eps, momentum, gamma, beta, running_mean, running_var
            try writer.writeInt(u32, @intCast(bn.size), .little);
            try writer.writeAll(std.mem.asBytes(&bn.eps));
            try writer.writeAll(std.mem.asBytes(&bn.momentum));
            bytes_written = try std.math.add(usize, bytes_written, 4 + 8 + 8);

            // Write gamma, beta, running_mean, running_var
            // SECURITY FIX: Use overflow-checked size calculations
            try writer.writeAll(std.mem.sliceAsBytes(bn.gamma.slice[0..bn.size]));
            try writer.writeAll(std.mem.sliceAsBytes(bn.beta.slice[0..bn.size]));
            try writer.writeAll(std.mem.sliceAsBytes(bn.running_mean.slice[0..bn.size]));
            try writer.writeAll(std.mem.sliceAsBytes(bn.running_var.slice[0..bn.size]));
            const bn_param_bytes = try calculateBufferSize2D(bn.size, 4, @sizeOf(f32));
            bytes_written = try std.math.add(usize, bytes_written, bn_param_bytes);
        },
        .layer_norm => |ln| {
            // LayerNorm: size, eps, gamma, beta
            try writer.writeInt(u32, @intCast(ln.size), .little);
            try writer.writeAll(std.mem.asBytes(&ln.eps));
            bytes_written = try std.math.add(usize, bytes_written, 4 + 8);

            // Write gamma, beta
            // SECURITY FIX: Use overflow-checked size calculations
            try writer.writeAll(std.mem.sliceAsBytes(ln.gamma.slice[0..ln.size]));
            try writer.writeAll(std.mem.sliceAsBytes(ln.beta.slice[0..ln.size]));
            const ln_param_bytes = try calculateBufferSize2D(ln.size, 2, @sizeOf(f32));
            bytes_written = try std.math.add(usize, bytes_written, ln_param_bytes);
        },
        .dropout => |dr| {
            // Dropout: size, rate
            try writer.writeInt(u32, @intCast(dr.size), .little);
            try writer.writeAll(std.mem.asBytes(&dr.rate));
            bytes_written = try std.math.add(usize, bytes_written, 4 + 4);
        },
        .lstm => |lst| {
            // LSTM: input_size, hidden_size, max_seq_len, weights
            try writer.writeInt(u32, @intCast(lst.input_size), .little);
            try writer.writeInt(u32, @intCast(lst.hidden_size), .little);
            try writer.writeInt(u32, @intCast(lst.max_seq_len), .little);
            bytes_written = try std.math.add(usize, bytes_written, 12);

            // Calculate total weights size: 4 gates * (input + hidden) * hidden
            // SECURITY FIX: Use overflow-checked size calculations
            const lstm_input_hidden = try std.math.add(usize, lst.input_size, lst.hidden_size);
            const weights_per_gate = try safeMul(lstm_input_hidden, lst.hidden_size);
            const total_weights = try safeMul(4, weights_per_gate);
            try writer.writeAll(std.mem.sliceAsBytes(lst.weights.slice[0..total_weights]));
            const lstm_weights_bytes = try calculateBufferSize(total_weights, @sizeOf(f32));
            bytes_written = try std.math.add(usize, bytes_written, lstm_weights_bytes);

            // Biases: 4 * hidden_size
            const total_biases = try safeMul(4, lst.hidden_size);
            try writer.writeAll(std.mem.sliceAsBytes(lst.bias.slice[0..total_biases]));
            const lstm_bias_bytes = try calculateBufferSize(total_biases, @sizeOf(f32));
            bytes_written = try std.math.add(usize, bytes_written, lstm_bias_bytes);
        },
        .gru => |g| {
            // GRU: input_size, hidden_size, max_seq_len, weights
            try writer.writeInt(u32, @intCast(g.input_size), .little);
            try writer.writeInt(u32, @intCast(g.hidden_size), .little);
            try writer.writeInt(u32, @intCast(g.max_seq_len), .little);
            bytes_written = try std.math.add(usize, bytes_written, 12);

            // Calculate total weights size: 3 gates * (input + hidden) * hidden
            // SECURITY FIX: Use overflow-checked size calculations
            const gru_input_hidden = try std.math.add(usize, g.input_size, g.hidden_size);
            const weights_per_gate = try safeMul(gru_input_hidden, g.hidden_size);
            const total_weights = try safeMul(3, weights_per_gate);
            try writer.writeAll(std.mem.sliceAsBytes(g.weights.slice[0..total_weights]));
            const gru_weights_bytes = try calculateBufferSize(total_weights, @sizeOf(f32));
            bytes_written = try std.math.add(usize, bytes_written, gru_weights_bytes);

            // Biases: 3 * hidden_size
            const total_biases = try safeMul(3, g.hidden_size);
            try writer.writeAll(std.mem.sliceAsBytes(g.bias.slice[0..total_biases]));
            const gru_bias_bytes = try calculateBufferSize(total_biases, @sizeOf(f32));
            bytes_written = try std.math.add(usize, bytes_written, gru_bias_bytes);
        },
        .rnn => |r| {
            // RNN: input_size, hidden_size, activation, weights
            try writer.writeInt(u32, @intCast(r.input_size), .little);
            try writer.writeInt(u32, @intCast(r.hidden_size), .little);
            try writer.writeInt(u32, @intFromEnum(r.act), .little);
            bytes_written = try std.math.add(usize, bytes_written, 12);

            // SECURITY FIX: Use overflow-checked size calculations
            const rnn_input_hidden = try std.math.add(usize, r.input_size, r.hidden_size);
            const weights_size = try safeMul(rnn_input_hidden, r.hidden_size);
            try writer.writeAll(std.mem.sliceAsBytes(r.weights.slice[0..weights_size]));
            const rnn_weights_bytes = try calculateBufferSize(weights_size, @sizeOf(f32));
            bytes_written = try std.math.add(usize, bytes_written, rnn_weights_bytes);

            // Bias
            try writer.writeAll(std.mem.sliceAsBytes(r.bias.slice[0..r.hidden_size]));
            const rnn_bias_bytes = try calculateBufferSize(r.hidden_size, @sizeOf(f32));
            bytes_written = try std.math.add(usize, bytes_written, rnn_bias_bytes);
        },
        .conv1d => |c| {
            // Conv1D: input_channels, output_channels, kernel_size, stride, weights
            try writer.writeInt(u32, @intCast(c.input_channels), .little);
            try writer.writeInt(u32, @intCast(c.output_channels), .little);
            try writer.writeInt(u32, @intCast(c.kernel_size), .little);
            try writer.writeInt(u32, @intCast(c.stride), .little);
            bytes_written = try std.math.add(usize, bytes_written, 16);

            // SECURITY FIX: Use overflow-checked size calculations
            const weights_size = try calculateBufferSize3D(c.input_channels, c.output_channels, c.kernel_size, 1);
            try writer.writeAll(std.mem.sliceAsBytes(c.weights.slice[0..weights_size]));
            const conv_weights_bytes = try calculateBufferSize(weights_size, @sizeOf(f32));
            bytes_written = try std.math.add(usize, bytes_written, conv_weights_bytes);

            // Bias
            try writer.writeAll(std.mem.sliceAsBytes(c.bias.slice[0..c.output_channels]));
            const conv_bias_bytes = try calculateBufferSize(c.output_channels, @sizeOf(f32));
            bytes_written = try std.math.add(usize, bytes_written, conv_bias_bytes);
        },
        .attention => |a| {
            // Attention: d_model, num_heads, seq_len, weights
            try writer.writeInt(u32, @intCast(a.d_model), .little);
            try writer.writeInt(u32, @intCast(a.num_heads), .little);
            try writer.writeInt(u32, @intCast(a.seq_len), .little);
            bytes_written = try std.math.add(usize, bytes_written, 12);

            // W_q, W_k, W_v, W_o: each d_model * d_model
            // SECURITY FIX: Use overflow-checked size calculations
            const weights_size = try safeMul(a.d_model, a.d_model);
            try writer.writeAll(std.mem.sliceAsBytes(a.w_q.slice[0..weights_size]));
            try writer.writeAll(std.mem.sliceAsBytes(a.w_k.slice[0..weights_size]));
            try writer.writeAll(std.mem.sliceAsBytes(a.w_v.slice[0..weights_size]));
            try writer.writeAll(std.mem.sliceAsBytes(a.w_o.slice[0..weights_size]));
            const attention_weights_bytes = try calculateBufferSize2D(4, weights_size, @sizeOf(f32));
            bytes_written = try std.math.add(usize, bytes_written, attention_weights_bytes);
        },
        .sampling => |s| {
            // Sampling: vocab_size, d_model, max_len, temperature
            try writer.writeInt(u32, @intCast(s.vocab_size), .little);
            try writer.writeInt(u32, @intCast(s.d_model), .little);
            try writer.writeInt(u32, @intCast(s.max_len), .little);
            try writer.writeAll(std.mem.asBytes(&s.temperature));
            bytes_written = try std.math.add(usize, bytes_written, 12 + 4);
        },
        .bidirectional => |b| {
            // Bidirectional: save underlying layer type and both directions
            try writer.writeInt(u32, @intFromEnum(getLayerType(.{ .rnn = b.fw_layer })), .little);
            bytes_written = try std.math.add(usize, bytes_written, 4);
            // Note: bidirectional wrapper doesn't store extra weights
            // The inner layers handle their own serialization
        },
        .twopath => |t| {
            // TwoPath: save both paths
            try writer.writeInt(u32, @intFromEnum(getLayerType(.{ .rnn = t.path1 })), .little);
            bytes_written = try std.math.add(usize, bytes_written, 4);
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
            // SECURITY FIX: Use overflow-checked size calculations
            const weights_len = try safeMul(input_size, output_size);
            const weights_bytes_count = try calculateBufferSize(weights_len, @sizeOf(f32));
            const weights_bytes = try reader.readExact(std.mem.sliceAsBytes(d.weights.slice[0..weights_len]), weights_bytes_count);
            _ = weights_bytes;

            // Read bias
            const bias_bytes_count = try calculateBufferSize(output_size, @sizeOf(f32));
            const bias_bytes = try reader.readExact(std.mem.sliceAsBytes(d.bias.slice[0..output_size]), bias_bytes_count);
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
            // SECURITY FIX: Use overflow-checked size calculations
            const bn_param_size = try calculateBufferSize(size, @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(bn.gamma.slice[0..size]), bn_param_size);
            _ = try reader.readExact(std.mem.sliceAsBytes(bn.beta.slice[0..size]), bn_param_size);
            _ = try reader.readExact(std.mem.sliceAsBytes(bn.running_mean.slice[0..size]), bn_param_size);
            _ = try reader.readExact(std.mem.sliceAsBytes(bn.running_var.slice[0..size]), bn_param_size);

            return layer.Layer{ .batch_norm = bn };
        },
        .layer_norm => {
            const size = try reader.readInt(u32, .little);
            var eps: f32 = undefined;
            _ = try reader.readExact(std.mem.asBytes(&eps), @sizeOf(f32));

            const ln = try layer.LayerNorm.init(allocator, size, backend_inst);
            ln.eps = eps;

            // Read gamma, beta
            // SECURITY FIX: Use overflow-checked size calculations
            const ln_param_size = try calculateBufferSize(size, @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(ln.gamma.slice[0..size]), ln_param_size);
            _ = try reader.readExact(std.mem.sliceAsBytes(ln.beta.slice[0..size]), ln_param_size);

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
            // SECURITY FIX: Use overflow-checked size calculations
            const lstm_input_hidden = try std.math.add(usize, input_size, hidden_size);
            const weights_per_gate = try safeMul(lstm_input_hidden, hidden_size);
            const total_weights = try safeMul(4, weights_per_gate);
            const lstm_weights_bytes = try calculateBufferSize(total_weights, @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(l.weights.slice[0..total_weights]), lstm_weights_bytes);

            // Read biases
            const total_biases = try safeMul(4, hidden_size);
            const lstm_bias_bytes = try calculateBufferSize(total_biases, @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(l.bias.slice[0..total_biases]), lstm_bias_bytes);

            return layer.Layer{ .lstm = l };
        },
        .gru => {
            const input_size = try reader.readInt(u32, .little);
            const hidden_size = try reader.readInt(u32, .little);
            const max_seq_len = try reader.readInt(u32, .little);

            const g = try layer.GRU.init(allocator, input_size, hidden_size, max_seq_len, backend_inst);

            // Read weights
            // SECURITY FIX: Use overflow-checked size calculations
            const gru_input_hidden = try std.math.add(usize, input_size, hidden_size);
            const weights_per_gate = try safeMul(gru_input_hidden, hidden_size);
            const total_weights = try safeMul(3, weights_per_gate);
            const gru_weights_bytes = try calculateBufferSize(total_weights, @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(g.weights.slice[0..total_weights]), gru_weights_bytes);

            // Read biases
            const total_biases = try safeMul(3, hidden_size);
            const gru_bias_bytes = try calculateBufferSize(total_biases, @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(g.bias.slice[0..total_biases]), gru_bias_bytes);

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
            // SECURITY FIX: Use overflow-checked size calculations
            const rnn_input_hidden = try std.math.add(usize, input_size, hidden_size);
            const weights_size = try safeMul(rnn_input_hidden, hidden_size);
            const rnn_weights_bytes = try calculateBufferSize(weights_size, @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(r.weights.slice[0..weights_size]), rnn_weights_bytes);

            // Read bias
            const rnn_bias_bytes = try calculateBufferSize(hidden_size, @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(r.bias.slice[0..hidden_size]), rnn_bias_bytes);

            return layer.Layer{ .rnn = r };
        },
        .conv1d => {
            const input_channels = try reader.readInt(u32, .little);
            const output_channels = try reader.readInt(u32, .little);
            const kernel_size = try reader.readInt(u32, .little);
            const stride = try reader.readInt(u32, .little);

            const c = try layer.Conv1D.init(allocator, input_channels, output_channels, kernel_size, stride, backend_inst);

            // Read weights
            // SECURITY FIX: Use overflow-checked size calculations
            const weights_size = try calculateBufferSize3D(input_channels, output_channels, kernel_size, 1);
            const conv_weights_bytes = try calculateBufferSize(weights_size, @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(c.weights.slice[0..weights_size]), conv_weights_bytes);

            // Read bias
            const conv_bias_bytes = try calculateBufferSize(output_channels, @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(c.bias.slice[0..output_channels]), conv_bias_bytes);

            return layer.Layer{ .conv1d = c };
        },
        .attention => {
            const d_model = try reader.readInt(u32, .little);
            const num_heads = try reader.readInt(u32, .little);
            const seq_len = try reader.readInt(u32, .little);

            const a = try layer.Attention.init(allocator, d_model, num_heads, seq_len, backend_inst);

            // Read weights
            // SECURITY FIX: Use overflow-checked size calculations
            const weights_size = try safeMul(d_model, d_model);
            const attention_weights_bytes = try calculateBufferSize(weights_size, @sizeOf(f32));
            _ = try reader.readExact(std.mem.sliceAsBytes(a.w_q.slice[0..weights_size]), attention_weights_bytes);
            _ = try reader.readExact(std.mem.sliceAsBytes(a.w_k.slice[0..weights_size]), attention_weights_bytes);
            _ = try reader.readExact(std.mem.sliceAsBytes(a.w_v.slice[0..weights_size]), attention_weights_bytes);
            _ = try reader.readExact(std.mem.sliceAsBytes(a.w_o.slice[0..weights_size]), attention_weights_bytes);

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
    LayerCountOverflow,
};

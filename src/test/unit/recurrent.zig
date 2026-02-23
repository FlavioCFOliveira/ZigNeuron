/// Unit tests for recurrent layers
const std = @import("std");
const testing = std.testing;

const zn = @import("ZigNeuron");
const activation = zn.activation;
const backend = zn.backend;
const recurrent = zn.recurrent;

test "recurrent vanilla_rnn forward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var rnn = try recurrent.VanillaRNN.init(allocator, 2, 3, .tanh, cpu_backend);
    defer rnn.deinit();

    // Constant weights for predictability
    @memset(rnn.weights_ih.slice, 0.1);
    @memset(rnn.weights_hh.slice, 0.1);
    @memset(rnn.bias.slice, 0.0);

    const input = [_]f32{ 1.0, 1.0, 0.5, 0.5 }; // seq_len = 2, input_size = 2
    var output = [_]f32{0} ** 6; // seq_len = 2, hidden_size = 3

    try rnn.forward(&input, null, &output, null);

    // Verify first step: h_1 = tanh(W_ih * x_1 + W_hh * h_0 + b)
    // x_1 = [1, 1], h_0 = [0, 0, 0]
    // W_ih * x_1 = [0.2, 0.2, 0.2]
    // h_1 = tanh([0.2, 0.2, 0.2]) = [0.19737, 0.19737, 0.19737]
    try testing.expect(output[0] > 0.19 and output[0] < 0.20);
    try testing.expect(output[1] > 0.19 and output[1] < 0.20);
    try testing.expect(output[2] > 0.19 and output[2] < 0.20);
}

test "recurrent vanilla_rnn backward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var rnn = try recurrent.VanillaRNN.init(allocator, 2, 2, .tanh, cpu_backend);
    defer rnn.deinit();

    const input = [_]f32{ 1.0, 1.0 }; // seq_len = 1, input_size = 2
    const h_states = [_]f32{ 0.5, 0.5 };
    const grad_output = [_]f32{ 1.0, 1.0 };
    var grad_input = [_]f32{0, 0};

    try rnn.backward(&input, null, &grad_output, null, &grad_input, null, &h_states, null);

    // Verify gradients are accumulated
    var sum: f32 = 0;
    for (rnn.grad_weights_ih.slice) |g| sum += @abs(g);
    try testing.expect(sum > 0);
}

test "recurrent lstm forward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var lstm = try recurrent.LSTM.init(allocator, 2, 2, 5, cpu_backend);
    defer lstm.deinit();

    // Constant weights for predictability
    @memset(lstm.weights_ih.slice, 0.1);
    @memset(lstm.weights_hh.slice, 0.1);
    @memset(lstm.bias.slice, 0.0);

    const input = [_]f32{ 1.0, 1.0 }; // seq_len = 1, input_size = 2
    var output = [_]f32{0, 0}; // seq_len = 1, hidden_size = 2

    try lstm.forward(&input, null, &output, null);

    // Verify output is reasonable (between -1 and 1 due to tanh/sigmoid)
    try testing.expect(output[0] >= -1.0 and output[0] <= 1.0);
    try testing.expect(output[1] >= -1.0 and output[1] <= 1.0);
}

test "recurrent lstm backward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var lstm = try recurrent.LSTM.init(allocator, 2, 2, 5, cpu_backend);
    defer lstm.deinit();

    const input = [_]f32{ 1.0, 1.0 }; // seq_len = 1, input_size = 2
    var output = [_]f32{ 0.5, 0.5 };
    const grad_output = [_]f32{ 1.0, 1.0 };
    var grad_input = [_]f32{0, 0};

    // Run forward first to populate internal states (cell_states, gate_activations)
    try lstm.forward(&input, null, &output, null);

    try lstm.backward(&input, null, &grad_output, null, &grad_input, null, &output, null);

    // Verify gradients are accumulated
    var sum: f32 = 0;
    for (lstm.grad_weights_ih.slice) |g| sum += @abs(g);
    try testing.expect(sum > 0);
}

test "recurrent gru forward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var gru = try recurrent.GRU.init(allocator, 2, 2, 5, cpu_backend);
    defer gru.deinit();

    // Constant weights for predictability
    @memset(gru.weights_ih.slice, 0.1);
    @memset(gru.weights_hh.slice, 0.1);
    @memset(gru.bias.slice, 0.0);

    const input = [_]f32{ 1.0, 1.0 }; // seq_len = 1, input_size = 2
    var output = [_]f32{0, 0}; // seq_len = 1, hidden_size = 2

    try gru.forward(&input, null, &output, null);

    // Verify output is reasonable (between -1 and 1 due to tanh)
    try testing.expect(output[0] >= -1.0 and output[0] <= 1.0);
    try testing.expect(output[1] >= -1.0 and output[1] <= 1.0);
}

test "recurrent gru backward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var gru = try recurrent.GRU.init(allocator, 2, 2, 5, cpu_backend);
    defer gru.deinit();

    const input = [_]f32{ 1.0, 1.0 }; // seq_len = 1, input_size = 2
    var output = [_]f32{ 0.5, 0.5 };
    const grad_output = [_]f32{ 1.0, 1.0 };
    var grad_input = [_]f32{0, 0};

    // Run forward first to populate internal states
    try gru.forward(&input, null, &output, null);

    try gru.backward(&input, null, &grad_output, null, &grad_input, null, &output, null);

    // Verify gradients are accumulated
    var sum: f32 = 0;
    for (gru.grad_weights_ih.slice) |g| sum += @abs(g);
    try testing.expect(sum > 0);
}

test "recurrent bidirectional lstm forward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var bi = try recurrent.Bidirectional.init(allocator, .lstm, 2, 2, 5, .tanh, cpu_backend);
    defer bi.deinit();

    const input = [_]f32{ 1.0, 1.0, 0.5, 0.5 }; // seq_len = 2, input_size = 2
    var output = [_]f32{0} ** 8; // seq_len = 2, output_size = 2 * hidden_size = 4

    try bi.forward(&input, null, &output, null);

    // Verify output size and reasonable values
    try testing.expect(output[0] >= -1.0 and output[0] <= 1.0);
}

test "recurrent bidirectional lstm backward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var bi = try recurrent.Bidirectional.init(allocator, .lstm, 2, 2, 5, .tanh, cpu_backend);
    defer bi.deinit();

    const input = [_]f32{ 1.0, 1.0 }; // seq_len = 1
    var output = [_]f32{ 0.5, 0.5, 0.5, 0.5 }; // seq_len = 1, hidden * 2 = 4
    const grad_output = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var grad_input = [_]f32{0, 0};

    // Run forward first
    try bi.forward(&input, null, &output, null);

    try bi.backward(&input, null, &grad_output, null, &grad_input, null, &output, null);

    // Verify gradients
    var sum_fw: f32 = 0;
    for (bi.fw_layer.lstm.grad_weights_ih.slice) |g| sum_fw += @abs(g);
    try testing.expect(sum_fw > 0);

    var sum_bw: f32 = 0;
    for (bi.bw_layer.lstm.grad_weights_ih.slice) |g| sum_bw += @abs(g);
    try testing.expect(sum_bw > 0);
}

test "recurrent twopath forward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var tp = try recurrent.TwoPath.init(allocator, .lstm, 2, .gru, 3, 2, 5, .tanh, cpu_backend);
    defer tp.deinit();

    const input = [_]f32{ 1.0, 1.0 }; // seq_len = 1, input_size = 2
    var output = [_]f32{0} ** 5; // seq_len = 1, hidden1 + hidden2 = 2 + 3 = 5

    try tp.forward(&input, null, &output, null);

    try testing.expect(output[0] >= -1.0 and output[0] <= 1.0);
}

test "recurrent twopath backward" {
    const allocator = testing.allocator;
    var cpu_backend = try backend.Backend.init(allocator);
    cpu_backend.type = .cpu;
    defer cpu_backend.deinit();

    var tp = try recurrent.TwoPath.init(allocator, .lstm, 2, .gru, 2, 2, 5, .tanh, cpu_backend);
    defer tp.deinit();

    const input = [_]f32{ 1.0, 1.0 }; // seq_len = 1
    var output = [_]f32{ 0.5, 0.5, 0.5, 0.5 }; // seq_len = 1, 2 + 2 = 4
    const grad_output = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var grad_input = [_]f32{0, 0};

    try tp.forward(&input, null, &output, null);
    try tp.backward(&input, null, &grad_output, null, &grad_input, null, &output, null);

    // Verify gradients
    var sum1: f32 = 0;
    for (tp.path1.lstm.grad_weights_ih.slice) |g| sum1 += @abs(g);
    try testing.expect(sum1 > 0);

    var sum2: f32 = 0;
    for (tp.path2.gru.grad_weights_ih.slice) |g| sum2 += @abs(g);
    try testing.expect(sum2 > 0);
}

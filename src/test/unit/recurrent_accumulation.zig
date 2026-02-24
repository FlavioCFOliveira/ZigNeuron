const std = @import("std");
const zn = @import("ZigNeuron");

test "RNN Gradient Accumulation" {
    const allocator = std.testing.allocator;
    const backend = try zn.backend.Backend.init(allocator);
    defer {
        var mutable_backend = backend;
        mutable_backend.deinit();
    }

    const input_size = 1;
    const hidden_size = 1;
    const rnn = try zn.recurrent.VanillaRNN.init(allocator, input_size, hidden_size, .tanh, backend);
    defer rnn.deinit();

    // Set weights to known values
    rnn.weights_ih.slice[0] = 1.0;
    rnn.weights_hh.slice[0] = 1.0;
    rnn.bias.slice[0] = 0.0;
    try backend.copyData(rnn.weights_ih.slice, null, rnn.weights_ih.slice, rnn.weights_ih.getMtlBuffer());
    try backend.copyData(rnn.weights_hh.slice, null, rnn.weights_hh.slice, rnn.weights_hh.getMtlBuffer());
    try backend.copyData(rnn.bias.slice, null, rnn.bias.slice, rnn.bias.getMtlBuffer());

    // Sequence length = 2
    const input = [_]f32{ 1.0, 1.0 };
    const h_states = [_]f32{ 0.7616, 0.9415 }; // tanh(1*1), tanh(1*1 + 1*0.7616)
    const grad_output = [_]f32{ 1.0, 1.0 };

    var grad_input = [_]f32{ 0.0, 0.0 };

    // Clear gradients
    @memset(rnn.grad_weights_ih.slice, 0);
    @memset(rnn.grad_weights_hh.slice, 0);
    @memset(rnn.grad_bias.slice, 0);
    try backend.copyData(rnn.grad_weights_ih.slice, null, rnn.grad_weights_ih.slice, rnn.grad_weights_ih.getMtlBuffer());
    try backend.copyData(rnn.grad_weights_hh.slice, null, rnn.grad_weights_hh.slice, rnn.grad_weights_hh.getMtlBuffer());
    try backend.copyData(rnn.grad_bias.slice, null, rnn.grad_bias.slice, rnn.grad_bias.getMtlBuffer());

    try rnn.backward(&input, null, &grad_output, null, &grad_input, null, &h_states, null);

    // Sync back
    try backend.copyData(rnn.grad_weights_ih.slice, rnn.grad_weights_ih.getMtlBuffer(), rnn.grad_weights_ih.slice, null);

    // Expected (accumulated): ~0.5813
    // Expected (overwritten): ~0.4677
    try std.testing.expect(rnn.grad_weights_ih.slice[0] > 0.5);
}

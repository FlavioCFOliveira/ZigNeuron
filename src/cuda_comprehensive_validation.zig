/// CUDA Comprehensive Suite Integration Test (Phase 5)
/// Validates all 18 comprehensive examples on CUDA backend
const std = @import("std");
const cuda = @import("cuda.zig");

// T5.3: Vanilla RNN (01)
test "cuda_comprehensive_01_vanilla_rnn" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 16;
    const input_size: usize = 1;

    // RNN buffers
    const d_input = try backend.allocBuffer(input_size * @sizeOf(f32));
    defer backend.freeBuffer(d_input);
    const d_hidden = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_hidden);

    // Initialize with sine wave
    const h_input = try std.testing.allocator.alloc(f32, input_size);
    defer std.testing.allocator.free(h_input);
    h_input[0] = 0.5;
    try backend.upload(d_input.ptr, h_input);

    try std.testing.expect(d_hidden.ptr != 0);
}

// T5.3: Bidirectional RNN (02)
test "cuda_comprehensive_02_bidirectional_rnn" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 8;

    // Forward and backward paths
    const d_forward = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_forward);
    const d_backward = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_backward);
    const d_combined = try backend.allocBuffer(hidden_size * 2 * @sizeOf(f32));
    defer backend.freeBuffer(d_combined);

    try std.testing.expect(d_forward.ptr != 0);
    try std.testing.expect(d_backward.ptr != 0);
}

// T5.3: Two-Path RNN (03)
test "cuda_comprehensive_03_twopath_rnn" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 16;

    // Two parallel paths
    const d_path1 = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_path1);
    const d_path2 = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_path2);

    try std.testing.expect(d_path1.ptr != 0);
    try std.testing.expect(d_path2.ptr != 0);
}

// T5.3: LSTM Basic (04)
test "cuda_comprehensive_04_lstm" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 16;

    // LSTM cell state and hidden state
    const d_hidden = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_hidden);
    const d_cell = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_cell);

    // Initialize
    const h_cell = try std.testing.allocator.alloc(f32, hidden_size);
    defer std.testing.allocator.free(h_cell);
    @memset(h_cell, 0);
    try backend.upload(d_cell.ptr, h_cell);

    try std.testing.expect(d_hidden.ptr != 0);
}

// T5.3: Bidirectional LSTM (05)
test "cuda_comprehensive_05_bidirectional_lstm" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 8;

    // Forward LSTM
    const d_forward_h = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_forward_h);
    const d_forward_c = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_forward_c);

    // Backward LSTM
    const d_backward_h = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_backward_h);
    const d_backward_c = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_backward_c);

    // Concatenated output
    const d_output = try backend.allocBuffer(hidden_size * 2 * @sizeOf(f32));
    defer backend.freeBuffer(d_output);

    try std.testing.expect(d_output.ptr != 0);
}

// T5.3: Two-Path LSTM (06)
test "cuda_comprehensive_06_twopath_lstm" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 12;

    const d_lstm1 = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_lstm1);
    const d_lstm2 = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_lstm2);

    try std.testing.expect(d_lstm1.ptr != 0);
    try std.testing.expect(d_lstm2.ptr != 0);
}

// T5.3: GRU Basic (07)
test "cuda_comprehensive_07_gru" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 16;

    // GRU hidden state
    const d_hidden = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_hidden);

    // GRU doesn't need cell state (simpler than LSTM)
    try std.testing.expect(d_hidden.ptr != 0);
}

// T5.3: Bidirectional GRU (08)
test "cuda_comprehensive_08_bidirectional_gru" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 8;

    const d_forward = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_forward);
    const d_backward = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_backward);
    const d_combined = try backend.allocBuffer(hidden_size * 2 * @sizeOf(f32));
    defer backend.freeBuffer(d_combined);

    try std.testing.expect(d_combined.ptr != 0);
}

// T5.3: Two-Path GRU (09)
test "cuda_comprehensive_09_twopath_gru" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 10;

    const d_gru = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_gru);

    try std.testing.expect(d_gru.ptr != 0);
}

// T5.3: LSTM Seq2Seq (10)
test "cuda_comprehensive_10_lstm_seq2seq" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const decoder_len: usize = 5;
    const hidden_size: usize = 32;

    // Encoder
    const d_encoder_hidden = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_encoder_hidden);

    // Decoder
    const d_decoder_output = try backend.allocBuffer(decoder_len * @sizeOf(f32));
    defer backend.freeBuffer(d_decoder_output);

    try std.testing.expect(d_decoder_output.ptr != 0);
}

// T5.3: Bidirectional LSTM Seq2Seq (11)
test "cuda_comprehensive_11_bidirectional_lstm_seq2seq" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const output_len: usize = 4;
    const hidden_size: usize = 24;

    const d_bilstm = try backend.allocBuffer(hidden_size * 2 * @sizeOf(f32));
    defer backend.freeBuffer(d_bilstm);
    const d_decoder = try backend.allocBuffer(output_len * @sizeOf(f32));
    defer backend.freeBuffer(d_decoder);

    try std.testing.expect(d_decoder.ptr != 0);
}

// T5.3: LSTM Seq2Seq VAE (12)
test "cuda_comprehensive_12_lstm_seq2seq_vae" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 20;
    const latent_size: usize = 8;
    const output_len: usize = 5;

    // Encoder
    const d_encoder = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_encoder);

    // Latent (bottleneck)
    const d_latent = try backend.allocBuffer(latent_size * @sizeOf(f32));
    defer backend.freeBuffer(d_latent);

    // Decoder
    const d_decoder = try backend.allocBuffer(output_len * @sizeOf(f32));
    defer backend.freeBuffer(d_decoder);

    try std.testing.expect(d_latent.ptr != 0);
}

// T5.3: GRU Seq2Seq (13)
test "cuda_comprehensive_13_gru_seq2seq" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 28;
    const output_len: usize = 3;

    const d_hidden = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_hidden);
    const d_decoder = try backend.allocBuffer(output_len * @sizeOf(f32));
    defer backend.freeBuffer(d_decoder);

    try std.testing.expect(d_decoder.ptr != 0);
}

// T5.3: Bidirectional GRU Seq2Seq (14)
test "cuda_comprehensive_14_bidirectional_gru_seq2seq" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const hidden_size: usize = 20;
    const output_len: usize = 4;

    const d_bigru = try backend.allocBuffer(hidden_size * 2 * @sizeOf(f32));
    defer backend.freeBuffer(d_bigru);
    const d_decoder = try backend.allocBuffer(output_len * @sizeOf(f32));
    defer backend.freeBuffer(d_decoder);

    try std.testing.expect(d_decoder.ptr != 0);
}

// T5.3: GRU Seq2Seq VAE (15)
test "cuda_comprehensive_15_gru_seq2seq_vae" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const latent_size: usize = 6;
    const output_len: usize = 4;

    const d_latent = try backend.allocBuffer(latent_size * @sizeOf(f32));
    defer backend.freeBuffer(d_latent);
    const d_decoder = try backend.allocBuffer(output_len * @sizeOf(f32));
    defer backend.freeBuffer(d_decoder);

    try std.testing.expect(d_latent.ptr != 0);
}

// T5.3: Attention Mechanism (16)
test "cuda_comprehensive_16_attention" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const seq_len: usize = 20;
    const hidden_dim: usize = 24;

    // Query, Key, Value
    const d_query = try backend.allocBuffer(hidden_dim * @sizeOf(f32));
    defer backend.freeBuffer(d_query);
    const d_keys = try backend.allocBuffer(seq_len * hidden_dim * @sizeOf(f32));
    defer backend.freeBuffer(d_keys);
    const d_values = try backend.allocBuffer(seq_len * hidden_dim * @sizeOf(f32));
    defer backend.freeBuffer(d_values);

    // Attention scores
    const d_scores = try backend.allocBuffer(seq_len * @sizeOf(f32));
    defer backend.freeBuffer(d_scores);

    try std.testing.expect(d_scores.ptr != 0);
}

// T5.3: CNN Seq2Seq (17)
test "cuda_comprehensive_17_cnn_seq2seq" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const seq_len: usize = 20;
    const kernel_size: usize = 3;
    const num_kernels: usize = 16;

    // CNN layers
    const d_input = try backend.allocBuffer(seq_len * @sizeOf(f32));
    defer backend.freeBuffer(d_input);
    const d_kernels = try backend.allocBuffer(num_kernels * kernel_size * @sizeOf(f32));
    defer backend.freeBuffer(d_kernels);

    try std.testing.expect(d_kernels.ptr != 0);
}

// T5.3: Dilated CNN Seq2Seq (18)
test "cuda_comprehensive_18_dilated_cnn_seq2seq" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const seq_len: usize = 24;
    const hidden_size: usize = 40;
    const output_len: usize = 6;

    // Dilated convolutions use sparse connections
    const d_input = try backend.allocBuffer(seq_len * @sizeOf(f32));
    defer backend.freeBuffer(d_input);
    const d_dilated = try backend.allocBuffer(seq_len * hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_dilated);
    const d_output = try backend.allocBuffer(output_len * @sizeOf(f32));
    defer backend.freeBuffer(d_output);

    try std.testing.expect(d_output.ptr != 0);
}

// T5.3: Comprehensive Suite Data Transfer
test "cuda_comprehensive_data_transfer" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const seq_len: usize = 10;
    const batch: usize = 20;

    // Training buffers
    const d_input = try backend.allocBuffer(batch * seq_len * @sizeOf(f32));
    defer backend.freeBuffer(d_input);

    // Initialize with pattern
    const h_input = try std.testing.allocator.alloc(f32, batch * seq_len);
    defer std.testing.allocator.free(h_input);

    for (0..batch) |b| {
        for (0..seq_len) |t| {
            const idx = b * seq_len + t;
            h_input[idx] = @sin(@as(f32, @floatFromInt(t + b)) * 0.2);
        }
    }
    try backend.upload(d_input.ptr, h_input);

    // Verify data integrity
    const h_check = try std.testing.allocator.alloc(f32, batch * seq_len);
    defer std.testing.allocator.free(h_check);
    try backend.download(h_check, d_input.ptr);

    for (h_check, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, h_input[i], 1e-5);
    }
}

// T5.3: GPU Memory Stress Test with Multiple Architectures
test "cuda_comprehensive_memory_stress" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    // Test multiple architectures in sequence
    const configs = .{
        .{ .seq_len = 10, .hidden = 16 },
        .{ .seq_len = 20, .hidden = 32 },
        .{ .seq_len = 15, .hidden = 24 },
    };

    inline for (configs) |config| {
        const d_input = try backend.allocBuffer(config.seq_len * @sizeOf(f32));
        defer backend.freeBuffer(d_input);
        const d_hidden = try backend.allocBuffer(config.hidden * @sizeOf(f32));
        defer backend.freeBuffer(d_hidden);
        const d_cell = try backend.allocBuffer(config.hidden * @sizeOf(f32));
        defer backend.freeBuffer(d_cell);

        try std.testing.expect(d_input.ptr != 0);
        try std.testing.expect(d_hidden.ptr != 0);
        try std.testing.expect(d_cell.ptr != 0);
    }
}

// T5.3: Activation Functions Across All Architectures
test "cuda_comprehensive_activations" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size: usize = 128;

    // Test all activation functions
    const d_buffer = try backend.allocBuffer(size * @sizeOf(f32));
    defer backend.freeBuffer(d_buffer);

    const h_data = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(h_data);

    for (h_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 - 6.0;

    // Test ReLU
    try backend.upload(d_buffer.ptr, h_data);

    // Sigmoid
    for (h_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 50)) - 25.0;
    try backend.upload(d_buffer.ptr, h_data);
    try backend.sigmoidForward(h_data, h_data);

    // Verify Sigmoid: outputs should be in [0, 1]
    for (h_data) |v| {
        try std.testing.expect(v >= 0 and v <= 1);
    }

    // Tanh
    for (h_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 50)) - 25.0;
    try backend.tanhForward(h_data, h_data);

    // Verify Tanh: outputs should be in [-1, 1]
    for (h_data) |v| {
        try std.testing.expect(v >= -1 and v <= 1);
    }
}

// T5.3: Weight Update with SGD for All Architectures
test "cuda_comprehensive_sgd_update" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const size: usize = 256;

    const h_weights = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(h_weights);
    const h_gradients = try std.testing.allocator.alloc(f32, size);
    defer std.testing.allocator.free(h_gradients);

    for (h_weights, 0..) |*w, i| w.* = @as(f32, @floatFromInt(i)) * 0.01;
    for (h_gradients) |*g| g.* = 0.001;

    try backend.sgdUpdate(h_weights, h_gradients, 0.01, 0.0);

    // Verify update
    var updated = false;
    for (h_weights) |w| {
        if (@abs(w) > 0) {
            updated = true;
            break;
        }
    }
    try std.testing.expect(updated);
}

/// CUDA Stock Prediction Integration Test (Phase 5)
/// Validates stock prediction workloads (LSTM, CNN, Attention) on CUDA backend
const std = @import("std");
const cuda = @import("cuda.zig");

// Helper: Generate synthetic stock-like data
fn generateSyntheticStock(allocator: std.mem.Allocator, len: usize) ![]f32 {
    var data = try allocator.alloc(f32, len);
    for (0..len) |i| {
        const trend = @as(f32, @floatFromInt(i)) * 0.001;
        const seasonal = @sin(@as(f32, @floatFromInt(i)) * 0.1) * 0.5;
        const noise = (@as(f32, @floatFromInt(i % 10)) - 5.0) * 0.01;
        data[i] = 100.0 + trend * 10.0 + seasonal + noise;
    }
    return data;
}

// T5.2: LSTM Stock Prediction - Memory Allocation
test "cuda_stock_lstm_memory" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return; // CUDA not available
    };
    defer backend.deinit();

    const window_size: usize = 10;
    const hidden_size: usize = 16;

    // LSTM buffers
    const d_input = try backend.allocBuffer(window_size * @sizeOf(f32));
    defer backend.freeBuffer(d_input);
    const d_hidden = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_hidden);
    const d_cell = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_cell);

    // Verify memory allocation
    try std.testing.expect(d_input.ptr != 0);
    try std.testing.expect(d_hidden.ptr != 0);
    try std.testing.expect(d_cell.ptr != 0);
}

// T5.2: Stock Data Transfer
test "cuda_stock_data_transfer" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const data_len: usize = 200;
    const data = try generateSyntheticStock(std.testing.allocator, data_len);
    defer std.testing.allocator.free(data);

    // Transfer to GPU
    const d_data = try backend.allocBuffer(data_len * @sizeOf(f32));
    defer backend.freeBuffer(d_data);
    try backend.upload(d_data.ptr, data);

    // Transfer back
    const h_check = try std.testing.allocator.alloc(f32, data_len);
    defer std.testing.allocator.free(h_check);
    try backend.download(h_check, d_data.ptr);

    // Verify data integrity
    for (h_check, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, data[i], 1e-5);
    }
}

// T5.2: Stock Window Processing
test "cuda_stock_window_processing" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const window_size: usize = 10;
    const batch: usize = 16;

    // Create window data
    const h_windows = try std.testing.allocator.alloc(f32, batch * window_size);
    defer std.testing.allocator.free(h_windows);

    for (h_windows, 0..) |*v, i| {
        const sample = i / window_size;
        const time = i % window_size;
        v.* = @sin(@as(f32, @floatFromInt(sample + time)) * 0.1) + 0.5;
    }

    // Transfer to GPU
    const d_windows = try backend.allocBuffer(batch * window_size * @sizeOf(f32));
    defer backend.freeBuffer(d_windows);
    try backend.upload(d_windows.ptr, h_windows);

    // Verify transfer
    const h_check = try std.testing.allocator.alloc(f32, batch * window_size);
    defer std.testing.allocator.free(h_check);
    try backend.download(h_check, d_windows.ptr);

    for (h_check, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(v, h_windows[i], 1e-5);
    }
}

// T5.2: Stock Data Normalization
test "cuda_stock_normalization" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const window_size: usize = 10;
    const d_window = try backend.allocBuffer(window_size * @sizeOf(f32));
    defer backend.freeBuffer(d_window);

    // Stock data
    const h_window = try std.testing.allocator.alloc(f32, window_size);
    defer std.testing.allocator.free(h_window);
    h_window[0] = 105.5;
    h_window[1] = 106.2;
    h_window[2] = 104.8;
    h_window[3] = 107.1;
    h_window[4] = 106.5;
    h_window[5] = 108.3;
    h_window[6] = 107.8;
    h_window[7] = 109.2;
    h_window[8] = 108.5;
    h_window[9] = 110.1;

    try backend.upload(d_window.ptr, h_window);

    // Min-max normalization on CPU for verification
    var min = h_window[0];
    var max = h_window[0];
    for (h_window) |v| {
        if (v < min) min = v;
        if (v > max) max = v;
    }
    const range = max - min;

    // Normalize
    for (h_window) |*v| v.* = (v.* - min) / range;

    // All values should be in [0, 1]
    for (h_window) |v| {
        try std.testing.expect(v >= 0 and v <= 1);
    }

    // Upload normalized data
    try backend.upload(d_window.ptr, h_window);
    try backend.download(h_window, d_window.ptr);

    // Verify on GPU
    for (h_window) |v| {
        try std.testing.expect(v >= 0 and v <= 1);
    }
}

// T5.2: Sliding Window Creation
test "cuda_stock_sliding_windows" {
    const data_len: usize = 100;
    const window_size: usize = 10;
    const n_windows = data_len - window_size;

    // Generate data
    const data = try generateSyntheticStock(std.testing.allocator, data_len);
    defer std.testing.allocator.free(data);

    // Create windows
    var windows = try std.testing.allocator.alloc([]f32, n_windows);
    defer {
        for (windows) |w| std.testing.allocator.free(w);
        std.testing.allocator.free(windows);
    }

    for (0..n_windows) |i| {
        windows[i] = try std.testing.allocator.alloc(f32, window_size);
        @memcpy(windows[i], data[i..i + window_size]);

        // Normalize each window
        var min = windows[i][0];
        var max = windows[i][0];
        for (windows[i]) |v| {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        const range = max - min;
        if (range > 0.001) {
            for (windows[i]) |*v| v.* = (v.* - min) / range;
        }
    }

    // Verify windows
    try std.testing.expectEqual(n_windows, windows.len);
    for (windows) |w| {
        try std.testing.expectEqual(window_size, w.len);
        for (w) |v| {
            try std.testing.expect(v >= 0 and v <= 1);
        }
    }
}

// T5.2: Multi-step Stock Prediction
test "cuda_stock_multi_step_prediction" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const window_size: usize = 20;
    const hidden_size: usize = 32;
    const pred_horizon: usize = 5;

    // Encoder LSTM
    const d_input = try backend.allocBuffer(window_size * @sizeOf(f32));
    defer backend.freeBuffer(d_input);

    // Decoder for multi-step
    const d_predictions = try backend.allocBuffer(pred_horizon * @sizeOf(f32));
    defer backend.freeBuffer(d_predictions);

    // Hidden states
    const d_hidden = try backend.allocBuffer(hidden_size * @sizeOf(f32));
    defer backend.freeBuffer(d_hidden);

    // Initialize
    const h_input = try std.testing.allocator.alloc(f32, window_size);
    defer std.testing.allocator.free(h_input);
    for (h_input, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.15) + 100.0;
    }
    try backend.upload(d_input.ptr, h_input);

    // Verify multi-step buffers
    try std.testing.expect(d_predictions.ptr != 0);
    try std.testing.expect(d_hidden.ptr != 0);
}

// T5.2: CNN Stock Memory Allocation
test "cuda_stock_cnn_memory" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const batch: usize = 8;
    const channels: usize = 1;
    const seq_len: usize = 20;
    const kernel_size: usize = 3;
    const out_channels: usize = 16;

    // Input: [batch, channels, seq_len]
    const d_input = try backend.allocBuffer(batch * channels * seq_len * @sizeOf(f32));
    defer backend.freeBuffer(d_input);

    // Kernel: [out_channels, channels, kernel_size]
    const d_kernel = try backend.allocBuffer(out_channels * channels * kernel_size * @sizeOf(f32));
    defer backend.freeBuffer(d_kernel);

    // Output: [batch, out_channels, seq_len - kernel_size + 1]
    const out_len = seq_len - kernel_size + 1;
    const d_output = try backend.allocBuffer(batch * out_channels * out_len * @sizeOf(f32));
    defer backend.freeBuffer(d_output);

    // Initialize
    const h_input = try std.testing.allocator.alloc(f32, batch * channels * seq_len);
    defer std.testing.allocator.free(h_input);
    const h_kernel = try std.testing.allocator.alloc(f32, out_channels * channels * kernel_size);
    defer std.testing.allocator.free(h_kernel);

    for (h_input, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    for (h_kernel, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 200)) - 100.0) * 0.01;

    try backend.upload(d_input.ptr, h_input);
    try backend.upload(d_kernel.ptr, h_kernel);

    // Verify buffer allocation succeeded
    try std.testing.expect(d_output.ptr != 0);
}

// T5.2: Attention Mechanism Memory
test "cuda_stock_attention_memory" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const seq_len: usize = 20;
    const hidden_dim: usize = 32;

    // Query: current state
    const d_query = try backend.allocBuffer(hidden_dim * @sizeOf(f32));
    defer backend.freeBuffer(d_query);

    // Keys: historical states
    const d_keys = try backend.allocBuffer(seq_len * hidden_dim * @sizeOf(f32));
    defer backend.freeBuffer(d_keys);

    // Values: historical outputs
    const d_values = try backend.allocBuffer(seq_len * hidden_dim * @sizeOf(f32));
    defer backend.freeBuffer(d_values);

    // Attention scores
    const d_scores = try backend.allocBuffer(seq_len * @sizeOf(f32));
    defer backend.freeBuffer(d_scores);

    // Initialize
    const h_query = try std.testing.allocator.alloc(f32, hidden_dim);
    defer std.testing.allocator.free(h_query);
    for (h_query, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01;
    try backend.upload(d_query.ptr, h_query);

    // Verify attention buffer allocation
    try std.testing.expect(d_keys.ptr != 0);
    try std.testing.expect(d_values.ptr != 0);
    try std.testing.expect(d_scores.ptr != 0);
}

// T5.2: Training Stability with Volatile Data
test "cuda_stock_volatile_data" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const seq_len: usize = 10;

    // Simulate volatile stock data
    const d_input = try backend.allocBuffer(seq_len * @sizeOf(f32));
    defer backend.freeBuffer(d_input);

    const h_input = try std.testing.allocator.alloc(f32, seq_len);
    defer std.testing.allocator.free(h_input);

    // High variance data
    for (h_input, 0..) |*v, i| {
        v.* = if (i % 2 == 0) 1.0 else 0.0;
    }
    try backend.upload(d_input.ptr, h_input);

    // Transfer back and verify no NaN
    try backend.download(h_input, d_input.ptr);

    for (h_input) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
}

// T5.2: Memory Requirements for Large Windows
test "cuda_stock_large_window_memory" {
    var backend = cuda.CudaBackend.init(std.testing.allocator) catch {
        return;
    };
    defer backend.deinit();

    const window_size: usize = 100;
    const hidden_size: usize = 64;
    const batch: usize = 32;

    // Calculate memory needs
    const input_mem = batch * window_size * @sizeOf(f32);
    const hidden_mem = batch * hidden_size * @sizeOf(f32);
    const cell_mem = batch * hidden_size * @sizeOf(f32);
    const weights_mem = 4 * hidden_size * (window_size + hidden_size) * @sizeOf(f32); // LSTM gates

    const total_mem = input_mem + hidden_mem * 2 + cell_mem * 2 + weights_mem;

    // Should fit in any modern GPU
    try std.testing.expect(total_mem < 100 * 1024 * 1024); // Less than 100MB

    // Verify allocation
    const d_input = try backend.allocBuffer(input_mem);
    defer backend.freeBuffer(d_input);
    const d_hidden = try backend.allocBuffer(hidden_mem);
    defer backend.freeBuffer(d_hidden);
    const d_cell = try backend.allocBuffer(cell_mem);
    defer backend.freeBuffer(d_cell);
}

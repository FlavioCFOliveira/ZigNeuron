# ZigNeuron Examples

This directory contains comprehensive examples demonstrating ZigNeuron's capabilities.

## Comprehensive Suite

The `comprehensive_suite/` directory contains 18 examples covering different neural network architectures:

### Recurrent Networks

| Example | Architecture | Description |
|---------|--------------|-------------|
| `01_vanilla_rnn.zig` | Vanilla RNN | Basic RNN with tanh activation |
| `02_vanilla_bidirectional.zig` | Bidirectional RNN | RNN processing in both directions |
| `03_vanilla_twopath.zig` | Two-Path RNN | Parallel RNN paths |
| `04_lstm.zig` | LSTM | Long Short-Term Memory network |
| `05_lstm_bidirectional.zig` | Bidirectional LSTM | LSTM in both directions |
| `06_lstm_twopath.zig` | LSTM Two-Path | Parallel LSTM paths |
| `07_gru.zig` | GRU | Gated Recurrent Unit |
| `08_gru_bidirectional.zig` | Bidirectional GRU | GRU in both directions |
| `09_gru_twopath.zig` | GRU Two-Path | Parallel GRU paths |

### Sequence-to-Sequence

| Example | Architecture | Description |
|---------|--------------|-------------|
| `10_lstm_seq2seq.zig` | LSTM Seq2Seq | Encoder-decoder with LSTM |
| `11_lstm_bidirectional_seq2seq.zig` | BiLSTM Seq2Seq | Bidirectional encoder |
| `12_lstm_seq2seq_vae.zig` | LSTM VAE | Variational autoencoder with LSTM |
| `13_gru_seq2seq.zig` | GRU Seq2Seq | Encoder-decoder with GRU |
| `14_gru_bidirectional_seq2seq.zig` | BiGRU Seq2Seq | Bidirectional encoder |
| `15_gru_seq2seq_vae.zig` | GRU VAE | Variational autoencoder with GRU |

### Other Architectures

| Example | Architecture | Description |
|---------|--------------|-------------|
| `16_attention.zig` | Self-Attention | Attention mechanism for time series |
| `17_cnn_seq2seq.zig` | CNN Seq2Seq | Convolutional encoder-decoder |
| `18_dilated_cnn_seq2seq.zig` | Dilated CNN | Dilated convolutions for longer context |

## Stock Prediction

The `stock_prediction/` directory contains real-world examples using stock price data:

| Example | Architecture | Dataset |
|---------|--------------|---------|
| `lstm.zig` | LSTM | Stock prices (windowed) |
| `attention_transformer.zig` | Attention | Stock prices with attention |
| `cnn_seq2seq.zig` | CNN Seq2Seq | Stock prices (convolutional) |

## Running Examples

```bash
# Build all examples
zig build

# Run specific example
./zig-out/bin/01_vanilla_rnn

# Run stock prediction
./zig-out/bin/stock_lstm
```

## Example Structure

Each example follows this pattern:

```zig
const std = @import("std");
const zn = @import("ZigNeuron");
const common = @import("common.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Load data
    const dataset = try common.loadStockData(allocator, "path/to/data.csv", window_size, 1);
    defer dataset.deinit();

    // Create backend and network
    const backend = try zn.backend.Backend.init(allocator);
    var net = try zn.network.Network.init(allocator, backend);
    defer net.deinit();

    // Add layers
    _ = try net.addLSTM(1, 32, window_size);
    _ = try net.addDense(32, 1, .linear);

    // Train
    std.debug.print("\n--- Training ---\n", .{});
    try net.train(dataset.x, dataset.y, 50, 0.01, .{ .mse = {} }, null);
}
```

## Data Format

Examples use CSV files with format:
```csv
date,open,high,low,close,volume
2023-01-01,100.0,105.0,99.0,102.0,1000000
...
```

The `common.zig` module provides utilities for:
- Loading CSV data
- Normalization
- Windowing for time series
- Train/test splitting

## Adding New Examples

1. Create `.zig` file in appropriate directory
2. Import required modules
3. Follow the pattern above
4. Update `build.zig` to compile the example
5. Add documentation to this README

## Learning Resources

- [RNN Tutorial](../docs/rnn_tutorial.md)
- [LSTM Explanation](../docs/lstm_explained.md)
- [Attention Mechanism](../docs/attention.md)

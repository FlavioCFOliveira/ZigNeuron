/// Recurrent Neural Network layers
const std = @import("std");
const activation = @import("activation.zig");
const backend_module = @import("backend.zig");
const tensor = @import("tensor.zig");
const metal = @import("metal.zig");

/// Vanilla RNN Layer
/// h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
pub const VanillaRNN = struct {
    weights_ih: tensor.Tensor, // [input_size, hidden_size]
    weights_hh: tensor.Tensor, // [hidden_size, hidden_size]
    bias: tensor.Tensor,       // [hidden_size]

    grad_weights_ih: tensor.Tensor,
    grad_weights_hh: tensor.Tensor,
    grad_bias: tensor.Tensor,

    // Buffers for zero-allocation
    grad_h_next: tensor.Tensor, // [hidden_size] - gradient flowing from next time step
    h_prev_work: tensor.Tensor, // [hidden_size] - previous hidden state buffer
    tmp_hh_work: tensor.Tensor, // [hidden_size] - temporary buffer for HH matmul
    grad_after_act_work: tensor.Tensor, // [hidden_size] - temporary buffer for activation derivative

    input_size: usize,
    hidden_size: usize,
    act: activation.Activation,
    backend: backend_module.Backend,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, act: activation.Activation, backend: backend_module.Backend) !*VanillaRNN {
        const self = try allocator.create(VanillaRNN);
        errdefer allocator.destroy(self);

        self.weights_ih = try tensor.Tensor.init(allocator, &.{ input_size, hidden_size }, backend);
        errdefer self.weights_ih.deinit();

        self.weights_hh = try tensor.Tensor.init(allocator, &.{ hidden_size, hidden_size }, backend);
        errdefer self.weights_hh.deinit();

        self.bias = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.bias.deinit();

        self.grad_weights_ih = try tensor.Tensor.init(allocator, &.{ input_size, hidden_size }, backend);
        errdefer self.grad_weights_ih.deinit();

        self.grad_weights_hh = try tensor.Tensor.init(allocator, &.{ hidden_size, hidden_size }, backend);
        errdefer self.grad_weights_hh.deinit();

        self.grad_bias = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.grad_bias.deinit();

        self.grad_h_next = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.grad_h_next.deinit();

        self.h_prev_work = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.h_prev_work.deinit();

        self.tmp_hh_work = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.tmp_hh_work.deinit();

        self.grad_after_act_work = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.grad_after_act_work.deinit();

        // Xavier/He initialization
        var prng = std.Random.DefaultPrng.init(@intCast(@as(u64, @bitCast(std.time.timestamp())) +% input_size +% hidden_size));
        const random = prng.random();
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(input_size + hidden_size)));

        for (self.weights_ih.slice) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        for (self.weights_hh.slice) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        @memset(self.bias.slice, 0);
        @memset(self.grad_h_next.slice, 0);

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.act = act;
        self.backend = backend;
        self.allocator = allocator;

        return self;
    }

    pub fn deinit(self: *VanillaRNN) void {
        self.weights_ih.deinit();
        self.weights_hh.deinit();
        self.bias.deinit();
        self.grad_weights_ih.deinit();
        self.grad_weights_hh.deinit();
        self.grad_bias.deinit();
        self.grad_h_next.deinit();
        self.h_prev_work.deinit();
        self.tmp_hh_work.deinit();
        self.grad_after_act_work.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *VanillaRNN,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        const seq_len = input.len / self.input_size;
        const hidden_size = self.hidden_size;
        const batch_size = 1;

        // Initial hidden state h_0 is all zeros
        @memset(self.h_prev_work.slice, 0);

        for (0..seq_len) |t| {
            // If output buffer is large enough, store all hidden states (many-to-many)
            // Otherwise, we only care about the last hidden state (many-to-one)
            const h_t = if (output.len >= seq_len * hidden_size)
                output[t * hidden_size .. (t + 1) * hidden_size]
            else if (t == seq_len - 1)
                output[0..hidden_size]
            else
                self.grad_after_act_work.slice; // Use a work buffer if we don't store intermediate states

            // 1. Calculate W_ih * x_t
            // If we're not storing all states, we need a temporary buffer for this step
            // because in the many-to-many case we used the output buffer directly.
            const x_t = input[t * self.input_size .. (t + 1) * self.input_size];
            try self.backend.matMul(
                x_t, input_buf,
                self.weights_ih.slice, self.weights_ih.getMtlBuffer(),
                h_t, output_buf,
                batch_size, hidden_size, self.input_size
            );

            // 2. tmp_hh = W_hh * h_{t-1}
            try self.backend.matMul(
                self.h_prev_work.slice, self.h_prev_work.getMtlBuffer(),
                self.weights_hh.slice, self.weights_hh.getMtlBuffer(),
                self.tmp_hh_work.slice, self.tmp_hh_work.getMtlBuffer(),
                batch_size, hidden_size, hidden_size
            );

            // 3. h_t = tanh(h_t + tmp_hh + bias)
            try self.backend.rnnForwardStep(
                h_t, output_buf,
                self.tmp_hh_work.slice, self.tmp_hh_work.getMtlBuffer(),
                self.bias.slice, self.bias.getMtlBuffer(),
                h_t, output_buf,
                hidden_size
            );

            // Update h_prev for next step
            try self.backend.copyData(h_t, output_buf, self.h_prev_work.slice, self.h_prev_work.getMtlBuffer());
        }
    }

    pub fn backward(self: *VanillaRNN,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer,
        h_states: []const f32, h_states_buf: ?*const metal.MTLBuffer
    ) !void {
        const seq_len = input.len / self.input_size;
        const hidden_size = self.hidden_size;

        // Reset grad_h_next for the beginning of BPTT
        @memset(self.grad_h_next.slice, 0);

        var i: usize = seq_len;
        while (i > 0) {
            i -= 1;
            const t = i;

            const x_t = input[t * self.input_size .. (t + 1) * self.input_size];
            const h_t = h_states[t * hidden_size .. (t + 1) * hidden_size];
            const grad_h_t_out = grad_output[t * hidden_size .. (t + 1) * hidden_size];
            const h_prev = if (t > 0) h_states[(t - 1) * hidden_size .. t * hidden_size] else null;

            // 1. grad_after_act = (grad_output[t] + grad_h_next) * tanh'(h_t)
            try self.backend.rnnBackwardStep(
                grad_h_t_out, grad_output_buf,
                self.grad_h_next.slice, self.grad_h_next.getMtlBuffer(),
                h_t, h_states_buf,
                self.grad_after_act_work.slice, self.grad_after_act_work.getMtlBuffer(),
                hidden_size
            );

            // 2. Accumulate gradients for W_ih and bias
            try self.backend.matMulTransposeA(
                x_t, input_buf,
                self.grad_after_act_work.slice, self.grad_after_act_work.getMtlBuffer(),
                self.grad_weights_ih.slice, self.grad_weights_ih.getMtlBuffer(),
                self.input_size, hidden_size, 1
            );

            try self.backend.accumulateBias(
                self.grad_bias.slice, self.grad_bias.getMtlBuffer(),
                self.grad_after_act_work.slice, self.grad_after_act_work.getMtlBuffer()
            );

            // 3. Accumulate gradients for W_hh and compute grad_h_prev
            if (h_prev) |hp| {
                try self.backend.matMulTransposeA(
                    hp, h_states_buf,
                    self.grad_after_act_work.slice, self.grad_after_act_work.getMtlBuffer(),
                    self.grad_weights_hh.slice, self.grad_weights_hh.getMtlBuffer(),
                    hidden_size, hidden_size, 1
                );

                // grad_h_prev = grad_after_act * W_hh^T
                try self.backend.matMulTransposeB(
                    self.grad_after_act_work.slice, self.grad_after_act_work.getMtlBuffer(),
                    self.weights_hh.slice, self.weights_hh.getMtlBuffer(),
                    self.grad_h_next.slice, self.grad_h_next.getMtlBuffer(),
                    1, hidden_size, hidden_size
                );
            } else {
                @memset(self.grad_h_next.slice, 0);
            }

            // 4. Compute grad_input_t = grad_after_act * W_ih^T
            const gi_t = grad_input[t * self.input_size .. (t + 1) * self.input_size];
            try self.backend.matMulTransposeB(
                self.grad_after_act_work.slice, self.grad_after_act_work.getMtlBuffer(),
                self.weights_ih.slice, self.weights_ih.getMtlBuffer(),
                gi_t, grad_input_buf,
                1, self.input_size, hidden_size
            );
        }
    }
};

/// LSTM Layer
pub const LSTM = struct {
    weights_ih: tensor.Tensor, // [input_size, 4 * hidden_size]
    weights_hh: tensor.Tensor, // [hidden_size, 4 * hidden_size]
    bias: tensor.Tensor,       // [4 * hidden_size]

    grad_weights_ih: tensor.Tensor,
    grad_weights_hh: tensor.Tensor,
    grad_bias: tensor.Tensor,

    // Buffers for zero-allocation BPTT
    cell_states: tensor.Tensor,      // [max_seq_len + 1, hidden_size]
    gate_activations: tensor.Tensor, // [max_seq_len, 4 * hidden_size]

    // Workspace for zero-allocation
    grad_h_next: tensor.Tensor,      // [hidden_size]
    grad_c_next: tensor.Tensor,      // [hidden_size]
    grad_gates_work: tensor.Tensor,  // [4 * hidden_size]
    h_prev_work: tensor.Tensor,      // [hidden_size]
    gates_ih_work: tensor.Tensor,    // [4 * hidden_size]
    gates_hh_work: tensor.Tensor,    // [4 * hidden_size]
    total_grad_h_work: tensor.Tensor, // [hidden_size]

    input_size: usize,
    hidden_size: usize,
    max_seq_len: usize,
    backend: backend_module.Backend,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, max_seq_len: usize, backend: backend_module.Backend) !*LSTM {
        const self = try allocator.create(LSTM);
        errdefer allocator.destroy(self);

        self.weights_ih = try tensor.Tensor.init(allocator, &.{ input_size, 4 * hidden_size }, backend);
        errdefer self.weights_ih.deinit();
        self.weights_hh = try tensor.Tensor.init(allocator, &.{ hidden_size, 4 * hidden_size }, backend);
        errdefer self.weights_hh.deinit();
        self.bias = try tensor.Tensor.init(allocator, &.{4 * hidden_size}, backend);
        errdefer self.bias.deinit();

        self.grad_weights_ih = try tensor.Tensor.init(allocator, &.{ input_size, 4 * hidden_size }, backend);
        errdefer self.grad_weights_ih.deinit();
        self.grad_weights_hh = try tensor.Tensor.init(allocator, &.{ hidden_size, 4 * hidden_size }, backend);
        errdefer self.grad_weights_hh.deinit();
        self.grad_bias = try tensor.Tensor.init(allocator, &.{4 * hidden_size}, backend);
        errdefer self.grad_bias.deinit();

        self.cell_states = try tensor.Tensor.init(allocator, &.{ max_seq_len + 1, hidden_size }, backend);
        errdefer self.cell_states.deinit();
        self.gate_activations = try tensor.Tensor.init(allocator, &.{ max_seq_len, 4 * hidden_size }, backend);
        errdefer self.gate_activations.deinit();

        self.grad_h_next = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.grad_h_next.deinit();
        self.grad_c_next = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.grad_c_next.deinit();
        self.grad_gates_work = try tensor.Tensor.init(allocator, &.{4 * hidden_size}, backend);
        errdefer self.grad_gates_work.deinit();

        self.h_prev_work = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.h_prev_work.deinit();
        self.gates_ih_work = try tensor.Tensor.init(allocator, &.{4 * hidden_size}, backend);
        errdefer self.gates_ih_work.deinit();
        self.gates_hh_work = try tensor.Tensor.init(allocator, &.{4 * hidden_size}, backend);
        errdefer self.gates_hh_work.deinit();
        self.total_grad_h_work = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.total_grad_h_work.deinit();

        // Xavier/He initialization
        var prng = std.Random.DefaultPrng.init(@intCast(@as(u64, @bitCast(std.time.timestamp())) +% input_size +% hidden_size));
        const random = prng.random();
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(input_size + hidden_size)));

        for (self.weights_ih.slice) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        for (self.weights_hh.slice) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        @memset(self.bias.slice, 0);

        // Initialize forget bias to 1.0
        const forget_offset = hidden_size;
        for (0..hidden_size) |i| {
            self.bias.slice[forget_offset + i] = 1.0;
        }

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.max_seq_len = max_seq_len;
        self.backend = backend;
        self.allocator = allocator;

        return self;
    }

    pub fn deinit(self: *LSTM) void {
        self.weights_ih.deinit();
        self.weights_hh.deinit();
        self.bias.deinit();
        self.grad_weights_ih.deinit();
        self.grad_weights_hh.deinit();
        self.grad_bias.deinit();
        self.cell_states.deinit();
        self.gate_activations.deinit();
        self.grad_h_next.deinit();
        self.grad_c_next.deinit();
        self.grad_gates_work.deinit();
        self.h_prev_work.deinit();
        self.gates_ih_work.deinit();
        self.gates_hh_work.deinit();
        self.total_grad_h_work.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *LSTM,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        const seq_len = input.len / self.input_size;
        if (seq_len > self.max_seq_len) return error.SequenceTooLong;

        const hidden_size = self.hidden_size;
        const batch_size = 1;

        // 1. Batch input matmul for the whole sequence: gate_activations = input * weights_ih
        // gate_activations: [seq_len, 4 * hidden_size]
        try self.backend.matMul(
            input, input_buf,
            self.weights_ih.slice, self.weights_ih.getMtlBuffer(),
            self.gate_activations.slice, self.gate_activations.getMtlBuffer(),
            seq_len, 4 * hidden_size, self.input_size
        );

        // Initial states are zero
        @memset(self.cell_states.slice[0..hidden_size], 0);
        @memset(self.h_prev_work.slice, 0);

        for (0..seq_len) |t| {
            const gates_t = self.gate_activations.slice[t * 4 * hidden_size .. (t + 1) * 4 * hidden_size];

            // h_t should be slice of output if output matches seq_len * hidden_size.
            // If output is just [hidden_size], it might be intended for many-to-one.
            // Let's check output length.
            const h_t = if (output.len >= seq_len * hidden_size)
                output[t * hidden_size .. (t + 1) * hidden_size]
            else if (t == seq_len - 1)
                output[0..hidden_size]
            else
                self.h_prev_work.slice; // Dummy if we only care about last state

            const c_prev = self.cell_states.slice[t * hidden_size .. (t + 1) * hidden_size];
            const c_t = self.cell_states.slice[(t + 1) * hidden_size .. (t + 2) * hidden_size];

            // 2. gates_hh = W_hh * h_{t-1}
            try self.backend.matMul(
                self.h_prev_work.slice, self.h_prev_work.getMtlBuffer(),
                self.weights_hh.slice, self.weights_hh.getMtlBuffer(),
                self.gates_hh_work.slice, self.gates_hh_work.getMtlBuffer(),
                batch_size, 4 * hidden_size, hidden_size
            );

            // 3. Combined LSTM step
            // Note: gates_t already contains W_ih * x_t (from batch matmul)
            try self.backend.lstmForwardStep(
                gates_t, self.gate_activations.getMtlBuffer(),
                self.gates_hh_work.slice, self.gates_hh_work.getMtlBuffer(),
                self.bias.slice, self.bias.getMtlBuffer(),
                c_prev, self.cell_states.getMtlBuffer(),
                c_t, self.cell_states.getMtlBuffer(),
                h_t, output_buf,
                gates_t, self.gate_activations.getMtlBuffer(),
                hidden_size
            );

            // Update h_prev for next step
            try self.backend.copyData(h_t, output_buf, self.h_prev_work.slice, self.h_prev_work.getMtlBuffer());
        }
    }

    pub fn backward(self: *LSTM,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer,
        h_states: []const f32, h_states_buf: ?*const metal.MTLBuffer
    ) !void {
        const seq_len = input.len / self.input_size;
        const hidden_size = self.hidden_size;

        @memset(self.grad_h_next.slice, 0);
        @memset(self.grad_c_next.slice, 0);

        var i: usize = seq_len;
        while (i > 0) {
            i -= 1;
            const t = i;

            const x_t = input[t * self.input_size .. (t + 1) * self.input_size];
            const h_prev = if (t > 0) h_states[(t - 1) * hidden_size .. t * hidden_size] else null;
            const c_t = self.cell_states.slice[(t + 1) * hidden_size .. (t + 2) * hidden_size];
            const c_prev = self.cell_states.slice[t * hidden_size .. (t + 1) * hidden_size];
            const gates_t = self.gate_activations.slice[t * 4 * hidden_size .. (t + 1) * 4 * hidden_size];
            const grad_h_t = grad_output[t * hidden_size .. (t + 1) * hidden_size];

            const d_gates = self.grad_gates_work.slice;

            try self.backend.lstmBackwardStep(
                grad_h_t, grad_output_buf,
                self.grad_h_next.slice, self.grad_h_next.getMtlBuffer(),
                self.grad_c_next.slice, self.grad_c_next.getMtlBuffer(),
                gates_t, self.gate_activations.getMtlBuffer(),
                c_t, self.cell_states.getMtlBuffer(),
                c_prev, self.cell_states.getMtlBuffer(),
                d_gates, self.grad_gates_work.getMtlBuffer(),
                self.grad_c_next.slice, self.grad_c_next.getMtlBuffer(),
                self.h_prev_work.slice, self.h_prev_work.getMtlBuffer(), // Reuse h_prev_work as partial grad_h_prev
                hidden_size
            );

            // 2. Accumulate gradients for W_ih, W_hh, bias
            try self.backend.matMulTransposeA(x_t, input_buf, d_gates, self.grad_gates_work.getMtlBuffer(), self.grad_weights_ih.slice, self.grad_weights_ih.getMtlBuffer(), self.input_size, 4 * hidden_size, 1);
            if (h_prev) |hp| {
                try self.backend.matMulTransposeA(hp, h_states_buf, d_gates, self.grad_gates_work.getMtlBuffer(), self.grad_weights_hh.slice, self.grad_weights_hh.getMtlBuffer(), hidden_size, 4 * hidden_size, 1);
            }
            try self.backend.accumulateBias(self.grad_bias.slice, self.grad_bias.getMtlBuffer(), d_gates, self.grad_gates_work.getMtlBuffer());

            // 3. Compute grad_h_next for next BPTT iteration: d_h_prev = d_gates * W_hh^T
            try self.backend.matMulTransposeB(d_gates, self.grad_gates_work.getMtlBuffer(), self.weights_hh.slice, self.weights_hh.getMtlBuffer(), self.grad_h_next.slice, self.grad_h_next.getMtlBuffer(), 1, hidden_size, 4 * hidden_size);

            // 4. Compute grad_input_t = d_gates * W_ih^T
            const gi_t = grad_input[t * self.input_size .. (t + 1) * self.input_size];
            try self.backend.matMulTransposeB(d_gates, self.grad_gates_work.getMtlBuffer(), self.weights_ih.slice, self.weights_ih.getMtlBuffer(), gi_t, grad_input_buf, 1, self.input_size, 4 * hidden_size);
        }
    }
};

/// GRU Layer
pub const GRU = struct {
    weights_ih: tensor.Tensor, // [input_size, 3 * hidden_size]
    weights_hh: tensor.Tensor, // [hidden_size, 3 * hidden_size]
    bias: tensor.Tensor,       // [3 * hidden_size]

    grad_weights_ih: tensor.Tensor,
    grad_weights_hh: tensor.Tensor,
    grad_bias: tensor.Tensor,

    // Buffers for zero-allocation BPTT
    gate_activations: tensor.Tensor, // [max_seq_len, 3 * hidden_size] (z, r, n)
    hh_components: tensor.Tensor,    // [max_seq_len, 3 * hidden_size] (W_hh * h_{t-1})

    // Workspace for zero-allocation
    grad_h_next: tensor.Tensor,      // [hidden_size]
    grad_gates_work: tensor.Tensor,  // [3 * hidden_size]
    grad_gates_hh_work: tensor.Tensor, // [3 * hidden_size]
    h_prev_work: tensor.Tensor,      // [hidden_size]
    gates_ih_work: tensor.Tensor,    // [3 * hidden_size]
    gates_hh_work: tensor.Tensor,    // [3 * hidden_size]

    input_size: usize,
    hidden_size: usize,
    max_seq_len: usize,
    backend: backend_module.Backend,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, max_seq_len: usize, backend: backend_module.Backend) !*GRU {
        const self = try allocator.create(GRU);
        errdefer allocator.destroy(self);

        self.weights_ih = try tensor.Tensor.init(allocator, &.{ input_size, 3 * hidden_size }, backend);
        errdefer self.weights_ih.deinit();
        self.weights_hh = try tensor.Tensor.init(allocator, &.{ hidden_size, 3 * hidden_size }, backend);
        errdefer self.weights_hh.deinit();
        self.bias = try tensor.Tensor.init(allocator, &.{3 * hidden_size}, backend);
        errdefer self.bias.deinit();

        self.grad_weights_ih = try tensor.Tensor.init(allocator, &.{ input_size, 3 * hidden_size }, backend);
        errdefer self.grad_weights_ih.deinit();
        self.grad_weights_hh = try tensor.Tensor.init(allocator, &.{ hidden_size, 3 * hidden_size }, backend);
        errdefer self.grad_weights_hh.deinit();
        self.grad_bias = try tensor.Tensor.init(allocator, &.{3 * hidden_size}, backend);
        errdefer self.grad_bias.deinit();

        self.gate_activations = try tensor.Tensor.init(allocator, &.{ max_seq_len, 3 * hidden_size }, backend);
        errdefer self.gate_activations.deinit();
        self.hh_components = try tensor.Tensor.init(allocator, &.{ max_seq_len, 3 * hidden_size }, backend);
        errdefer self.hh_components.deinit();

        self.grad_h_next = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.grad_h_next.deinit();
        self.grad_gates_work = try tensor.Tensor.init(allocator, &.{3 * hidden_size}, backend);
        errdefer self.grad_gates_work.deinit();
        self.grad_gates_hh_work = try tensor.Tensor.init(allocator, &.{3 * hidden_size}, backend);
        errdefer self.grad_gates_hh_work.deinit();

        self.h_prev_work = try tensor.Tensor.init(allocator, &.{hidden_size}, backend);
        errdefer self.h_prev_work.deinit();
        self.gates_ih_work = try tensor.Tensor.init(allocator, &.{3 * hidden_size}, backend);
        errdefer self.gates_ih_work.deinit();
        self.gates_hh_work = try tensor.Tensor.init(allocator, &.{3 * hidden_size}, backend);
        errdefer self.gates_hh_work.deinit();

        // Xavier/He initialization
        var prng = std.Random.DefaultPrng.init(@intCast(@as(u64, @bitCast(std.time.timestamp())) +% input_size +% hidden_size));
        const random = prng.random();
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(input_size + hidden_size)));

        for (self.weights_ih.slice) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        for (self.weights_hh.slice) |*w| w.* = (random.float(f32) * 2.0 - 1.0) * scale;
        @memset(self.bias.slice, 0);

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.max_seq_len = max_seq_len;
        self.backend = backend;
        self.allocator = allocator;

        return self;
    }

    pub fn deinit(self: *GRU) void {
        self.weights_ih.deinit();
        self.weights_hh.deinit();
        self.bias.deinit();
        self.grad_weights_ih.deinit();
        self.grad_weights_hh.deinit();
        self.grad_bias.deinit();
        self.gate_activations.deinit();
        self.hh_components.deinit();
        self.grad_h_next.deinit();
        self.grad_gates_work.deinit();
        self.grad_gates_hh_work.deinit();
        self.h_prev_work.deinit();
        self.gates_ih_work.deinit();
        self.gates_hh_work.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *GRU,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        const seq_len = input.len / self.input_size;
        if (seq_len > self.max_seq_len) return error.SequenceTooLong;

        const hidden_size = self.hidden_size;
        const batch_size = 1;

        @memset(self.h_prev_work.slice, 0);

        for (0..seq_len) |t| {
            const h_t = if (output.len >= seq_len * hidden_size)
                output[t * hidden_size .. (t + 1) * hidden_size]
            else if (t == seq_len - 1)
                output[0..hidden_size]
            else
                self.h_prev_work.slice; // Reuse h_prev_work as dummy if needed

            // We still need to store intermediate gate activations and hh_components for backward pass.
            // These were allocated with max_seq_len, so it's fine.
            const gates_t = self.gate_activations.slice[t * 3 * hidden_size .. (t + 1) * 3 * hidden_size];
            const hh_t = self.hh_components.slice[t * 3 * hidden_size .. (t + 1) * 3 * hidden_size];

            // 1. Calculate W_ih * x_t
            const x_t = input[t * self.input_size .. (t + 1) * self.input_size];
            try self.backend.matMul(
                x_t, input_buf,
                self.weights_ih.slice, self.weights_ih.getMtlBuffer(),
                gates_t, self.gate_activations.getMtlBuffer(),
                batch_size, 3 * hidden_size, self.input_size
            );

            // 2. gates_hh = W_hh * h_{t-1}
            try self.backend.matMul(self.h_prev_work.slice, self.h_prev_work.getMtlBuffer(), self.weights_hh.slice, self.weights_hh.getMtlBuffer(), self.gates_hh_work.slice, self.gates_hh_work.getMtlBuffer(), batch_size, 3 * hidden_size, hidden_size);

            // 3. Combined GRU forward step
            // Note: gates_t already contains W_ih * x_t
            try self.backend.gruForwardStep(
                gates_t, self.gate_activations.getMtlBuffer(),
                self.gates_hh_work.slice, self.gates_hh_work.getMtlBuffer(),
                self.bias.slice, self.bias.getMtlBuffer(),
                self.h_prev_work.slice, self.h_prev_work.getMtlBuffer(),
                h_t, output_buf,
                gates_t, self.gate_activations.getMtlBuffer(),
                hh_t[2 * hidden_size .. 3 * hidden_size], self.hh_components.getMtlBuffer(), // Store n_hh component
                hidden_size
            );

            // Update h_prev for next step
            try self.backend.copyData(h_t, output_buf, self.h_prev_work.slice, self.h_prev_work.getMtlBuffer());
        }
    }

    pub fn backward(self: *GRU,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer,
        h_states: []const f32, h_states_buf: ?*const metal.MTLBuffer
    ) !void {
        const seq_len = input.len / self.input_size;
        const hidden_size = self.hidden_size;

        @memset(self.grad_h_next.slice, 0);

        var i: usize = seq_len;
        while (i > 0) {
            i -= 1;
            const t = i;

            const x_t = input[t * self.input_size .. (t + 1) * self.input_size];
            const h_prev = if (t > 0) h_states[(t - 1) * hidden_size .. t * hidden_size] else null;
            const h_prev_buf = if (t > 0) h_states_buf else null;
            const gates_t = self.gate_activations.slice[t * 3 * hidden_size .. (t + 1) * 3 * hidden_size];
            const hh_t = self.hh_components.slice[t * 3 * hidden_size .. (t + 1) * 3 * hidden_size];
            const grad_h_t = grad_output[t * hidden_size .. (t + 1) * hidden_size];
            const n_hh = hh_t[2 * hidden_size .. 3 * hidden_size];

            const d_gates = self.grad_gates_work.slice;
            const d_gates_hh = self.grad_gates_hh_work.slice;

            // Combined GRU backward step
            try self.backend.gruBackwardStep(
                grad_h_t, grad_output_buf,
                self.grad_h_next.slice, self.grad_h_next.getMtlBuffer(),
                gates_t, self.gate_activations.getMtlBuffer(),
                h_prev orelse self.h_prev_work.slice, h_prev_buf orelse self.h_prev_work.getMtlBuffer(),
                n_hh, self.hh_components.getMtlBuffer(),
                d_gates, self.grad_gates_work.getMtlBuffer(),
                d_gates_hh, self.grad_gates_hh_work.getMtlBuffer(),
                self.grad_h_next.slice, self.grad_h_next.getMtlBuffer(), // Updates in-place
                hidden_size
            );

            // Accumulate gradients for W_ih, W_hh, bias
            try self.backend.matMulTransposeA(x_t, input_buf, d_gates, self.grad_gates_work.getMtlBuffer(), self.grad_weights_ih.slice, self.grad_weights_ih.getMtlBuffer(), self.input_size, 3 * hidden_size, 1);
            if (h_prev) |hp| {
                try self.backend.matMulTransposeA(hp, h_prev_buf, d_gates_hh, self.grad_gates_hh_work.getMtlBuffer(), self.grad_weights_hh.slice, self.grad_weights_hh.getMtlBuffer(), hidden_size, 3 * hidden_size, 1);
            }
            try self.backend.accumulateBias(self.grad_bias.slice, self.grad_bias.getMtlBuffer(), d_gates, self.grad_gates_work.getMtlBuffer());

            // Compute grad_input_t = d_gates * W_ih^T
            const gi_t = grad_input[t * self.input_size .. (t + 1) * self.input_size];
            try self.backend.matMulTransposeB(d_gates, self.grad_gates_work.getMtlBuffer(), self.weights_ih.slice, self.weights_ih.getMtlBuffer(), gi_t, grad_input_buf, 1, self.input_size, 3 * hidden_size);
        }
    }
};

/// Tagged union for recurrent layers
pub const RecurrentLayerType = enum {
    rnn,
    lstm,
    gru,
};

pub const RecurrentLayer = union(RecurrentLayerType) {
    rnn: *VanillaRNN,
    lstm: *LSTM,
    gru: *GRU,

    pub fn deinit(self: RecurrentLayer) void {
        switch (self) {
            .rnn => |l| l.deinit(),
            .lstm => |l| l.deinit(),
            .gru => |l| l.deinit(),
        }
    }

    pub fn forward(self: RecurrentLayer, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        switch (self) {
            .rnn => |l| try l.forward(input, input_buf, output, output_buf),
            .lstm => |l| try l.forward(input, input_buf, output, output_buf),
            .gru => |l| try l.forward(input, input_buf, output, output_buf),
        }
    }

    pub fn backward(self: RecurrentLayer,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer,
        h_states: []const f32, h_states_buf: ?*const metal.MTLBuffer
    ) !void {
        switch (self) {
            .rnn => |l| try l.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, h_states, h_states_buf),
            .lstm => |l| try l.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, h_states, h_states_buf),
            .gru => |l| try l.backward(input, input_buf, grad_output, grad_output_buf, grad_input, grad_input_buf, h_states, h_states_buf),
        }
    }

    pub fn updateWeights(self: RecurrentLayer, learning_rate: f32, weight_decay: f32) !void {
        switch (self) {
            .rnn => |l| {
                try l.backend.sgdUpdate(l.weights_ih.slice, l.weights_ih.getMtlBuffer(), l.grad_weights_ih.slice, l.grad_weights_ih.getMtlBuffer(), learning_rate, weight_decay);
                try l.backend.sgdUpdate(l.weights_hh.slice, l.weights_hh.getMtlBuffer(), l.grad_weights_hh.slice, l.grad_weights_hh.getMtlBuffer(), learning_rate, weight_decay);
                try l.backend.sgdUpdateBias(l.bias.slice, l.bias.getMtlBuffer(), l.grad_bias.slice, l.grad_bias.getMtlBuffer(), learning_rate);
            },
            .lstm => |l| {
                try l.backend.sgdUpdate(l.weights_ih.slice, l.weights_ih.getMtlBuffer(), l.grad_weights_ih.slice, l.grad_weights_ih.getMtlBuffer(), learning_rate, weight_decay);
                try l.backend.sgdUpdate(l.weights_hh.slice, l.weights_hh.getMtlBuffer(), l.grad_weights_hh.slice, l.grad_weights_hh.getMtlBuffer(), learning_rate, weight_decay);
                try l.backend.sgdUpdateBias(l.bias.slice, l.bias.getMtlBuffer(), l.grad_bias.slice, l.grad_bias.getMtlBuffer(), learning_rate);
            },
            .gru => |l| {
                try l.backend.sgdUpdate(l.weights_ih.slice, l.weights_ih.getMtlBuffer(), l.grad_weights_ih.slice, l.grad_weights_ih.getMtlBuffer(), learning_rate, weight_decay);
                try l.backend.sgdUpdate(l.weights_hh.slice, l.weights_hh.getMtlBuffer(), l.grad_weights_hh.slice, l.grad_weights_hh.getMtlBuffer(), learning_rate, weight_decay);
                try l.backend.sgdUpdateBias(l.bias.slice, l.bias.getMtlBuffer(), l.grad_bias.slice, l.grad_bias.getMtlBuffer(), learning_rate);
            },
        }
    }

    pub fn inputSize(self: RecurrentLayer) usize {
        switch (self) {
            .rnn => |l| return l.input_size,
            .lstm => |l| return l.input_size,
            .gru => |l| return l.input_size,
        }
    }

    pub fn hiddenSize(self: RecurrentLayer) usize {
        switch (self) {
            .rnn => |l| return l.hidden_size,
            .lstm => |l| return l.hidden_size,
            .gru => |l| return l.hidden_size,
        }
    }

    pub fn getHPrevWork(self: RecurrentLayer) *tensor.Tensor {
        switch (self) {
            .rnn => |l| return &l.h_prev_work,
            .lstm => |l| return &l.h_prev_work,
            .gru => |l| return &l.h_prev_work,
        }
    }

    pub fn getWeights(self: RecurrentLayer) *tensor.Tensor {
        switch (self) {
            .rnn => |l| return &l.weights_ih,
            .lstm => |l| return &l.weights_ih,
            .gru => |l| return &l.weights_ih,
        }
    }

    pub fn getGradWeights(self: RecurrentLayer) *tensor.Tensor {
        switch (self) {
            .rnn => |l| return &l.grad_weights_ih,
            .lstm => |l| return &l.grad_weights_ih,
            .gru => |l| return &l.grad_weights_ih,
        }
    }

    pub fn getBias(self: RecurrentLayer) *tensor.Tensor {
        switch (self) {
            .rnn => |l| return &l.bias,
            .lstm => |l| return &l.bias,
            .gru => |l| return &l.bias,
        }
    }

    pub fn getGradBias(self: RecurrentLayer) *tensor.Tensor {
        switch (self) {
            .rnn => |l| return &l.grad_bias,
            .lstm => |l| return &l.grad_bias,
            .gru => |l| return &l.grad_bias,
        }
    }

    pub fn getCPrevWork(self: RecurrentLayer) ?*tensor.Tensor {
        switch (self) {
            .lstm => |l| {
                 // Cell states are stored in cell_states buffer.
                 // This is a bit complex for a simple getter since we need the last one.
                 return &l.cell_states;
            },
            else => return null,
        }
    }
};

/// Seq2seq Wrapper
/// Encapsulates an encoder and a decoder recurrent layer
pub const Seq2seq = struct {
    encoder: RecurrentLayer,
    decoder: RecurrentLayer,
    input_size: usize,
    hidden_size: usize,
    max_seq_len: usize,
    allocator: std.mem.Allocator,
    backend: backend_module.Backend,

    // Buffers for zero-allocation
    enc_outputs_work: tensor.Tensor, // [max_seq_len * hidden_size]
    dec_input_work: tensor.Tensor,   // [max_seq_len * hidden_size]

    pub fn init(allocator: std.mem.Allocator, encoder_type: RecurrentLayerType, decoder_type: RecurrentLayerType, input_size: usize, hidden_size: usize, max_seq_len: usize, act: activation.Activation, backend: backend_module.Backend) !*Seq2seq {
        const self = try allocator.create(Seq2seq);
        errdefer allocator.destroy(self);

        self.encoder = switch (encoder_type) {
            .rnn => .{ .rnn = try VanillaRNN.init(allocator, input_size, hidden_size, act, backend) },
            .lstm => .{ .lstm = try LSTM.init(allocator, input_size, hidden_size, max_seq_len, backend) },
            .gru => .{ .gru = try GRU.init(allocator, input_size, hidden_size, max_seq_len, backend) },
        };
        errdefer self.encoder.deinit();

        self.decoder = switch (decoder_type) {
            .rnn => .{ .rnn = try VanillaRNN.init(allocator, hidden_size, hidden_size, act, backend) }, // Decoder input is encoder hidden
            .lstm => .{ .lstm = try LSTM.init(allocator, hidden_size, hidden_size, max_seq_len, backend) },
            .gru => .{ .gru = try GRU.init(allocator, hidden_size, hidden_size, max_seq_len, backend) },
        };
        errdefer self.decoder.deinit();

        self.enc_outputs_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size }, backend);
        errdefer self.enc_outputs_work.deinit();
        self.dec_input_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size }, backend);
        errdefer self.dec_input_work.deinit();

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.max_seq_len = max_seq_len;
        self.backend = backend;
        self.allocator = allocator;

        return self;
    }

    pub fn deinit(self: *Seq2seq) void {
        self.encoder.deinit();
        self.decoder.deinit();
        self.enc_outputs_work.deinit();
        self.dec_input_work.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Seq2seq,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        const enc_seq_len = input.len / self.input_size;
        const dec_seq_len = output.len / self.hidden_size;
        if (enc_seq_len > self.max_seq_len or dec_seq_len > self.max_seq_len) return error.SequenceTooLong;
        const hidden_size = self.hidden_size;

        // 1. Encoder forward
        const enc_outputs = self.enc_outputs_work.slice[0 .. enc_seq_len * hidden_size];
        try self.encoder.forward(input, input_buf, enc_outputs, self.enc_outputs_work.getMtlBuffer());

        // 2. Transfer state from encoder to decoder
        // Last hidden state of encoder becomes initial hidden state of decoder
        // Sync to CPU if needed to find last_h
        try self.backend.copyData(enc_outputs, self.enc_outputs_work.getMtlBuffer(), enc_outputs, null);
        const last_h = enc_outputs[(enc_seq_len - 1) * hidden_size .. enc_seq_len * hidden_size];
        const dec_h_prev = self.decoder.getHPrevWork();
        try self.backend.copyData(last_h, null, dec_h_prev.slice, dec_h_prev.getMtlBuffer());

        // 3. Decoder forward
        const dec_input = self.dec_input_work.slice[0 .. dec_seq_len * hidden_size];
        @memset(dec_input, 0);
        try self.backend.copyData(dec_input, null, dec_input, self.dec_input_work.getMtlBuffer());

        try self.decoder.forward(dec_input, self.dec_input_work.getMtlBuffer(), output, output_buf);
    }

    // Backward and updateWeights omitted for brevity in this complex multi-layer wrapper,
    // but follow similar patterns of state transfer.
};

/// Bidirectional Wrapper
pub const Bidirectional = struct {
    fw_layer: RecurrentLayer,
    bw_layer: RecurrentLayer,
    input_size: usize,
    hidden_size: usize,
    max_seq_len: usize,
    allocator: std.mem.Allocator,
    backend: backend_module.Backend,

    // Buffers for zero-allocation
    fw_output_work: tensor.Tensor,      // [max_seq_len * hidden_size]
    bw_output_raw_work: tensor.Tensor,  // [max_seq_len * hidden_size]
    reversed_input_work: tensor.Tensor, // [max_seq_len * input_size]
    grad_fw_work: tensor.Tensor,       // [max_seq_len * hidden_size]
    grad_bw_rev_work: tensor.Tensor,   // [max_seq_len * hidden_size]
    fw_h_states_work: tensor.Tensor,   // [max_seq_len * hidden_size]
    bw_h_states_rev_work: tensor.Tensor, // [max_seq_len * hidden_size]
    grad_input_fw_work: tensor.Tensor, // [max_seq_len * input_size]
    grad_input_bw_rev_work: tensor.Tensor, // [max_seq_len * input_size]

    pub fn init(allocator: std.mem.Allocator, layer_type: RecurrentLayerType, input_size: usize, hidden_size: usize, max_seq_len: usize, act: activation.Activation, backend: backend_module.Backend) !*Bidirectional {
        const self = try allocator.create(Bidirectional);
        errdefer allocator.destroy(self);

        self.fw_layer = switch (layer_type) {
            .rnn => .{ .rnn = try VanillaRNN.init(allocator, input_size, hidden_size, act, backend) },
            .lstm => .{ .lstm = try LSTM.init(allocator, input_size, hidden_size, max_seq_len, backend) },
            .gru => .{ .gru = try GRU.init(allocator, input_size, hidden_size, max_seq_len, backend) },
        };
        errdefer self.fw_layer.deinit();

        self.bw_layer = switch (layer_type) {
            .rnn => .{ .rnn = try VanillaRNN.init(allocator, input_size, hidden_size, act, backend) },
            .lstm => .{ .lstm = try LSTM.init(allocator, input_size, hidden_size, max_seq_len, backend) },
            .gru => .{ .gru = try GRU.init(allocator, input_size, hidden_size, max_seq_len, backend) },
        };
        errdefer self.bw_layer.deinit();

        self.fw_output_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size }, backend);
        errdefer self.fw_output_work.deinit();
        self.bw_output_raw_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size }, backend);
        errdefer self.bw_output_raw_work.deinit();
        self.reversed_input_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * input_size }, backend);
        errdefer self.reversed_input_work.deinit();

        self.grad_fw_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size }, backend);
        errdefer self.grad_fw_work.deinit();
        self.grad_bw_rev_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size }, backend);
        errdefer self.grad_bw_rev_work.deinit();
        self.fw_h_states_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size }, backend);
        errdefer self.fw_h_states_work.deinit();
        self.bw_h_states_rev_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size }, backend);
        errdefer self.bw_h_states_rev_work.deinit();
        self.grad_input_fw_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * input_size }, backend);
        errdefer self.grad_input_fw_work.deinit();
        self.grad_input_bw_rev_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * input_size }, backend);
        errdefer self.grad_input_bw_rev_work.deinit();

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.max_seq_len = max_seq_len;
        self.allocator = allocator;
        self.backend = backend;

        return self;
    }

    pub fn deinit(self: *Bidirectional) void {
        self.fw_layer.deinit();
        self.bw_layer.deinit();
        self.fw_output_work.deinit();
        self.bw_output_raw_work.deinit();
        self.reversed_input_work.deinit();
        self.grad_fw_work.deinit();
        self.grad_bw_rev_work.deinit();
        self.fw_h_states_work.deinit();
        self.bw_h_states_rev_work.deinit();
        self.grad_input_fw_work.deinit();
        self.grad_input_bw_rev_work.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Bidirectional,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        const seq_len = input.len / self.input_size;
        if (seq_len > self.max_seq_len) return error.SequenceTooLong;
        const hidden_size = self.hidden_size;

        // 1. Forward pass for forward layer
        const fw_output = self.fw_output_work.slice[0 .. seq_len * hidden_size];
        try self.fw_layer.forward(input, input_buf, fw_output, self.fw_output_work.getMtlBuffer());

        // 2. Backward pass for backward layer (process reversed input)
        const reversed_input = self.reversed_input_work.slice[0..input.len];
        for (0..seq_len) |t| {
            const src_t = seq_len - 1 - t;
            @memcpy(reversed_input[t * self.input_size .. (t + 1) * self.input_size], input[src_t * self.input_size .. (src_t + 1) * self.input_size]);
        }
        // Sync to GPU if needed
        try self.backend.copyData(reversed_input, null, reversed_input, self.reversed_input_work.getMtlBuffer());

        const bw_output_raw = self.bw_output_raw_work.slice[0 .. seq_len * hidden_size];
        try self.bw_layer.forward(reversed_input, self.reversed_input_work.getMtlBuffer(), bw_output_raw, self.bw_output_raw_work.getMtlBuffer());

        // 3. Reverse backward output and concatenate
        // Sync back to CPU for concatenation if needed (if output_buf is null)
        if (output_buf == null) {
            try self.backend.copyData(fw_output, self.fw_output_work.getMtlBuffer(), fw_output, null);
            try self.backend.copyData(bw_output_raw, self.bw_output_raw_work.getMtlBuffer(), bw_output_raw, null);
        }

        for (0..seq_len) |t| {
            const fw_t = fw_output[t * hidden_size .. (t + 1) * hidden_size];
            const src_t = seq_len - 1 - t;
            const bw_t = bw_output_raw[src_t * hidden_size .. (src_t + 1) * hidden_size];

            const out_t = output[t * 2 * hidden_size .. (t + 1) * 2 * hidden_size];
            @memcpy(out_t[0..hidden_size], fw_t);
            @memcpy(out_t[hidden_size .. 2 * hidden_size], bw_t);
        }

        // Sync concatenated output to GPU if needed
        if (output_buf != null) {
            try self.backend.copyData(output, null, output, output_buf);
        }
    }

    pub fn backward(self: *Bidirectional,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer,
        h_states: []const f32, h_states_buf: ?*const metal.MTLBuffer
    ) !void {
        const seq_len = input.len / self.input_size;
        const hidden_size = self.hidden_size;

        // 1. Split grad_output and h_states
        // Sync from GPU if needed
        if (grad_output_buf != null) {
             // We need to work on slices, so sync to local buffer first
             try self.backend.copyData(grad_output, grad_output_buf, @constCast(grad_output), null);
        }
        if (h_states_buf != null) {
             try self.backend.copyData(h_states, h_states_buf, @constCast(h_states), null);
        }

        const grad_fw = self.grad_fw_work.slice[0 .. seq_len * hidden_size];
        const grad_bw_rev = self.grad_bw_rev_work.slice[0 .. seq_len * hidden_size];
        const fw_h_states = self.fw_h_states_work.slice[0 .. seq_len * hidden_size];
        const bw_h_states_rev = self.bw_h_states_rev_work.slice[0 .. seq_len * hidden_size];

        for (0..seq_len) |t| {
            const g_out_t = grad_output[t * 2 * hidden_size .. (t + 1) * 2 * hidden_size];
            @memcpy(grad_fw[t * hidden_size .. (t + 1) * hidden_size], g_out_t[0..hidden_size]);

            const bw_t = seq_len - 1 - t;
            @memcpy(grad_bw_rev[bw_t * hidden_size .. (bw_t + 1) * hidden_size], g_out_t[hidden_size .. 2 * hidden_size]);

            const h_t = h_states[t * 2 * hidden_size .. (t + 1) * 2 * hidden_size];
            @memcpy(fw_h_states[t * hidden_size .. (t + 1) * hidden_size], h_t[0..hidden_size]);
            @memcpy(bw_h_states_rev[bw_t * hidden_size .. (bw_t + 1) * hidden_size], h_t[hidden_size .. 2 * hidden_size]);
        }

        // Sync split buffers to GPU
        try self.backend.copyData(grad_fw, null, grad_fw, self.grad_fw_work.getMtlBuffer());
        try self.backend.copyData(grad_bw_rev, null, grad_bw_rev, self.grad_bw_rev_work.getMtlBuffer());
        try self.backend.copyData(fw_h_states, null, fw_h_states, self.fw_h_states_work.getMtlBuffer());
        try self.backend.copyData(bw_h_states_rev, null, bw_h_states_rev, self.bw_h_states_rev_work.getMtlBuffer());

        // 3. Reversed input for backward layer
        const reversed_input = self.reversed_input_work.slice[0..input.len];
        for (0..seq_len) |t| {
            const src_t = seq_len - 1 - t;
            @memcpy(reversed_input[t * self.input_size .. (t + 1) * self.input_size], input[src_t * self.input_size .. (src_t + 1) * self.input_size]);
        }
        try self.backend.copyData(reversed_input, null, reversed_input, self.reversed_input_work.getMtlBuffer());

        // 4. Backprop through both layers
        const grad_input_fw = self.grad_input_fw_work.slice[0..input.len];
        try self.fw_layer.backward(input, input_buf, grad_fw, self.grad_fw_work.getMtlBuffer(), grad_input_fw, self.grad_input_fw_work.getMtlBuffer(), fw_h_states, self.fw_h_states_work.getMtlBuffer());

        const grad_input_bw_rev = self.grad_input_bw_rev_work.slice[0..input.len];
        try self.bw_layer.backward(reversed_input, self.reversed_input_work.getMtlBuffer(), grad_bw_rev, self.grad_bw_rev_work.getMtlBuffer(), grad_input_bw_rev, self.grad_input_bw_rev_work.getMtlBuffer(), bw_h_states_rev, self.bw_h_states_rev_work.getMtlBuffer());

        // 5. Combine grad_input
        // Sync back to CPU for combination
        try self.backend.copyData(grad_input_fw, self.grad_input_fw_work.getMtlBuffer(), grad_input_fw, null);
        try self.backend.copyData(grad_input_bw_rev, self.grad_input_bw_rev_work.getMtlBuffer(), grad_input_bw_rev, null);

        for (0..seq_len) |t| {
            const fw_gi_t = grad_input_fw[t * self.input_size .. (t + 1) * self.input_size];
            const bw_t = seq_len - 1 - t;
            const bw_gi_t = grad_input_bw_rev[bw_t * self.input_size .. (bw_t + 1) * self.input_size];

            const gi_t = grad_input[t * self.input_size .. (t + 1) * self.input_size];
            for (gi_t, fw_gi_t, bw_gi_t) |*out, f, b| out.* = f + b;
        }

        // Sync final grad_input to GPU
        if (grad_input_buf != null) {
            try self.backend.copyData(grad_input, null, grad_input, grad_input_buf);
        }
    }
};

/// TwoPath Wrapper (Parallel paths)
pub const TwoPath = struct {
    path1: RecurrentLayer,
    path2: RecurrentLayer,
    input_size: usize,
    hidden_size1: usize,
    hidden_size2: usize,
    max_seq_len: usize,
    allocator: std.mem.Allocator,
    backend: backend_module.Backend,

    // Buffers for zero-allocation
    out1_work: tensor.Tensor,      // [max_seq_len * hidden_size1]
    out2_work: tensor.Tensor,      // [max_seq_len * hidden_size2]
    grad_h1_work: tensor.Tensor,   // [max_seq_len * hidden_size1]
    grad_h2_work: tensor.Tensor,   // [max_seq_len * hidden_size2]
    h1_states_work: tensor.Tensor, // [max_seq_len * hidden_size1]
    h2_states_work: tensor.Tensor, // [max_seq_len * hidden_size2]
    gi1_work: tensor.Tensor,       // [max_seq_len * input_size]
    gi2_work: tensor.Tensor,       // [max_seq_len * input_size]

    pub fn init(allocator: std.mem.Allocator, layer_type1: RecurrentLayerType, hidden_size1: usize, layer_type2: RecurrentLayerType, hidden_size2: usize, input_size: usize, max_seq_len: usize, act: activation.Activation, backend: backend_module.Backend) !*TwoPath {
        const self = try allocator.create(TwoPath);
        errdefer allocator.destroy(self);

        self.path1 = switch (layer_type1) {
            .rnn => .{ .rnn = try VanillaRNN.init(allocator, input_size, hidden_size1, act, backend) },
            .lstm => .{ .lstm = try LSTM.init(allocator, input_size, hidden_size1, max_seq_len, backend) },
            .gru => .{ .gru = try GRU.init(allocator, input_size, hidden_size1, max_seq_len, backend) },
        };
        errdefer self.path1.deinit();

        self.path2 = switch (layer_type2) {
            .rnn => .{ .rnn = try VanillaRNN.init(allocator, input_size, hidden_size2, act, backend) },
            .lstm => .{ .lstm = try LSTM.init(allocator, input_size, hidden_size2, max_seq_len, backend) },
            .gru => .{ .gru = try GRU.init(allocator, input_size, hidden_size2, max_seq_len, backend) },
        };
        errdefer self.path2.deinit();

        self.out1_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size1 }, backend);
        errdefer self.out1_work.deinit();
        self.out2_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size2 }, backend);
        errdefer self.out2_work.deinit();

        self.grad_h1_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size1 }, backend);
        errdefer self.grad_h1_work.deinit();
        self.grad_h2_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size2 }, backend);
        errdefer self.grad_h2_work.deinit();
        self.h1_states_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size1 }, backend);
        errdefer self.h1_states_work.deinit();
        self.h2_states_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * hidden_size2 }, backend);
        errdefer self.h2_states_work.deinit();

        self.gi1_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * input_size }, backend);
        errdefer self.gi1_work.deinit();
        self.gi2_work = try tensor.Tensor.init(allocator, &.{ max_seq_len * input_size }, backend);
        errdefer self.gi2_work.deinit();

        self.input_size = input_size;
        self.hidden_size1 = hidden_size1;
        self.hidden_size2 = hidden_size2;
        self.max_seq_len = max_seq_len;
        self.allocator = allocator;
        self.backend = backend;

        return self;
    }

    pub fn deinit(self: *TwoPath) void {
        self.path1.deinit();
        self.path2.deinit();
        self.out1_work.deinit();
        self.out2_work.deinit();
        self.grad_h1_work.deinit();
        self.grad_h2_work.deinit();
        self.h1_states_work.deinit();
        self.h2_states_work.deinit();
        self.gi1_work.deinit();
        self.gi2_work.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *TwoPath,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        output: []f32, output_buf: ?*const metal.MTLBuffer
    ) !void {
        const seq_len = input.len / self.input_size;
        if (seq_len > self.max_seq_len) return error.SequenceTooLong;

        const out1 = self.out1_work.slice[0 .. seq_len * self.hidden_size1];
        try self.path1.forward(input, input_buf, out1, self.out1_work.getMtlBuffer());

        const out2 = self.out2_work.slice[0 .. seq_len * self.hidden_size2];
        try self.path2.forward(input, input_buf, out2, self.out2_work.getMtlBuffer());

        // Sync back to CPU for concatenation if needed
        if (output_buf == null) {
            try self.backend.copyData(out1, self.out1_work.getMtlBuffer(), out1, null);
            try self.backend.copyData(out2, self.out2_work.getMtlBuffer(), out2, null);
        }

        for (0..seq_len) |t| {
            const out_t = output[t * (self.hidden_size1 + self.hidden_size2) .. (t + 1) * (self.hidden_size1 + self.hidden_size2)];
            @memcpy(out_t[0..self.hidden_size1], out1[t * self.hidden_size1 .. (t + 1) * self.hidden_size1]);
            @memcpy(out_t[self.hidden_size1..], out2[t * self.hidden_size2 .. (t + 1) * self.hidden_size2]);
        }

        if (output_buf != null) {
            try self.backend.copyData(output, null, output, output_buf);
        }
    }

    pub fn backward(self: *TwoPath,
        input: []const f32, input_buf: ?*const metal.MTLBuffer,
        grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer,
        grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer,
        h_states: []const f32, h_states_buf: ?*const metal.MTLBuffer
    ) !void {
        const seq_len = input.len / self.input_size;

        // Split grad_output and h_states
        if (grad_output_buf != null) {
             try self.backend.copyData(grad_output, grad_output_buf, @constCast(grad_output), null);
        }
        if (h_states_buf != null) {
             try self.backend.copyData(h_states, h_states_buf, @constCast(h_states), null);
        }

        const grad_h1 = self.grad_h1_work.slice[0 .. seq_len * self.hidden_size1];
        const grad_h2 = self.grad_h2_work.slice[0 .. seq_len * self.hidden_size2];
        const h1_states = self.h1_states_work.slice[0 .. seq_len * self.hidden_size1];
        const h2_states = self.h2_states_work.slice[0 .. seq_len * self.hidden_size2];

        for (0..seq_len) |t| {
            const g_out_t = grad_output[t * (self.hidden_size1 + self.hidden_size2) .. (t + 1) * (self.hidden_size1 + self.hidden_size2)];
            @memcpy(grad_h1[t * self.hidden_size1 .. (t + 1) * self.hidden_size1], g_out_t[0..self.hidden_size1]);
            @memcpy(grad_h2[t * self.hidden_size2 .. (t + 1) * self.hidden_size2], g_out_t[self.hidden_size1..]);

            const h_t = h_states[t * (self.hidden_size1 + self.hidden_size2) .. (t + 1) * (self.hidden_size1 + self.hidden_size2)];
            @memcpy(h1_states[t * self.hidden_size1 .. (t + 1) * self.hidden_size1], h_t[0..self.hidden_size1]);
            @memcpy(h2_states[t * self.hidden_size2 .. (t + 1) * self.hidden_size2], h_t[self.hidden_size1..]);
        }

        // Sync to GPU
        try self.backend.copyData(grad_h1, null, grad_h1, self.grad_h1_work.getMtlBuffer());
        try self.backend.copyData(grad_h2, null, grad_h2, self.grad_h2_work.getMtlBuffer());
        try self.backend.copyData(h1_states, null, h1_states, self.h1_states_work.getMtlBuffer());
        try self.backend.copyData(h2_states, null, h2_states, self.h2_states_work.getMtlBuffer());

        const gi1 = self.gi1_work.slice[0..input.len];
        try self.path1.backward(input, input_buf, grad_h1, self.grad_h1_work.getMtlBuffer(), gi1, self.gi1_work.getMtlBuffer(), h1_states, self.h1_states_work.getMtlBuffer());

        const gi2 = self.gi2_work.slice[0..input.len];
        try self.path2.backward(input, input_buf, grad_h2, self.grad_h2_work.getMtlBuffer(), gi2, self.gi2_work.getMtlBuffer(), h2_states, self.h2_states_work.getMtlBuffer());

        // Sync back to CPU for combination
        try self.backend.copyData(gi1, self.gi1_work.getMtlBuffer(), gi1, null);
        try self.backend.copyData(gi2, self.gi2_work.getMtlBuffer(), gi2, null);

        for (grad_input, gi1, gi2) |*out, g1, g2| out.* = g1 + g2;

        if (grad_input_buf != null) {
            try self.backend.copyData(grad_input, null, grad_input, grad_input_buf);
        }
    }
};

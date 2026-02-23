#include <metal_stdlib>
using namespace metal;

// LSTM forward step: gate activations and state updates
// gates_ih and gates_hh are each [4 * hidden_size]
// bias is [4 * hidden_size]
// c_prev is [hidden_size]
// c_curr and h_curr are [hidden_size]
kernel void lstm_forward_step(
    device const float* gates_ih [[buffer(0)]],
    device const float* gates_hh [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* c_prev [[buffer(3)]],
    device float* c_curr [[buffer(4)]],
    device float* h_curr [[buffer(5)]],
    device float* gate_acts [[buffer(6)]], // To store activated gates for backward pass
    constant uint& hidden_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < hidden_size) {
        uint h = hidden_size;

        // Gate indices
        float gi_i = gates_ih[gid];
        float gi_f = gates_ih[h + gid];
        float gi_g = gates_ih[2 * h + gid];
        float gi_o = gates_ih[3 * h + gid];

        float gh_i = gates_hh[gid];
        float gh_f = gates_hh[h + gid];
        float gh_g = gates_hh[2 * h + gid];
        float gh_o = gates_hh[3 * h + gid];

        float b_i = bias[gid];
        float b_f = bias[h + gid];
        float b_g = bias[2 * h + gid];
        float b_o = bias[3 * h + gid];

        // Activate gates
        float i = 1.0f / (1.0f + exp(-(gi_i + gh_i + b_i)));
        float f = 1.0f / (1.0f + exp(-(gi_f + gh_f + b_f)));
        float g = tanh(gi_g + gh_g + b_g);
        float o = 1.0f / (1.0f + exp(-(gi_o + gh_o + b_o)));

        // Store activations
        gate_acts[gid] = i;
        gate_acts[h + gid] = f;
        gate_acts[2 * h + gid] = g;
        gate_acts[3 * h + gid] = o;

        // Update cell state
        float cp = c_prev[gid];
        float cc = f * cp + i * g;
        c_curr[gid] = cc;

        // Update hidden state
        h_curr[gid] = o * tanh(cc);
    }
}

// GRU forward step
kernel void gru_forward_step(
    device const float* gates_ih [[buffer(0)]], // [3 * hidden_size]
    device const float* gates_hh [[buffer(1)]], // [3 * hidden_size]
    device const float* bias [[buffer(2)]],     // [3 * hidden_size]
    device const float* h_prev [[buffer(3)]],   // [hidden_size]
    device float* h_curr [[buffer(4)]],         // [hidden_size]
    device float* gate_acts [[buffer(5)]],      // [3 * hidden_size] (z, r, n)
    device float* n_hh_out [[buffer(6)]],       // [hidden_size] Store for backward
    constant uint& hidden_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < hidden_size) {
        uint h = hidden_size;

        // z_t = sigmoid(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_{hz})
        float z = 1.0f / (1.0f + exp(-(gates_ih[gid] + gates_hh[gid] + bias[gid])));

        // r_t = sigmoid(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_{hr})
        float r = 1.0f / (1.0f + exp(-(gates_ih[h + gid] + gates_hh[h + gid] + bias[h + gid])));

        // n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_{t-1} + b_{hn}))
        // Note: PyTorch handles bias differently but we use a combined one here
        float n_hh = gates_hh[2 * h + gid];
        float n = tanh(gates_ih[2 * h + gid] + bias[2 * h + gid] + r * n_hh);

        // h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        float hp = h_prev[gid];
        h_curr[gid] = (1.0f - z) * n + z * hp;

        // Store for backward
        gate_acts[gid] = z;
        gate_acts[h + gid] = r;
        gate_acts[2 * h + gid] = n;
        n_hh_out[gid] = n_hh;
    }
}

// LSTM backward step: calculate d_gates and d_c_prev
kernel void lstm_backward_step(
    device const float* grad_h_curr [[buffer(0)]],
    device const float* grad_h_next [[buffer(1)]],
    device const float* grad_c_next [[buffer(2)]],
    device const float* gate_acts [[buffer(3)]],
    device const float* c_curr [[buffer(4)]],
    device const float* c_prev [[buffer(5)]],
    device float* grad_gates [[buffer(6)]], // [4 * hidden_size]
    device float* grad_c_prev [[buffer(7)]],
    device float* grad_h_prev_part [[buffer(8)]], // Partial grad_h_prev to be used for W_hh matmul
    constant uint& hidden_size [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < hidden_size) {
        uint h = hidden_size;
        float i = gate_acts[gid];
        float f = gate_acts[h + gid];
        float g = gate_acts[2 * h + gid];
        float o = gate_acts[3 * h + gid];

        float gh = grad_h_curr[gid] + grad_h_next[gid];
        float tanh_cc = tanh(c_curr[gid]);

        // d_o = d_h * tanh(c_t) * sigmoid'(o_t)
        float d_o = gh * tanh_cc * (o * (1.0f - o));

        // d_c = d_h * o * tanh'(c_t) + d_c_next
        float d_c = gh * o * (1.0f - tanh_cc * tanh_cc) + grad_c_next[gid];

        // d_f = d_c * c_{t-1} * sigmoid'(f_t)
        float d_f = d_c * c_prev[gid] * (f * (1.0f - f));
        // d_i = d_c * g_t * sigmoid'(i_t)
        float d_i = d_c * g * (i * (1.0f - i));
        // d_g = d_c * i_t * tanh'(g_t)
        float d_g = d_c * i * (1.0f - g * g);

        grad_gates[gid] = d_i;
        grad_gates[h + gid] = d_f;
        grad_gates[2 * h + gid] = d_g;
        grad_gates[3 * h + gid] = d_o;

        // grad_c_prev = d_c * f
        grad_c_prev[gid] = d_c * f;
    }
}

// GRU backward step
kernel void gru_backward_step(
    device const float* grad_h_curr [[buffer(0)]],
    device const float* grad_h_next [[buffer(1)]],
    device const float* gate_acts [[buffer(2)]], // (z, r, n)
    device const float* h_prev [[buffer(3)]],
    device const float* n_hh [[buffer(4)]],     // W_hh * h_prev [n component]
    device float* grad_gates_ih [[buffer(5)]],  // [3 * hidden_size]
    device float* grad_gates_hh [[buffer(6)]],  // [3 * hidden_size]
    device float* grad_h_prev [[buffer(7)]],
    constant uint& hidden_size [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < hidden_size) {
        uint h = hidden_size;
        float z = gate_acts[gid];
        float r = gate_acts[h + gid];
        float n = gate_acts[2 * h + gid];

        float gh = grad_h_curr[gid] + grad_h_next[gid];
        float hp = h_prev[gid];

        // h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        // d_nt = d_h * (1 - z_t)
        float d_nt = gh * (1.0f - z);
        // d_zt = d_h * (h_{t-1} - n_t)
        float d_zt = gh * (hp - n);

        // n_t = tanh(n_ih + r_t * n_hh)
        float d_n_raw = d_nt * (1.0f - n * n);
        float d_n_ih = d_n_raw;
        float d_n_hh = d_n_raw * r;

        // d_rt = d_n_raw * n_hh * sigmoid'(r_t)
        float d_rt = d_n_raw * n_hh[gid] * (r * (1.0f - r));

        // z_t = sigmoid(z_raw) => d_z_raw = d_zt * z_t * (1 - z_t)
        float d_z_raw = d_zt * z * (1.0f - z);

        // Gates IH grads
        grad_gates_ih[gid] = d_z_raw;
        grad_gates_ih[h + gid] = d_rt;
        grad_gates_ih[2 * h + gid] = d_n_ih;

        // Gates HH grads
        grad_gates_hh[gid] = d_z_raw;
        grad_gates_hh[h + gid] = d_rt;
        grad_gates_hh[2 * h + gid] = d_n_hh;

        // grad_h_prev = d_h * z_t
        grad_h_prev[gid] = gh * z;
    }
}

// Vanilla RNN forward step
kernel void rnn_forward_step(
    device const float* gates_ih [[buffer(0)]],
    device const float* gates_hh [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* h_curr [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < hidden_size) {
        float val = gates_ih[gid] + gates_hh[gid] + bias[gid];
        h_curr[gid] = tanh(val);
    }
}

// Vanilla RNN backward step
kernel void rnn_backward_step(
    device const float* grad_h_curr [[buffer(0)]],
    device const float* grad_h_next [[buffer(1)]],
    device const float* h_curr [[buffer(2)]],
    device float* grad_after_act [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < hidden_size) {
        float gh = grad_h_curr[gid] + grad_h_next[gid];
        float h = h_curr[gid];
        grad_after_act[gid] = gh * (1.0f - h * h);
    }
}

// CUDA Kernels for Recurrent Layers
// LSTM, GRU, and vanilla RNN implementations

#include "common.h"

// LSTM forward step
// gates_ih: [4 * hidden_size]
// gates_hh: [4 * hidden_size]
// bias: [4 * hidden_size]
// c_prev: [hidden_size]
// c_curr: [hidden_size]
// h_curr: [hidden_size]
// gate_acts: [4 * hidden_size] (output for backward)
extern "C" __global__ void lstm_forward_step(
    const float* __restrict__ gates_ih,
    const float* __restrict__ gates_hh,
    const float* __restrict__ bias,
    const float* __restrict__ c_prev,
    float* __restrict__ c_curr,
    float* __restrict__ h_curr,
    float* __restrict__ gate_acts,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= hidden_size) return;

    int h = hidden_size;

    // Gate indices
    float gi_i = gates_ih[idx];
    float gi_f = gates_ih[h + idx];
    float gi_g = gates_ih[2 * h + idx];
    float gi_o = gates_ih[3 * h + idx];

    float gh_i = gates_hh[idx];
    float gh_f = gates_hh[h + idx];
    float gh_g = gates_hh[2 * h + idx];
    float gh_o = gates_hh[3 * h + idx];

    float b_i = bias[idx];
    float b_f = bias[h + idx];
    float b_g = bias[2 * h + idx];
    float b_o = bias[3 * h + idx];

    // Activate gates
    float i_gate = sigmoidf_stable(gi_i + gh_i + b_i);
    float f_gate = sigmoidf_stable(gi_f + gh_f + b_f);
    float g_gate = tanhf(gi_g + gh_g + b_g);
    float o_gate = sigmoidf_stable(gi_o + gh_o + b_o);

    // Store gate activations for backward
    gate_acts[idx] = i_gate;
    gate_acts[h + idx] = f_gate;
    gate_acts[2 * h + idx] = g_gate;
    gate_acts[3 * h + idx] = o_gate;

    // Update cell state
    float cp = c_prev[idx];
    float cc = f_gate * cp + i_gate * g_gate;
    c_curr[idx] = cc;

    // Update hidden state
    h_curr[idx] = o_gate * tanhf(cc);
}

// LSTM backward step
extern "C" __global__ void lstm_backward_step(
    const float* __restrict__ grad_h_curr,
    const float* __restrict__ grad_h_next,
    const float* __restrict__ grad_c_next,
    const float* __restrict__ gate_acts,
    const float* __restrict__ c_curr,
    const float* __restrict__ c_prev,
    float* __restrict__ grad_gates,
    float* __restrict__ grad_c_prev,
    float* __restrict__ grad_h_prev_part,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= hidden_size) return;

    int h = hidden_size;
    float i_gate = gate_acts[idx];
    float f_gate = gate_acts[h + idx];
    float g_gate = gate_acts[2 * h + idx];
    float o_gate = gate_acts[3 * h + idx];

    float gh = grad_h_curr[idx] + grad_h_next[idx];
    float tanh_cc = tanhf(c_curr[idx]);

    // d_o = d_h * tanh(c_t) * sigmoid'(o_t)
    float d_o = gh * tanh_cc * (o_gate * (1.0f - o_gate));

    // d_c = d_h * o * tanh'(c_t) + d_c_next
    float d_c = gh * o_gate * (1.0f - tanh_cc * tanh_cc) + grad_c_next[idx];

    // d_f = d_c * c_{t-1} * sigmoid'(f_t)
    float d_f = d_c * c_prev[idx] * (f_gate * (1.0f - f_gate));

    // d_i = d_c * g_t * sigmoid'(i_t)
    float d_i = d_c * g_gate * (i_gate * (1.0f - i_gate));

    // d_g = d_c * i_t * tanh'(g_t)
    float d_g = d_c * i_gate * (1.0f - g_gate * g_gate);

    grad_gates[idx] = d_i;
    grad_gates[h + idx] = d_f;
    grad_gates[2 * h + idx] = d_g;
    grad_gates[3 * h + idx] = d_o;

    // grad_c_prev = d_c * f
    grad_c_prev[idx] = d_c * f_gate;
}

// GRU forward step
extern "C" __global__ void gru_forward_step(
    const float* __restrict__ gates_ih,  // [3 * hidden_size]
    const float* __restrict__ gates_hh,  // [3 * hidden_size]
    const float* __restrict__ bias,      // [3 * hidden_size]
    const float* __restrict__ h_prev,    // [hidden_size]
    float* __restrict__ h_curr,          // [hidden_size]
    float* __restrict__ gate_acts,       // [3 * hidden_size] (z, r, n)
    float* __restrict__ n_hh_out,        // [hidden_size] Store for backward
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= hidden_size) return;

    int h = hidden_size;

    // z_t = sigmoid(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)
    float z_gate = sigmoidf_stable(gates_ih[idx] + gates_hh[idx] + bias[idx]);

    // r_t = sigmoid(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)
    float r_gate = sigmoidf_stable(gates_ih[h + idx] + gates_hh[h + idx] + bias[h + idx]);

    // n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_{t-1} + b_hn))
    float n_hh = gates_hh[2 * h + idx];
    float n_gate = tanhf(gates_ih[2 * h + idx] + bias[2 * h + idx] + r_gate * n_hh);

    // h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    float hp = h_prev[idx];
    h_curr[idx] = (1.0f - z_gate) * n_gate + z_gate * hp;

    // Store for backward
    gate_acts[idx] = z_gate;
    gate_acts[h + idx] = r_gate;
    gate_acts[2 * h + idx] = n_gate;
    n_hh_out[idx] = n_hh;
}

// GRU backward step
extern "C" __global__ void gru_backward_step(
    const float* __restrict__ grad_h_curr,
    const float* __restrict__ grad_h_next,
    const float* __restrict__ gate_acts,
    const float* __restrict__ h_prev,
    const float* __restrict__ n_hh,
    float* __restrict__ grad_gates_ih,
    float* __restrict__ grad_gates_hh,
    float* __restrict__ grad_h_prev,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= hidden_size) return;

    int h = hidden_size;
    float z_gate = gate_acts[idx];
    float r_gate = gate_acts[h + idx];
    float n_gate = gate_acts[2 * h + idx];

    float gh = grad_h_curr[idx] + grad_h_next[idx];
    float hp = h_prev[idx];

    // h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    // d_nt = d_h * (1 - z_t)
    float d_nt = gh * (1.0f - z_gate);

    // d_zt = d_h * (h_{t-1} - n_t)
    float d_zt = gh * (hp - n_gate);

    // n_t = tanh(n_ih + r_t * n_hh)
    float d_n_raw = d_nt * (1.0f - n_gate * n_gate);
    float d_n_ih = d_n_raw;
    float d_n_hh = d_n_raw * r_gate;

    // d_rt = d_n_raw * n_hh * sigmoid'(r_t)
    float d_rt = d_n_raw * n_hh[idx] * (r_gate * (1.0f - r_gate));

    // d_z_raw = d_zt * z_t * (1 - z_t)
    float d_z_raw = d_zt * z_gate * (1.0f - z_gate);

    // Gates IH grads
    grad_gates_ih[idx] = d_z_raw;
    grad_gates_ih[h + idx] = d_rt;
    grad_gates_ih[2 * h + idx] = d_n_ih;

    // Gates HH grads
    grad_gates_hh[idx] = d_z_raw;
    grad_gates_hh[h + idx] = d_rt;
    grad_gates_hh[2 * h + idx] = d_n_hh;

    // grad_h_prev = d_h * z_t
    grad_h_prev[idx] = gh * z_gate;
}

// Vanilla RNN forward step
extern "C" __global__ void rnn_forward_step(
    const float* __restrict__ gates_ih,
    const float* __restrict__ gates_hh,
    const float* __restrict__ bias,
    float* __restrict__ h_curr,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= hidden_size) return;

    float val = gates_ih[idx] + gates_hh[idx] + bias[idx];
    h_curr[idx] = tanhf(val);
}

// Vanilla RNN backward step
extern "C" __global__ void rnn_backward_step(
    const float* __restrict__ grad_h_curr,
    const float* __restrict__ grad_h_next,
    const float* __restrict__ h_curr,
    float* __restrict__ grad_after_act,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= hidden_size) return;

    float gh = grad_h_curr[idx] + grad_h_next[idx];
    float h = h_curr[idx];
    grad_after_act[idx] = gh * (1.0f - h * h);
}

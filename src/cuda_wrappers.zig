
    // =========================================================================
    // CUDA Backend Operations
    // =========================================================================

    /// CUDA matrix multiplication wrapper
    fn cudaMatMul(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, m: usize, n: usize, k: usize, transpose_b: bool, accumulate: bool) !void {
        _ = a_buf; _ = b_buf; _ = c_buf;
        if (self.cuda_ctx) |ctx| {
            const backend = &ctx.backend;
            try backend.matMul(a, b, c, m, n, k, transpose_b, false, accumulate);
        } else {
            return error.CudaNotAvailable;
        }
    }

    /// CUDA batched matrix multiplication wrapper
    fn cudaMatMulBatch(self: Backend, a: []const f32, a_buf: ?*const metal.MTLBuffer, b: []const f32, b_buf: ?*const metal.MTLBuffer, c: []f32, c_buf: ?*const metal.MTLBuffer, batch_size: usize, n: usize, k: usize, accumulate: bool) !void {
        _ = a_buf; _ = b_buf; _ = c_buf;
        if (self.cuda_ctx) |ctx| {
            const backend = &ctx.backend;
            try backend.matMulBatch(a, b, c, batch_size, n, k, accumulate);
        } else {
            return error.CudaNotAvailable;
        }
    }

    /// CUDA activation forward wrapper
    fn cudaActivationForward(self: Backend, act: activation.Activation, input: []const f32, input_buf: ?*const metal.MTLBuffer, output: []f32, output_buf: ?*const metal.MTLBuffer) !void {
        _ = input_buf; _ = output_buf;
        if (self.cuda_ctx) |ctx| {
            const backend = &ctx.backend;
            switch (act) {
                .relu => try backend.reluForward(input, output),
                .sigmoid => try backend.sigmoidForward(input, output),
                .tanh => try backend.tanhForward(input, output),
                .softmax => try backend.softmaxForward(input, output, 1, input.len),
                else => return error.UnsupportedActivation,
            }
        } else {
            return error.CudaNotAvailable;
        }
    }

    /// CUDA activation backward wrapper
    fn cudaActivationBackward(self: Backend, act: activation.Activation, input: []const f32, input_buf: ?*const metal.MTLBuffer, grad_output: []const f32, grad_output_buf: ?*const metal.MTLBuffer, grad_input: []f32, grad_input_buf: ?*const metal.MTLBuffer) !void {
        _ = input_buf; _ = grad_output_buf; _ = grad_input_buf;
        if (self.cuda_ctx) |ctx| {
            const backend = &ctx.backend;
            switch (act) {
                .relu => try backend.reluBackward(input, grad_output, grad_input),
                .sigmoid => try backend.sigmoidBackward(input, grad_output, grad_input),
                .tanh => try backend.tanhBackward(input, grad_output, grad_input),
                else => return error.UnsupportedActivation,
            }
        } else {
            return error.CudaNotAvailable;
        }
    }

    /// CUDA loss backward wrapper
    fn cudaLossBackward(self: Backend, loss_fn: loss.Loss, output: []const f32, output_buf: ?*const metal.MTLBuffer, target: []const f32, target_buf: ?*const metal.MTLBuffer, grad_output: []f32, grad_output_buf: ?*const metal.MTLBuffer) !void {
        _ = output_buf; _ = target_buf; _ = grad_output_buf;
        if (self.cuda_ctx) |ctx| {
            const backend = &ctx.backend;
            switch (loss_fn) {
                .mse => try backend.mseBackward(output, target, grad_output),
                .cross_entropy => try backend.crossEntropyBackward(output, target, grad_output),
                else => return error.UnsupportedLoss,
            }
        } else {
            return error.CudaNotAvailable;
        }
    }

    /// CUDA SGD update wrapper
    fn cudaSgdUpdate(self: Backend, weights: []f32, weights_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, learning_rate: f32, weight_decay: f32) !void {
        _ = weights_buf; _ = gradients_buf;
        if (self.cuda_ctx) |ctx| {
            _ = ctx;
            _ = weights;
            _ = gradients;
            _ = learning_rate;
            _ = weight_decay;
            // TODO: Implement CUDA SGD
            return error.NotImplemented;
        } else {
            return error.CudaNotAvailable;
        }
    }

    /// CUDA Adam update wrapper
    fn cudaAdamUpdate(self: Backend, weights: []f32, weights_buf: ?*const metal.MTLBuffer, gradients: []const f32, gradients_buf: ?*const metal.MTLBuffer, m: []f32, m_buf: ?*const metal.MTLBuffer, v: []f32, v_buf: ?*const metal.MTLBuffer, lr: f32, beta1: f32, beta2: f32, eps: f32, bias_corr1: f32, bias_corr2: f32) !void {
        _ = weights_buf; _ = gradients_buf; _ = m_buf; _ = v_buf;
        if (self.cuda_ctx) |ctx| {
            const backend = &ctx.backend;
            try backend.adamUpdate(weights, gradients, m, v, lr, beta1, beta2, eps, bias_corr1, bias_corr2);
        } else {
            return error.CudaNotAvailable;
        }
    }
};

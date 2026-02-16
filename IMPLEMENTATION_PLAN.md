# Plano de Implementação: Otimizações de Desempenho Críticas

**Data:** 2026-02-16
**Objetivo:** Implementar shaders Metal, processamento em lote e vetorização SIMD

---

## 1. Implementar Shaders Metal para Aceleração Real do GPU

### **Fase 1.1: Estrutura de Shaders Metal**

**Arquivos a Criar:**
```
shaders/
├── metal/
│   ├── matmul.metal          # Multiplicação de matrizes
│   ├── activation.metal      # Funções de ativação
│   ├── loss.metal            # Funções de perda
│   └── common.h              # Cabeçalhos comuns
```

**Exemplo: matmul.metal**
```metal
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

kernel void matmul_batch(
    device const float* A [[buffer(0)]],  // [batch_size, K]
    device const float* B [[buffer(1)]],  // [K, N]
    device float* C [[buffer(2)]],        // [batch_size, N]
    constant uint& batch_size [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (batch < batch_size && row < 1 && col < N) {
        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            sum += A[batch * K + k] * B[k * N + col];
        }
        C[batch * N + col] = sum;
    }
}
```

**Exemplo: activation.metal**
```metal
kernel void relu_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = input[gid];
        output[gid] = max(0.0, x);
    }
}

kernel void sigmoid_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = input[gid];
        output[gid] = 1.0 / (1.0 + exp(-x));
    }
}

kernel void relu_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) {
        float x = input[gid];
        grad_input[gid] = (x > 0.0) ? grad_output[gid] : 0.0;
    }
}
```

### **Fase 1.2: Pipeline de Compilação**

**build.zig - Adicionar compilação de shaders Metal:**
```zig
// Metal shader compilation step
const enable_metal = b.option(bool, "metal", "Build with Metal support") orelse true;

if (enable_metal) {
    const compile_metal_step = b.step("compile-metal", "Compile Metal shaders to metallib");

    // Define Metal shader files
    const metal_shaders = [_][]const u8{
        "shaders/metal/matmul.metal",
        "shaders/metal/activation.metal",
        "shaders/metal/loss.metal",
    };

    for (metal_shaders) |shader| {
        const compile_cmd = b.addSystemCommand(&.{
            "xcrun", "-sdk", "macosx", "metal",
            "-c", shader,
            "-o", b.pathJoin(&.{ b.cache_root, std.fs.path.basename(shader) ++ ".air" }),
        });
        compile_shaders_step.dependOn(&compile_cmd.step);
    }

    // Link to metallib
    const link_cmd = b.addSystemCommand(&.{
        "xcrun", "-sdk", "macosx", "metallib",
    });
    // Add all .air files
    link_cmd.addArg("-o");
    link_cmd.addArg("shaders/metal/default.metallib");
    compile_metal_step.dependOn(link_cmd);
}
```

### **Fase 1.3: Integração Zig-Metal**

**src/metal.zig - Ligação Metal API:**
```zig
const std = @import("std");
const objc = @import("objc");

// Metal API bindings
pub const MTLDevice = opaque {
    pub fn create() !*MTLDevice {
        // Get default Metal device
        const cls = objc.getClass("MTLDevice");
        const device = objc.msgSend(cls, objc.sel("defaultDevice"), .{});
        return device;
    }

    pub fn newCommandQueue(self: *MTLDevice) !*MTLCommandQueue {
        return objc.msgSend(self, objc.sel("newCommandQueue"), .{});
    }

    pub fn newBufferWithBytes(self: *MTLDevice,
        bytes: [*]const u8,
        length: usize,
        options: MTLResourceOptions
    ) !*MTLBuffer {
        return objc.msgSend(self, objc.sel("newBufferWithBytes:length:options:"), .{
            bytes, length, options
        });
    }
};

pub const MTLCommandQueue = opaque {
    pub fn commandBuffer(self: *MTLCommandQueue) !*MTLCommandBuffer {
        return objc.msgSend(self, objc.sel("commandBuffer"), .{});
    }
};

pub const MTLCommandBuffer = opaque {
    pub fn computeCommandEncoder(self: *MTLCommandBuffer) !*MTLComputeCommandEncoder {
        return objc.msgSend(self, objc.sel("computeCommandEncoder"), .{});
    }

    pub fn commit(self: *MTLCommandBuffer) void {
        objc.msgSend(self, objc.sel("commit"), .{});
    }

    pub fn waitUntilCompleted(self: *MTLCommandBuffer) void {
        objc.msgSend(self, objc.sel("waitUntilCompleted"), .{});
    }
};

pub const MTLComputeCommandEncoder = opaque {
    pub fn setComputePipelineState(self: *MTLComputeCommandEncoder, state: *MTLComputePipelineState) void {
        objc.msgSend(self, objc.sel("setComputePipelineState:"), .{state});
    }

    pub fn setBuffer(self: *MTLComputeCommandEncoder,
        buffer: *MTLBuffer,
        offset: usize,
        index: usize
    ) void {
        objc.msgSend(self, objc.sel("setBuffer:offset:atIndex:"), .{ buffer, offset, index });
    }

    pub func dispatchThreads(self: *MTLComputeCommandEncoder,
        threads: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) void {
        objc.msgSend(self, objc.sel("dispatchThreads:threadsPerThreadgroup:"), .{
            threads, threadsPerThreadgroup
        });
    }

    pub func endEncoding(self: *MTLComputeCommandEncoder) void {
        objc.msgSend(self, objc.sel("endEncoding"), .{});
    }
};

// Simplified Objective-C interop for Metal
pub const objc = struct {
    pub fn getClass(name: [*:0]const u8) *anyopaque {
        // Implementation depends on platform
        // On macOS, this would call objc_getClass
        return @ptrFromInt(usize); // Placeholder
    }

    pub fn sel(name: [*:0]const u8) *anyopaque {
        // Would call sel_registerName
        return @ptrFromInt(usize); // Placeholder
    }

    pub fn msgSend(obj: anytype, sel: anytype, args: anytype) anytype {
        // Would call objc_msgSend with appropriate signature
        return undefined; // Placeholder
    }
};
```

**src/backend.zig - Implementação Metal:**
```zig
fn metalMatMulGPU(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) !void {
    // 1. Get Metal device
    const device = try metal.MTLDevice.create();
    defer device.release();

    // 2. Create command queue
    const cmd_queue = try device.newCommandQueue();
    defer cmd_queue.release();

    // 3. Load compute pipeline (pre-compiled metallib)
    const pipeline = try device.newComputePipelineStateWithFunction("matmul");
    defer pipeline.release();

    // 4. Create buffers
    const buffer_a = try device.newBufferWithBytes(a.ptr, a.len * @sizeOf(f32), .StorageModeShared);
    const buffer_b = try device.newBufferWithBytes(b.ptr, b.len * @sizeOf(f32), .StorageModeShared);
    const buffer_c = try device.newBufferWithBytes(c.ptr, c.len * @sizeOf(f32), .StorageModeShared);

    defer buffer_a.release();
    defer buffer_b.release();
    defer buffer_c.release();

    // 5. Create command buffer and encoder
    const cmd_buffer = try cmd_queue.commandBuffer();
    defer cmd_buffer.release();

    const encoder = try cmd_buffer.computeCommandEncoder();
    defer encoder.release();

    // 6. Set pipeline and buffers
    encoder.setComputePipelineState(pipeline);
    encoder.setBuffer(buffer_a, 0, 0);
    encoder.setBuffer(buffer_b, 0, 1);
    encoder.setBuffer(buffer_c, 0, 2);

    // 7. Set constants
    const m_val = @as(u32, @intCast(m));
    const n_val = @as(u32, @intCast(n));
    const k_val = @as(u32, @intCast(k));
    encoder.setBytes(&m_val, @sizeOf(u32), 3);
    encoder.setBytes(&n_val, @sizeOf(u32), 4);
    encoder.setBytes(&k_val, @sizeOf(u32), 5);

    // 8. Dispatch threads
    const threads = metal.MTLSizeMake(n, m, 1);
    const threadsPerGroup = metal.MTLSizeMake(16, 16, 1);
    encoder.dispatchThreads(threads, threadsPerGroup);

    // 9. End encoding and commit
    encoder.endEncoding();
    cmd_buffer.commit();
    cmd_buffer.waitUntilCompleted();
}
```

### **Fase 1.4: Timeline e Prioridades**

**Semana 1-2: Estrutura básica**
- [ ] Criar arquivos shader Metal
- [ ] Configurar pipeline de compilação
- [ ] Implementar ligação básica Metal API
- [ ] Testar compilação de shaders

**Semana 3-4: Operações principais**
- [ ] Implementar matmul Metal
- [ ] Implementar ativações Metal
- [ ] Testar operações individuais
- [ ] Benchmark contra CPU

**Semana 5-6: Integração e otimização**
- [ ] Integrar com backend existente
- [ ] Otimizar tamanhos de thread group
- [ ] Testar cargas de trabalho completas
- [ ] Ajustar desempenho

**Resultados Esperados:**
- Speedup de 10-50x para operações GPU
- Redução de latência de kernel (<1ms)
- Utilização de GPU de 70-90%

---

## 2. Adicionar Suporte a Processamento em Lote

### **Fase 2.1: API de Operações em Lote**

**src/backend.zig - Funções em lote:**
```zig
pub const Backend = union(enum) {
    // ... existing functions ...

    /// Batched matrix multiplication
    pub fn matMulBatch(self: Backend,
        a: []const f32,
        b: []const f32,
        c: []f32,
        batch_size: usize,
        n: usize,
        k: usize
    ) !void {
        switch (self) {
            .gpu => |gpu| switch (gpu) {
                .metal => try metalMatMulBatch(a, b, c, batch_size, n, k),
                .vulkan => try vulkanMatMulBatch(a, b, c, batch_size, n, k),
            },
            .cpu => cpuMatMulBatch(a, b, c, batch_size, n, k),
        }
    }

    /// Batched activation forward
    pub fn activationForwardBatch(self: Backend,
        act: activation.Activation,
        input: []f32,
        output: []f32,
        batch_size: usize,
        size: usize
    ) !void {
        switch (self) {
            .gpu => |gpu| switch (gpu) {
                .metal => try metalActivationForwardBatch(act, input, output, batch_size, size),
                .vulkan => try vulkanActivationForwardBatch(act, input, output, batch_size, size),
            },
            .cpu => cpuActivationForwardBatch(act, input, output, batch_size, size),
        }
    }
};
```

**Implementações CPU:**
```zig
fn cpuMatMulBatch(a: []const f32, b: []const f32, c: []f32, batch_size: usize, n: usize, k: usize) void {
    // Each batch element: a[batch] (1xk) * b (kxn) = c[batch] (1xn)
    for (0..batch_size) |batch| {
        const a_offset = batch * k;
        const c_offset = batch * n;

        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |p| {
                sum += a[a_offset + p] * b[p * n + j];
            }
            c[c_offset + j] = sum;
        }
    }
}

fn cpuActivationForwardBatch(
    act: activation.Activation,
    input: []f32,
    output: []f32,
    batch_size: usize,
    size: usize
) void {
    for (0..batch_size) |batch| {
        const offset = batch * size;
        const input_batch = input[offset..offset+size];
        const output_batch = output[offset..offset+size];

        // Handle softmax specially
        if (act == .softmax) {
            act.softmaxForward(input_batch, output_batch) catch unreachable;
        } else {
            for (0..size) |i| {
                output_batch[i] = act.forward(input_batch[i]);
            }
        }
    }
}
```

### **Fase 2.2: Integração com Rede Neural**

**src/network.zig - Passagem direta em lote:**
```zig
pub fn forwardBatch(self: *Network, batch_data: []const []const f32, batch_output: []f32) !void {
    const batch_size = batch_data.len;
    if (batch_size == 0) return;

    const input_size = batch_data[0].len;

    // Stack inputs into single matrix
    const stacked_input = try self.allocator.alloc(f32, batch_size * input_size);
    defer self.allocator.free(stacked_input);

    for (batch_data, 0..) |sample, i| {
        @memcpy(stacked_input[i*input_size..(i+1)*input_size], sample);
    }

    // Process through layers
    var current = stacked_input;

    for (self.layers.items) |layer| {
        const output_size = layer.output_size;
        const next_size = batch_size * output_size;

        const output = try self.allocator.alloc(f32, next_size);
        defer self.allocator.free(output);

        // Batched matrix multiplication
        try self.backend.matMulBatch(
            current,                    // [batch_size, input_size]
            layer.weights,              // [input_size, output_size]
            output,                     // [batch_size, output_size]
            batch_size,
            output_size,
            layer.input_size
        );

        // Add bias (broadcasted)
        for (0..batch_size) |b| {
            for (0..output_size) |j| {
                output[b * output_size + j] += layer.bias[j];
            }
        }

        // Apply activation
        try self.backend.activationForwardBatch(
            layer.act,
            output,
            output,
            batch_size,
            output_size
        );

        current = output;
    }

    // Copy final output
    @memcpy(batch_output, current);
}
```

**src/network.zig - Treino em lote:**
```zig
pub fn trainBatch(self: *Network,
    batch_data: []const []const f32,
    batch_targets: []const []const f32,
    learning_rate: f32,
    loss_fn: loss.Loss
) !f32 {
    const batch_size = batch_data.len;
    if (batch_size == 0) return 0;

    // Forward pass
    const outputs = try self.forwardBatch(batch_data);
    defer self.allocator.free(outputs);

    // Compute loss
    var total_loss: f32 = 0;
    for (batch_targets, 0..) |target, i| {
        const output = outputs[i*target.len..(i+1)*target.len];
        total_loss += try loss_fn.forward(output, target);
    }

    // Backward pass (simplified - actual implementation would be more complex)
    // This is a placeholder - real implementation needs batched backprop

    return total_loss / @as(f32, @floatFromInt(batch_size));
}
```

### **Fase 2.3: Otimizações de Lote**

**Otimização 1: Tamanho de lote dinâmico**
```zig
pub fn getOptimalBatchSize(memory_limit: usize, sample_size: usize) usize {
    // Considerar limites de memória e tamanho de amostra
    const max_batch = memory_limit / (sample_size * @sizeOf(f32) * 4); // *4 para buffers
    return @min(max_batch, 128); // Limitar a 128 para estabilidade
}
```

**Otimização 2: Pré-carregamento de dados**
```zig
pub fn createDataLoader(data: []const []const f32,
                       targets: []const []const f32,
                       batch_size: usize) DataLoader {
    return DataLoader{
        .data = data,
        .targets = targets,
        .batch_size = batch_size,
        .num_batches = (data.len + batch_size - 1) / batch_size,
    };
}
```

**Otimização 3: Lotes em memória**
```zig
pub fn preloadBatches(allocator: Allocator,
                     data: []const []const f32,
                     batch_size: usize) ![][]const f32 {
    var batches = try allocator.alloc([]const f32, (data.len + batch_size - 1) / batch_size);

    for (batches, 0..) |*batch, i| {
        const start = i * batch_size;
        const end = @min(start + batch_size, data.len);
        batch.* = data[start..end];
    }

    return batches;
}
```

### **Fase 2.4: Timeline e Prioridades**

**Semana 1: Infraestrutura de lote**
- [ ] Implementar funções em lote no backend
- [ ] Adicionar testes unitários para operações em lote
- [ ] Benchmark contra implementação única

**Semana 2: Integração com rede**
- [ ] Implementar forwardBatch
- [ ] Adicionar treino em lote básico
- [ ] Testar com diferentes tamanhos de lote

**Semana 3: Otimizações e testes**
- [ ] Adicionar tamanho de lote dinâmico
- [ ] Implementar pré-carregamento de dados
- [ ] Testes de desempenho completos

**Resultados Esperados:**
- Speedup de 5-10x para treino com batch_size=32+
- Melhor utilização do GPU (70-90%)
- Menor overhead de kernel

---

## 3. Otimizar com Vetorização SIMD

### **Fase 3.1: Detecção e Configuração SIMD**

**src/simd.zig - Detecção de recursos:**
```zig
const std = @import("std");
const builtin = @import("builtin");

pub const SIMD = struct {
    /// Verificar se NEON está disponível (Apple Silicon)
    pub fn hasNEON() bool {
        return builtin.cpu.arch == .aarch64;
    }

    /// Verificar se AVX2 está disponível (x86_64)
    pub fn hasAVX2() bool {
        return builtin.cpu.arch == .x86_64; // Simplificado
    }

    /// Largura do vetor em floats
    pub const VECTOR_WIDTH: usize = switch (builtin.cpu.arch) {
        .aarch64 => 4, // NEON processa 4 floats
        .x86_64 => 8,  // AVX2 processa 8 floats
        else => 1,
    };
};
```

### **Fase 3.2: Funções Vetorizadas**

**src/simd.zig - Ativações vetorizadas:**
```zig
/// Vetorizado ReLU usando NEON ou AVX2
pub fn reluVectorized(input: []const f32, output: []f32) void {
    if (!SIMD.hasNEON()) {
        // Fallback para implementação escalar
        for (input, output) |x, *out| {
            out.* = @max(0.0, x);
        }
        return;
    }

    const vec_width = SIMD.VECTOR_WIDTH;
    var i: usize = 0;

    // Processar elementos do vetor
    while (i + vec_width <= input.len) : (i += vec_width) {
        // Carregar vetor
        const vec = simdLoad(&input[i]);

        // Criar vetor zero
        const zero = simdSetZero();

        // Calcular max(vec, 0)
        const result = simdMax(vec, zero);

        // Armazenar resultado
        simdStore(&output[i], result);
    }

    // Processar elementos restantes
    while (i < input.len) : (i += 1) {
        output[i] = @max(0.0, input[i]);
    }
}

/// Implementações NEON
fn simdLoad(ptr: [*]const f32) @Vector(4, f32) {
    return @vecLoad(ptr, 0, @Vector(4, f32));
}

fn simdStore(ptr: [*]f32, vec: @Vector(4, f32)) void {
    @vecStore(ptr, 0, vec);
}

fn simdSetZero() @Vector(4, f32) {
    return @as(@Vector(4, f32), @splat(0.0));
}

fn simdMax(a: @Vector(4, f32), b: @Vector(4, f32)) @Vector(4, f32) {
    return @select(f32, a > b, a, b);
}

/// Vetorizado Sigmoid com aproximação
pub fn sigmoidVectorized(input: []const f32, output: []f32) void {
    const vec_width = SIMD.VECTOR_WIDTH;
    var i: usize = 0;

    while (i + vec_width <= input.len) : (i += vec_width) {
        // Carregar vetor
        const vec = simdLoad(&input[i]);

        // Calcular sigmoid usando aproximação vetorizada
        // sigmoid(x) ≈ 0.5 + 0.197 * x / (1 + 0.197 * |x|)
        const abs_x = simdAbs(vec);
        const coeff = @as(@Vector(4, f32), @splat(0.197));
        const denom = simdAdd(simdSetOne(), simdMul(coeff, abs_x));
        const numer = simdMul(coeff, vec);
        const frac = simdDiv(numer, denom);
        const result = simdAdd(simdSetHalf(), frac);

        simdStore(&output[i], result);
    }

    // Processar elementos restantes
    while (i < input.len) : (i += 1) {
        const x = input[i];
        output[i] = 1.0 / (1.0 + @exp(-x));
    }
}
```

### **Fase 3.3: Integração com Backend**

**src/backend.zig - Integração SIMD:**
```zig
const simd = @import("simd.zig");

fn cpuActivationForward(act: activation.Activation, input: []f32, output: []f32) void {
    // Usar implementação vetorizada se disponível e tamanho for grande o suficiente
    if (simd.SIMD.VECTOR_WIDTH > 1 and input.len >= simd.SIMD.VECTOR_WIDTH * 4) {
        switch (act) {
            .relu => {
                simd.reluVectorized(input, output);
                return;
            },
            .sigmoid => {
                simd.sigmoidVectorized(input, output);
                return;
            },
            // Adicionar mais ativações vetorizadas
            else => {},
        }
    }

    // Fallback para implementação escalar
    // ... existing code ...
}

fn cpuMatMul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
    // Usar multiplicação vetorizada para tamanhos grandes
    if (simd.SIMD.VECTOR_WIDTH > 1 and m * n * k >= 4096) {
        simd.matmulVectorized(a, b, c, m, n, k);
        return;
    }

    // Fallback para implementação escalar
    // ... existing code ...
}
```

### **Fase 3.4: Funções SIMD Avançadas**

**Multiplicação de matrizes vetorizada:**
```zig
pub fn matmulVectorized(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
    const vec_width = VECTOR_WIDTH;

    // Tamanho de bloco para cache
    const block_size = 64;

    for (0..m) |i| {
        for (0..n) |j| {
            var sum_vec = simdSetZero();

            // Processar múltiplos elementos do vetor
            var p: usize = 0;
            while (p + vec_width <= k) : (p += vec_width) {
                const a_vec = simdLoad(&a[i * k + p]);
                const b_vec = simdLoad(&b[p * n + j]);
                sum_vec = simdMulAdd(a_vec, b_vec, sum_vec);
            }

            // Reduzir vetor para escalar
            var sum: f32 = simdReduceAdd(sum_vec);

            // Processar elementos restantes
            while (p < k) : (p += 1) {
                sum += a[i * k + p] * b[p * n + j];
            }

            c[i * n + j] = sum;
        }
    }
}
```

### **Fase 3.5: Timeline e Prioridades**

**Semana 1: Infraestrutura SIMD**
- [ ] Configurar detecção de recursos SIMD
- [ ] Implementar operações SIMD básicas (load, store, add, mul)
- [ ] Adicionar testes unitários para operações SIMD

**Semana 2: Ativações vetorizadas**
- [ ] Implementar ReLU vetorizado
- [ ] Implementar Sigmoid vetorizado
- [ ] Implementar Tanh vetorizado
- [ ] Benchmark contra implementação escalar

**Semana 3: Operações matriciais**
- [ ] Implementar multiplicação matricial vetorizada
- [ ] Adicionar suporte a múltiplas arquiteturas (NEON, AVX2)
- [ ] Otimizar tamanhos de bloco

**Semana 4: Integração e otimização**
- [ ] Integrar SIMD com backend existente
- [ ] Adicionar heurísticas de seleção (quando usar SIMD)
- [ ] Testes de desempenho completos

**Resultados Esperados:**
- Speedup de 2-4x para operações vetorizadas
- Melhor utilização da CPU (80-95%)
- Redução no uso de CPU para operações paralelas

---

## Prioridades de Implementação

### **Alta Prioridade (Semanas 1-4)**

1. **Shaders Metal (Semanas 1-4)**
   - Impacto: 10-50x speedup
   - Complexidade: Alta
   - Riscos: Ligação Metal API

2. **Infraestrutura de lote (Semanas 1-2)**
   - Impacto: Habilita otimizações GPU
   - Complexidade: Média
   - Riscos: Mudanças de API

### **Prioridade Média (Semanas 3-6)**

3. **Integração de lote (Semanas 3-4)**
   - Impacto: 5-10x speedup para treino
   - Complexidade: Alta
   - Riscos: Complexidade de backprop

4. **Ativações SIMD (Semanas 3-4)**
   - Impacto: 2-4x speedup para CPU
   - Complexidade: Média
   - Riscos: Compatibilidade de plataforma

### **Prioridade Baixa (Semanas 5-8)**

5. **Operações matriciais SIMD (Semanas 5-6)**
   - Impacto: 1.5-3x speedup para CPU
   - Complexidade: Alta
   - Riscos: Cache locality

6. **Otimizações avançadas (Semanas 7-8)**
   - Impacto: 1.1-1.5x speedup
   - Complexidade: Média
   - Riscos: Diminuição de retorno

---

## Resultados Esperados Combinados

### **Speedup de Desempenho**

| Otimização | Speedup | Status |
|------------|---------|--------|
| Shaders Metal | 10-50x | ⏳ Não implementado |
| Processamento em lote | 5-10x | ⏳ Parcialmente implementado |
| SIMD (ativações) | 2-4x | ⏳ Não implementado |
| SIMD (matmul) | 1.5-3x | ⏳ Não implementado |
| **COMBINADO** | **20-100x** | 🎯 Alvo final |

### **Timeline de Implementação**

**Mês 1: Fundação**
- Semanas 1-2: Shaders Metal + Infraestrutura de lote
- Semanas 3-4: Integração de lote + Ativações SIMD

**Mês 2: Otimização**
- Semanas 5-6: Matmul SIMD + Otimizações de lote
- Semanas 7-8: Testes + Ajustes de desempenho

**Mês 3: Polimento**
- Semanas 9-10: Benchmarks + Otimizações finais
- Semanas 11-12: Documentação + Testes de produção

### **Métricas de Sucesso**

1. **Desempenho:** 20-100x speedup para operações de rede neural
2. **Utilização:** 70-90% utilização do GPU, 80-95% utilização da CPU
3. **Memória:** Mesma pegada com zero-copy no Apple Silicon
4. **Corretude:** Resultados idênticos entre CPU e GPU (dentro de tolerância)
5. **Estabilidade:** Sem vazamentos de memória ou corrupção de dados

---

## Próximos Passos

### **Semana 1: Começar com Metal**
1. Configurar estrutura de shader Metal
2. Implementar compilação de shader no build.zig
3. Criar ligação básica Metal API

### **Semana 2: Adicionar Lote**
1. Implementar funções em lote no backend
2. Adicionar testes para operações em lote
3. Benchmark contra execução única

### **Semana 3: Adicionar SIMD**
1. Configurar detecção de recursos SIMD
2. Implementar ReLU vetorizado
3. Benchmark contra implementação escalar

---

**Plano Criado:** 2026-02-16
**Alvo de Conclusão:** 3 meses
**Alvo de Desempenho:** 20-100x speedup
**Prioridade:** CRÍTICO para desempenho de produção

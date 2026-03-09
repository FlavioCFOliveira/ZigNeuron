# ZigNeuron Roadmap

**Baseado na Auditoria Completa de 2026-03-09**
**Versão:** 2.0 - Atualizado com novos findings de segurança e performance

---

## Resumo Executivo

Esta versão do roadmap incorpora os resultados da **auditoria exaustiva de 2026-03-09** realizada por 4 especialistas:
- Neural Network Architect
- Zig Performance Architect
- Security Architect
- ML Architecture Expert

**Novos Findings Críticos Identificados:**
- 1 vulnerabilidade de segurança **crítica** (use-after-free)
- 3 vulnerabilidades **altas** (race conditions, double-free)
- 5 problemas de performance **críticos** em hot paths
- 10+ funcionalidades essenciais em falta (AdamW, pooling, etc.)

---

## Fase 1: Segurança Crítica (Semanas 1-2)

> ⚠️ **ESTA FASE É BLOQUEANTE - Não use em produção sem completar**

### 1.1 VULN-001: Use-After-Free em MetalContext
**CVSS:** 9.1 (Crítica) | **Ficheiro:** `src/metal_context.zig:60-68`

**Problema:** O `shader_source` é libertado por `defer` antes do uso confirmado por `newLibraryWithSource`.

**Tarefas:**
1. [ ] Remover `defer allocator.free(shader_source)` imediatamente após concatenação
2. [ ] Libertar memória APÓS confirmação de uso da library
3. [ ] Adicionar teste de stress para carregamento de shaders

**Código Problemático:**
```zig
const shader_source = try std.mem.concat(allocator, u8, &sources);
defer allocator.free(shader_source); // ❌ Libertado aqui
self.library = try self.device.newLibraryWithSource(shader_source); // Usado aqui
```

**Solução:**
```zig
const shader_source = try std.mem.concat(allocator, u8, &sources);
// NO defer aqui
self.library = try self.device.newLibraryWithSource(shader_source);
allocator.free(shader_source); // ✅ Libertar após uso confirmado
```

---

### 1.2 VULN-002: Integer Overflow em Tensor::init
**CVSS:** 8.6 (Alta) | **Ficheiro:** `src/tensor.zig:14-21`

**Problema:** Não há validação de shapes vazios ou dimensões zero.

**Tarefas:**
1. [ ] Validar que shape não é vazio (`shape.len == 0` → erro)
2. [ ] Validar que todas as dimensões são > 0
3. [ ] Usar `std.math.mul` com overflow checking
4. [ ] Adicionar limite máximo de tamanho (2^32 elementos)
5. [ ] Adicionar limite de dimensões (max 8)

**Implementação Necessária:**
```zig
pub fn init(allocator: std.mem.Allocator, shape: []const usize, backend: backend_module.Backend) !Tensor {
    if (shape.len == 0) return error.EmptyShape;
    if (shape.len > 8) return error.ShapeTooLarge;

    var size: usize = 1;
    for (shape) |dim| {
        if (dim == 0) return error.ZeroDimension;
        if (dim > 1_000_000_000) return error.DimensionTooLarge;

        size = std.math.mul(usize, size, dim) catch {
            return error.TensorSizeOverflow;
        };
    }
    // ... resto da inicialização
}
```

---

### 1.3 VULN-003: Race Condition em Command Buffer
**CVSS:** 7.5 (Alta) | **Ficheiro:** `src/backend.zig`

**Problema:** Acesso não-sincronizado ao `active_command_buffer`.

**Tarefas:**
1. [ ] Implementar mutex para proteger `active_command_buffer`
2. [ ] Adicionar `std.Thread.Mutex` à struct Backend
3. [ ] Garantir cleanup em caso de erro
4. [ ] Adicionar asserts em builds de debug

---

### 1.4 VULN-004: Double-Free em CUDA Context
**CVSS:** 7.1 (Alta) | **Ficheiro:** `src/cuda_context.zig:367-379`

**Problema:** Lógica de retorno de buffer pode causar double-free em cenários de erro.

**Tarefas:**
1. [ ] Revisar toda a lógica de lifecycle de buffers CUDA
2. [ ] Usar estado explícito para trackear ownership
3. [ ] Adicionar logs em builds de debug
4. [ ] Implementar contador de referências simples

---

### 1.5 VULN-005 a VULN-009: Vulnerabilidades Médias
**CVSS:** 4.0-6.9 | **Ficheiros:** Vários

**Lista:**
- [ ] **VULN-005:** Buffer Overflow em `tensor.zig:index2D`
- [ ] **VULN-006:** Deserialização Insegura em `serialization.zig:loadModel`
- [ ] **VULN-007:** Kernel Launch Sem Validação em CUDA
- [ ] **VULN-008:** Memory Leak em `Tensor::deinit`
- [ ] **VULN-009:** Command Buffer Não Finalizado em Erro

---

### 1.6 Validações de Input Adicionais
**Prioridade:** 🟡 Média

**Tarefas:**
- [ ] Validar learning_rate em (0, 1) em todos os optimizers
- [ ] Validar dropout rate em [0, 1]
- [ ] Verificar NaN/Inf em inputs de loss functions
- [ ] Limitar número máximo de camadas (e.g., 10.000)
- [ ] Validar dimensões compatíveis em operações matriciais
- [ ] Validar que kernel_size <= input_len em Conv1D
- [ ] Validar que stride >= 1 em todas as camadas

---

## Fase 2: Performance Crítica (Semanas 2-4)

### 2.1 Remover Alocação em Hot Loop (Attention)
**Prioridade:** 🔴 Crítica | **Impacto:** 2-3x speedup em seq2seq
**Ficheiro:** `src/backend.zig:2805-2837`

**Problema:** CPU attention forward usa `std.heap.page_allocator.alloc` para scores buffer dentro do loop.

**Tarefas:**
1. [ ] Pré-alocar buffer de scores na inicialização da layer Attention
2. [ ] Modificar `cpuAttentionForward` para aceitar buffer pré-alocado
3. [ ] Atualizar todos os callsites
4. [ ] Benchmark para confirmar melhoria

**API Atual (PROBLEMA):**
```zig
const scores: []f32 = if (seq_len <= STACK_BUFFER_SIZE)
    stack_buffer[0..seq_len]
else
    try std.heap.page_allocator.alloc(f32, seq_len); // ❌ Alocação no hot path
defer if (seq_len > STACK_BUFFER_SIZE) std.heap.page_allocator.free(scores);
```

**API Nova (SOLUÇÃO):**
```zig
pub fn cpuAttentionForward(
    self: Backend,
    q: []const f32, k: []const f32, v: []const f32,
    output: []f32,
    scores_buffer: []f32,  // ✅ Buffer pré-alocado
    seq_len: usize, d_k: usize
) !void
```

---

### 2.2 Otimizar LayerNorm (Algoritmo Welford)
**Prioridade:** 🔴 Alta | **Impacto:** 2x menos operações de memória
**Ficheiro:** `src/backend.zig:2495-2513`

**Problema:** LayerNorm atual usa 2 passes sobre os dados (mean, depois variance).

**Tarefas:**
1. [ ] Implementar algoritmo de Welford para cálculo online de mean/variance
2. [ ] Reduzir de 2 passes para 1 pass
3. [ ] Benchmark para confirmar melhoria
4. [ ] Manter precisão numérica

**Referência:** Welford's online algorithm (Welford, 1962)

---

### 2.3 Adicionar SIMD em Normalização
**Prioridade:** 🔴 Alta | **Impacto:** 4x speedup em CPU
**Ficheiros:** `src/optimization.zig`, `src/backend.zig`

**Tarefas:**
1. [ ] Implementar `batchNormForwardVectorized` usando @Vector(4, f32)
2. [ ] Implementar `layerNormForwardVectorized`
3. [ ] Implementar `batchNormBackwardVectorized`
4. [ ] Adicionar deteção automática de alignment
5. [ ] Fallback para versão scalar quando necessário

**Implementação Exemplo:**
```zig
pub fn batchNormForwardVectorized(
    input: []const f32,
    output: []f32,
    gamma: []const f32,
    beta: []const f32,
    mean: f32,
    var_inv: f32
) void {
    const Vec4 = @Vector(4, f32);
    const mean_vec: Vec4 = @splat(mean);
    const var_inv_vec: Vec4 = @splat(var_inv);

    var i: usize = 0;
    while (i + 4 <= input.len) : (i += 4) {
        const x = @as(Vec4, @ptrCast(@alignCast(input[i..i+4].ptr))).*;
        const g = @as(Vec4, @ptrCast(@alignCast(gamma[i..i+4].ptr))).*;
        const b = @as(Vec4, @ptrCast(@alignCast(beta[i..i+4].ptr))).*;
        const normalized = (x - mean_vec) * var_inv_vec;
        const result = normalized * g + b;
        @as(*Vec4, @ptrCast(@alignCast(output[i..i+4].ptr))).* = result;
    }
    // Handle remainder...
}
```

---

### 2.4 Otimizar MatMul Blocking
**Prioridade:** 🟡 Média | **Impacto:** 30-50% speedup
**Ficheiro:** `src/backend.zig:2260-2347`

**Tarefas:**
1. [ ] Usar block sizes adaptativos baseados em cache size (L1/L2)
2. [ ] Pre-check alignment uma vez, não por bloco
3. [ ] Usar `comptime` para selecionar vector width ótimo
4. [ ] Considerar integração com BLAS (Accelerate no macOS)
5. [ ] Benchmark diferentes tile sizes (16, 32, 64)

---

### 2.5 Otimizações Metal Adicionais
**Prioridade:** 🟡 Média

**Tarefas:**
1. [ ] Aumentar TILE_SIZE para 32 em Apple Silicon (atualmente 16)
2. [ ] Adicionar kernel fused `matmul_bias_gelu`
3. [ ] Otimizar threadgroup dispatch sizes baseado no workload
4. [ ] Profile kernel occupancy em M1/M2/M3

---

## Fase 3: Funcionalidades Essenciais (Semanas 4-8)

### 3.1 Implementar AdamW Optimizer
**Prioridade:** 🔴 Crítica | **Impacto:** Essencial para SOTA
**Referência:** Loshchilov & Hutter, 2017

**Descrição:** AdamW (Adam com weight decay decoupled) é o optimizer padrão em modelos modernos. A principal diferença é que o weight decay é aplicado diretamente aos pesos, não aos gradients.

**Tarefas:**
1. [ ] Implementar struct AdamW em `optimizer.zig`
2. [ ] Adicionar parâmetro `weight_decay: f32` (default 0.01)
3. [ ] Aplicar weight decay diretamente aos pesos (não aos gradients)
4. [ ] Implementar método `step()` com lógica AdamW
5. [ ] Adicionar testes de convergência vs PyTorch
6. [ ] Atualizar documentação
7. [ ] Criar exemplo comparativo

**Implementação:**
```zig
pub const AdamW = struct {
    m_weights: ?tensor.Tensor = null,
    v_weights: ?tensor.Tensor = null,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,  // Parâmetro chave
    t: usize = 0,

    pub fn step(self: *AdamW, lyr: *layer_module.Layer, learning_rate: f32) !void {
        // Adam update normal para m e v
        // ...

        // AdamW: weight decay DECOUPLED do gradient
        const w = lyr.getWeights();
        for (w.slice) |*weight| {
            weight.* = weight.* - learning_rate * self.weight_decay * weight.*;
        }
        // ... depois aplicar update Adam normal
    }
};
```

---

### 3.2 Implementar MaxPool1D e MaxPool2D
**Prioridade:** 🔴 Alta | **Impacto:** Essencial para CNNs
**Ficheiro:** `src/layer.zig`

**Descrição:** Pooling é essencial para redução dimensional em CNNs. Sem pooling, as CNNs são muito limitadas.

**Tarefas:**
1. [ ] Implementar struct MaxPool1D com:
   - `kernel_size: usize`
   - `stride: usize` (default = kernel_size)
   - `padding: usize` (default = 0)
2. [ ] Implementar forward pass (argmax para backward)
3. [ ] Implementar backward pass (max unpooling)
4. [ ] Implementar MaxPool2D para imagens (2D spatial)
5. [ ] Adicionar kernel Metal para GPU
6. [ ] Adicionar testes unitários (forward/backward)
7. [ ] Criar exemplo de CNN com pooling
8. [ ] Documentar API

**API Proposta:**
```zig
pub const MaxPool2D = struct {
    kernel_size: usize,
    stride: usize,
    padding: usize,

    pub fn init(allocator: std.mem.Allocator, kernel_size: usize, stride: usize, backend: Backend) !*MaxPool2D;
    pub fn forward(self: *MaxPool2D, input: []const f32, output: []f32) !void;
    pub fn backward(self: *MaxPool2D, grad_output: []const f32, grad_input: []f32) !void;
};
```

---

### 3.3 Implementar AvgPool1D e AvgPool2D
**Prioridade:** 🟡 Média
**Ficheiro:** `src/layer.zig`

**Tarefas:**
1. [ ] Implementar AvgPool1D (similar a MaxPool mas faz average)
2. [ ] Implementar AvgPool2D
3. [ ] Implementar backward pass
4. [ ] Adicionar kernel Metal
5. [ ] Testes unitários

---

### 3.4 Implementar LeakyReLU e ELU
**Prioridade:** 🟡 Média | **Impacto:** Evita dying ReLU
**Ficheiro:** `src/activation.zig`

**Tarefas:**
1. [ ] Implementar LeakyReLU com parâmetro alpha (default 0.01)
2. [ ] Implementar backward de LeakyReLU
3. [ ] Implementar ELU com parâmetro alpha (default 1.0)
4. [ ] Implementar backward de ELU
5. [ ] Adicionar kernels Metal correspondentes
6. [ ] Adicionar testes de gradiente
7. [ ] Documentar vantagens vs ReLU

**Implementação:**
```zig
pub const Activation = union(enum) {
    relu,
    leaky_relu: f32,  // alpha
    elu: f32,         // alpha
    // ... resto

    pub fn forward(self: Activation, x: f32) f32 {
        return switch (self) {
            .relu => reluForward(x),
            .leaky_relu => |alpha| if (x > 0) x else alpha * x,
            .elu => |alpha| if (x > 0) x else alpha * (std.math.exp(x) - 1),
            // ...
        };
    }
};
```

---

### 3.5 Implementar Weight Decay nos Otimizadores
**Prioridade:** 🟡 Média | **Impacto:** Regularização L2
**Ficheiro:** `src/optimizer.zig`

**Tarefas:**
1. [ ] Adicionar parâmetro `weight_decay: f32` a SGD
2. [ ] Adicionar parâmetro `weight_decay: f32` a Adam
3. [ ] Adicionar parâmetro `weight_decay: f32` a RMSprop
4. [ ] Implementar L2 regularization (penalidade no gradiente)
5. [ ] Diferenciar claramente de AdamW (que é decoupled)
6. [ ] Adicionar documentação explicando diferença

---

### 3.6 Completar Conv2D
**Prioridade:** 🟡 Média | **Impacto:** Computer Vision
**Ficheiro:** `src/layer.zig`

**Tarefas:**
1. [ ] Completar backward pass de Conv2D
2. [ ] Otimizar implementação atual
3. [ ] Adicionar suporte a dilation
4. [ ] Adicionar suporte a groups (depthwise separable)
5. [ ] Kernel Metal otimizado

---

## Fase 4: Funcionalidades Avançadas (Semanas 8-12)

### 4.1 Implementar Focal Loss
**Prioridade:** 🟡 Média | **Impacto:** Class imbalance
**Referência:** Lin et al., 2017
**Ficheiro:** `src/loss.zig`

**Descrição:** Focal Loss é essencial para datasets desbalanceados (deteção de objetos, etc.). Reduz o peso de exemplos fáceis durante o treino.

**Tarefas:**
1. [ ] Implementar Focal Loss com parâmetros alpha e gamma
2. [ ] Implementar backward pass
3. [ ] Adicionar testes com dataset desbalanceado
4. [ ] Criar exemplo prático (e.g., deteção de objetos simulada)

**Fórmula:** `FL = -α(1-p)^γ log(p)`

---

### 4.2 Implementar Swish Activation
**Prioridade:** 🟡 Média | **Impacto:** Melhor convergência
**Referência:** Ramachandran et al., 2017
**Ficheiro:** `src/activation.zig`

**Descrição:** Swish (x * sigmoid(x)) é uma ativação smooth não-monotónica que frequentemente supera ReLU.

**Tarefas:**
1. [ ] Implementar forward: `f(x) = x * sigmoid(x)`
2. [ ] Implementar backward
3. [ ] Adicionar kernel Metal
4. [ ] Benchmark vs ReLU

---

### 4.3 Implementar Transformer Block
**Prioridade:** 🟡 Média | **Impacto:** Arquitetura moderna
**Referência:** Vaswani et al., 2017
**Ficheiro:** `src/layer.zig` ou novo `src/transformer.zig`

**Descrição:** Bloco completo de Transformer (Multi-Head Attention + Feed Forward + Residuals + LayerNorm).

**Tarefas:**
1. [ ] Implementar Multi-Head Attention
2. [ ] Implementar Positional Encoding
3. [ ] Implementar Feed Forward Network (2 layers)
4. [ ] Implementar Residual Connections
5. [ ] Integrar LayerNorm
6. [ ] Adicionar masking (causal/padding)
7. [ ] Criar exemplo de tradução ou classificação de texto
8. [ ] Documentar arquitetura

---

### 4.4 Implementar Label Smoothing
**Prioridade:** 🟢 Baixa | **Impacto:** Regularização
**Ficheiro:** `src/loss.zig` ou `network.zig`

**Tarefas:**
1. [ ] Implementar função `applyLabelSmoothing(targets, num_classes, smoothing)`
2. [ ] Integrar em `network.train()`
3. [ ] Documentar benefícios

---

### 4.5 Implementar Embedding Layer
**Prioridade:** 🟡 Média | **Impacto:** NLP
**Ficheiro:** `src/layer.zig`

**Tarefas:**
1. [ ] Implementar lookup table para embeddings
2. [ ] Suporte a vocab_size e embedding_dim
3. [ ] Implementar forward (index lookup)
4. [ ] Implementar backward (sparse gradients)
5. [ ] Adicionar exemplo de NLP

---

## Fase 5: Completar CUDA Backend (Semanas 10-16)

### 5.1 Implementar CUDA Core
**Prioridade:** 🔴 Alta (para Linux/Windows) | **Impacto:** GPU em não-macOS
**Ficheiros:** `src/cuda.zig`, `src/cuda_context.zig`, `src/cuda_driver.zig`

**Descrição:** A estrutura base existe mas precisa de implementação completa. Atualmente retorna `error.CudaNotYetImplemented`.

**Tarefas:**
1. [ ] Implementar kernel compilation (PTX loading)
2. [ ] Implementar matMul CUDA
3. [ ] Implementar kernels de ativação
4. [ ] Implementar batch operations
5. [ ] Testar em NVIDIA GPU real
6. [ ] Otimizar block/grid sizes
7. [ ] Adicionar memory pooling

**Checklist de Kernels:**
- [ ] matmul
- [ ] matmul_bias
- [ ] relu_forward/backward
- [ ] sigmoid_forward/backward
- [ ] tanh_forward/backward
- [ ] softmax
- [ ] batch_norm
- [ ] layer_norm
- [ ] dropout
- [ ] conv1d
- [ ] conv2d
- [ ] lstm_gru_cells

---

### 5.2 Otimizações CUDA Avançadas
**Prioridade:** 🟡 Média

**Tarefas:**
1. [ ] Usar shared memory eficientemente
2. [ ] Implementar kernel fusion
3. [ ] Adicionar suporte a FP16 (half precision)
4. [ ] Otimizar para Tensor Cores (where available)
5. [ ] Implementar stream parallelism

---

## Fase 6: Melhorias de Qualidade (Semanas 12-16)

### 6.1 Melhorar Serialização
**Prioridade:** 🟡 Média | **Segurança:** Deserialização insegura
**Ficheiro:** `src/serialization.zig`

**Tarefas:**
1. [ ] Adicionar magic number e version checking
2. [ ] Implementar checksum/integrity verification
3. [ ] Limitar layer_count máximo (10.000)
4. [ ] Validar tamanhos de tensores lidos
5. [ ] Rejeitar arquivos muito grandes (>1GB)
6. [ ] Documentar formato de ficheiro

**Estrutura do Formato .znn:**
```
[Magic: 4 bytes "ZNN\0"]
[Version: 4 bytes]
[Checksum: 32 bytes SHA256]
[Layer Count: 8 bytes]
[Layers...]
[Weights Data...]
```

---

### 6.2 Adicionar Mais Ativações
**Prioridade:** 🟢 Baixa

**Lista:**
- [ ] SELU (Self-Normalizing Neural Networks)
- [ ] GELU (melhorar implementação atual - backward mais preciso)
- [ ] Mish
- [ ] PReLU (Parametric ReLU)
- [ ] Hard Tanh
- [ ] Hard Sigmoid

---

### 6.3 Adicionar Mais Loss Functions
**Prioridade:** 🟢 Baixa

**Lista:**
- [ ] Triplet Loss (para embeddings)
- [ ] Contrastive Loss
- [ ] Dice Loss (segmentação)
- [ ] Huber Loss (robust regression)
- [ ] Tversky Loss

---

### 6.4 Adicionar Mais Otimizadores
**Prioridade:** 🟢 Baixa

**Lista:**
- [ ] Adagrad
- [ ] Adadelta
- [ ] Nadam (Adam + Nesterov)
- [ ] Lion (Chen et al., 2023)
- [ ] LAMB (para large batch training)

---

### 6.5 Melhorar GELU
**Prioridade:** 🟡 Média
**Ficheiro:** `src/activation.zig:120-144`

**Problema:** O backward atual usa aproximação numérica.

**Tarefas:**
1. [ ] Guardar valor x durante forward pass
2. [ ] Usar x para calcular derivada exata no backward
3. [ ] Alternativa: Usar aproximação mais precisa

---

## Fase 7: Performance Avançada (Futuro)

### 7.1 Mixed Precision Training (FP16/BF16)
**Prioridade:** 🟢 Baixa | **Impacto:** 2x memória + speed

**Tarefas:**
1. [ ] Adicionar suporte a half precision (f16)
2. [ ] Implementar loss scaling para estabilidade
3. [ ] Kernels Metal otimizados para f16
4. [ ] Testar em modelos grandes

---

### 7.2 Integração BLAS
**Prioridade:** 🟢 Baixa | **Impacto:** 5-10x matmul

**Tarefas:**
1. [ ] Integrar Accelerate Framework no macOS
2. [ ] Integrar OpenBLAS no Linux
3. [ ] Fallback automático para implementação nativa
4. [ ] Benchmarks comparativos

---

### 7.3 Multi-GPU / Distributed
**Prioridade:** 🟢 Baixa

**Tarefas:**
1. [ ] Implementar data parallelism
2. [ ] Gradient synchronization
3. [ ] Model parallelism (para modelos muito grandes)
4. [ ] Suporte a clusters

---

## Fase 8: Ecosystem e Tooling (Futuro)

### 8.1 Suporte ONNX
**Prioridade:** 🟢 Baixa | **Impacto:** Interoperabilidade

**Descrição:** Permitir importar/exportar modelos no formato ONNX.

**Tarefas:**
1. [ ] Implementar export para ONNX
2. [ ] Implementar import de ONNX (operadores core)
3. [ ] Documentar limitações

---

### 8.2 Visualização e Debugging
**Prioridade:** 🟢 Baixa

**Tarefas:**
1. [ ] Logging estruturado de treino
2. [ ] Exportação para TensorBoard
3. [ ] Visualização de grafos de computação
4. [ ] Debugging de gradients (gradient flow)

---

### 8.3 Datasets e Preprocessing
**Prioridade:** 🟢 Baixa

**Tarefas:**
1. [ ] Loaders para datasets comuns (MNIST, CIFAR, etc.)
2. [ ] Data augmentation (flip, rotate, crop, etc.)
3. [ ] Normalização utilities
4. [ ] Batch collation

---

## Timeline Resumida

```
Semana  1:  [1.1] [1.2] [1.3] [1.4]      Segurança Crítica
Semana  2:  [1.5] [1.6] [2.1]             Segurança + Performance
Semana  3:  [2.2] [2.3]                   Performance
Semana  4:  [2.4] [2.5] [3.1]             Performance + AdamW
Semana  5:  [3.2] [3.3]                   Pooling
Semana  6:  [3.4] [3.5] [3.6]             Ativações + Conv2D
Semana  7:  [4.1] [4.2]                   Losses + Ativações
Semana  8:  [4.3] [4.4]                   Transformer
Semanas 9-12: [5.x]                       CUDA Backend
Semanas 13-16: [6.x]                      Qualidade
Semanas 17+:   [7.x] [8.x]                 Avançado
```

---

## Dependências entre Tarefas

```
Fase 1 (Segurança) ───────────────────────────────────┐
                                                       ▼
Fase 2 (Performance) ──────────────────────────────► Fase 3 (Essencial)
                                                      │
F3.1 (AdamW) ──────────────────────────────────────► F4.x (Avançado)
                                                      │
F3.2 (MaxPool) ────────────────────────────────────► F8.3 (Datasets)
                                                      │
F5.x (CUDA) ───────────────────────────────────────► F7.x (Multi-GPU)
```

---

## Métricas de Sucesso

### Segurança
- [ ] 0 vulnerabilidades Críticas/Altas
- [ ] 100% dos inputs validados em funções públicas
- [ ] Fuzzing tests passando
- [ ] Valgrind/ASAN clean

### Performance
- [ ] 2x speedup em seq2seq (fix attention)
- [ ] 4x speedup em CPU normalização (SIMD)
- [ ] Metal matmul competitivo com MPS
- [ ] CUDA backend funcional

### Funcionalidade
- [ ] AdamW implementado e testado
- [ ] MaxPool funcional com exemplo
- [ ] Transformer block completo
- [ ] 90%+ coverage de testes

### Qualidade
- [ ] Documentação completa
- [ ] Exemplos para todas as features principais
- [ ] Benchmarks automatizados
- [ ] CI/CD com tests

---

## Tabela de Prioridades

| Tarefa | Prioridade | Esforço | Impacto | Status |
|--------|------------|---------|---------|--------|
| VULN-001 (Use-after-free) | 🔴 Crítica | Baixo | Segurança | ⏳ Pendente |
| VULN-002 (Overflow) | 🔴 Crítica | Baixo | Segurança | ⏳ Pendente |
| VULN-003 (Race) | 🔴 Alta | Médio | Segurança | ⏳ Pendente |
| VULN-004 (Double-free) | 🔴 Alta | Médio | Segurança | ⏳ Pendente |
| F2.1 (Attention alloc) | 🔴 Crítica | Médio | 2-3x speedup | ⏳ Pendente |
| F2.2 (Welford) | 🔴 Alta | Médio | 2x memória | ⏳ Pendente |
| F2.3 (SIMD norm) | 🔴 Alta | Médio | 4x CPU | ⏳ Pendente |
| F3.1 (AdamW) | 🔴 Crítica | Médio | SOTA | ⏳ Pendente |
| F3.2 (MaxPool) | 🔴 Alta | Alto | CNNs | ⏳ Pendente |
| F3.4 (LeakyReLU) | 🟡 Média | Baixo | Qualidade | ⏳ Pendente |
| F4.1 (Focal Loss) | 🟡 Média | Médio | Class imbalance | ⏳ Pendente |
| F4.3 (Transformer) | 🟡 Média | Alto | Arquitetura | ⏳ Pendente |
| F5.1 (CUDA core) | 🔴 Alta | Alto | GPU Linux | ⏳ Pendente |
| F6.1 (Serialização) | 🟡 Média | Médio | Segurança | ⏳ Pendente |

---

## Notas

- **Prioridades podem mudar** baseado em feedback de utilizadores
- **Contribuições externas** são bem-vindas para tarefas de prioridade baixa/média
- **Segurança é não-negociável** - Fase 1 deve ser completa antes de produção
- **Performance crítica** - Fase 2 melhora UX significativamente
- **Funcionalidades essenciais** - Fase 3 torna a biblioteca competitiva

---

## Anexos

### Ficheiros de Referência

- Relatório Completo: `AUDITORIA_COMPLETA_ZIGNEURON.md`
- Relatório Segurança: `SECURITY_AUDIT_REPORT_V2.md`
- Relatório Neural Net: `NEURAL_NET_AUDIT_REPORT_COMPLETE.md`
- Relatório Performance: `PERFORMANCE_AUDIT_REPORT.md`

### Convenções de Commits

- `security:` - Fixes de segurança
- `fix:` - Correção de bugs
- `perf:` - Otimizações
- `feat:` - Novas funcionalidades
- `docs:` - Documentação

---

*Roadmap atualizado em 2026-03-09 com base na auditoria exaustiva*

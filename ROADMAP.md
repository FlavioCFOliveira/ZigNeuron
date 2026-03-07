# Roadmap ZigNeuron - Pós Auditoria

**Data:** 2026-03-06
**Versão:** 1.0
**Baseado em:** Auditoria Completa com 9 Especialistas

---

## Visão Geral

Este roadmap organiza as tarefas resultantes da auditoria profunda do ZigNeuron por ordem de prioridade e dependências. As tarefas estão divididas em 4 fases:

- **Fase 1 (Crítica):** Correções de segurança e bugs bloqueantes
- **Fase 2 (Alta):** Otimizações de performance e correções funcionais
- **Fase 3 (Média):** Novas funcionalidades e melhorias
- **Fase 4 (Baixa):** Melhorias de qualidade de vida e features adicionais

---

## FASE 1: CRÍTICA (Semanas 1-2)

> ⚠️ **Estas tarefas devem ser concluídas antes de qualquer release.**
>
> Impactam segurança, estabilidade ou funcionalidades core.

### 1.1 Segurança - Integer Overflows
- [ ] **#3** Corrigir integer overflow em cálculo de tamanho de tensor
  - Ficheiro: `src/tensor.zig:14-16`
  - Usar `std.math.mul` com verificação de overflow

- [ ] **#16** Corrigir overflow em cálculos de índice 3D
  - Ficheiro: `src/tensor.zig:90-98`
  - Verificar overflow em multiplicações de índice

### 1.2 Segurança - Divisão por Zero
- [ ] **#15** Corrigir divisão por zero em Dense layer
  - Ficheiro: `src/layer.zig`
  - Validar `input_size > 0` e `output_size > 0` no init

- [ ] **#17** Corrigir divisão por zero em Softmax
  - Ficheiro: `src/activation.zig:51`
  - Proteger quando `sum = 0`

### 1.3 Backend Metal
- [ ] **#11** Corrigir erro de const qualifier no Metal backend
  - Ficheiro: `src/backend.zig:79`
  - Permitir modificação do command buffer

- [ ] **#10** Adicionar bounds checking em kernels Metal
  - Ficheiro: `shaders/metal/activation.metal`
  - Verificar limites quando size não é múltiplo de 4

### 1.4 Funcionalidade Core
- [ ] **#18** Implementar backward da camada Attention
  - Ficheiro: `src/layer.zig:633-636`
  - **CRÍTICO:** Atualmente apenas copia gradientes
  - Implementar cálculo completo de dL/dQ, dL/dK, dL/dV

**Dependências:** Nenhuma (podem ser executadas em paralelo)

**Métricas de Sucesso:**
- Todos os testes de segurança passam
- `zig build` compila sem erros
- Testes de attention convergem corretamente

---

## FASE 2: ALTA (Semanas 3-5)

> 🔧 Melhorias de performance e correções funcionais importantes.

### 2.1 Correções de Segurança/Estabilidade
- [ ] **#14** Corrigir underflow em Conv1D
  - Ficheiro: `src/layer.zig:454`
  - Verificar `kernel_size > input_len`

- [ ] **#49** Corrigir unwraps inseguros
  - Ficheiros: `src/network.zig:328, 350`
  - Substituir `.?` por `orelse return error`

### 2.2 Performance - SIMD
- [ ] **#28** Implementar SIMD nas operações CPU
  - Usar `@Vector(8, f32)` do Zig
  - Operações: `cpuMatMul`, `cpuElementWise`, ativações
  - **Impacto estimado:** 4-8x speedup em CPU

- [ ] **#13** Otimizar threadgroup sizes Metal
  - Configurar para múltiplos de 32 (SIMD width)
  - Atingir melhor occupancy

### 2.3 Performance - Memory
- [ ] **#25** Corrigir alocação em hot path da Attention
  - Pré-alocar buffer de scores na struct
  - Eliminar alocação em `cpuAttentionForward`

- [ ] **#26** Corrigir kernel de Attention (stack overflow)
  - Reimplementar sem array fixo de 1024
  - Usar shared memory ou processar em blocos

### 2.4 Funcionalidade
- [ ] **#8** Corrigir SGD momentum
  - Ficheiro: `src/optimizer.zig:73-80`
  - Implementar velocity update corretamente

**Dependências:**
- #28 (SIMD) pode ser paralelo com outras tarefas
- #25 e #26 são independentes entre si

**Métricas de Sucesso:**
- Speedup de 4x em operações CPU vetorizadas
- Zero alocações em hot paths
- SGD momentum funciona corretamente

---

## FASE 3: MÉDIA (Semanas 6-12)

> ✨ Novas funcionalidades e melhorias significativas.

### 3.1 Novos Backends
- [ ] **#29** Implementar Backend CUDA
  - **Estimativa:** 12-18 semanas (tarefa grande)
  - 40+ kernels CUDA necessários
  - Detecção automática de GPUs NVIDIA
  - Alternativa: Dividir em subtarefas por categoria de kernel

- [x] **#36** Remover Backend Vulkan (decisão: remover e focar em Metal + CUDA futuro)
  - Decisão: Implementar completamente ou remover
  - Recomendação: Remover e focar em Metal + CUDA

### 3.2 Otimizações Metal
- [ ] **#35** Adicionar kernels fused (Metal)
  - `matmul_bias_relu` combinado
  - Reduzir bandwidth de memória

### 3.3 Testes e Validação
- [ ] **#37** Implementar gradient checking nos testes
  - Validar backward passes via finite differences
  - Cobrir todas as camadas e ativações

### 3.4 Documentação
- [ ] **#38** Criar ficheiros de documentação em falta
  - `GPU_OPTIMIZATIONS.md`
  - `MemoryTests.md`
  - `CONTRIBUTING.md`

- [ ] **#21** Criar ficheiros de documentação em falta (duplicado)
  - Verificar overlap com #38

- [ ] **#5** Adicionar referências académicas ao código
  - Cit papers para Adam, LSTM, Attention, etc.

- [ ] **#9** Melhorar documentação dos exemplos
  - README em `examples/`
  - report.md para cada exemplo

### 3.5 Funcionalidades de Treino
- [ ] **#39** Implementar learning rate scheduling
  - Step decay, exponential, cosine annealing
  - Integrar em `network.train()`

**Dependências:**
- #29 (CUDA) pode ser desenvolvido em paralelo
- **#36** ~~Completar ou remover Backend Vulkan~~ - **REMOVIDO**

**Métricas de Sucesso:**
- CUDA funcional em Linux/Windows
- Todos os backward passes validados numericamente
- Documentação completa e atualizada

---

## FASE 4: BAIXA (Semanas 13-16)

> 🎯 Melhorias de qualidade de vida e features adicionais.

### 4.1 Qualidade e Robustez
- [ ] **#4** Corrigir comentários enganosos sobre BCE
  - Esclarecer dependência do sigmoid

- [ ] **#47** Melhorar seed de RNG
  - Usar `std.os.getrandom()` em vez de `timestamp()`

- [ ] **#42** Implementar early stopping
  - Monitorar validation loss
  - Restaurar melhores pesos

### 4.2 Novas Camadas
- [ ] **#44** Implementar Conv2D
  - Para processamento de imagens
  - Forward e backward completos

- [ ] **#45** Implementar BatchNorm
  - Forward (train/inference)
  - Backward completo

### 4.3 Serialização
- [ ] **#40** Implementar saving/loading de modelos
  - Formato binário próprio (.znn)
  - JSON opcional
  - ONNX (futuro)

### 4.4 Exemplos
- [ ] **#48** Adicionar testes de classificação multiclasse
  - MNIST (dígitos)
  - Iris (flores)

- [ ] **#7** Adicionar testes de classificação multiclasse (duplicado)
  - Verificar overlap com #48

**Dependências:**
- #40 (save/load) pode ser paralelo
- #44 e #45 são independentes

**Métricas de Sucesso:**
- Modelos podem ser guardados e carregados
- Exemplos de classificação funcionais
- Seed de RNG não previsível

---

## Resumo por Prioridade

| Prioridade | Quantidade | Tempo Estimado |
|------------|------------|----------------|
| 🔴 Crítica | 7 tarefas | 2 semanas |
| 🟠 Alta | 8 tarefas | 3 semanas |
| 🟡 Média | 11 tarefas | 6 semanas |
| 🟢 Baixa | 10 tarefas | 4 semanas |
| **Total** | **36 tarefas** | **15 semanas** |

---

## Dependências Entre Tarefas

```
Fase 1 (Crítica)
├── #3, #16 (overflows) ──┐
├── #15, #17 (div zero) ─┤
├── #11, #10 (Metal) ─────┤──> Fase 2 pode começar
└── #18 (Attention) ────────┘

Fase 2 (Alta)
├── #14 (Conv1D) ─────────┐
├── #28, #13 (SIMD) ──────┤
├── #25, #26 (Memory) ─────┤──> Fase 3 pode começar
├── #8 (SGD) ─────────────┤
└── #49 (unwraps) ────────┘

Fase 3 (Média)
├── #29 (CUDA) ───────────┐
├── #36 (Vulkan - REMOVIDO) ─┤
├── #35 (fused) ───────────┤
├── #37 (gradcheck) ───────┤──> Fase 4 pode começar
├── #38, #9 (docs) ────────┤
└── #39 (scheduler) ──────┘

Fase 4 (Baixa)
├── #4 (comentários) ─────┐
├── #47 (RNG) ─────────────┤
├── #42 (early stop) ──────┤
├── #44, #45 (layers) ─────┤──> Projeto completo
├── #40 (save/load) ───────┤
└── #48 (exemplos) ───────┘
```

---

## Checklist de Progresso

### Sprint 1 (Semanas 1-2) - Segurança Crítica
- [ ] Todos os integer overflows corrigidos
- [ ] Divisões por zero protegidas
- [ ] Metal backend compila e executa
- [ ] Attention backward implementado
- [ ] Todos os testes passam

### Sprint 2 (Semanas 3-5) - Performance
- [ ] SIMD implementado nas operações CPU
- [ ] Threadgroup sizes otimizados
- [ ] Alocações em hot paths eliminadas
- [ ] SGD momentum funciona
- [ ] Benchmarks mostram melhoria significativa

### Sprint 3 (Semanas 6-8) - Backend CUDA (Parte 1)
- [ ] Módulo cuda.zig criado
- [ ] Contexto CUDA funcional
- [ ] Kernels core (matmul, ativações) funcionais
- [ ] Testes de paridade CUDA vs CPU

### Sprint 4 (Semanas 9-12) - Backend CUDA (Parte 2) + Docs
- [ ] Todos os kernels CUDA implementados
- [ ] Documentação em falta criada
- [ ] Gradient checking implementado
- [ ] Learning rate scheduling funcional

### Sprint 5 (Semanas 13-16) - Polish
- [ ] Save/load de modelos funcional
- [ ] Conv2D e BatchNorm implementados
- [ ] Exemplos de classificação adicionados
- [ ] Código documentado e pronto para release

---

## Notas para Implementação

### Ordem Recomendada dentro de cada Sprint

1. **Começar pelos testes:** Para cada fix, primeiro escrever um teste que falha
2. **Fazer commits atómicos:** Um commit por tarefa ou subtarefa
3. **Atualizar CHANGELOG:** Documentar cada mudança significativa
4. **Atualizar CLAUDE.md:** Manter status dos componentes atualizado

### Convenções

- Usar prefixos nos commits:
  - `security:` para fixes de segurança
  - `fix:` para bugs
  - `perf:` para otimizações
  - `feat:` para novas funcionalidades
  - `docs:` para documentação

### Recursos

- **Time estimado:** 1 desenvolvedor full-time por ~4 meses
- **Ou:** 2 desenvolvedores por ~2 meses
- **Prioridade máxima:** Fase 1 (segurança) não é negociável

---

## Anexos

### Tarefas Duplicadas a Consolidar
- #21 e #38 (documentação em falta)
- #7 e #48 (exemplos de classificação)

### Tarefas Que Podem Ser Divididas
- #29 (CUDA) → Dividir por categoria de kernel:
  - Core (matmul)
  - Ativações
  - Loss
  - Recorrentes
  - Optimizers

### Links Úteis
- [Auditoria Completa](./AUDIT_REPORT.md)
- [CLAUDE.md](./CLAUDE.md)
- [README.md](./README.md)

---

*Roadmap gerado automaticamente a partir da auditoria completa do ZigNeuron*

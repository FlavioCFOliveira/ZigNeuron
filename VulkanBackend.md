# Backend Vulkan para ZigNeuron

Este documento descreve a implementação do backend Vulkan para o ZigNeuron.

## Visão Geral

O backend Vulkan fornece suporte cross-platform para execução GPU em compute shaders. A implementação utiliza Vulkan 1.1+ para aceleração de:

- Matrix multiplication (matmul)
- Activation functions (ReLU, Sigmoid, Tanh)
- Loss functions (MSE, Cross-Entropy, Binary Cross-Entropy)

## Arquitetura

### Componentes Principais

```
src/
├── vulkan.zig          # FFI bindings + wrapper do device
├── backend.zig         # Integração com backends GPU
└── main.zig            # Exportações da biblioteca

shaders/
├── matmul.comp         # Compute shader para multiplicação de matrizes
├── activation_forward.comp
├── activation_backward.comp
├── loss_backward.comp
└── *.spv               # Shaders compilados (SPIR-V)
```

### Estrutura do Vulkan Module (`vulkan.zig`)

#### Types

| Type | Descrição |
|------|-----------|
| `Device` | Wrapper para VkInstance, VkPhysicalDevice, VkDevice, VkQueue |
| `Buffer` | Wrapper para VkBuffer + VkDeviceMemory |
| `CommandPool` | Wrapper para VkCommandPool |
| `CommandBuffer` | Wrapper para VkCommandBuffer |
| `DescriptorSetLayout` | Wrapper para VkDescriptorSetLayout |
| `DescriptorPool` | Wrapper para VkDescriptorPool |
| `PipelineLayout` | Wrapper para VkPipelineLayout |
| `Pipeline` | Wrapper para VkPipeline |
| `ShaderModule` | Wrapper para VkShaderModule |

#### Funções Principais

```zig
// Criar e destruir device Vulkan
pub fn Device.init() !Device
pub fn deinit(self: *Device) void

// Criar buffers de GPU
pub fn createBuffer(self: *Device, size: usize, usage: u32) !Buffer

// Criar pipelines computacionais
pub fn createPipeline(
    self: *Device,
    layout: vk.PipelineLayout,
    shader_module: vk.ShaderModule,
    entry_point: []const u8,
    work_group_size: [3]u32,
) !Pipeline

// Executar operações via Vulkan
pub fn vulkanMatMul(device: *Device, a, b, c: []const f32, m, n, k: usize) !void
pub fn vulkanActivationForward(device: *Device, act: anytype, input, output: []f32) !void
pub fn vulkanActivationBackward(device: *Device, act: anytype, input, grad_output, grad_input: []const f32) !void
pub fn vulkanLossBackward(device: *Device, loss_fn: anytype, output, target, grad_output: []const f32) !void
```

### Compute Shaders

#### 1. Matrix Multiplication (`matmul.comp`)

```glsl
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer BufferA { float a[]; };
layout(set = 0, binding = 1) readonly buffer BufferB { float b[]; };
layout(set = 0, binding = 2) writeonly buffer BufferC { float c[]; };

layout(push_constant) uniform Constants {
    uint m, n, k;
} consts;

void main() {
    uint global_x = gl_GlobalInvocationID.x;
    uint global_y = gl_GlobalInvocationID.y;

    if (global_x >= consts.m || global_y >= consts.n) return;

    float sum = 0.0;
    for (uint i = 0; i < consts.k; i++) {
        sum += a[global_x * consts.k + i] * b[i * consts.n + global_y];
    }
    c[global_x * consts.n + global_y] = sum;
}
```

#### 2. Activation Forward (`activation_forward.comp`)

Computa: `output = activation(input)`

- ReLU: `max(0, x)`
- Sigmoid: `1 / (1 + exp(-x))`
- Tanh: `tanh(x)`

#### 3. Activation Backward (`activation_backward.comp`)

Computa: `grad_input = grad_output * activation_derivative(input)`

#### 4. Loss Backward (`loss_backward.comp`)

Computa gradientes para:
- MSE: `2 * (output - target)`
- Cross-Entropy: `-target / output`
- Binary Cross-Entropy: `(output - target) / (output * (1 - output))`

## Build System

### Compilando os Shaders

```bash
# Compilar todos os shaders SPIR-V
zig build compile-shaders

# Ou manualmente com glslc
glslc -fshader-stage=compute shaders/matmul.comp -o shaders/matmul.comp.spv
glslc -fshader-stage=compute shaders/activation_forward.comp -o shaders/activation_forward.comp.spv
glslc -fshader-stage=compute shaders/activation_backward.comp -o shaders/activation_backward.comp.spv
glslc -fshader-stage=compute shaders/loss_backward.comp -o shaders/loss_backward.comp.spv
```

### Build da biblioteca

```bash
# Build padrão (com Vulkan)
zig build

# Build sem Vulkan
zig build -Dvulkan=false
```

## Requisitos

### Dependências

- **Vulkan SDK**: Necessário para compilar shaders SPIR-V
  - macOS: `brew install --cask vulkan-sdk`
  - Ubuntu/Debian: `apt install vulkan-sdk glslc`
  - Windows: Download do [LunarG Vulkan SDK](https://vulkan.lunarg.com/)

### Hardware

- GPU com suporte a Vulkan (Vulkan 1.1+)
- Compute queue support
- Memory property: `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`

## Uso

### Detectar e usar o backend Vulkan

```zig
const zn = @import("ZigNeuron");

// O backend Vulkan é usado automaticamente pelo Backend.default()
const backend = zn.backend.Backend.default();

// Criar rede neural com o backend
const net = try zn.network.Network.init(allocator, backend);
defer net.deinit();

// Adicionar camadas...
```

### Execução manual com Vulkan

```zig
const vulkan = @import("vulkan.zig");

// Criar device Vulkan
const device = try vulkan.Device.init();
defer device.deinit();

// Executar matmul com Vulkan
try vulkan.vulkanMatMul(&device, a, b, c, m, n, k);
```

## Fallbacks

### Fallback para CPU

Quando Vulkan não está disponível, o código automaticamente cai para implementações CPU:

1. **Device.init()** - Retorna erro `NoGPU` se não houver GPU Vulkan
2. **MatMul pequeno (< 4096 elementos)** - CPU é mais rápido devido ao overhead
3. **Activation/Loss pequeno (< 256 elementos)** - CPU é mais eficiente

### Fallback para Metal (Apple Silicon)

No macOS, o backend Metal tem prioridade高于 Vulkan.

## Performance

### Thresholds de otimização

| Operação | Threshold | Fallback |
|----------|-----------|----------|
| MatMul | < 4096 elementos | CPU |
| Activation | < 256 elementos | CPU |
| Loss | < 256 elementos | CPU |

### Expectativas de speedup

- MatMul (grande): 5x-50x sobre CPU
- Activation (grande): 3x-10x sobre CPU
- Loss (grande): 3x-10x sobre CPU

## Debugging

### Verificar GPU Vulkan disponível

```bash
vulkaninfo | grep -A 10 "gpuId"
```

### Verificar drivers

```bash
vulkaninfo | grep "deviceName"
```

### Erros comuns

| Erro | Solução |
|------|---------|
| `InstanceCreateFailed` | Vulkan SDK não instalado |
| `NoGPU` | GPU não suporta Vulkan 1.1+ |
| `DeviceCreateFailed` | Driver Vulkan desatualizado |

## Status da Implementação

| Componente | Status | Notas |
|------------|--------|-------|
| FFI Vulkan | Done | Bindings completas |
| Device management | Done | Instance, Device, Queue |
| Buffer management | Done | VkBuffer + Memory |
| Descriptor sets | Done | Layout, Pool, Sets |
| Pipeline | Done | Compute pipelines |
| Shaders SPIR-V | Done | 4 shaders compilados |
| MatMul | Done | Com GPU fallback |
| Activation | Done | Forward/Backward |
| Loss | Done | MSE/CE/BCE |
| Build system | Done | compile-shaders step |

## Próximos Passos

- [ ] Pooling de buffers para reuso
- [ ] Async compute execution
- [ ] Memory deduplication
- [ ] Support para mais tipos de dados (f16, f64)
- [ ] Batch processing support

## Referências

- [Vulkan Spec](https://www.khronos.org/registry/vulkan/specs/1.3/html/)
- [Zig FFI](https://ziglang.org/learn/overview/#foreign-function-interface)
- [SPIR-V Specification](https://www.khronos.org/registry/spir-v/)

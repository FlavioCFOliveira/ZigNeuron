# Makefile for ZigNeuron CUDA Kernels
# Compiles all .cu files to PTX for runtime loading

# CUDA Compiler
NVCC = nvcc

# CUDA Architecture Flags
# Supports: Turing (7.5), Ampere (8.0, 8.6), Ada (8.9), Hopper (9.0)
ARCH_FLAGS = -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90

# Compilation flags
NVCC_FLAGS = -O3 -dc -std=c++17 $(ARCH_FLAGS)

# For PTX generation (JIT compilation)
PTX_FLAGS = -O3 -ptx $(ARCH_FLAGS)

# Source files
SOURCES = matmul.cu activation.cu optimizer.cu loss.cu \
          convolution.cu normalization.cu attention.cu \
          dropout.cu recurrent.cu auxiliary.cu

# Object files (for static linking)
OBJECTS = $(SOURCES:.cu=.o)

# PTX files (for runtime loading)
PTX_FILES = $(SOURCES:.cu=.ptx)

# Targets
.PHONY: all ptx objects clean

all: ptx objects

# Compile to PTX for runtime loading
ptx: $(PTX_FILES)

%.ptx: %.cu common.h
	$(NVCC) $(PTX_FLAGS) -o $@ $<

# Compile to object files for static linking
objects: $(OBJECTS)

%.o: %.cu common.h
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Create static library
libzigneuron_cuda.a: $(OBJECTS)
	nvcc -lib -o $@ $^

# Clean
clean:
	rm -f *.o *.ptx *.a

# Compile and link everything into a single PTX for Zig embed
single_ptx: $(SOURCES) common.h
	$(NVCC) $(PTX_FLAGS) -o zigneuron_kernels.ptx $(SOURCES)

# Help
help:
	@echo "ZigNeuron CUDA Kernels Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  ptx          - Compile to PTX files for runtime loading"
	@echo "  objects      - Compile to object files"
	@echo "  single_ptx   - Compile all kernels to single PTX file"
	@echo "  clean        - Remove generated files"
	@echo "  help         - Show this help"

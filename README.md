In progress & evolving design of NUMA aware tensor parallel CPU-only AMX inference engine.

### Main points:
- Modern C++ (cpp23)
- Modules
- No external dependencies
- Megatron Tensor-Parallel Row/Col interleaving
- NUMA Awareness
- AVX512 + AMX exclusive
- Pure AMX GEMM

### AMXQ
- AMXQ (Grouped asymmetric mean-centered quantization)
- Fused AMXQ AMX GEMM (Reduces bandwidth pressure by shrinking the accumulator)

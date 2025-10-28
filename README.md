### Actively in Development
This is currently in progress and is not feature complete. Below are the existing features, but there's still quite a lot of work before inference can be done.

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

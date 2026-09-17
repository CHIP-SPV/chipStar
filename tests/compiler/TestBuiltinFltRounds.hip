// __builtin_flt_rounds emits llvm.get.rounding, which llvm-spirv rejects.
#include <hip/hip_runtime.h>

__global__ void k(int *Out) { Out[0] = __builtin_flt_rounds(); }

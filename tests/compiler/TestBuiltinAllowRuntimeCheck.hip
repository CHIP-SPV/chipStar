// __builtin_allow_runtime_check emits llvm.allow.runtime.check, which llvm-spirv rejects.
#include <hip/hip_runtime.h>

__global__ void k(int *Out, int N) {
  if (__builtin_allow_runtime_check("bounds") && N > 4096)
    return;
  Out[0] = N;
}

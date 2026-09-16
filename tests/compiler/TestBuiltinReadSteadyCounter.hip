// __builtin_readsteadycounter emits llvm.readsteadycounter, which llvm-spirv rejects.
#include <hip/hip_runtime.h>

__global__ void k(unsigned long long *Out) {
  Out[0] = __builtin_readsteadycounter();
}

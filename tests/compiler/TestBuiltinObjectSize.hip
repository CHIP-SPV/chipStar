// __builtin_object_size emits llvm.objectsize, which llvm-spirv rejects.
#include <hip/hip_runtime.h>

__global__ void k(char *P, unsigned long long *Out) {
  Out[0] = __builtin_object_size(P, 0);
  Out[1] = __builtin_object_size(P, 2);
}

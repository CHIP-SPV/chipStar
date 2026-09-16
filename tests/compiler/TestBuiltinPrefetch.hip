// __builtin_prefetch emits llvm.prefetch, which llvm-spirv rejects.
#include <hip/hip_runtime.h>

__global__ void k(const float *In, float *Out) {
  __builtin_prefetch(In + 16);
  __builtin_prefetch(Out + 16, 1);
  __builtin_prefetch(In + 32, 0, 3);
  Out[0] = In[0];
}

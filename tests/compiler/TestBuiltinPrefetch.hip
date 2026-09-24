// __builtin_prefetch emits llvm.prefetch, which llvm-spirv rejects.
#include <hip/hip_runtime.h>

__device__ float G[64];

__global__ void k(const float *In, float *Out) {
  __shared__ float S[64];
  __builtin_prefetch(In + 16);
  __builtin_prefetch(Out + 16, 1);
  __builtin_prefetch(G + 1, 0, 3);
  __builtin_prefetch(S + 1);
  Out[0] = In[0];
}

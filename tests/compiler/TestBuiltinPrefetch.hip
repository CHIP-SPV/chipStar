// __builtin_prefetch emits llvm.prefetch, which llvm-spirv rejects.
#include <hip/hip_runtime.h>

__device__ float G[64];

__global__ void k(const float *In, float *Out) {
  __shared__ float S[64];
  __builtin_prefetch(In + 16);
  __builtin_prefetch(Out + 16, 1);
  __builtin_prefetch(G + 1, 0, 3);
  using GP = const float __attribute__((address_space(1))) *;
  using GI = const size_t __attribute__((address_space(1))) *;
  __builtin_prefetch((const void *)(GP)((size_t)(GP)S + threadIdx.x));
  __builtin_prefetch(S + 1);
  __builtin_prefetch((const void *)(GP)*(GI)S);
  Out[0] = In[0];
}

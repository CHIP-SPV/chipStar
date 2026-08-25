// __builtin_prefetch in device code emits llvm.prefetch, which no SPIR-V
// producer chipStar uses accepts. The llvm-spirv translator rejects it with
// "InvalidFunctionCall: Unexpected llvm intrinsic: llvm.prefetch.p4" and the
// in-tree SPIR-V backend with "unable to legalize instruction: G_PREFETCH".
// The intrinsic is a pure hint, so HipLowerHintIntrinsicsPass erases it.
#include <hip/hip_runtime.h>

__global__ void k(const float *In, float *Out) {
  __builtin_prefetch(In + 16);
  __builtin_prefetch(Out + 16, 1);
  __builtin_prefetch(In + 32, 0, 3);
  Out[0] = In[0];
}

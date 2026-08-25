// __builtin_readcyclecounter emits llvm.readcyclecounter, which no SPIR-V
// producer chipStar uses accepts. A SPIR-V kernel has no cycle counter
// register, and LangRef says a backend without support lowers the intrinsic
// to a constant 0, so HipLowerHintIntrinsicsPass folds it to 0.
#include <hip/hip_runtime.h>

__global__ void k(unsigned long long *Out) {
  Out[0] = __builtin_readcyclecounter();
}

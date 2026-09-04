// __builtin_readsteadycounter emits llvm.readsteadycounter, the fixed
// frequency sibling of llvm.readcyclecounter. Neither SPIR-V producer
// accepts it and LangRef gives both the same "lowered to a constant 0"
// fallback, so HipLowerHintIntrinsicsPass folds it to 0.
#include <hip/hip_runtime.h>

__global__ void k(unsigned long long *Out) {
  Out[0] = __builtin_readsteadycounter();
}

// __builtin_allow_runtime_check emits llvm.allow.runtime.check, which the
// llvm-spirv translator rejects. LangRef requires the program to be correct
// whichever of true or false the intrinsic yields, so
// HipLowerHintIntrinsicsPass folds it to false, eliding the guarded check.
#include <hip/hip_runtime.h>

__global__ void k(int *Out, int N) {
  if (__builtin_allow_runtime_check("bounds") && N > 4096)
    return;
  Out[0] = N;
}

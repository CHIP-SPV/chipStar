// __builtin_object_size emits llvm.objectsize, which the llvm-spirv
// translator rejects. The intrinsic never reads memory and is always
// resolvable at compile time, so HipLowerHintIntrinsicsPass resolves it with
// LLVM's own lowerObjectSizeCall. Compiled at -O0 only because from -O1 up
// LLVM's LowerConstantIntrinsics pass already folds it away before the device
// link, which hides the bug.
#include <hip/hip_runtime.h>

__global__ void k(char *P, unsigned long long *Out) {
  Out[0] = __builtin_object_size(P, 0);
  Out[1] = __builtin_object_size(P, 2);
}

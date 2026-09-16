// __builtin_memcpy_inline emits llvm.memcpy.inline, which llvm-spirv rejects.
#include <hip/hip_runtime.h>

__global__ void k(char *D, const char *S) {
  __builtin_memcpy_inline(D, S, 16);
}

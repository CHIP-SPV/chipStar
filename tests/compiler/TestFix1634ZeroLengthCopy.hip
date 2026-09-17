// A constant zero length copy must not reach SPIR-V: OpCopyMemorySized
// forbids a constant zero Size (CHIP-SPV/chipStar#1634).
#include <hip/hip_runtime.h>

// Lowering this zero sized global emits a zero length copy of its initializer.
__device__ int ZeroLenArray[0];

__global__ void k(char *D, const char *S, unsigned long N) {
  __builtin_memcpy(D, S, 0);
  __builtin_memmove(D, S, 0);
  // Only a constant zero length copy may be erased; these two must survive.
  __builtin_memcpy(D + 1, S, 16);
  __builtin_memmove(D + 32, S, N);
  D[0] = (char)(long)&ZeroLenArray;
}

// The length is constant zero only after __builtin_object_size is folded, and
// the goto puts the copy ahead of it in block order.
__global__ void foldedLength(char *D, const char *S) {
  unsigned long N;
  goto length;
copy:
  __builtin_memcpy(D, S, N);
  return;
length:
  N = __builtin_object_size(S, 2);
  goto copy;
}

#include <hip/hip_runtime.h>

struct  __align__(32) s0_t {
  int a;
};

__global__ void
check(size_t *IsAligned,
      int Dummy, //< This attempts to throw off the alignment of Val.
      s0_t Val) {
  auto AlignmentMask = ~(alignof(s0_t) - 1u);
  printf("&Val=%p\n", &Val);
  *IsAligned = AlignmentMask & reinterpret_cast<uintptr_t>(&Val);
}

int main() {
  size_t OutH = 0, *OutD;
  hipMalloc(&OutD, sizeof(size_t));
  s0_t s0{123};
  check<<<1, 1>>>(OutD, 0, s0);
  hipMemcpy(&OutH, OutD, sizeof(size_t), hipMemcpyDeviceToHost);
  if (!OutH)
    return 1;
  return 0;
}

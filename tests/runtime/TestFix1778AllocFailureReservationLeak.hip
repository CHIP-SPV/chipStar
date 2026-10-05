// Reproduces #1778: when the backend allocation threw, Context::allocate kept
// the size it had reserved, so a failed allocation lowered hipMemGetInfo's free
// memory for good.
#include "CHIPBackend.hh"
#include <hip/hip_runtime_api.h>
#include <cstdio>

int main() {
  size_t Free0, Free1, Total;
  if (hipMemGetInfo(&Free0, &Total) != hipSuccess)
    return 1;

  // Level Zero and OpenCL bufferdevaddr throw for this type without allocating.
  try {
    // Not freed: eraseRecord expects a hipHostRegister-counted Managed record.
    Backend->getActiveContext()->allocate(1 << 20, hipMemoryTypeManaged);
    printf("HIP_SKIP_THIS_TEST: the backend accepted the allocation\n");
    return CHIP_SKIP_TEST;
  } catch (...) {
  }

  if (hipMemGetInfo(&Free1, &Total) != hipSuccess)
    return 1;
  if (Free1 != Free0) {
    printf("FAIL: a failed allocation lowered free memory by %zu\n",
           Free0 - Free1);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

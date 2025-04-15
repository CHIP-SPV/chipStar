// Test various HIP API functions.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <hip/hip_runtime.h>
#include <cassert>
#include <iostream>

__global__ void k() {}

int main() {
  // Non-conformant from C++ spec PoV. Not sure if HIP supports this.
  auto *KPtr = reinterpret_cast<const void *>(k);

  for (auto Cfg : {hipFuncCachePreferNone, hipFuncCachePreferShared,
                   hipFuncCachePreferL1, hipFuncCachePreferEqual}) {
    // Preferences should not return error code.
    assert(hipFuncSetCacheConfig(KPtr, Cfg) == hipSuccess);
    assert(hipFuncSetCacheConfig(nullptr, Cfg) != hipSuccess);
  }

  std::cout << "PASSED\n";
  return 0;
}

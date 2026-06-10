// Verify that hipPointerAttribute_t exposes both 'type' (CUDA-aligned name)
// and 'memoryType' (deprecated AMD/ROCm name) as aliases for the same field.
// Many older HIP/ROCm and ported CUDA codebases (e.g. HecBench's
// warpselect-hip/DeviceUtils.cu) read `attr.memoryType` and fail to compile
// when only `type` is provided. AMD historically kept both via an anonymous
// union (see ROCm/HIP commit 99c48bfc / SWDEV-399623), which we mirror.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <hip/hip_runtime.h>
#include <cassert>
#include <iostream>

int main() {
  void *dPtr = nullptr;
  assert(hipMalloc(&dPtr, 1024) == hipSuccess);

  hipPointerAttribute_t attr{};
  assert(hipPointerGetAttributes(&attr, dPtr) == hipSuccess);

  // Both names must compile and must refer to the same storage. The exact
  // memory-type value chipStar reports for hipMalloc'd memory is backend-
  // dependent (Device on real GPUs, but possibly Managed/Unified on pocl-CPU
  // where the device is the host), so don't assert on a specific enum value —
  // only that the two names see identical bits.
  static_assert(sizeof(attr.type) == sizeof(attr.memoryType),
                "type and memoryType must alias the same field");
  assert(static_cast<int>(attr.type) == static_cast<int>(attr.memoryType));

  // Writing through one alias must be visible via the other.
  attr.memoryType = hipMemoryTypeHost;
  assert(attr.type == hipMemoryTypeHost);
  attr.type = hipMemoryTypeDevice;
  assert(attr.memoryType == hipMemoryTypeDevice);

  assert(hipFree(dPtr) == hipSuccess);
  std::cout << "PASSED\n";
  return 0;
}

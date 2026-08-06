// Payload for the module cache invalidation test.
//
// Driven repeatedly by tests/run_module_cache_invalidation_test.cmake, which
// varies the compilation inputs between runs and checks whether the module
// cache key changed. This program only has to compile a device module and
// produce a correct result; the assertions about caching live in that script.
//
// atomicAdd on a float is deliberate: it makes the module import a symbol from
// the runtime device library, so the compile takes the compile+link path that
// links an rtdevlib module in after the cache is consulted. Those modules are
// part of what the key must cover, and a module that needs none of them would
// take the direct clBuildProgram path and never exercise it.

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void accumulate(float *Out) { atomicAdd(Out, 1.0f); }

int main() {
  constexpr int NumThreads = 64;

  float *DevOut = nullptr;
  if (hipMalloc(&DevOut, sizeof(float)) != hipSuccess) {
    printf("FAILED: hipMalloc\n");
    return 1;
  }
  if (hipMemset(DevOut, 0, sizeof(float)) != hipSuccess) {
    printf("FAILED: hipMemset\n");
    return 1;
  }

  accumulate<<<1, NumThreads>>>(DevOut);

  float HostOut = 0.0f;
  if (hipMemcpy(&HostOut, DevOut, sizeof(float), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    printf("FAILED: hipMemcpy\n");
    return 1;
  }
  (void)hipFree(DevOut);

  if (HostOut != static_cast<float>(NumThreads)) {
    printf("FAILED: expected %d, got %f\n", NumThreads, HostOut);
    return 1;
  }

  printf("PASSED\n");
  return 0;
}

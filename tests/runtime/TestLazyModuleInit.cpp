#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <hip/hip_runtime.h>

#include "CHIPDriver.hh"
#include "SPVRegister.hh"

// Remove device variable that causes hipspv-link to hang
// __device__ int Foo = 123;
__global__ void bar(int *Dst) { *Dst = 42; } // Use constant instead

int main() {
  const char *LazyJit = std::getenv("CHIP_LAZY_JIT");
  if (LazyJit && std::string_view(LazyJit) == "0") {
    printf("CHIP_LAZY_JIT is set to 0. Skip testing.\n");
    return CHIP_SKIP_TEST;
  }

  // Check the source binary is registered.
  assert(getSPVRegister().getNumSources() == 1);

  int OutH, *OutD;
  (void)hipMalloc(&OutD, sizeof(int));

  // Check the module is not initialized/compiled yet.
  auto *RuntimeDev = Backend->getActiveDevice();
  assert(RuntimeDev);
  assert(RuntimeDev->getNumCompiledModules() == 0);

  // Launch kernel - this should trigger lazy compilation
  bar<<<1, 1>>>(OutD);
  hipDeviceSynchronize();

  // Check the module is now compiled
  assert(RuntimeDev->getNumCompiledModules() == 1);

  hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost);
  assert(OutH == 42);

  hipFree(OutD);
  std::cout << "PASSED\n";
  return 0;
}

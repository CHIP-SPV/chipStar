#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <hip/hip_runtime.h>

#include "CHIPDriver.hh"
#include "SPVRegister.hh"

__device__ int Foo = 123;
__global__ void bar(int *Dst) { *Dst = Foo; }

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

  bar<<<1, 1>>>(OutD);

  // Check getNumCompiledModules() reports correctly.
  assert(RuntimeDev->getNumCompiledModules() == 1);

  (void)hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost);
  bool passed = (OutH == 123);
  if (passed)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";
  return !passed;
}

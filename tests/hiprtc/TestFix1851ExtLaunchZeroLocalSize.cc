// Reproduces #1851: hipExtModuleLaunchKernel with a zero local work size
// divides by zero (SIGFPE) instead of returning hipErrorInvalidConfiguration.

#include "TestCommon.hh"

// Declared here, not via hip/hip_ext.h: C++ linkage in libCHIP (#1857).
hipError_t hipExtModuleLaunchKernel(hipFunction_t, uint32_t, uint32_t,
                                    uint32_t, uint32_t, uint32_t, uint32_t,
                                    size_t, hipStream_t, void **, void **,
                                    hipEvent_t = nullptr, hipEvent_t = nullptr,
                                    uint32_t = 0);

int main() {
  auto Program = HiprtcAssertCreateProgram(
      R"---(extern "C" __global__ void k() {})---");
  auto Code = HiprtcAssertCompileProgram(Program);
  hipModule_t Module;
  hipFunction_t Kernel;
  HIP_CHECK(hipModuleLoadData(&Module, Code.data()));
  HIP_CHECK(hipModuleGetFunction(&Kernel, Module, "k"));

  const uint32_t Local[][3] = {{0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
  for (const auto &L : Local)
    TEST_ASSERT(hipExtModuleLaunchKernel(Kernel, 1, 1, 1, L[0], L[1], L[2], 0,
                                         nullptr, nullptr, nullptr) ==
                hipErrorInvalidConfiguration);

  HIPRTC_CHECK(hiprtcDestroyProgram(&Program));
  HIP_CHECK(hipModuleUnload(Module));
  std::cout << "PASSED\n";
  return 0;
}

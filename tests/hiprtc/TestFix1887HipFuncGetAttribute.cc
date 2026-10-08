// hipFuncGetAttribute on a module-loaded kernel must answer, not return
// hipErrorNotSupported.

#include "TestCommon.hh"

static constexpr auto Source = R"---(
extern "C" __global__ void k(int *Out) { Out[threadIdx.x] = threadIdx.x; }
)---";

int main() {
  auto Prog = HiprtcAssertCreateProgram(Source);
  auto Code = HiprtcAssertCompileProgram(Prog);
  hipModule_t Module;
  HIP_CHECK(hipModuleLoadData(&Module, Code.data()));
  hipFunction_t Kernel;
  HIP_CHECK(hipModuleGetFunction(&Kernel, Module, "k"));

  int MaxThreads = 0;
  HIP_CHECK(hipFuncGetAttribute(
      &MaxThreads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, Kernel));
  hipDeviceProp_t Props;
  HIP_CHECK(hipGetDeviceProperties(&Props, 0));
  std::cerr << "maxThreadsPerBlock=" << MaxThreads << " device max="
            << Props.maxThreadsPerBlock << "\n";
  TEST_ASSERT(MaxThreads > 0 && MaxThreads <= Props.maxThreadsPerBlock);

  HIP_CHECK(hipModuleUnload(Module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&Prog));
  std::cout << "PASSED\n";
  return 0;
}

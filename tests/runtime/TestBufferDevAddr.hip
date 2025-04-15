// OpenCL-only test. Check CHIP_OCL_USE_ALLOC_STRATEGY=bufferdevaddr
// and its excepted consequences. Notably unified virtual addressing (UVA)
// won't not supported.

#include <hip/hip_runtime.h>
#include "backend/OpenCL/CHIPBackendOpenCL.hh"

#include <unistd.h>

void check(bool Ok, size_t Line) {
  if (!Ok) {
    std::cerr << "Error at line " << Line << ".\n";
    std::quick_exit(1);
  }
}

#define CHECK(_Cond) check(_Cond, __LINE__)

void setupTest(int ArgC, char *ArgV[]) {
  // Run the test if we are running on an OpenCL implementation with
  // cl_ext_buffer_device_address available and set the allocation
  // strategy to 'bufferdevaddr'

  auto *OclBE = dynamic_cast<CHIPBackendOpenCL *>(::Backend);
  if (!OclBE) {
    printf("Skip: not running on OpenCL.\n");
    exit(CHIP_SKIP_TEST);
  }

  auto *OclDev = static_cast<CHIPDeviceOpenCL *>(OclBE->getActiveDevice());
  std::string DevExts = OclDev->ClDevice->getInfo<CL_DEVICE_EXTENSIONS>();
  if (DevExts.find("cl_ext_buffer_device_address") == std::string::npos) {
    printf("Skip: target does not have cl_ext_buffer_device_address.\n");
    exit(CHIP_SKIP_TEST);
  }

  auto *Env = std::getenv("CHIP_OCL_USE_ALLOC_STRATEGY");
  if (!Env || std::string_view(Env) != "bufferdevaddr") {
    if (ArgC > 1) // Infinite recursion!
      exit(3);

    printf("Relaunch with CHIP_OCL_USE_ALLOC_STRATEGY=bufferdevaddr.\n");

    // Relaunch test with CHIP_OCL_USE_ALLOC_STRATEGY=bufferdevaddr. Simpler to
    // do this way than trying to construct new backend the right way. Add an
    // extra argument to catch accidental inifinite recursion.
    const char *Args[] = {ArgV[0], "1", nullptr};
    setenv("CHIP_OCL_USE_ALLOC_STRATEGY", "bufferdevaddr", 1);
    execve(ArgV[0], const_cast<char *const *>(Args), environ);
    exit(4); // Something went wrong.
  }

  printf("Test setup complete.\n");
}

struct SomeStruct {
  int A;
  int *Ptr;
};

__global__ void k0(int *Dst, SomeStruct Src) { *Dst = *Src.Ptr; }

int main(int ArgC, char *ArgV[]) {
  setupTest(ArgC, ArgV);

  int AttrVal = -1;
  CHECK(hipDeviceGetAttribute(&AttrVal, hipDeviceAttributeUnifiedAddressing,
                              0) == hipSuccess);
  // Can't comply UVA expectations with bufferdevaddr allocation strategy, hence
  // it must be off.
  CHECK(AttrVal == 0);

  int SrcH = 123, *SrcD;
  CHECK(hipMalloc(&SrcD, sizeof(int)) == hipSuccess);

  // Device and host allocations may alias, thus copy direction can't
  // be determined robustly.
  CHECK(hipMemcpy(SrcD, &SrcH, sizeof(int), hipMemcpyDefault) ==
        hipErrorInvalidMemcpyDirection);

  CHECK(hipMemcpy(SrcD, &SrcH, sizeof(int), hipMemcpyHostToDevice) ==
        hipSuccess);

  int *DstD;
  CHECK(hipMalloc(&DstD, sizeof(int)) == hipSuccess);

  int *DstH;
  // With UVA off, hipHostMalloc should still work - the allocation just won't
  // be mapped and portable implicitly.
  CHECK(hipHostMalloc(&DstH, sizeof(int)) == hipSuccess);

  k0<<<1, 1>>>(DstD, {123, SrcD});
  CHECK(hipMemcpy(DstH, DstD, sizeof(int), hipMemcpyDeviceToHost) ==
        hipSuccess);
  CHECK(*DstH == 123);

  CHECK(hipFree(DstD) == hipSuccess);
  CHECK(hipFree(SrcD) == hipSuccess);
  CHECK(hipHostFree(DstH) == hipSuccess);

  printf("PASSED.\n");
  return 0;
}

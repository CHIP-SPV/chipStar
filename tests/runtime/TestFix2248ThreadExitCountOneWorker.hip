// https://github.com/CHIP-SPV/chipStar/issues/2248
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

int main() {
  uintptr_t Handles[8] = {};
  int Err = hipGetBackendNativeHandles(0, Handles, nullptr);
  if (Err == hipErrorInitializationError || Err == hipErrorNoDevice) {
    printf("HIP_SKIP_THIS_TEST: no backend available\n");
    return CHIP_SKIP_TEST;
  }
  if (Err != hipSuccess) {
    printf("FAIL: hipGetBackendNativeHandles returned %d\n", Err);
    return 1;
  }
  if (!strcmp(reinterpret_cast<const char *>(Handles[0]), "opencl")) {
    printf("HIP_SKIP_THIS_TEST: OpenCL uninitialize has its own copy of this wait, not fixed here\n");
    return CHIP_SKIP_TEST;
  }

  std::atomic<bool> Started{false};
  std::thread([&] {
    int C;
    hipGetDeviceCount(&C);
    Started = true;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    printf("PASS\n");
    fflush(stdout);
  }).detach();
  while (!Started)
    std::this_thread::yield();
  return 0;
}

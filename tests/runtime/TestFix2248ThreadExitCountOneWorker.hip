// https://github.com/CHIP-SPV/chipStar/issues/2248
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

int main() {
  int N = 0, NumHandles = 0;
  hipGetDeviceCount(&N);
  hipGetBackendNativeHandles(0, nullptr, &NumHandles);
  uintptr_t Handles[8] = {};
  hipGetBackendNativeHandles(0, Handles, nullptr);
  if (!strcmp(reinterpret_cast<const char *>(Handles[0]), "opencl")) {
    printf("HIP_SKIP_THIS_TEST: OpenCL teardown has its own thread wait\n");
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

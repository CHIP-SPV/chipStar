// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2259: reinitializing
// the backend freed the modules and device variables it kept using.
#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

// Weak so this also links on a build without the OpenCL backend.
extern "C" __attribute__((weak)) int clRetainContext(void *Context);

__device__ int Global = 42;
__global__ void kernel(int *Out) { *Out = Global; }

static bool launchOk() {
  int *D = nullptr, H = 0;
  hipMalloc(&D, sizeof(int));
  kernel<<<1, 1>>>(D);
  hipDeviceSynchronize();
  hipMemcpy(&H, D, sizeof(int), hipMemcpyDeviceToHost);
  hipFree(D);
  return H == 42;
}

int main() {
  if (!launchOk()) {
    printf("FAIL: first launch\n");
    return 1;
  }
  int N = 0;
  hipGetBackendNativeHandles(0, nullptr, &N);
  std::vector<uintptr_t> Handles(N);
  hipGetBackendNativeHandles(0, Handles.data(), nullptr);
  // Level Zero's old backend destroys its own context and queue, so a
  // backend's own handles cannot be reinitialized from there.
  if (strcmp(reinterpret_cast<const char *>(Handles[0]), "opencl")) {
    printf("HIP_SKIP_THIS_TEST: needs the OpenCL backend\n");
    return CHIP_SKIP_TEST;
  }
  // The new backend adopts the context without retaining it.
  clRetainContext(reinterpret_cast<void *>(Handles[3]));
  if (hipInitFromNativeHandles(&Handles[1], 4) != hipSuccess) {
    printf("FAIL: hipInitFromNativeHandles\n");
    return 1;
  }
  if (!launchOk()) {
    printf("FAIL: launch after reinitialization\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

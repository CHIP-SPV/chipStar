// Reproduces #2284: hipStreamPerThread is unusable after hipInitFromNativeHandles.
#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

extern "C" __attribute__((weak)) int clRetainContext(void *Context);

static bool streamOk() {
  int *D = nullptr, H = 0;
  if (hipMalloc(&D, sizeof(int)) != hipSuccess)
    return false;
  bool Ok = hipMemsetAsync(D, 7, sizeof(int), hipStreamPerThread) == hipSuccess &&
            hipStreamSynchronize(hipStreamPerThread) == hipSuccess &&
            hipMemcpy(&H, D, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess &&
            H == 0x07070707;
  return hipFree(D) == hipSuccess && Ok;
}

static bool streamOkOnThread() {
  bool Ok = false;
  std::thread([&Ok] { Ok = streamOk(); }).join();
  return Ok;
}

int main() {
  bool Ok = streamOk() && streamOkOnThread();

  int N = 0;
  hipGetBackendNativeHandles(0, nullptr, &N);
  std::vector<uintptr_t> Handles(N);
  hipGetBackendNativeHandles(0, Handles.data(), nullptr);
  if (strcmp(reinterpret_cast<const char *>(Handles[0]), "opencl")) {
    printf("HIP_SKIP_THIS_TEST: needs the OpenCL backend\n");
    return CHIP_SKIP_TEST;
  }
  // The new backend adopts the context without retaining it.
  clRetainContext(reinterpret_cast<void *>(Handles[3]));
  Ok = Ok && hipInitFromNativeHandles(&Handles[1], 4) == hipSuccess;

  Ok = Ok && streamOk() && streamOkOnThread();
  printf(Ok ? "PASS\n" : "FAIL\n");
  return Ok ? 0 : 1;
}

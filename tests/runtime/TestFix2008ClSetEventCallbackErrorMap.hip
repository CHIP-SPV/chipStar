// https://github.com/CHIP-SPV/chipStar/issues/2008: a failing clSetEventCallback
// must come back as a HIP error, not abort. Run under an interposer.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

static void cb(hipStream_t, hipError_t, void *) {}

int main() {
  unsetenv("CLSETEVENTCALLBACK_FAILED");
  hipError_t Err = hipStreamAddCallback(0, cb, nullptr, 0);
  bool Injected = getenv("CLSETEVENTCALLBACK_FAILED");
  if (Err == hipSuccess && !Injected) {
    printf("SKIP: interposer not reached, not the OpenCL backend\n");
    return 77; // CHIP_SKIP_TEST
  }
  if (Err != hipErrorOutOfMemory || !Injected) {
    printf("FAIL: %s, injected=%d\n", hipGetErrorString(Err), Injected);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

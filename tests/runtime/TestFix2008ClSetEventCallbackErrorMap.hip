// https://github.com/CHIP-SPV/chipStar/issues/2008: a failing clSetEventCallback
// must come back as a HIP error, not abort. Run under an interposer.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

static void cb(hipStream_t, hipError_t, void *) {}

int main() {
  hipError_t Err = hipStreamAddCallback(0, cb, nullptr, 0);
  bool Injected = getenv("CLSETEVENTCALLBACK_FAILED");
  if (Err == hipSuccess && !Injected) {
    printf("HIP_SKIP_THIS_TEST\n"); // interposer not reached: not OpenCL
    return 0;
  }
  if (Err != hipErrorOutOfMemory || !Injected) {
    printf("FAIL: %s, injected=%d\n", hipGetErrorString(Err), Injected);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

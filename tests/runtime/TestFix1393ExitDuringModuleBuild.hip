// Reproduces #1393: the device compiler terminating the process during a
// module build leaves chipStar deadlocked in its own exit handler instead of
// exiting.
//
// The first launch of a kernel is what compiles its module, so reaching the
// launch below is all this program has to do. The process exit is injected
// from inside the backend build call by an LD_PRELOAD interposer, which keeps
// the test independent of whether a given backend compiler happens to reject a
// module today. tests/run_exit_during_module_build_test.cmake drives the run
// and owns the pass/fail decision.

#include <hip/hip_runtime.h>

#include <cstdio>

__global__ void touch(int *Out) { Out[threadIdx.x] = threadIdx.x; }

int main() {
  int *Buf = nullptr;
  if (hipMalloc(&Buf, 64 * sizeof(int)) != hipSuccess) {
    std::printf("hipMalloc failed\n");
    return 1;
  }

  std::printf("launching\n");
  std::fflush(stdout);

  hipLaunchKernelGGL(touch, dim3(1), dim3(64), 0, 0, Buf);
  (void)hipDeviceSynchronize();
  (void)hipFree(Buf);

  std::printf("launch returned\n");
  std::fflush(stdout);
  return 0;
}

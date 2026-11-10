// https://github.com/CHIP-SPV/chipStar/issues/2264: uninitialize with a live HIP thread must not relock EventsMtx.
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <atomic>
#include <cstdio>
#include <thread>
#include <unistd.h>

__global__ void K() { printf("drained\n"); }

#define REQUIRE(C) if (!(C)) { printf("FAIL: %s\n", #C); fflush(stdout); _exit(1); }

int main() {
  hipStream_t S;
  REQUIRE(hipStreamCreate(&S) == hipSuccess); // never destroyed, so the timeout path finishes it
  K<<<1, 1, 0, S>>>();
  REQUIRE(hipGetLastError() == hipSuccess);
  std::atomic<bool> Up{false};
  std::thread([&] { int N; REQUIRE(hipGetDeviceCount(&N) == hipSuccess); Up = true; for (;;) pause(); }).detach();
  while (!Up) usleep(1000);
  alarm(30); // SIGALRM kills the process if the lock is taken twice
  uintptr_t H[4] = {};
  REQUIRE(hipInitFromNativeHandles(H, 4) == hipErrorNotSupported);
  printf("PASS\n");
  fflush(stdout);
  _exit(0);
}

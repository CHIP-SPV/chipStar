// https://github.com/CHIP-SPV/chipStar/issues/2264: uninitialize with a live HIP thread must not relock EventsMtx.
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <atomic>
#include <cstdio>
#include <thread>
#include <unistd.h>

__global__ void K() { printf("x\n"); }

int main() {
  hipStream_t S;
  hipStreamCreate(&S); // never destroyed, so the timeout path finishes it
  K<<<1, 1, 0, S>>>();
  hipStreamSynchronize(S);
  std::atomic<bool> Up{false};
  std::thread([&] { int N; hipGetDeviceCount(&N); Up = true; for (;;) pause(); }).detach();
  while (!Up) usleep(1000);
  alarm(30); // SIGALRM kills the process if the lock is taken twice
  uintptr_t H[4] = {};
  hipInitFromNativeHandles(H, 4);
  printf("PASS\n");
  fflush(stdout);
  _exit(0);
}

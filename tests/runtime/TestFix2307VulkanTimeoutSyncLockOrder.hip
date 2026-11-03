// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2307:
// hipInitFromNativeHandles while another thread is in hipDeviceSynchronize
// must not deadlock on EventsMtx and QueueAddRemoveMtx.
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <thread>
#include <unistd.h>

static thread_local bool Stall = false;
static std::atomic<bool> Stalled{false};

// Holds the drain in vkQueueWaitIdle, under QueueAddRemoveMtx, once.
extern "C" int vkQueueWaitIdle(void *Q) {
  static auto Real = (int (*)(void *))dlsym(RTLD_NEXT, "vkQueueWaitIdle");
  if (Stall) {
    Stall = false;
    Stalled = true;
    sleep(8);
  }
  return Real(Q);
}

#define REQUIRE(C)                                                             \
  if (!(C)) {                                                                  \
    printf("FAIL: %s\n", #C);                                                  \
    fflush(stdout);                                                            \
    _exit(1);                                                                  \
  }

__global__ void kernel() { printf("hello\n"); }

int main() {
  alarm(60); // SIGALRM kills the process if a deadlock occurs
  hipStream_t S;
  REQUIRE(hipStreamCreateWithFlags(&S, hipStreamNonBlocking) == hipSuccess);
  kernel<<<1, 1, 0, S>>>();
  REQUIRE(hipStreamSynchronize(S) == hipSuccess);
  std::thread([] {
    Stall = true;
    REQUIRE(hipDeviceSynchronize() == hipSuccess);
    for (;;)
      pause(); // stays alive so the thread-exit wait times out
  }).detach();
  while (!Stalled)
    usleep(1000);
  uintptr_t H[4] = {};
  REQUIRE(hipInitFromNativeHandles(H, 4) == hipErrorNotSupported);
  printf("PASS\n");
  fflush(stdout);
  _exit(0);
}

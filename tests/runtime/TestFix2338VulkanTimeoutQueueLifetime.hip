// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2338:
// the thread-exit timeout must not destroy a queue a surviving thread is in.
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <thread>
#include <unistd.h>

static thread_local bool Stall = false;
static std::atomic<bool> Stalled{false};

// The worker stays inside its submit, so it never resumes after teardown.
extern "C" int vkQueueSubmit(void *Q, uint32_t N, const void *S, void *F) {
  static auto Real =
      (int (*)(void *, uint32_t, const void *, void *))dlsym(RTLD_NEXT,
                                                              "vkQueueSubmit");
  int R = Real(Q, N, S, F);
  if (Stall) {
    Stalled = true;
    for (;;)
      pause();
  }
  return R;
}

static hipStream_t Stream;
static std::atomic<bool> Done{false};

// The stream is the queue the worker is inside; it must never be freed.
static void checkFree(void *P) {
  if (Stalled && P == (void *)Stream) {
    printf("FAIL: queue freed while a thread is inside it\n");
    fflush(stdout);
    _exit(1);
  }
}
void operator delete(void *P) noexcept {
  checkFree(P);
  free(P);
}
void operator delete(void *P, size_t) noexcept {
  checkFree(P);
  free(P);
}

#define REQUIRE(C)                                                             \
  if (!(C)) {                                                                  \
    printf("FAIL: %s\n", #C);                                                  \
    fflush(stdout);                                                            \
    _exit(1);                                                                  \
  }

int main() {
  alarm(60); // SIGALRM kills the process if teardown hangs
  REQUIRE(hipStreamCreateWithFlags(&Stream, hipStreamNonBlocking) == hipSuccess);
  void *A, *B;
  REQUIRE(hipMalloc(&A, 64) == hipSuccess);
  REQUIRE(hipMalloc(&B, 64) == hipSuccess);
  std::thread([=] {
    Stall = true;
    hipMemcpyAsync(A, B, 64, hipMemcpyDeviceToDevice, Stream);
    Done = true; // reached only where vkQueueSubmit is not the backend
    for (;;)
      pause();
  }).detach();
  while (!Stalled && !Done)
    usleep(1000);
  if (!Stalled) {
    printf("HIP_SKIP_THIS_TEST: not the Vulkan backend\n");
    return CHIP_SKIP_TEST;
  }
  uintptr_t H[4] = {};
  REQUIRE(hipInitFromNativeHandles(H, 4) == hipErrorNotSupported);
  printf("PASS\n");
  fflush(stdout);
  _exit(0);
}

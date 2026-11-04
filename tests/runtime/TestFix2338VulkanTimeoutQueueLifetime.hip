// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2338:
// the thread-exit timeout must not destroy a queue a surviving thread is in.
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

// Only ~CHIPQueueVulkan destroys a command pool during teardown.
extern "C" void vkDestroyCommandPool(void *D, void *P, const void *A) {
  static auto Real = (void (*)(void *, void *, const void *))dlsym(
      RTLD_NEXT, "vkDestroyCommandPool");
  if (Stalled) {
    printf("FAIL: queue destroyed while a thread is inside it\n");
    fflush(stdout);
    _exit(1);
  }
  Real(D, P, A);
}

#define REQUIRE(C)                                                             \
  if (!(C)) {                                                                  \
    printf("FAIL: %s\n", #C);                                                  \
    fflush(stdout);                                                            \
    _exit(1);                                                                  \
  }

int main() {
  alarm(60); // SIGALRM kills the process if teardown hangs
  hipStream_t S;
  REQUIRE(hipStreamCreateWithFlags(&S, hipStreamNonBlocking) == hipSuccess);
  void *A, *B;
  REQUIRE(hipMalloc(&A, 64) == hipSuccess);
  REQUIRE(hipMalloc(&B, 64) == hipSuccess);
  std::thread([=] {
    Stall = true;
    hipMemcpyAsync(A, B, 64, hipMemcpyDeviceToDevice, S);
    for (;;)
      pause();
  }).detach();
  while (!Stalled)
    usleep(1000);
  uintptr_t H[4] = {};
  REQUIRE(hipInitFromNativeHandles(H, 4) == hipErrorNotSupported);
  printf("PASS\n");
  fflush(stdout);
  _exit(0);
}

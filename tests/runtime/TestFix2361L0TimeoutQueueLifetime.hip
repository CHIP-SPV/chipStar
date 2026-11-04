// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2361:
// the thread-exit timeout must not destroy a queue a surviving thread is in.
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <thread>
#include <unistd.h>
#include <vector>

static thread_local bool Stall = false;
static std::atomic<bool> Stalled{false};

// The worker stays inside the copy, before the queue's event lock is taken.
extern "C" int zeEventHostReset(void *E) {
  static auto Real = (int (*)(void *))dlsym(RTLD_NEXT, "zeEventHostReset");
  if (Stall) {
    Stalled = true;
    for (;;)
      pause();
  }
  return Real(E);
}

static hipStream_t Stream;

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

static bool isLevel0() {
  int N = 0;
  if (hipGetBackendNativeHandles(0, nullptr, &N) != hipSuccess || N < 1)
    return false;
  std::vector<uintptr_t> H(N);
  if (hipGetBackendNativeHandles(0, H.data(), nullptr) != hipSuccess)
    return false;
  auto *Name = reinterpret_cast<const char *>(H[0]);
  return Name && !strcmp(Name, "level0");
}

int main() {
  alarm(60); // SIGALRM kills the process if teardown hangs
  if (!isLevel0()) {
    printf("HIP_SKIP_THIS_TEST: not the Level Zero backend\n");
    return CHIP_SKIP_TEST;
  }
  REQUIRE(hipStreamCreateWithFlags(&Stream, hipStreamNonBlocking) == hipSuccess);
  void *A, *B;
  REQUIRE(hipMalloc(&A, 64) == hipSuccess);
  REQUIRE(hipMalloc(&B, 64) == hipSuccess);
  // The early exit for a single surviving thread needs a second one.
  std::atomic<bool> BReady{false};
  std::thread([&] {
    REQUIRE(hipSetDevice(0) == hipSuccess);
    BReady = true;
    for (;;)
      pause();
  }).detach();
  while (!BReady)
    usleep(1000);
  std::thread([=] {
    Stall = true;
    hipMemcpyAsync(A, B, 64, hipMemcpyDeviceToDevice, Stream);
    for (;;)
      pause();
  }).detach();
  for (int I = 0; !Stalled && I < 10000; ++I)
    usleep(1000);
  REQUIRE(Stalled);
  return 0;
}

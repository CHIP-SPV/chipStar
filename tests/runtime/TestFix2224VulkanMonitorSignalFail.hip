// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2224: a failed
// vkSignalSemaphore on the Vulkan monitor thread must be reported to the
// stream's waiters, not terminate the process or leave them waiting forever.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

static std::atomic<bool> Fail{false};
static std::atomic<int> Hits{0}, Ran{0};
static std::atomic<bool> Release{false};

extern "C" int vkSignalSemaphore(void *Dev, const void *Info) {
  if (Fail) {
    Hits++;
    return -1; // VK_ERROR_OUT_OF_HOST_MEMORY
  }
  static auto Real = (int (*)(void *, const void *))dlsym(RTLD_NEXT,
                                                          "vkSignalSemaphore");
  return Real(Dev, Info);
}

static void cb(hipStream_t, hipError_t, void *) {
  while (!Release)
    usleep(100);
  Ran++;
}

#define CHECK(Cond)                                                            \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      printf("FAIL: %s (line %d)\n", #Cond, __LINE__);                         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  void *Buf;
  CHECK(hipMalloc(&Buf, 64) == hipSuccess);
  Fail = true;
  CHECK(hipStreamAddCallback(0, cb, nullptr, 0) == hipSuccess);
  CHECK(hipStreamAddCallback(0, cb, nullptr, 0) == hipSuccess);
  // The record submits behind the first callback's timeline value.
  hipEvent_t E;
  CHECK(hipEventCreate(&E) == hipSuccess);
  CHECK(hipEventRecord(E, 0) == hipSuccess);
  Release = true;
  hipError_t Sync = hipStreamSynchronize(0);
  if (!Hits) {
    printf("HIP_SKIP_THIS_TEST: not the Vulkan backend\n");
    return CHIP_SKIP_TEST;
  }
  CHECK(Sync == hipErrorTbd);
  CHECK(Ran >= 1);
  CHECK(hipEventSynchronize(E) == hipErrorTbd);
  CHECK(hipStreamQuery(0) == hipErrorTbd);
  CHECK(hipMemset(Buf, 0, 64) == hipErrorTbd);
  printf("PASS\n");
  return 0;
}

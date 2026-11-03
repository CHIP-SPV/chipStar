// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2214: a hard
// vkGetFenceStatus error must not terminate the event monitor thread.
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <thread>

static std::atomic<bool> Fail{false};
static std::atomic<int> Hits{0};
static std::atomic<int> WaitHits{0};
static std::atomic<bool> Ran{false};
static const int VkErrorDeviceLost = -4;

extern "C" int vkGetFenceStatus(void *Dev, void *Fence) {
  static auto Real = (int (*)(void *, void *))dlsym(RTLD_NEXT, "vkGetFenceStatus");
  if (Fail) {
    Hits++;
    return VkErrorDeviceLost;
  }
  return Real(Dev, Fence);
}

extern "C" int vkWaitForFences(void *Dev, unsigned N, const void *Fences,
                               unsigned WaitAll, unsigned long long Timeout) {
  static auto Real = (int (*)(void *, unsigned, const void *, unsigned,
                              unsigned long long))dlsym(RTLD_NEXT,
                                                        "vkWaitForFences");
  WaitHits++;
  if (Fail)
    return VkErrorDeviceLost;
  return Real(Dev, N, Fences, WaitAll, Timeout);
}

static void Callback(hipStream_t, hipError_t, void *) { Ran = true; }

#define CHECK(X)                                                               \
  if ((X) != hipSuccess) {                                                     \
    Fail = false;                                                              \
    printf("FAIL: %s\n", #X);                                                  \
    return 1;                                                                  \
  }

int main() {
  Fail = true;
  hipEvent_t E;
  CHECK(hipEventCreate(&E));
  CHECK(hipEventRecord(E, 0));
  CHECK(hipStreamAddCallback(0, Callback, nullptr, 0));

  float Ms;
  hipError_t ElapsedErr = hipEventElapsedTime(&Ms, E, E);
  hipError_t Err = hipEventSynchronize(E);

  if (WaitHits == 0) {
    Fail = false;
    printf("HIP_SKIP_THIS_TEST: not the Vulkan backend\n");
    return CHIP_SKIP_TEST;
  }
  if (Err != hipErrorTbd) {
    Fail = false;
    printf("FAIL: hipEventSynchronize returned %d\n", (int)Err);
    return 1;
  }
  if (ElapsedErr != hipErrorTbd) {
    Fail = false;
    printf("FAIL: hipEventElapsedTime returned %d\n", (int)ElapsedErr);
    return 1;
  }

  int H = Hits;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (Hits == H) {
    Fail = false;
    printf("FAIL: the monitor stopped polling after a hard error\n");
    return 1;
  }

  Fail = false;
  CHECK(hipDeviceSynchronize());
  if (!Ran) {
    printf("FAIL: the callback never ran\n");
    return 1;
  }
  CHECK(hipEventDestroy(E));
  printf("PASS\n");
  return 0;
}

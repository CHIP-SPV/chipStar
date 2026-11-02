// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2214: a hard
// vkGetFenceStatus error must not terminate the process from the event monitor
// thread, and must not be polled again once seen.
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <thread>

static std::atomic<bool> Fail{false};
static std::atomic<int> Hits{0};
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
  if (Fail)
    return VkErrorDeviceLost;
  return Real(Dev, N, Fences, WaitAll, Timeout);
}

__global__ void kernel(int *P) { *P = 1; }

int main() {
  int *P;
  (void)hipMalloc(&P, sizeof(int));
  Fail = true;
  kernel<<<1, 1>>>(P);
  hipEvent_t E;
  (void)hipEventCreate(&E);
  (void)hipEventRecord(E, 0);

  for (int I = 0; I < 1000 && Hits == 0; ++I)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  if (Hits == 0) {
    Fail = false;
    printf("HIP_SKIP_THIS_TEST: vkGetFenceStatus was never called\n");
    return CHIP_SKIP_TEST;
  }

  // A monitor that keeps polling would hit this thousands of times.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  if (Hits != 1) {
    printf("FAIL: vkGetFenceStatus polled %d times after a hard error\n",
           Hits.load());
    return 1;
  }
  if (hipEventSynchronize(E) == hipSuccess) {
    printf("FAIL: hipEventSynchronize did not report the fence error\n");
    return 1;
  }

  Fail = false;
  if (hipDeviceSynchronize() != hipSuccess) {
    printf("FAIL: hipDeviceSynchronize failed after the error cleared\n");
    return 1;
  }
  (void)hipEventDestroy(E);
  printf("PASS\n");
  return 0;
}

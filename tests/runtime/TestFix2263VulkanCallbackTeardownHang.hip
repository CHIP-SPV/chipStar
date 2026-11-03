// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2263: teardown must run pending stream callbacks.
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

static std::atomic<bool> CallbackRan{false}, Hold{false};
static std::atomic<long long> ReleaseNs{LLONG_MAX};

static long long nowNs() {
  return std::chrono::steady_clock::now().time_since_epoch().count();
}

// Keeps the callback's marker unready until ReleaseNs, past the monitor's Stop.
extern "C" int vkGetFenceStatus(void *Dev, void *F) {
  static auto Real = (int (*)(void *, void *))dlsym(RTLD_NEXT, "vkGetFenceStatus");
  if (Hold && nowNs() < ReleaseNs)
    return 1; // VK_NOT_READY
  return Real(Dev, F);
}

// Never waits for real: the unsignaled callback value would hang it forever.
extern "C" int vkDeviceWaitIdle(void *Dev) {
  static auto Real = (int (*)(void *))dlsym(RTLD_NEXT, "vkDeviceWaitIdle");
  if (!CallbackRan) {
    printf("FAIL: vkDeviceWaitIdle reached with the callback never run\n");
    fflush(stdout);
    _exit(1);
  }
  return Real(Dev);
}

__global__ void kernel() {}

static void CB(hipStream_t, hipError_t, void *) { CallbackRan = true; }

int main() {
  alarm(60);
  (void)hipFree(nullptr);
  hipStream_t S;
  (void)hipStreamCreate(&S);
  kernel<<<1, 1, 0, S>>>();
  Hold = true;
  (void)hipStreamAddCallback(S, CB, nullptr, 0);
  kernel<<<1, 1, 0, S>>>(); // Waits on the callback's timeline value.
  ReleaseNs = nowNs() + 1000000000LL;
  uintptr_t Handles[4] = {};
  (void)hipInitFromNativeHandles(Handles, 4); // Tears the backend down.
  printf(CallbackRan ? "PASS\n" : "FAIL: callback never ran\n");
  fflush(stdout);
  _exit(CallbackRan ? 0 : 1); // The backend is gone; skip atexit.
}

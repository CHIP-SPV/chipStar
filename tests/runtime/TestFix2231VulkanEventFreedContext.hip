// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2231: a tracked
// event still recording when the monitor stops must hand its fence back before
// the device is destroyed. Fails if the submitted fence is not destroyed then.
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>

#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

static void *SubmitFence;
static bool DeviceDestroyed, FenceDestroyedEarly;

extern "C" int vkQueueSubmit(void *Q, unsigned N, const void *Info, void *F) {
  static auto Real = (int (*)(void *, unsigned, const void *, void *))dlsym(
      RTLD_NEXT, "vkQueueSubmit");
  int R = Real(Q, N, Info, F);
  if (R == 0 && F)
    SubmitFence = F;
  return R;
}

// Keeps the event tracked and RECORDING until the backend stops.
extern "C" int vkGetFenceStatus(void *, void *) { return 1; }

extern "C" void vkDestroyFence(void *Dev, void *F, const void *A) {
  static auto Real = (void (*)(void *, void *, const void *))dlsym(
      RTLD_NEXT, "vkDestroyFence");
  if (F == SubmitFence && !DeviceDestroyed)
    FenceDestroyedEarly = true;
  Real(Dev, F, A);
}

extern "C" void vkDestroyDevice(void *Dev, const void *A) {
  static auto Real =
      (void (*)(void *, const void *))dlsym(RTLD_NEXT, "vkDestroyDevice");
  DeviceDestroyed = true;
  Real(Dev, A);
}

int main() {
  (void)hipFree(nullptr);
  SubmitFence = nullptr;
  void *S = nullptr, *D = nullptr;
  if (hipMalloc(&S, 4) != hipSuccess || hipMalloc(&D, 4) != hipSuccess ||
      hipMemcpyAsync(D, S, 4, hipMemcpyDeviceToDevice) != hipSuccess) {
    printf("FAIL: hipMalloc/hipMemcpyAsync failed\n");
    return 1;
  }
  if (!SubmitFence) {
    printf("HIP_SKIP_THIS_TEST: no Vulkan fence was submitted\n");
    return CHIP_SKIP_TEST;
  }
  uintptr_t Handles[4] = {};
  (void)hipInitFromNativeHandles(Handles, 4); // Tears the backend down.
  printf(FenceDestroyedEarly ? "PASS\n" : "FAIL: event fence not released\n");
  fflush(stdout);
  _exit(FenceDestroyedEarly ? 0 : 1); // The backend is gone; skip atexit.
}

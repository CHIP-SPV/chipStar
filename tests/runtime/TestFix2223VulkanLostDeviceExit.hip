// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2223: a lost
// Vulkan device must not abort the process from the library destructor.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <dlfcn.h>

static std::atomic<bool> Fail{false};
static std::atomic<int> Hits{0};

extern "C" int vkWaitSemaphores(void *Dev, const void *Info, unsigned long T) {
  static auto Real = (int (*)(void *, const void *, unsigned long))dlsym(
      RTLD_NEXT, "vkWaitSemaphores");
  Hits++;
  return Fail ? -4 : Real(Dev, Info, T); // VK_ERROR_DEVICE_LOST
}

extern "C" int vkWaitForFences(void *Dev, unsigned N, const void *F,
                               unsigned All, unsigned long T) {
  static auto Real = (int (*)(void *, unsigned, const void *, unsigned,
                              unsigned long))dlsym(RTLD_NEXT, "vkWaitForFences");
  return Fail ? -4 : Real(Dev, N, F, All, T);
}

int main() {
  (void)hipStreamSynchronize(0);
  if (!Hits) {
    printf("HIP_SKIP_THIS_TEST: not the Vulkan backend\n");
    return CHIP_SKIP_TEST;
  }
  Fail = true;
  printf("PASS\n");
  return 0;
}

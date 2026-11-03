// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2223: a lost
// Vulkan device must not abort the process from the library destructor, with
// an undestroyed stream and HIP threads still alive at exit.
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <thread>

static std::atomic<bool> Fail{false};
static std::atomic<int> Hits{0};

extern "C" int vkWaitSemaphores(void *Dev, const void *Info, unsigned long T) {
  static auto Real = (int (*)(void *, const void *, unsigned long))dlsym(
      RTLD_NEXT, "vkWaitSemaphores");
  Hits++;
  return Fail ? -4 : Real(Dev, Info, T); // VK_ERROR_DEVICE_LOST
}

int main() {
  (void)hipStreamSynchronize(0);
  if (!Hits) {
    printf("HIP_SKIP_THIS_TEST: no vkWaitSemaphores call was intercepted\n");
    return CHIP_SKIP_TEST;
  }
  hipStream_t S;
  (void)hipStreamCreate(&S); // left undestroyed
  std::atomic<int> Up{0};
  for (int I = 0; I < 2; I++)
    std::thread([&] {
      int N;
      (void)hipGetDeviceCount(&N);
      Up++;
      std::this_thread::sleep_for(std::chrono::hours(1));
    }).detach();
  while (Up < 2)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  Fail = true;
  return 0;
}

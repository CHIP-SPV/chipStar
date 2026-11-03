// https://github.com/CHIP-SPV/chipStar/issues/2242: a lost device's staging
// buffers must be destroyed before the device.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

static std::atomic<int> Created{0}, Destroyed{0};
static std::atomic<bool> Lost{false}, Exiting{false};

extern "C" int vkCreateBuffer(void *Dev, const void *Info, const void *Alloc,
                              void **Out) {
  static auto Real = (int (*)(void *, const void *, const void *, void **))
      dlsym(RTLD_NEXT, "vkCreateBuffer");
  int R = Real(Dev, Info, Alloc, Out);
  if (R == 0)
    Created++;
  return R;
}

extern "C" void vkDestroyBuffer(void *Dev, void *Buf, const void *Alloc) {
  static auto Real = (void (*)(void *, void *, const void *))dlsym(
      RTLD_NEXT, "vkDestroyBuffer");
  if (Buf)
    Destroyed++;
  Real(Dev, Buf, Alloc);
}

// A lost device may report any fence status; 1 is VK_NOT_READY.
extern "C" int vkGetFenceStatus(void *Dev, void *Fence) {
  static auto Real =
      (int (*)(void *, void *))dlsym(RTLD_NEXT, "vkGetFenceStatus");
  return Lost ? 1 : Real(Dev, Fence);
}

// The device destructor calls this after the event monitor has stopped.
extern "C" int vkDeviceWaitIdle(void *Dev) {
  static auto Real = (int (*)(void *))dlsym(RTLD_NEXT, "vkDeviceWaitIdle");
  Lost = Exiting.load();
  return Real(Dev);
}

// The verdict: every buffer the runtime frees at exit is gone by now.
extern "C" void vkDestroyDevice(void *Dev, const void *Alloc) {
  static auto Real = (void (*)(void *, const void *))dlsym(
      RTLD_NEXT, "vkDestroyDevice");
  if (Lost) {
    bool Ok = Created == Destroyed;
    if (Ok)
      printf("PASS\n");
    else
      printf("FAIL: created %d buffers, destroyed %d\n", Created.load(),
             Destroyed.load());
    fflush(stdout);
    _exit(Ok ? 0 : 1);
  }
  Real(Dev, Alloc);
}

int main() {
  static char Src[1024];
  void *Dst = nullptr;
  if (hipMalloc(&Dst, sizeof(Src)) != hipSuccess)
    return 1;
  int C0 = Created;
  if (hipMemcpyAsync(Dst, Src, sizeof(Src), hipMemcpyHostToDevice, nullptr) !=
          hipSuccess ||
      hipDeviceSynchronize() != hipSuccess)
    return 1;
  if (Created == C0) {
    printf("HIP_SKIP_THIS_TEST: no Vulkan staging buffer was created\n");
    return CHIP_SKIP_TEST;
  }
  Exiting = true;
  // vkDestroyDevice at exit decides; reaching this status means it never ran.
  return 1;
}

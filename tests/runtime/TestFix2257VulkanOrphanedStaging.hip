// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2257: a copy or
// fill that fails after allocating its staging buffer must not orphan it.
// Injects Vulkan failures and checks vkCreateBuffer/vkDestroyBuffer balance.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

static std::atomic<int> Created{0}, Destroyed{0};
static std::atomic<bool> Exiting{false};
static std::atomic<bool> Failed{false}, LostSubmitPending{false};
// Main-thread only, so the event monitor thread never sees an injected failure.
static thread_local int FailWait = 0, FailSubmit = 0;

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
  if (LostSubmitPending.exchange(false)) {
    printf("FAIL: buffer destroyed before waiting on a lost submit\n");
    Failed = true;
  }
  Real(Dev, Buf, Alloc);
}

extern "C" int vkWaitForFences(void *Dev, unsigned N, const void *F, unsigned W,
                               unsigned long long T) {
  static auto Real = (int (*)(void *, unsigned, const void *, unsigned,
                              unsigned long long))dlsym(RTLD_NEXT,
                                                        "vkWaitForFences");
  int R = Real(Dev, N, F, W, T);
  return R == 0 && FailWait ? FailWait : R;
}

// Does the real submit but reports a lost device, which leaves it pending.
extern "C" int vkQueueSubmit(void *Q, unsigned N, const void *S, void *F) {
  static auto Real =
      (int (*)(void *, unsigned, const void *, void *))dlsym(RTLD_NEXT,
                                                              "vkQueueSubmit");
  int R = Real(Q, N, S, F);
  if (R == 0 && FailSubmit) {
    LostSubmitPending = true;
    return FailSubmit;
  }
  return R;
}

extern "C" int vkQueueWaitIdle(void *Q) {
  static auto Real = (int (*)(void *))dlsym(RTLD_NEXT, "vkQueueWaitIdle");
  LostSubmitPending = false;
  return Real(Q);
}

extern "C" void vkDestroyDevice(void *Dev, const void *Alloc) {
  static auto Real = (void (*)(void *, const void *))dlsym(
      RTLD_NEXT, "vkDestroyDevice");
  if (Exiting) {
    bool Ok = Created == Destroyed && !Failed;
    if (Ok)
      printf("PASS\n");
    else if (Created != Destroyed)
      printf("FAIL: created %d buffers, destroyed %d\n", Created.load(),
             Destroyed.load());
    fflush(stdout);
    _exit(Ok ? 0 : 1);
  }
  Real(Dev, Alloc);
}

// Runs Call with a failure injected; expects an error and Expect extra live buffers.
template <class F>
static void check(const char *Name, int Wait, int Submit, int Expect, F Call) {
  int Before = Created - Destroyed;
  FailWait = Wait;
  FailSubmit = Submit;
  hipError_t E = Call();
  FailWait = FailSubmit = 0;
  (void)hipGetLastError();
  int Delta = Created - Destroyed - Before;
  if (E == hipSuccess || Delta != Expect) {
    printf("FAIL %s: call returned %d, live buffers changed by %d, expected "
           "%d\n", Name, (int)E, Delta, Expect);
    Failed = true;
  }
}

int main() {
  static char Host[1024];
  char *Dev;
  if (hipMalloc(&Dev, sizeof(Host)) != hipSuccess)
    return 1;
  int C0 = Created;
  if (hipMemcpy(Host, Dev, sizeof(Host), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  if (Created == C0) {
    printf("HIP_SKIP_THIS_TEST: no Vulkan staging buffer was created\n");
    return CHIP_SKIP_TEST;
  }
  // -4 is VK_ERROR_DEVICE_LOST, -2 is VK_ERROR_OUT_OF_DEVICE_MEMORY.
  check("D2H wait lost", -4, 0, 0, [&] {
    return hipMemcpy(Host, Dev, sizeof(Host), hipMemcpyDeviceToHost); });
  check("H2D submit lost", 0, -4, 0, [&] {
    return hipMemcpy(Dev, Host, sizeof(Host), hipMemcpyHostToDevice); });
  check("Memset submit lost", 0, -4, 0, [&] {
    return hipMemset(Dev, 0x5a, 1000); });
  check("Memset2D submit lost", 0, -4, 0, [&] {
    return hipMemset2D(Dev, 16, 0x5a, 7, 4); });
  check("Memset3D submit lost", 0, -4, 0, [&] {
    return hipMemset3D(make_hipPitchedPtr(Dev, 16, 7, 4), 0x5a,
                       make_hipExtent(7, 4, 2)); });
  // The device may still use the buffer, so it must stay alive until drained.
  check("D2H wait OOM", -2, 0, 1, [&] {
    return hipMemcpy(Host, Dev, sizeof(Host), hipMemcpyDeviceToHost); });
  (void)hipMemcpy(Host, Dev, sizeof(Host), hipMemcpyDeviceToHost);
  Exiting = true;
  // vkDestroyDevice at exit decides; reaching this status means it never ran.
  return 1;
}

// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2334: a failed
// vkQueueSubmit in hipEventRecord must not leak or strand its timestamp slot.
// Never waits on the queue after a failed submit (see #2273).
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

static std::atomic<int> Submits{0};
static thread_local int FailSubmit = 0;
static std::atomic<int> LastSlot{-1}, UnwrittenSlot{-1};
static std::atomic<bool> ReadUnwritten{false};

extern "C" int vkQueueSubmit(void *Q, unsigned N, const void *S, void *F) {
  static auto Real = (int (*)(void *, unsigned, const void *, void *))dlsym(
      RTLD_NEXT, "vkQueueSubmit");
  Submits++;
  if (FailSubmit) {
    FailSubmit = 0;
    return -2; // VK_ERROR_OUT_OF_DEVICE_MEMORY
  }
  return Real(Q, N, S, F);
}

extern "C" void vkResetQueryPool(void *Dev, void *Pool, uint32_t First,
                                 uint32_t Count) {
  static auto Real = (void (*)(void *, void *, uint32_t, uint32_t))dlsym(
      RTLD_NEXT, "vkResetQueryPool");
  if (Count == 1) // Device init resets the whole pool at once.
    LastSlot = First;
  Real(Dev, Pool, First, Count);
}

extern "C" int vkGetQueryPoolResults(void *Dev, void *Pool, uint32_t First,
                                     uint32_t Count, size_t Size, void *Data,
                                     uint64_t Stride, uint32_t Flags) {
  static auto Real =
      (int (*)(void *, void *, uint32_t, uint32_t, size_t, void *, uint64_t,
               uint32_t))dlsym(RTLD_NEXT, "vkGetQueryPoolResults");
  if ((int)First == UnwrittenSlot) {
    ReadUnwritten = true;
    return 5; // VK_NOT_READY instead of waiting forever
  }
  return Real(Dev, Pool, First, Count, Size, Data, Stride, Flags);
}

int main() {
  hipEvent_t Ev, Ev2;
  if (hipEventCreate(&Ev) != hipSuccess || hipEventCreate(&Ev2) != hipSuccess ||
      hipEventRecord(Ev, nullptr) != hipSuccess ||
      hipEventSynchronize(Ev) != hipSuccess) {
    printf("FAIL: setup\n");
    return 1;
  }
  if (Submits == 0 || LastSlot < 0) {
    printf("HIP_SKIP_THIS_TEST: no Vulkan timestamp slot was used\n");
    return CHIP_SKIP_TEST;
  }
  int Failures = 0;

  // Re-record fails: the event must not point at a slot nothing will write.
  FailSubmit = 1;
  bool Failed = hipEventRecord(Ev, nullptr) != hipSuccess;
  bool Reached = !FailSubmit;
  (void)hipGetLastError();
  if (!Failed || !Reached) {
    printf("FAIL: injected re-record did not fail at vkQueueSubmit\n");
    fflush(stdout);
    _exit(1);
  }
  UnwrittenSlot = LastSlot.load();
  float Ms;
  if (hipEventElapsedTime(&Ms, Ev, Ev) != hipSuccess) {
    printf("FAIL: hipEventElapsedTime after failed re-record\n");
    Failures++;
  }
  if (ReadUnwritten) {
    printf("FAIL: elapsed time read a slot no submit will write\n");
    Failures++;
  }
  UnwrittenSlot = -1;

  // First record fails and is retried: the slot must be reused.
  FailSubmit = 1;
  Failed = hipEventRecord(Ev2, nullptr) != hipSuccess;
  Reached = !FailSubmit;
  int First = LastSlot;
  (void)hipGetLastError();
  if (!Failed || !Reached) {
    printf("FAIL: injected record did not fail at vkQueueSubmit\n");
    fflush(stdout);
    _exit(1);
  }
  if (hipEventRecord(Ev2, nullptr) != hipSuccess) {
    printf("FAIL: retried record\n");
    Failures++;
  } else if (LastSlot != First) {
    printf("FAIL: failed record leaked timestamp slot\n");
    Failures++;
  }

  if (!Failures)
    printf("PASS\n");
  fflush(stdout);
  _exit(Failures ? 1 : 0); // Exit and destroy would wait on the queue.
}

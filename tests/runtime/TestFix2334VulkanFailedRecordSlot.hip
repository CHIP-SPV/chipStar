// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2334: a failed
// vkQueueSubmit or vkEndCommandBuffer in hipEventRecord must not leak or
// strand its timestamp slot, and a lost submit must keep its slot.
// Never waits on the queue after a failed submit (see #2273).
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

static std::atomic<int> Submits{0};
static thread_local int FailSubmit = 0, FailEnd = 0; // VkResult or flag
static std::atomic<int> LastSlot{-1}, UnwrittenSlot{-1};
static std::atomic<bool> ReadUnwritten{false};

extern "C" int vkQueueSubmit(void *Q, unsigned N, const void *S, void *F) {
  static auto Real = (int (*)(void *, unsigned, const void *, void *))dlsym(
      RTLD_NEXT, "vkQueueSubmit");
  Submits++;
  if (FailSubmit) {
    int R = FailSubmit;
    FailSubmit = 0;
    return R;
  }
  return Real(Q, N, S, F);
}

extern "C" int vkEndCommandBuffer(void *Cb) {
  static auto Real =
      (int (*)(void *))dlsym(RTLD_NEXT, "vkEndCommandBuffer");
  int R = Real(Cb);
  if (FailEnd) {
    FailEnd = 0;
    return -2; // VK_ERROR_OUT_OF_DEVICE_MEMORY
  }
  return R;
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

// Records E with *Flag armed; true when it failed at the injection point.
static bool recordFails(hipEvent_t E, int *Flag, int Val) {
  *Flag = Val;
  bool Failed = hipEventRecord(E, nullptr) != hipSuccess;
  bool Reached = !*Flag;
  *Flag = 0;
  (void)hipGetLastError();
  return Failed && Reached;
}

int main() {
  hipEvent_t Ev, Ev2, Ev3, Ev4;
  if (hipEventCreate(&Ev) != hipSuccess || hipEventCreate(&Ev2) != hipSuccess ||
      hipEventCreate(&Ev3) != hipSuccess || hipEventCreate(&Ev4) != hipSuccess ||
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
  if (!recordFails(Ev, &FailSubmit, -2)) {
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
  bool Ok = recordFails(Ev2, &FailSubmit, -2);
  int First = LastSlot;
  if (!Ok) {
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

  // A lost submit still counts as pending: its slot must not be handed out.
  Ok = recordFails(Ev3, &FailSubmit, -4); // VK_ERROR_DEVICE_LOST
  int Lost = LastSlot;
  if (!Ok) {
    printf("FAIL: injected device loss did not fail at vkQueueSubmit\n");
    fflush(stdout);
    _exit(1);
  }
  if (hipEventRecord(Ev3, nullptr) != hipSuccess) {
    printf("FAIL: retried record after device loss\n");
    Failures++;
  } else if (LastSlot == Lost) {
    printf("FAIL: slot reused while its lost submit counts as pending\n");
    Failures++;
  }

  // A failed vkEndCommandBuffer must give the slot back.
  Ok = recordFails(Ev4, &FailEnd, 1);
  int EndSlot = LastSlot;
  if (!Ok) {
    printf("FAIL: injected record did not fail at vkEndCommandBuffer\n");
    fflush(stdout);
    _exit(1);
  }
  if (hipEventRecord(Ev4, nullptr) != hipSuccess) {
    printf("FAIL: retried record after vkEndCommandBuffer\n");
    Failures++;
  } else if (LastSlot != EndSlot) {
    printf("FAIL: failed vkEndCommandBuffer leaked timestamp slot\n");
    Failures++;
  }

  if (!Failures)
    printf("PASS\n");
  fflush(stdout);
  _exit(Failures ? 1 : 0); // Exit and destroy would wait on the queue.
}

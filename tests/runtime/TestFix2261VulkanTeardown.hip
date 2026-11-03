// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2261: teardown
// must not destroy what a HIP thread still in flight uses. Modes: reinit
// (hipInitFromNativeHandles) and exit (return from main).
#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <pthread.h>
#include <thread>
#include <unistd.h>

static std::atomic<bool> Armed{false}, Blocked{false}, Resume{false},
    WorkerDone{false};
static thread_local bool IsWorker, StallNext;
static bool ExitMode;

static void fail(const char *What) {
  printf("FAIL: %s while a HIP thread was in flight\n", What);
  fflush(stdout);
  _exit(1);
}

extern "C" int vkQueueSubmit(void *Q, unsigned N, const void *I, void *F) {
  static auto Real = (int (*)(void *, unsigned, const void *, void *))dlsym(
      RTLD_NEXT, "vkQueueSubmit");
  int R = Real(Q, N, I, F);
  if (IsWorker && Armed.exchange(false))
    StallNext = true;
  return R;
}

// Stalls the worker, bounded, at its first lock after the submit.
extern "C" int pthread_mutex_lock(pthread_mutex_t *M) {
  static auto Real =
      (int (*)(pthread_mutex_t *))dlsym(RTLD_NEXT, "pthread_mutex_lock");
  if (StallNext) {
    StallNext = false;
    Blocked = true;
    for (int I = 0; I < 30000 && !Resume; I++)
      usleep(1000);
  }
  return Real(M);
}

extern "C" void CtxD0(void *T) __asm__("_ZN17CHIPContextVulkanD0Ev");
extern "C" void CtxD0(void *T) {
  static auto Real =
      (void (*)(void *))dlsym(RTLD_NEXT, "_ZN17CHIPContextVulkanD0Ev");
  if (Blocked && !Resume)
    fail("context deleted");
  Real(T);
}

extern "C" void vkDestroyDevice(void *Dev, const void *A) {
  static auto Real =
      (void (*)(void *, const void *))dlsym(RTLD_NEXT, "vkDestroyDevice");
  if (Blocked && !Resume)
    fail("device destroyed");
  Real(Dev, A);
}

__global__ void kernel() {}

int main(int argc, char **argv) {
  ExitMode = argc > 1 && !strcmp(argv[1], "exit");
  static void *DA, *DB;
  kernel<<<1, 1>>>();
  (void)hipDeviceSynchronize();
  (void)hipMalloc(&DA, 16);
  (void)hipMalloc(&DB, 16);
  std::thread([] {
    IsWorker = true;
    Armed = true;
    (void)hipMemcpyAsync(DA, DB, 16, hipMemcpyDeviceToDevice, hipStreamPerThread);
    WorkerDone = true;
    sleep(60);
  }).detach();
  for (int I = 0; I < 60000 && !Blocked; I++)
    usleep(1000);
  if (!Blocked) {
    printf("HIP_SKIP_THIS_TEST: worker never stalled after vkQueueSubmit\n");
    return CHIP_SKIP_TEST;
  }
  if (!ExitMode) {
    uintptr_t Handles[4] = {};
    (void)hipInitFromNativeHandles(Handles, 4);
    Resume = true; // The worker's late event tracking and drop run now.
    for (int I = 0; I < 20000 && !WorkerDone; I++)
      usleep(1000);
  }
  printf("PASS\n");
  fflush(stdout);
  if (!ExitMode)
    _exit(0);
  return 0; // Teardown runs on the normal exit path.
}

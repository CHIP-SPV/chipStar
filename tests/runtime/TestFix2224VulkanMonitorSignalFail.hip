// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2224: a failed
// vkSignalSemaphore on the Vulkan monitor thread must be reported to the
// stream's waiters, not terminate the process or leave them waiting forever.
// Another stream waiting on the failed value must report it too. The normal
// exit path is checked except on NVIDIA, which leaves with _exit.
#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>

static std::atomic<bool> Fail{false};
static std::atomic<int> Hits{0};
static std::atomic<bool> Release{false};

extern "C" int vkSignalSemaphore(void *Dev, const void *Info) {
  if (Fail) {
    Hits++;
    return -1; // VK_ERROR_OUT_OF_HOST_MEMORY
  }
  static auto Real = (int (*)(void *, const void *))dlsym(RTLD_NEXT,
                                                          "vkSignalSemaphore");
  return Real(Dev, Info);
}

static void cb(hipStream_t, hipError_t, void *) {
  while (!Release)
    usleep(100);
}

#define CHECK(Cond)                                                            \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      printf("FAIL: %s (line %d)\n", #Cond, __LINE__);                         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static bool isVulkan() {
  int N = 0;
  if (hipGetBackendNativeHandles(0, nullptr, &N) != hipSuccess || N < 1)
    return false;
  uintptr_t H[8] = {};
  if (N > 8 || hipGetBackendNativeHandles(0, H, nullptr) != hipSuccess)
    return false;
  return H[0] && !strcmp(reinterpret_cast<const char *>(H[0]), "vulkan");
}

int main() {
  alarm(60);
  if (!isVulkan()) {
    printf("HIP_SKIP_THIS_TEST: not the Vulkan backend\n");
    return CHIP_SKIP_TEST;
  }
  // Makes its thread_local queue exist so it is destroyed at exit.
  CHECK(hipStreamSynchronize(hipStreamPerThread) == hipSuccess);
  void *Buf;
  hipStream_t A, B;
  hipEvent_t E, E2;
  CHECK(hipMalloc(&Buf, 64) == hipSuccess);
  CHECK(hipStreamCreate(&A) == hipSuccess);
  CHECK(hipStreamCreate(&B) == hipSuccess);
  CHECK(hipEventCreate(&E) == hipSuccess);
  CHECK(hipEventCreate(&E2) == hipSuccess);
  char Name[256] = {};
  CHECK(hipDeviceGetName(Name, sizeof(Name), 0) == hipSuccess);
  Fail = true;
  CHECK(hipStreamAddCallback(A, cb, nullptr, 0) == hipSuccess);
  CHECK(hipStreamAddCallback(A, cb, nullptr, 0) == hipSuccess);
  // The record on A submits behind the first callback's timeline value; the
  // default stream's marker waits on A because A is a blocking stream.
  CHECK(hipEventRecord(E, A) == hipSuccess);
  CHECK(hipEventRecord(E2, 0) == hipSuccess);
  Release = true;
  CHECK(hipStreamSynchronize(0) == hipErrorTbd);
  CHECK(hipStreamSynchronize(A) == hipErrorTbd);
  CHECK(Hits > 0);
  CHECK(hipEventSynchronize(E) == hipErrorTbd);
  CHECK(hipEventQuery(E) == hipErrorTbd);
  CHECK(hipStreamQuery(A) == hipErrorTbd);
  CHECK(hipMemset(Buf, 0, 64) == hipErrorTbd);
  char Host[64];
  CHECK(hipMemcpy(Host, Buf, 64, hipMemcpyDeviceToHost) == hipErrorTbd);
  // Records fresh events until the command buffer ring wraps onto a stuck slot.
  hipEvent_t E3 = nullptr;
  hipError_t R = hipSuccess;
  for (int I = 0; I < 64 && R == hipSuccess; ++I) {
    CHECK(hipEventCreate(&E3) == hipSuccess);
    R = hipEventRecord(E3, A);
  }
  CHECK(R == hipErrorTbd);
  CHECK(hipStreamWaitEvent(A, E3, 0) == hipErrorTbd);
  CHECK(hipStreamWaitEvent(B, E, 0) == hipErrorTbd);
  CHECK(hipEventDestroy(E) == hipSuccess);
  CHECK(hipEventDestroy(E2) == hipSuccess);
  printf("PASS\n");
  // NVIDIA's driver blocks in its own exit teardown on a stuck submit.
  if (strstr(Name, "NVIDIA")) {
    fflush(stdout);
    _exit(0);
  }
  return 0;
}

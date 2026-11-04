// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2335: when the
// marker submit inside hipEventRecord fails, hipStreamWaitEvent on the event
// must not abort. Fails the second vkQueueSubmit through an interposer.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>

static std::atomic<int> Submits{0};
static thread_local int FailIn = -1;

extern "C" int vkQueueSubmit(void *Q, unsigned N, const void *S, void *F) {
  static auto Real =
      (int (*)(void *, unsigned, const void *, void *))dlsym(RTLD_NEXT,
                                                             "vkQueueSubmit");
  Submits++;
  if (FailIn >= 0 && FailIn-- == 0)
    return -2; // VK_ERROR_OUT_OF_DEVICE_MEMORY
  return Real(Q, N, S, F);
}

#define EXPECT(C, What)                                                        \
  if (!(C)) {                                                                  \
    printf("FAIL: %s\n", What);                                                \
    fflush(stdout);                                                            \
    _exit(1);                                                                  \
  }

int main() {
  char In[64];
  memset(In, 0x5a, sizeof(In));
  void *D;
  EXPECT(hipMalloc(&D, sizeof(In)) == hipSuccess, "hipMalloc");
  EXPECT(hipMemcpy(D, In, sizeof(In), hipMemcpyHostToDevice) == hipSuccess,
         "warmup copy");
  if (Submits == 0) {
    printf("HIP_SKIP_THIS_TEST: no vkQueueSubmit was seen\n");
    return CHIP_SKIP_TEST;
  }

  // Pending work for the failed record's event to complete behind.
  const size_t Big = 64u << 20;
  void *B;
  EXPECT(hipMalloc(&B, Big) == hipSuccess, "hipMalloc big");
  EXPECT(hipMemsetAsync(B, 1, Big, nullptr) == hipSuccess, "hipMemsetAsync");

  hipEvent_t Ev;
  EXPECT(hipEventCreate(&Ev) == hipSuccess, "hipEventCreate");
  FailIn = 1;
  EXPECT(hipEventRecord(Ev, nullptr) != hipSuccess, "injected record succeeded");
  EXPECT(FailIn == -1, "record did not reach its second vkQueueSubmit");
  EXPECT(hipStreamWaitEvent(nullptr, Ev, 0) == hipSuccess,
         "hipStreamWaitEvent after failed record");
  EXPECT(hipEventQuery(Ev) == hipSuccess,
         "event not complete after hipStreamWaitEvent");
  printf("PASS\n");
  // Teardown hangs on the timeline a failed submit left
  // behind, https://github.com/CHIP-SPV/chipStar/issues/2273.
  fflush(stdout);
  _exit(0);
}

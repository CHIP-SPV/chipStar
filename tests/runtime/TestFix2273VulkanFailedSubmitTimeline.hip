// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2273: after a
// vkQueueSubmit fails with an out of memory result, the next call on the queue
// must not hang. Fails the submit through an interposer; SIGALRM catches a hang.
#include <hip/hip_runtime.h>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>

static std::atomic<int> Submits{0};
static thread_local int FailSubmit = 0;

extern "C" int vkQueueSubmit(void *Q, unsigned N, const void *S, void *F) {
  static auto Real =
      (int (*)(void *, unsigned, const void *, void *))dlsym(RTLD_NEXT,
                                                             "vkQueueSubmit");
  Submits++;
  if (FailSubmit) {
    FailSubmit = 0;
    return -2; // VK_ERROR_OUT_OF_DEVICE_MEMORY
  }
  return Real(Q, N, S, F);
}

static void onAlarm(int) {
  const char Msg[] = "FAIL: hung after a failed submit\n";
  (void)!write(1, Msg, sizeof(Msg) - 1);
  _exit(1);
}

#define EXPECT(C, What)                                                        \
  if (!(C)) {                                                                  \
    printf("FAIL: %s\n", What);                                                \
    return 1;                                                                  \
  }

int main() {
  char In[64], Out[64] = {};
  memset(In, 0x5a, sizeof(In));
  void *D;
  EXPECT(hipMalloc(&D, sizeof(In)) == hipSuccess, "hipMalloc");
  EXPECT(hipMemcpy(D, In, sizeof(In), hipMemcpyHostToDevice) == hipSuccess,
         "warmup copy");
  if (Submits == 0) {
    printf("HIP_SKIP_THIS_TEST: no vkQueueSubmit was seen\n");
    return CHIP_SKIP_TEST;
  }
  signal(SIGALRM, onAlarm);
  alarm(60);

  FailSubmit = 1;
  EXPECT(hipMemcpy(D, In, sizeof(In), hipMemcpyHostToDevice) != hipSuccess,
         "injected copy succeeded");
  EXPECT(!FailSubmit, "copy did not reach vkQueueSubmit");
  (void)hipGetLastError();
  memset(In, 0xa5, sizeof(In));
  EXPECT(hipMemcpy(D, In, sizeof(In), hipMemcpyHostToDevice) == hipSuccess,
         "copy after failed copy");
  EXPECT(hipMemcpy(Out, D, sizeof(In), hipMemcpyDeviceToHost) == hipSuccess,
         "readback");
  EXPECT(!memcmp(In, Out, sizeof(In)), "data mismatch");
  EXPECT(hipDeviceSynchronize() == hipSuccess, "sync after failed copy");

  hipEvent_t Ev;
  EXPECT(hipEventCreate(&Ev) == hipSuccess, "hipEventCreate");
  FailSubmit = 1;
  EXPECT(hipEventRecord(Ev, nullptr) != hipSuccess, "injected record succeeded");
  EXPECT(!FailSubmit, "record did not reach vkQueueSubmit");
  (void)hipGetLastError();
  EXPECT(hipMemcpy(D, In, sizeof(In), hipMemcpyHostToDevice) == hipSuccess,
         "copy after failed record");
  EXPECT(hipDeviceSynchronize() == hipSuccess, "sync after failed record");
  (void)hipEventDestroy(Ev);
  printf("PASS\n");
  return 0;
}

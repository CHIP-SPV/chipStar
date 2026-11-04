// https://github.com/CHIP-SPV/chipStar/issues/2344: a printf record prints once even if the header clear fails.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <unistd.h>

static std::atomic<bool> Armed{false}, Injected{false};
static std::atomic<int> Begins{0}, Counting{0};

extern "C" int vkQueueWaitIdle(void *Q) {
  static auto Real = (int (*)(void *))dlsym(RTLD_NEXT, "vkQueueWaitIdle");
  int R = Real(Q);
  if (Armed.exchange(false))
    Counting = 1;
  return R;
}

// Device-type printf buffers have no host mapping: header read, record read, clear are begins 1, 2, 3.
extern "C" int vkBeginCommandBuffer(void *CB, const void *Info) {
  static auto Real =
      (int (*)(void *, const void *))dlsym(RTLD_NEXT, "vkBeginCommandBuffer");
  Begins++;
  if (Counting && Counting++ == 3) {
    Injected = true;
    return -2; // VK_ERROR_OUT_OF_DEVICE_MEMORY
  }
  return Real(CB, Info);
}

__global__ void k() { printf("kernel-printf-record\n"); }

int main() {
  hipStream_t S;
  (void)hipStreamCreate(&S);
  FILE *Tmp = tmpfile();
  fflush(stdout);
  int Saved = dup(1);
  dup2(fileno(Tmp), 1);
  k<<<1, 1>>>();
  Armed = true;
  hipError_t E1 = hipStreamSynchronize(nullptr);
  (void)hipGetLastError();
  hipError_t E2 = hipStreamSynchronize(S);
  fflush(stdout);
  dup2(Saved, 1);
  close(Saved);

  if (Begins == 0) {
    printf("HIP_SKIP_THIS_TEST: the Vulkan interposers were not called\n");
    return CHIP_SKIP_TEST;
  }
  if (!Injected || E1 == hipSuccess || E2 != hipSuccess) {
    printf("FAIL: injected=%d, syncs returned %d, %d\n", (int)Injected,
           (int)E1, (int)E2);
    return 1;
  }
  std::string Out;
  char Buf[256];
  rewind(Tmp);
  for (size_t N; (N = fread(Buf, 1, sizeof(Buf), Tmp)) > 0;)
    Out.append(Buf, N);
  int Count = 0;
  for (size_t P = 0; (P = Out.find("kernel-printf-record", P)) != std::string::npos; P++)
    Count++;
  if (Count != 1) {
    printf("FAIL: record printed %d times\n", Count);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

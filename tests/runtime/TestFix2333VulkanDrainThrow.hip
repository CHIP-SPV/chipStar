// https://github.com/CHIP-SPV/chipStar/issues/2333: kernel printf survives a copy that throws inside a drain.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>

static std::atomic<bool> ArmDrain{false}, FailBegin{false}, Injected{false};
static std::atomic<int> Begins{0};

extern "C" int vkQueueWaitIdle(void *Q) {
  static auto Real = (int (*)(void *))dlsym(RTLD_NEXT, "vkQueueWaitIdle");
  int R = Real(Q);
  if (ArmDrain.exchange(false))
    FailBegin = true;
  return R;
}

extern "C" int vkBeginCommandBuffer(void *CB, const void *Info) {
  static auto Real =
      (int (*)(void *, const void *))dlsym(RTLD_NEXT, "vkBeginCommandBuffer");
  Begins++;
  if (FailBegin.exchange(false)) {
    Injected = true;
    return -2; // VK_ERROR_OUT_OF_DEVICE_MEMORY
  }
  return Real(CB, Info);
}

__global__ void k(int V) { printf("kernel-printf-%d\n", V); }

int main() {
  k<<<1, 1>>>(1);
  ArmDrain = true;
  hipError_t E = hipDeviceSynchronize();
  (void)hipGetLastError();
  if (Begins == 0) {
    printf("HIP_SKIP_THIS_TEST: the Vulkan interposers were not called\n");
    return CHIP_SKIP_TEST;
  }
  if (!Injected || E == hipSuccess) {
    printf("FAIL: injected=%d, first sync returned %d\n", (int)Injected, (int)E);
    return 1;
  }

  FILE *Tmp = tmpfile();
  fflush(stdout);
  int Saved = dup(1);
  dup2(fileno(Tmp), 1);
  k<<<1, 1>>>(2);
  E = hipDeviceSynchronize();
  fflush(stdout);
  dup2(Saved, 1);
  close(Saved);

  char Buf[256] = {};
  rewind(Tmp);
  fread(Buf, 1, sizeof(Buf) - 1, Tmp);
  if (E != hipSuccess || !strstr(Buf, "kernel-printf-2")) {
    printf("FAIL: kernel-printf-2 missing (sync=%d)\n", (int)E);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

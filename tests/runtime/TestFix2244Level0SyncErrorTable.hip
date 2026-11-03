// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2244: a failed
// Level Zero queue or command list sync must surface as a HIP error, not abort.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <dlfcn.h>

// 0x70000001 is DEVICE_LOST, 1 is NOT_READY, 0x78000003 is UNSUPPORTED_FEATURE.
static std::atomic<int> QueueFail{0}, ListFail{0}, ListHits{0};

extern "C" int zeCommandQueueSynchronize(void *H, uint64_t T) {
  static auto Real = (int (*)(void *, uint64_t))dlsym(
      RTLD_NEXT, "zeCommandQueueSynchronize");
  if (int S = QueueFail.exchange(0))
    return S;
  return Real(H, T);
}

extern "C" int zeCommandListHostSynchronize(void *H, uint64_t T) {
  static auto Real = (int (*)(void *, uint64_t))dlsym(
      RTLD_NEXT, "zeCommandListHostSynchronize");
  ListHits++;
  if (int S = ListFail.exchange(0))
    return S;
  return Real(H, T);
}

static bool check(std::atomic<int> &Fail, int Status, hipError_t Want,
                  void *Buf) {
  Fail = Status;
  if (hipMemsetAsync(Buf, 0, 4, 0) != hipSuccess) {
    printf("FAIL: hipMemsetAsync\n");
    return false;
  }
  hipError_t Got = hipStreamSynchronize(0);
  if (Fail != 0 || Got != Want) {
    printf("FAIL: status %#x: armed status left=%d, got %d, want %d\n", Status,
           Fail.load(), Got, Want);
    return false;
  }
  return true;
}

int main() {
  void *Buf;
  if (hipMalloc(&Buf, 4) != hipSuccess ||
      hipMemsetAsync(Buf, 0, 4, 0) != hipSuccess ||
      hipStreamSynchronize(0) != hipSuccess) {
    printf("FAIL: setup\n");
    return 1;
  }
  if (!ListHits) {
    printf("HIP_SKIP_THIS_TEST: zeCommandListHostSynchronize not used\n");
    return CHIP_SKIP_TEST;
  }
  if (!check(QueueFail, 0x70000001, hipErrorNotInitialized, Buf) ||
      !check(QueueFail, 1, hipErrorNotReady, Buf) ||
      !check(ListFail, 0x70000001, hipErrorNotInitialized, Buf) ||
      !check(QueueFail, 0x78000003, hipErrorNotSupported, Buf))
    return 1;
  (void)hipMemsetAsync(Buf, 0, 4, 0);
  if (hipStreamSynchronize(0) != hipSuccess) {
    printf("FAIL: unarmed sync did not succeed\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

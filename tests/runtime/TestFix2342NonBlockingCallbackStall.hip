// A callback on a hipStreamNonBlocking stream must not stall the default stream.
// https://github.com/CHIP-SPV/chipStar/issues/2342
#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

static std::atomic<int> Flag{0};
static std::atomic<int> Seen{0};

static void spinUntilFlag(hipStream_t, hipError_t, void *) {
  for (int I = 0; I < 500; I++) {
    if (Flag.load()) {
      Seen = 1;
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

// Runs Op on the default stream while a callback blocks a non-blocking stream.
template <typename F> static bool run(const char *Name, F Op) {
  Flag = 0;
  Seen = 0;
  hipStream_t S;
  if (hipStreamCreateWithFlags(&S, hipStreamNonBlocking) != hipSuccess ||
      hipStreamAddCallback(S, spinUntilFlag, nullptr, 0) != hipSuccess) {
    printf("FAIL: %s: stream setup failed\n", Name);
    return false;
  }
  hipError_t Err = Op();
  Flag = 1; // Only after Op returns.
  (void)hipStreamSynchronize(S);
  (void)hipStreamDestroy(S);
  if (Err != hipSuccess || !Seen) {
    printf("FAIL: %s: err=%d callback_saw_flag=%d\n", Name, (int)Err,
           Seen.load());
    return false;
  }
  return true;
}

int main() {
  int N = 0;
  (void)hipGetBackendNativeHandles(0, nullptr, &N);
  uintptr_t H[8] = {};
  if (N < 1 || N > 8) {
    printf("HIP_SKIP_THIS_TEST: unexpected handle count %d\n", N);
    return CHIP_SKIP_TEST;
  }
  (void)hipGetBackendNativeHandles(0, H, nullptr);
  if (strcmp(reinterpret_cast<const char *>(H[0]), "level0") != 0) {
    printf("HIP_SKIP_THIS_TEST: not the level0 backend\n");
    return CHIP_SKIP_TEST;
  }

  void *D = nullptr;
  if (hipMalloc(&D, 256) != hipSuccess) {
    printf("FAIL: hipMalloc failed\n");
    return 1;
  }

  bool Ok = run("hipStreamSynchronize(0), empty default stream",
                [] { return hipStreamSynchronize(0); });
  Ok &= run("hipMemset on the default stream", [&] {
    hipError_t E = hipMemset(D, 0, 256);
    return E != hipSuccess ? E : hipStreamSynchronize(0);
  });

  (void)hipFree(D);
  printf(Ok ? "PASS\n" : "FAIL\n");
  return Ok ? 0 : 1;
}

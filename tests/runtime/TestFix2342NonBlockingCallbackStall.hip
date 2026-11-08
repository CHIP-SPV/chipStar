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

static std::atomic<int> Done{0};

static void sleepThenDone(hipStream_t, hipError_t, void *) {
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  Done = 1;
}

static void sleepThenWrite(hipStream_t, hipError_t, void *P) {
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  *static_cast<volatile int *>(P) = 1;
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

  // (c) A callback on a blocking stream is awaited by the null stream.
  {
    Done = 0;
    hipStream_t B;
    (void)hipStreamCreate(&B);
    (void)hipStreamAddCallback(B, sleepThenDone, nullptr, 0);
    (void)hipStreamSynchronize(0);
    if (!Done) {
      printf("FAIL: null stream sync did not wait for a blocking stream callback\n");
      Ok = false;
    }
    (void)hipStreamSynchronize(B);
    (void)hipStreamDestroy(B);
  }

  // (d) A stream with a running callback is not ready.
  {
    Flag = 0;
    Seen = 0;
    hipStream_t S;
    (void)hipStreamCreateWithFlags(&S, hipStreamNonBlocking);
    (void)hipStreamAddCallback(S, spinUntilFlag, nullptr, 0);
    hipError_t Busy = hipStreamQuery(S);
    Flag = 1;
    (void)hipStreamSynchronize(S);
    hipError_t Idle = hipStreamQuery(S);
    if (Busy != hipErrorNotReady || Idle != hipSuccess) {
      printf("FAIL: hipStreamQuery during callback=%d after sync=%d\n",
             (int)Busy, (int)Idle);
      Ok = false;
    }
    (void)hipStreamDestroy(S);
  }

  // (e) Work enqueued after a callback runs after it.
  int Managed = 0;
  (void)hipDeviceGetAttribute(&Managed, hipDeviceAttributeManagedMemory, 0);
  if (Managed) {
    int *M = nullptr;
    hipStream_t S;
    (void)hipMallocManaged(&M, sizeof(int));
    (void)hipStreamCreateWithFlags(&S, hipStreamNonBlocking);
    *M = 0;
    (void)hipStreamAddCallback(S, sleepThenWrite, M, 0);
    (void)hipMemsetAsync(M, 2, sizeof(int), S);
    (void)hipStreamSynchronize(S);
    if (*M != 0x02020202) {
      printf("FAIL: memset after callback lost, value=0x%x\n", (unsigned)*M);
      Ok = false;
    }
    (void)hipStreamDestroy(S);
    (void)hipFree(M);
  }

  (void)hipFree(D);
  printf(Ok ? "PASS\n" : "FAIL\n");
  return Ok ? 0 : 1;
}

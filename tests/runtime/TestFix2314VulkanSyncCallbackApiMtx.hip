#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdio>
#include <unistd.h>

// hipStreamSynchronize must not hold ApiMtx while waiting for a callback that
// makes a HIP call.
static hipStream_t S;
static void *A, *B;
static std::atomic<bool> Ran{false};
static hipError_t CopyRc = hipErrorUnknown;

static void Cb(hipStream_t, hipError_t, void *) {
  usleep(200000); // Lets main enter hipStreamSynchronize first.
  CopyRc = hipMemcpyAsync(B, A, 64, hipMemcpyDeviceToDevice, S);
  Ran = true;
}

int main() {
  alarm(60); // A deadlock must fail, not hang.
  if (hipStreamCreateWithFlags(&S, hipStreamNonBlocking) != hipSuccess ||
      hipMalloc(&A, 64) != hipSuccess || hipMalloc(&B, 64) != hipSuccess ||
      hipStreamAddCallback(S, Cb, nullptr, 0) != hipSuccess) {
    printf("FAIL: setup\n");
    return 1;
  }
  if (hipStreamSynchronize(S) != hipSuccess) {
    printf("FAIL: hipStreamSynchronize\n");
    return 1;
  }
  if (!Ran || CopyRc != hipSuccess) {
    printf("FAIL: callback ran=%d copy rc=%d\n", (int)Ran, (int)CopyRc);
    return 1;
  }
  if (hipStreamSynchronize(S) != hipSuccess) {
    printf("FAIL: second hipStreamSynchronize\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

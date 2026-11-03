// A stream callback that submits more than the ring's 16 slots on its own
// stream must not wait on itself. https://github.com/CHIP-SPV/chipStar/issues/2313
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdio>
#include <unistd.h>

static constexpr int N = 40; // more than the ring's 16 slots
static hipStream_t S;
static int *A;
static std::atomic<bool> Done{false};
static std::atomic<bool> CopyFailed{false};

static void Cb(hipStream_t, hipError_t, void *) {
  // Each copy moves the seed one element, so the last element shows every copy ran, in order.
  for (int I = 0; I < N; ++I)
    if (hipMemcpyAsync(&A[I + 1], &A[I], sizeof(int), hipMemcpyDeviceToDevice,
                       S) != hipSuccess)
      CopyFailed = true;
  Done = true;
}

int main() {
  alarm(60);
  int H[N + 1] = {7};
  bool Ok = hipStreamCreateWithFlags(&S, hipStreamNonBlocking) == hipSuccess &&
            hipMalloc(&A, sizeof(H)) == hipSuccess &&
            hipMemcpy(A, H, sizeof(H), hipMemcpyHostToDevice) == hipSuccess &&
            hipStreamAddCallback(S, Cb, nullptr, 0) == hipSuccess;
  while (Ok && !Done)
    usleep(1000);
  Ok = Ok && hipStreamSynchronize(S) == hipSuccess &&
       hipMemcpy(H, A, sizeof(H), hipMemcpyDeviceToHost) == hipSuccess &&
       !CopyFailed;
  for (int I = 0; Ok && I <= N; ++I)
    Ok = H[I] == 7;
  puts(Ok ? "PASS" : "FAIL");
  return Ok ? 0 : 1;
}

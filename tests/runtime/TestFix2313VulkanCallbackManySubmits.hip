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

__global__ void Step(int *A, int I) { A[I + 1] = A[I]; }

static void Cb(hipStream_t, hipError_t, void *) {
  // Each launch moves the seed one element, so the last element shows every launch ran, in order.
  for (int I = 0; I < N; ++I)
    hipLaunchKernelGGL(Step, dim3(1), dim3(1), 0, S, A, I);
  Done = true;
}

int main() {
  alarm(60);
  int H[N + 1] = {7};
  bool Ok = hipStreamCreateWithFlags(&S, hipStreamNonBlocking) == hipSuccess &&
            hipMalloc(&A, sizeof(H)) == hipSuccess &&
            hipMemcpy(A, H, sizeof(H), hipMemcpyHostToDevice) == hipSuccess;
  // Compile the kernel here so the callback only exercises the ring
  // (lazy JIT and device variable setup otherwise run on the monitor thread).
  if (Ok)
    hipLaunchKernelGGL(Step, dim3(1), dim3(1), 0, S, A, N - 1);
  Ok = Ok && hipStreamSynchronize(S) == hipSuccess &&
       hipStreamAddCallback(S, Cb, nullptr, 0) == hipSuccess;
  while (Ok && !Done)
    usleep(1000);
  Ok = Ok && hipStreamSynchronize(S) == hipSuccess &&
       hipMemcpy(H, A, sizeof(H), hipMemcpyDeviceToHost) == hipSuccess;
  for (int I = 0; Ok && I <= N; ++I)
    Ok = H[I] == 7;
  puts(Ok ? "PASS" : "FAIL");
  return Ok ? 0 : 1;
}

// A full command buffer ring must not hold the API lock against the stream callback that gates it.
// https://github.com/CHIP-SPV/chipStar/issues/2349
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdio>
#include <unistd.h>

static constexpr int Ring = 16;
static constexpr int N = 40; // more launches than the ring has slots
static hipStream_t S;
static int *P;
static std::atomic<int> Launched{0};
static std::atomic<bool> CbDone{false};

__global__ void add1(int *P) { *P += 1; }

static void Cb(hipStream_t, hipError_t, void *) {
  while (Launched < Ring)
    usleep(1000);
  sleep(1); // main is now blocked at its next launch
  hipLaunchKernelGGL(add1, dim3(1), dim3(1), 0, S, P);
  CbDone = true;
}

int main() {
  alarm(60);
  int V = 0;
  bool Ok = hipStreamCreate(&S) == hipSuccess &&
            hipMalloc(&P, sizeof(int)) == hipSuccess &&
            hipMemcpy(P, &V, sizeof(int), hipMemcpyHostToDevice) == hipSuccess;
  // Compile the kernel here so the callback does not JIT on the monitor thread.
  if (Ok)
    hipLaunchKernelGGL(add1, dim3(1), dim3(1), 0, S, P);
  Ok = Ok && hipStreamSynchronize(S) == hipSuccess &&
       hipStreamAddCallback(S, Cb, nullptr, 0) == hipSuccess;
  for (int I = 0; Ok && I < N; ++I) {
    ++Launched;
    hipLaunchKernelGGL(add1, dim3(1), dim3(1), 0, S, P);
  }
  while (Ok && !CbDone)
    usleep(1000);
  Ok = Ok && hipStreamSynchronize(S) == hipSuccess &&
       hipMemcpy(&V, P, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess;
  Ok = Ok && V == N + 2; // warm-up + N + callback
  puts(Ok ? "PASS" : "FAIL");
  return Ok ? 0 : 1;
}

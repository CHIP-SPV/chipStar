// Waits for submitted work must not hold the API lock against the stream callback that gates it.
// https://github.com/CHIP-SPV/chipStar/issues/2364
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <unistd.h>

static hipStream_t S;
static int *P;
static int *Big1, *Big2;
static constexpr size_t BigInts = 16 << 20; // 64 MB
static std::atomic<bool> CopyDone{false}, CopyOk{false};
static std::atomic<bool> Go{false};
static std::atomic<bool> CbDone{false};

__global__ void add1(int *P) { *P += 1; }

static void Cb(hipStream_t, hipError_t, void *) {
  while (!Go)
    usleep(1000);
  sleep(1); // main is now blocked inside its operation
  hipLaunchKernelGGL(add1, dim3(1), dim3(1), 0, S, P);
  CbDone = true;
}

// Runs on a non blocking stream's callback thread, where a synchronous copy
// must have landed by the time hipMemcpy returns.
static void CbCopy(hipStream_t, hipError_t, void *) {
  bool Ok = hipMemcpy(Big2, Big1, BigInts * sizeof(int), hipMemcpyDefault) ==
            hipSuccess;
  CopyOk = Ok && !memcmp(Big2, Big1, BigInts * sizeof(int));
  CopyDone = true;
}

template <class F> static bool Paused(const char *Name, F Op) {
  printf("%s\n", Name);
  fflush(stdout);
  Go = false;
  CbDone = false;
  if (hipStreamAddCallback(S, Cb, nullptr, 0) != hipSuccess)
    return false;
  Go = true;
  bool Ok = Op();
  while (!CbDone)
    usleep(1000);
  return hipStreamSynchronize(S) == hipSuccess && Ok;
}

int main() {
  alarm(60);
  constexpr int Warm = 1024; // the descriptor pool's maxSets
  constexpr int Scenarios = 6;
  constexpr int Marker = 0x1234;
  int *Q = nullptr;
  hipStream_t T;
  int *H1 = nullptr, *H2 = nullptr;
  bool Ok = hipStreamCreate(&S) == hipSuccess &&
            hipMalloc(&P, sizeof(int)) == hipSuccess &&
            hipMemset(P, 0, sizeof(int)) == hipSuccess &&
            hipHostMalloc(&H1, 64) == hipSuccess &&
            hipHostMalloc(&H2, 64) == hipSuccess &&
            hipMalloc(&Q, sizeof(int)) == hipSuccess &&
            hipMemcpy(Q, &Marker, sizeof(int), hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipStreamCreateWithFlags(&T, hipStreamNonBlocking) == hipSuccess &&
            hipHostMalloc(&Big1, BigInts * sizeof(int)) == hipSuccess &&
            hipHostMalloc(&Big2, BigInts * sizeof(int)) == hipSuccess;
  for (size_t I = 0; Ok && I < BigInts; ++I) {
    Big1[I] = I + 1;
    Big2[I] = 0;
  }
  for (int I = 0; Ok && I < Warm; ++I)
    hipLaunchKernelGGL(add1, dim3(1), dim3(1), 0, S, P);
  Ok = Ok && hipStreamSynchronize(S) == hipSuccess;

  Ok = Ok && Paused("pool", [] {
    hipLaunchKernelGGL(add1, dim3(1), dim3(1), 0, S, P);
    return true;
  });
  Ok = Ok && Paused("fill", [&] {
    return hipMemsetAsync(H1 + 8, 7, sizeof(int), S) == hipSuccess;
  });
  Ok = Ok && Paused("fill2d", [&] {
    return hipMemset2DAsync(H1, 16, 9, 4, 2, S) == hipSuccess;
  });
  Ok = Ok && Paused("fill3d", [&] {
    return hipMemset3DAsync(make_hipPitchedPtr(H1, 16, 4, 4), 5,
                            make_hipExtent(4, 2, 1), S) == hipSuccess;
  });
  Ok = Ok && Paused("copy mapped to mapped", [&] {
    return hipMemcpyAsync(H2, H1, sizeof(int), hipMemcpyHostToDevice, S) ==
           hipSuccess;
  });
  Ok = Ok && Paused("copy device to host", [&] {
    return hipMemcpyAsync(H1, Q, sizeof(int), hipMemcpyDeviceToHost, S) ==
           hipSuccess;
  });
  puts("callback thread copy");
  fflush(stdout);
  Ok = Ok && hipStreamAddCallback(T, CbCopy, nullptr, 0) == hipSuccess;
  while (Ok && !CopyDone)
    usleep(1000);
  Ok = Ok && hipStreamSynchronize(T) == hipSuccess && CopyOk;

  int V = 0;
  Ok = Ok && hipMemcpy(&V, P, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess;
  Ok = Ok && V == Warm + 1 + Scenarios;     // warm-up, pool, callbacks
  Ok = Ok && H1[0] == Marker && H1[8] == 0x07070707;
  Ok = Ok && H1[4] == 0x05050505 && H2[0] == 0x05050505;
  puts(Ok ? "PASS" : "FAIL");
  return Ok ? 0 : 1;
}

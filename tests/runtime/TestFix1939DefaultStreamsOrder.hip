// The legacy default stream and hipStreamPerThread order against each other.

#include <hip/hip_runtime.h>
#include <cstdio>

#define N (1024 * 1024 * 8)

__global__ void slowFill(int *P) {
  for (int I = 0; I < N; ++I)
    P[I] = I + 1;
}

static bool run(hipStream_t KernelStream, hipStream_t CopyStream) {
  int *H, *D;
  hipHostMalloc((void **)&H, sizeof(int) * N);
  hipMalloc((void **)&D, sizeof(int) * N);
  for (int I = 0; I < N; ++I)
    H[I] = -1;
  hipMemset(D, 0, sizeof(int) * N);
  hipDeviceSynchronize();
  slowFill<<<1, 1, 0, KernelStream>>>(D);
  hipMemcpyAsync(H, D, sizeof(int) * N, hipMemcpyDeviceToHost, CopyStream);
  hipStreamSynchronize(CopyStream);
  long Bad = 0;
  for (int I = 0; I < N; ++I)
    Bad += H[I] != I + 1;
  hipHostFree(H);
  hipFree(D);
  return Bad == 0;
}

int main() {
  bool Ok = run(nullptr, hipStreamPerThread);
  Ok &= run(hipStreamPerThread, nullptr);
  puts(Ok ? "PASS" : "FAIL");
  return Ok ? 0 : 1;
}

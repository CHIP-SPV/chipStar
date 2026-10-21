// hipStreamSynchronize(0) must wait for blocking streams and hipStreamPerThread.

#include <hip/hip_runtime.h>
#include <cstdio>

#define N (1024 * 1024 * 8)

__global__ void fill(int *P) {
  for (int I = 0; I < N; ++I)
    P[I] = I + 6;
}

static long stale(const int *H) {
  long B = 0;
  for (int I = N - 1; I >= 0; --I)
    B += H[I] != I + 6;
  return B;
}

static long run(hipStream_t S) {
  int *H, *D;
  hipHostMalloc((void **)&H, 4L * N);
  hipMalloc((void **)&D, 4L * N);
  for (int I = 0; I < N; ++I)
    H[I] = I;
  fill<<<1, 1, 0, S>>>(D);
  hipMemcpyAsync(H, D, 4L * N, hipMemcpyDeviceToHost, S);
  hipStreamSynchronize(0);
  long B = stale(H);
  hipDeviceSynchronize();
  hipHostFree(H);
  hipFree(D);
  return B;
}

int main() {
  long B1 = run(hipStreamPerThread);
  hipStream_t S;
  hipStreamCreate(&S);
  long B2 = run(S);
  hipStreamDestroy(S);
  if (B1 || B2) {
    printf("FAIL: stale elements perthread=%ld blocking=%ld\n", B1, B2);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

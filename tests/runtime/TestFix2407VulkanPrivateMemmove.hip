// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2407: a memmove
// within a private array compares two pointers into it, which Vulkan cannot.
#include <hip/hip_runtime.h>

#include <cstdio>

constexpr int Buckets = 8, Keep = 4, N = 200, Threads = 32;

// Keeps the Keep smallest values of each bucket, sorted, by insertion.
__device__ __host__ void smallest(const int *In, int T, int *Out) {
  int Best[Buckets * Keep];
  for (int J = 0; J < Buckets * Keep; ++J)
    Best[J] = 1 << 30;
  for (int I = 0; I < N; ++I) {
    int V = In[(I + T) % N];
    int Base = (V % Buckets) * Keep;
    for (int K = 0; K < Keep; ++K)
      if (V < Best[Base + K]) {
        for (int KK = Keep - 1; KK >= K + 1; --KK) // Becomes a memmove.
          Best[Base + KK] = Best[Base + KK - 1];
        Best[Base + K] = V;
        break;
      }
  }
  for (int J = 0; J < Buckets * Keep; ++J)
    Out[J] = Best[J];
}

__global__ void kernel(const int *In, int *Out) {
  smallest(In, threadIdx.x, Out + threadIdx.x * Buckets * Keep);
}

int main() {
  int H[N];
  for (int I = 0; I < N; ++I)
    H[I] = (I * 37 + 11) % 251;
  int *In, *Out;
  if (hipMalloc(&In, sizeof(H)) != hipSuccess ||
      hipMalloc(&Out, Threads * Buckets * Keep * sizeof(int)) != hipSuccess ||
      hipMemcpy(In, H, sizeof(H), hipMemcpyHostToDevice) != hipSuccess)
    return 1;
  kernel<<<1, Threads>>>(In, Out);
  int R[Threads * Buckets * Keep], E[Buckets * Keep];
  if (hipMemcpy(R, Out, sizeof(R), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  int Errors = 0;
  for (int T = 0; T < Threads; ++T) {
    smallest(H, T, E);
    for (int J = 0; J < Buckets * Keep; ++J)
      if (R[T * Buckets * Keep + J] != E[J] && Errors++ < 4)
        printf("thread %d slot %d: %d, expected %d\n", T, J,
               R[T * Buckets * Keep + J], E[J]);
  }
  hipFree(In);
  hipFree(Out);
  printf(Errors ? "FAILED\n" : "PASSED\n");
  return Errors != 0;
}

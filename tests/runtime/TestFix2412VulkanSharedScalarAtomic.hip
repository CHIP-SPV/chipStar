// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2412: an atomic
// on a __shared__ scalar passes its address as a constant expression.
#include <hip/hip_runtime.h>

#include <cstdio>

// One work item: contended shared float atomics hang on KosmicKrisp (#2563).
__global__ void sum(const float *In, float *Out, int N) {
  __shared__ float Partial;
  Partial = 0;
  for (int I = 0; I < N; ++I)
    atomicAdd(&Partial, In[I]);
  *Out = Partial;
}

int main() {
  constexpr int N = 64;
  float H[N];
  for (int I = 0; I < N; ++I)
    H[I] = I;
  float *In, *Out, R = 0;
  if (hipMalloc(&In, sizeof(H)) != hipSuccess ||
      hipMalloc(&Out, sizeof(float)) != hipSuccess ||
      hipMemcpy(In, H, sizeof(H), hipMemcpyHostToDevice) != hipSuccess)
    return 1;
  sum<<<1, 1>>>(In, Out, N);
  if (hipMemcpy(&R, Out, sizeof(R), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  hipFree(In);
  hipFree(Out);
  bool Ok = R == N * (N - 1) / 2;
  printf(Ok ? "PASSED\n" : "FAILED: %g\n", R);
  return !Ok;
}

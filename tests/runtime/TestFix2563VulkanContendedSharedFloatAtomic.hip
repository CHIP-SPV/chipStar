// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2563: contended
// float atomics on __shared__ memory hung on KosmicKrisp.
#include <hip/hip_runtime.h>

#include <cstdio>

__global__ void sum(const float *In, float *Out) {
  __shared__ float Partial;
  __shared__ float Bins[2];
  if (threadIdx.x == 0)
    Partial = Bins[0] = Bins[1] = 0;
  __syncthreads();
  atomicAdd(&Partial, In[threadIdx.x]);
  atomicAdd(&Bins[threadIdx.x % 2], In[threadIdx.x]);
  __syncthreads();
  if (threadIdx.x == 0) {
    Out[0] = Partial;
    Out[1] = Bins[0];
    Out[2] = Bins[1];
  }
}

int main() {
  constexpr int N = 64;
  float H[N], R[3] = {};
  for (int I = 0; I < N; ++I)
    H[I] = I;
  float *In, *Out;
  if (hipMalloc(&In, sizeof(H)) != hipSuccess ||
      hipMalloc(&Out, sizeof(R)) != hipSuccess ||
      hipMemcpy(In, H, sizeof(H), hipMemcpyHostToDevice) != hipSuccess)
    return 1;
  sum<<<1, N>>>(In, Out);
  if (hipMemcpy(R, Out, sizeof(R), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  (void)hipFree(In);
  (void)hipFree(Out);
  // Even and odd halves of 0..63.
  bool Ok = R[0] == 2016 && R[1] == 992 && R[2] == 1024;
  printf(Ok ? "PASSED\n" : "FAILED: %g %g %g\n", R[0], R[1], R[2]);
  return !Ok;
}

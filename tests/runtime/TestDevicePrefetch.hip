// A kernel with a __builtin_prefetch lowered to OpenCL.std prefetch builds and
// runs, and the prefetch does not change its result.
#include <hip/hip_runtime.h>
#include <cstdio>

#define N 64
__global__ void k(const float *In, float *Out) {
  int I = threadIdx.x;
  __builtin_prefetch(In + (I + 16) % N);
  Out[I] = In[I] * 2;
}

int main() {
  float H[N], R[N];
  for (int I = 0; I < N; I++)
    H[I] = I;
  float *In, *Out;
  (void)hipMalloc(&In, sizeof(H));
  (void)hipMalloc(&Out, sizeof(R));
  (void)hipMemcpy(In, H, sizeof(H), hipMemcpyHostToDevice);
  k<<<1, N>>>(In, Out);
  hipError_t E = hipMemcpy(R, Out, sizeof(R), hipMemcpyDeviceToHost);
  bool Ok = E == hipSuccess;
  for (int I = 0; I < N; I++)
    Ok &= R[I] == 2.0f * I;
  std::printf("err=%d %s\n", (int)E, Ok ? "PASS" : "FAIL");
  return !Ok;
}

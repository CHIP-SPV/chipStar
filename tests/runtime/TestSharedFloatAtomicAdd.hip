// Float atomic adds on __shared__ memory, which some devices (Mesa's ANV on
// Intel Arc) cannot do natively in Vulkan.
#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %s (line %d)\n", #Call, hipGetErrorName(Err),  \
             __LINE__);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void k(double *Out, float *OutF) {
  __shared__ double S[4];
  __shared__ float SF[4];
  if (threadIdx.x < 4) {
    S[threadIdx.x] = 0;
    SF[threadIdx.x] = 0;
  }
  __syncthreads();
  atomicAdd(&S[threadIdx.x % 4], 1.5);
  atomicAdd(&SF[threadIdx.x % 4], 1.5f);
  __syncthreads();
  if (threadIdx.x < 4) {
    Out[threadIdx.x] = S[threadIdx.x];
    OutF[threadIdx.x] = SF[threadIdx.x];
  }
}

int main() {
  double *Out, H[4];
  float *OutF, HF[4];
  CHECK(hipMalloc(&Out, sizeof(H)));
  CHECK(hipMalloc(&OutF, sizeof(HF)));
  k<<<1, 64>>>(Out, OutF);
  CHECK(hipGetLastError());
  CHECK(hipMemcpy(H, Out, sizeof(H), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(HF, OutF, sizeof(HF), hipMemcpyDeviceToHost));
  bool Ok = true;
  for (int I = 0; I < 4; ++I)
    if (H[I] != 24.0 || HF[I] != 24.0f) {
      printf("slot %d: %g %g, expected 24\n", I, H[I], HF[I]);
      Ok = false;
    }
  printf("%s\n", Ok ? "PASSED" : "FAILED");
  return !Ok;
}

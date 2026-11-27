// A byte swap lowers to an exported SPIR-V helper; its module must still load.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k(const unsigned *In, unsigned *Out) {
  Out[threadIdx.x] = __builtin_bswap32(In[threadIdx.x]);
}

int main() {
  constexpr int N = 64;
  unsigned In[N], Out[N], *DIn, *DOut;
  for (int I = 0; I < N; ++I)
    In[I] = 0x01020304u * (I + 1);
  if (hipMalloc(&DIn, sizeof(In)) != hipSuccess ||
      hipMalloc(&DOut, sizeof(Out)) != hipSuccess ||
      hipMemcpy(DIn, In, sizeof(In), hipMemcpyHostToDevice) != hipSuccess)
    return 1;
  hipLaunchKernelGGL(k, dim3(1), dim3(N), 0, 0, DIn, DOut);
  if (hipMemcpy(Out, DOut, sizeof(Out), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  int Wrong = 0;
  for (int I = 0; I < N; ++I)
    Wrong += Out[I] != __builtin_bswap32(In[I]);
  printf("%d wrong\n%s\n", Wrong, Wrong ? "FAIL" : "PASS");
  return Wrong != 0;
}

// Check memset and memcpy calls from device functions.
#include <hip/hip_runtime.h>
#include <numeric>

__global__ void performMemset(void *Dst, int Val, size_t Count) {
  memset(Dst, Val, Count);
}

__global__ void performMemcpy(void *Dst, void *Src, size_t Count) {
  memcpy(Dst, Src, Count);
}

int main() {
  constexpr size_t N = 10;
  char *DstD, DstH[N];
  (void)hipMalloc(&DstD, N);
  (void)hipMemset(DstD, 0, N);
  performMemset<<<1,1>>>(DstD, 100, N);
  (void)hipMemcpy(DstH, DstD, N, hipMemcpyDeviceToHost);
  for (size_t i = 0; i < N; i++)
    if (DstH[i] != 100)
      return 1;

  char SrcH[N];
  std::iota(SrcH, SrcH + N, 1);
  char *SrcD;
  (void)hipMalloc(&SrcD, N);
  (void)hipMemcpy(SrcD, SrcH, N, hipMemcpyHostToDevice);
  performMemcpy<<<1,1>>>(DstD, SrcD, N);
  (void)hipMemcpy(DstH, DstD, N, hipMemcpyDeviceToHost);
  for (size_t i = 0; i < N; i++)
    if (DstH[i] != i + 1)
      return 2;

  (void)hipFree(SrcD);
  (void)hipFree(DstD);

  return 0;
}

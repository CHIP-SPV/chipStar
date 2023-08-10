// Check mem{set,cpy} calls from device functions as well as
// mem{set,cpy}-like loops. There has been troubles with memset and
// memset-like loops triggering LLVM module verification failures
// during SPIR-V -> LLVM IR translation. For example:
//
//   Fails to verify module: Intrinsic name not mangled correctly for type
//   arguments! Should be: llvm.memset.p1i8.i64
#include <hip/hip_runtime.h>
#include <numeric>

__global__ void performMemset(void *Dst, int Val, size_t Count) {
  memset(Dst, Val, Count);
}

__global__ void performMemcpy(void *Dst, void *Src, size_t Count) {
  memcpy(Dst, Src, Count);
}

__global__ void performMemsetLoop(void *Dst, int Val, size_t Count) {
  auto *CharDst = static_cast<char *>(Dst);
  for (size_t i = 0; i < Count; i++)
    CharDst[i] = Val;
}

__global__ void performMemcpyLoop(void *Dst, void *Src, size_t Count) {
  auto *CharDst = static_cast<char *>(Dst);
  auto *CharSrc = static_cast<char *>(Src);
  for (size_t i = 0; i < Count; i++)
    CharDst[i] = CharSrc[i];
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

  performMemsetLoop<<<1,1>>>(DstD, 42, N);
  (void)hipMemcpy(DstH, DstD, N, hipMemcpyDeviceToHost);
  for (size_t i = 0; i < N; i++)
    if (DstH[i] != 42)
      return 2;

  char SrcH[N];
  std::iota(SrcH, SrcH + N, 1);
  char *SrcD;
  (void)hipMalloc(&SrcD, N);
  (void)hipMemcpy(SrcD, SrcH, N, hipMemcpyHostToDevice);
  performMemcpy<<<1,1>>>(DstD, SrcD, N);
  (void)hipMemcpy(DstH, DstD, N, hipMemcpyDeviceToHost);
  for (size_t i = 0; i < N; i++)
    if (DstH[i] != i + 1)
      return 3;

  std::iota(SrcH, SrcH + N, 64);
  (void)hipMemcpy(SrcD, SrcH, N, hipMemcpyHostToDevice);
  performMemcpyLoop<<<1,1>>>(DstD, SrcD, N);
  (void)hipMemcpy(DstH, DstD, N, hipMemcpyDeviceToHost);
  for (size_t i = 0; i < N; i++)
    if (DstH[i] != i + 64)
      return 4;

  (void)hipFree(SrcD);
  (void)hipFree(DstD);

  return 0;
}

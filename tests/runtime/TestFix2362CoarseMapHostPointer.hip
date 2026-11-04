// https://github.com/CHIP-SPV/chipStar/issues/2362: hipMemset on a
// hipHostRegister'ed pointer must not SVM map the caller's malloc memory on
// coarse grain SVM. Run under an interposer that reports coarse-only SVM.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
extern char *CoarseWatchBegin, *CoarseWatchEnd;
extern int CoarseCapsRewritten, CoarseMapCalls;
}

int main() {
  const size_t N = 4096;
  hipInit(0);
  if (!CoarseCapsRewritten) {
    printf("HIP_SKIP_THIS_TEST: no coarse grain SVM device behind the OpenCL "
           "backend\n");
    return CHIP_SKIP_TEST;
  }

  // Control: a mapped hipHostMalloc must go through clEnqueueSVMMap.
  int Before = CoarseMapCalls;
  void *H;
  hipHostMalloc(&H, N, hipHostMallocMapped);
  hipHostFree(H);
  if (CoarseMapCalls == Before) {
    printf("FAIL: control saw no clEnqueueSVMMap, coarse path is not active\n");
    return 1;
  }

  char *B = (char *)malloc(N);
  memset(B, 0, N);
  void *D;
  hipHostRegister(B, N, 0);
  hipHostGetDevicePointer(&D, B, 0);
  CoarseWatchBegin = B;
  CoarseWatchEnd = B + N;
  hipError_t Err = hipMemset(D, 0x5a, N);
  CoarseWatchBegin = CoarseWatchEnd = nullptr;
  char *Out = (char *)malloc(N);
  memset(Out, 0, N);
  hipError_t Err2 = hipMemcpy(Out, D, N, hipMemcpyDeviceToHost);
  int Bad = 0;
  for (size_t I = 0; I < N; ++I)
    Bad += B[I] != 0x5a || Out[I] != 0x5a;
  hipHostUnregister(B);
  free(B);
  free(Out);
  if (Err != hipSuccess || Err2 != hipSuccess || Bad) {
    printf("FAIL: memset=%d memcpy=%d badBytes=%d\n", Err, Err2, Bad);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

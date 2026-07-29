// Regression test for: llvm.{u,s}mul.with.overflow in device code must not
// reach the SPIR-V consumer.
//
// Bug (#1391): clang emits llvm.umul.with.overflow.i64 for the count times
// size computation of a device-side "new T[n]", and for __builtin_mul_overflow.
// Through the in-tree SPIR-V backend the intrinsic becomes OpUMulExtended,
// whose operands IGC's SPIR-V reader mangles as signed while its builtin
// module defines only the unsigned overloads:
//
//   error: undefined reference to `_Z20__spirv_UMulExtendedll'
//   error: backend compiler failed build.
//
// Through the llvm-spirv translator, llvm.smul.with.overflow has no lowering
// and translation aborts outright. chipStar puts all device code of a binary
// into one module, so either failure makes every kernel in the program fail to
// build, while hipDeviceSynchronize() keeps returning hipSuccess and the
// outputs keep their initial values.
//
// Fix: expand llvm.{u,s}mul.with.overflow into plain integer arithmetic in the
// post-link pass pipeline so neither construct reaches SPIR-V.
//
// This uses __builtin_mul_overflow rather than "new T[n]" so it exercises the
// intrinsic without also depending on the device heap.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void testUmulOverflow(uint64_t *Lo, uint64_t *Ov) {
  uint64_t R = 0;

  // 6 * 7 = 42, no overflow.
  Ov[0] = __builtin_mul_overflow((uint64_t)6, (uint64_t)7, &R) ? 1 : 0;
  Lo[0] = R;

  // UINT64_MAX * 2 overflows, low half is 0xFFFFFFFFFFFFFFFE.
  R = 0;
  Ov[1] = __builtin_mul_overflow(UINT64_MAX, (uint64_t)2, &R) ? 1 : 0;
  Lo[1] = R;

  // 0 * UINT64_MAX does not overflow. Guards the divide-by-zero edge of any
  // expansion that computes the overflow predicate with a division.
  R = 0;
  Ov[2] = __builtin_mul_overflow((uint64_t)0, UINT64_MAX, &R) ? 1 : 0;
  Lo[2] = R;

  // 2^32 * 2^32 = 2^64 overflows, low half is 0.
  R = 0;
  Ov[3] = __builtin_mul_overflow((uint64_t)1 << 32, (uint64_t)1 << 32, &R) ? 1 : 0;
  Lo[3] = R;
}

int main() {
  const int N = 4;
  uint64_t *DLo = nullptr, *DOv = nullptr;
  if (hipMalloc(&DLo, N * sizeof(uint64_t)) != hipSuccess) {
    printf("SKIP: no GPU\n");
    return 0;
  }
  if (hipMalloc(&DOv, N * sizeof(uint64_t)) != hipSuccess) {
    printf("SKIP: no GPU\n");
    return 0;
  }
  hipMemset(DLo, 0, N * sizeof(uint64_t));
  hipMemset(DOv, 0, N * sizeof(uint64_t));

  hipLaunchKernelGGL(testUmulOverflow, dim3(1), dim3(1), 0, 0, DLo, DOv);
  if (hipDeviceSynchronize() != hipSuccess) {
    printf("FAIL: kernel launch or sync failed\n");
    return 1;
  }

  uint64_t HLo[N] = {0}, HOv[N] = {0};
  hipMemcpy(HLo, DLo, sizeof(HLo), hipMemcpyDeviceToHost);
  hipMemcpy(HOv, DOv, sizeof(HOv), hipMemcpyDeviceToHost);
  hipFree(DLo);
  hipFree(DOv);

  const uint64_t WantLo[N] = {42, UINT64_MAX - 1, 0, 0};
  const uint64_t WantOv[N] = {0, 1, 0, 1};

  int Bad = 0;
  for (int I = 0; I < N; ++I) {
    if (HLo[I] != WantLo[I] || HOv[I] != WantOv[I]) {
      printf("  case %d: got lo=%llu ov=%llu, want lo=%llu ov=%llu\n", I,
             (unsigned long long)HLo[I], (unsigned long long)HOv[I],
             (unsigned long long)WantLo[I], (unsigned long long)WantOv[I]);
      ++Bad;
    }
  }

  printf(Bad == 0 ? "PASS\n" : "FAIL\n");
  return Bad == 0 ? 0 : 1;
}

// Reproducer: hipCUB's ThreadStore<STORE_CS/CG/WB> templates emit AMDGCN
// inline assembly (flat_store_dword ... glc, s_waitcnt vmcnt(0)) that
// SPIRV-LLVM-Translator cannot consume. On an unpatched chipStar the
// lowered bitcode still carries these inline-asm calls; invoking
// llvm-spirv then null-derefs inside transDirectCallInst
// (SPIRVWriter.cpp:5598, CI->getCalledFunction() is nullptr when the
// callee operand is an InlineAsm).
//
// This file mimics the minimal hipCUB pattern: a __device__ wrapper
// that emits "flat_store_dword $0, $1 glc" followed by
// "s_waitcnt vmcnt(0)" via `asm volatile`, called from a kernel. It
// must build cleanly when chipStar's HipStripAMDGCNAsm pass replaces
// the inline asm with plain LLVM stores and drops the waitcnt.
//
// Without the fix: hipcc aborts at hipspv-link stage with
// "clang++: error: hipspv-link command failed due to signal" and the
// llvm-spirv stack trace ending at SPIRV::LLVMToSPIRVBase::transDirectCallInst.
#include <hip/hip_runtime.h>
#include <cstdio>

__device__ __forceinline__ void amdgcn_store_cs(unsigned int* ptr,
                                                unsigned int val) {
#if defined(__HIP_DEVICE_COMPILE__)
  // Same shape as hipCUB thread_store.hpp HIPCUB_ASM_THREAD_STORE:
  //   asm volatile("flat_store_dword %0, %1 glc" : : "v"(ptr), "v"(val));
  //   asm volatile("s_waitcnt vmcnt(%0)" : : "I"(0x00));
  asm volatile("flat_store_dword %0, %1 glc" : : "v"(ptr), "v"(val));
  asm volatile("s_waitcnt vmcnt(%0)" : : "I"(0x00));
#else
  *ptr = val;
#endif
}

__global__ void test_kernel(unsigned int* out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
    amdgcn_store_cs(&out[tid], (unsigned int)(tid * 42));
}

#define CHECK(expr) do {                                                    \
  hipError_t e = (expr);                                                    \
  if (e != hipSuccess) {                                                    \
    printf("FAIL %s = %d (%s)\n", #expr, (int)e, hipGetErrorString(e));     \
    return 1;                                                               \
  }                                                                         \
} while (0)

int main() {
  constexpr int N = 64;
  unsigned int* d_out = nullptr;
  CHECK(hipMalloc(&d_out, N * sizeof(unsigned int)));
  CHECK(hipMemset(d_out, 0, N * sizeof(unsigned int)));

  hipLaunchKernelGGL(test_kernel, dim3(1), dim3(N), 0, 0, d_out, N);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());

  unsigned int host[N];
  CHECK(hipMemcpy(host, d_out, N * sizeof(unsigned int), hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) {
    if (host[i] != (unsigned int)(i * 42)) {
      printf("FAIL host[%d]=%u expected %u\n", i, host[i],
             (unsigned int)(i * 42));
      return 1;
    }
  }
  CHECK(hipFree(d_out));
  printf("PASS\n");
  return 0;
}

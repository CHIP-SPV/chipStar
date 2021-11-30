#ifndef __HIPLZ_SYCL_INTEROP_H__
#define __HIPLZ_SYCL_INTEROP_H__

extern "C" {
  // Run GEMM test via oneMKL
  int oneMKLGemmTest(unsigned long* nativeHandlers, float* A, float* B, float* C, int M, int N, int K,
		     int ldA, int ldB, int ldC, float alpha, float beta);
}

#endif

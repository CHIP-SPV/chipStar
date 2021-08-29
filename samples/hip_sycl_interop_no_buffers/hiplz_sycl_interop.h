#ifndef __HIPLZ_SYCL_INTEROP_H__
#define __HIPLZ_SYCL_INTEROP_H__

extern "C" {
  // Run GEMM test via oneMKL
  int oneMKLGemmTest(unsigned long* nativeHandlers, double* A, double* B, double* C, int M, int N, int K,
		     int ldA, int ldB, int ldC, double alpha, double beta);
}

#endif

#ifndef __sycl_chip_interop_H__
#define __sycl_chip_interop_H__

extern "C" {
// Run GEMM test via chipStar Level-Zero Backend via USM data transfer
int hipMatrixMultiplicationUSMTest(const float *A, const float *B, float *C,
                                   int M, int N);

// Run GEMM test via chipStar Level-Zero Backend
int hipMatrixMultiplicationTest(const float *A, const float *B, float *C, int M,
                                int N);
}
#endif

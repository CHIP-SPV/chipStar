#ifndef __sycl_chip_interop_H__
#define __sycl_chip_interop_H__

extern "C" {
// must declare this here since it's not part of the standard HIP API
int hipInitFromNativeHandles(const uintptr_t *NativeHandles, int NumHandles);

// Run GEMM test via CHIP-SPV Level-Zero Backend via USM data transfer
int hipMatrixMultiplicationUSMTest(const float *A, const float *B, float *C,
                                   int M, int N);

// Run GEMM test via CHIP-SPV Level-Zero Backend
int hipMatrixMultiplicationTest(const float *A, const float *B, float *C, int M,
                                int N);
}
#endif

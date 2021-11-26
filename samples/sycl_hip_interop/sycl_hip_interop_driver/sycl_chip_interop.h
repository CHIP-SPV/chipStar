#ifndef __sycl_chip_interop_H__
#define __sycl_chip_interop_H__

extern "C" {
// Initialize HipLZ via providing native runtime information
int hipInitFromOutside(void* driverPtr, void* deviePtr, void* contextPtr,
                       void* queueptr);

// Run GEMM test via HipLZ via USM data transfer
int hipMatrixMultiplicationUSMTest(const float* A, const float* B, float* C,
                                   int M, int N);

// Run GEMM test via HipLZ
int hipMatrixMultiplicationTest(const float* A, const float* B, float* C, int M,
                                int N);
}

#endif

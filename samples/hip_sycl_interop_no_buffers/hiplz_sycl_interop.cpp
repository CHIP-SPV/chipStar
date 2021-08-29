#include "hip/hip_runtime.h"

// STL classes
#include <exception>
#include <iostream>

#include <vector>

#include "hiplz_sycl_interop.h"

using namespace std;

const int WIDTH = 10;

bool ValueSame(double a, double b) {
  return std::fabs(a-b) < 1.0e-08;  
}

void VerifyResult(double *c_A, double *c_B) {
  bool MismatchFound = false;

  for (size_t i=0; i < WIDTH; i++) {
    for (size_t j=0; j < WIDTH; j++) {
      if (!ValueSame(c_A[i*WIDTH+j], c_B[i*WIDTH+j])) {
	std::cout << "fail - The result is incorrect for element: [" << i << ", " << j
		  << "], expected: " << c_A[i*WIDTH+j] << " , but got: " << c_B[i*WIDTH+j]
		  << std::endl;
	//	exit(1);
        MismatchFound = true;
      }
    }
  }

  if (!MismatchFound) {
    std::cout << "SUCCESS - The results are correct!" << std::endl;
    return;
  }

}

int main() { 
  double * A = (double *) malloc(WIDTH*WIDTH*sizeof(double));
  double * B = (double *) malloc(WIDTH*WIDTH*sizeof(double));
  double * C = (double *) malloc(WIDTH*WIDTH*sizeof(double));
  double * C_serial = (double *) malloc(WIDTH*WIDTH*sizeof(double));

  double *d_A, *d_B, *d_C;
  // matrix data sizes
  int m = WIDTH;
  int n = WIDTH;
  int k = WIDTH;
  int ldA, ldB, ldC;
  ldA = ldB = ldC = WIDTH;
  double alpha = 1.0;
  double beta  = 0.0;
  
  // Create HipLZ stream  
  hipStream_t stream = nullptr;
  hipStreamCreate(&stream);

  unsigned long nativeHandlers[4];
  int numItems = 0;
  hiplzStreamNativeInfo(stream, nativeHandlers, &numItems);

  // allocate memory
  hipMalloc( &d_A, WIDTH*WIDTH*sizeof(double));
  hipMalloc( &d_B, WIDTH*WIDTH*sizeof(double));
  hipMalloc( &d_C, WIDTH*WIDTH*sizeof(double));

  // initialize data on the host
    // prepare matrix data with ROW-major style
  // A(M, N)
  for (size_t i=0; i<WIDTH; i++)
    for (size_t j=0; j<WIDTH; j++)
      A[i*WIDTH + j] = i*WIDTH + j;
  // B(N, P)
  for (size_t i=0; i<WIDTH; i++)
    for (size_t j=0; j<WIDTH; j++)
      B[i*WIDTH + j] = i*WIDTH + j;

  // get CPU result
  // Resultant matrix: C_serial = A*B
  for (size_t i=0; i<WIDTH; i++) {
    for (size_t j=0; j<WIDTH; j++) {
      C_serial[i*WIDTH + j] = 0;
      for(size_t d=0; d<WIDTH; d++) {
	C_serial[i*WIDTH + j] += A[i*WIDTH + d] * B[d*WIDTH + j];
      }
    }
  }

  // copy A and B to the device
  hipMemcpy(d_A, A, WIDTH*WIDTH*sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, WIDTH*WIDTH*sizeof(double), hipMemcpyHostToDevice);

  // Invoke oneMKL GEMM
  oneMKLGemmTest(nativeHandlers, d_A, d_B, d_C, WIDTH, WIDTH, WIDTH, ldA, ldB, ldC, alpha, beta);

  // copy back C
  hipMemcpy(C, d_C, WIDTH*WIDTH*sizeof(double), hipMemcpyDeviceToHost);

  // check results
  std::cout << "Verify results between OneMKL & Serial: ";  
  VerifyResult(C, C_serial);

  return 0;
}

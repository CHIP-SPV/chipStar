#include "hip/hip_runtime.h"

// STL classes
#include <exception>
#include <iostream>

#include <vector>

#include "hiplz_sycl_interop.h"

using namespace std;

const int WIDTH = 10;

bool ValueSame(float a, float b) {
  return std::fabs(a-b) < 1.0e-08;
}

void VerifyResult(vector<float>& c_A, vector<float>& c_B) {
  bool MismatchFound = false;

  for (size_t i=0; i < WIDTH; i++) {
    for (size_t j=0; j < WIDTH; j++) {
      if (!ValueSame(c_A[i*WIDTH+j], c_B[i*WIDTH+j])) {
        std::cout << "fail - The result is incorrect for element: [" << i << ", " << j
                  << "], expected: " << c_A[i*WIDTH+j] << " , but got: " << c_B[i*WIDTH+j]
                  << std::endl;
       
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
  vector<float> A(100, 1);
  vector<float> B(100, 2);
  vector<float> C(100, 0);
  vector<float> C_serial(100, 0);
  int m, n, k;
  m = n = k = WIDTH;
  int ldA, ldB, ldC;
  ldA = ldB = ldC = WIDTH;
  double alpha = 1.0;
  double beta  = 0.0;

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

  // get CPU result for verification
  // Resultant matrix: C_serial = A*B
  for (size_t i=0; i<WIDTH; i++) {
    for (size_t j=0; j<WIDTH; j++) {
      C_serial[i*WIDTH + j] = 0;
      for(size_t d=0; d<WIDTH; d++) {
        C_serial[i*WIDTH + j] += A[i*WIDTH + d] * B[d*WIDTH + j];
      }
    }
  }
  
  // Create HipLZ stream  
  hipStream_t stream = nullptr;
  hipStreamCreate(&stream);

  unsigned long nativeHandlers[4];
  int numItems = 0;
  hiplzStreamNativeInfo(stream, nativeHandlers, &numItems);
  
  // Invoke oneMKL GEEM
  oneMKLGemmTest(nativeHandlers, A.data(), B.data(), C.data(), m, m, k, ldA, ldB, ldC, alpha, beta);

  // check results  
  std::cout << "Verify results between OneMKL & Serial: ";
  VerifyResult(C, C_serial);
  
  return 0;
}

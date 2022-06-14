#include "hip/hip_runtime.h"

// STL classes
#include <exception>
#include <iostream>

#include <vector>

#include "hip_sycl_interop.h"

using namespace std;

const int WIDTH = 10;

bool ValueSame(float a, float b) { return std::fabs(a - b) < 1.0e-08; }

void VerifyResult(float *c_A, float *c_B) {
  bool MismatchFound = false;

  for (size_t i = 0; i < WIDTH; i++) {
    for (size_t j = 0; j < WIDTH; j++) {
      if (!ValueSame(c_A[i * WIDTH + j], c_B[i * WIDTH + j])) {
        std::cout << "fail - The result is incorrect for element: [" << i
                  << ", " << j << "], expected: " << c_A[i * WIDTH + j]
                  << " , but got: " << c_B[i * WIDTH + j] << std::endl;
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
  std::string envVar = std::getenv("CHIP_BE");
  if (!envVar.compare("opencl")) {
    std::cout << "HIP_SKIP_THIS_TEST" << std::endl;
    exit(0);
  }
  float *A = (float *)malloc(WIDTH * WIDTH * sizeof(float));
  float *B = (float *)malloc(WIDTH * WIDTH * sizeof(float));
  float *C = (float *)malloc(WIDTH * WIDTH * sizeof(float));
  float *C_serial = (float *)malloc(WIDTH * WIDTH * sizeof(float));

  float *d_A, *d_B, *d_C;
  // matrix data sizes
  int m = WIDTH;
  int n = WIDTH;
  int k = WIDTH;
  int ldA, ldB, ldC;
  ldA = ldB = ldC = WIDTH;
  float alpha = 1.0;
  float beta = 0.0;

  // Create HipLZ stream
  hipStream_t stream = nullptr;
  hipStreamCreate(&stream);

  uintptr_t nativeHandlers[4];
  int numItems = 4;
  error = hipGetBackendNativeHandles(stream, nativeHandlers, &numItems);
  CHECK(error);

  // allocate memory
  hipMalloc(&d_A, WIDTH * WIDTH * sizeof(float));
  hipMalloc(&d_B, WIDTH * WIDTH * sizeof(float));
  hipMalloc(&d_C, WIDTH * WIDTH * sizeof(float));

  // initialize data on the host
  // prepare matrix data with ROW-major style
  // A(M, N)
  for (size_t i = 0; i < WIDTH; i++)
    for (size_t j = 0; j < WIDTH; j++)
      A[i * WIDTH + j] = i * WIDTH + j;
  // B(N, P)
  for (size_t i = 0; i < WIDTH; i++)
    for (size_t j = 0; j < WIDTH; j++)
      B[i * WIDTH + j] = i * WIDTH + j;

  // get CPU result
  // Resultant matrix: C_serial = A*B
  for (size_t i = 0; i < WIDTH; i++) {
    for (size_t j = 0; j < WIDTH; j++) {
      C_serial[i * WIDTH + j] = 0;
      for (size_t d = 0; d < WIDTH; d++) {
        C_serial[i * WIDTH + j] += A[i * WIDTH + d] * B[d * WIDTH + j];
      }
    }
  }

  // copy A and B to the device
  hipMemcpy(d_A, A, WIDTH * WIDTH * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, WIDTH * WIDTH * sizeof(float), hipMemcpyHostToDevice);

  // Invoke oneMKL GEMM
  oneMKLGemmTest(nativeHandlers, d_A, d_B, d_C, WIDTH, WIDTH, WIDTH, ldA, ldB,
                 ldC, alpha, beta);

  // copy back C
  hipMemcpy(C, d_C, WIDTH * WIDTH * sizeof(float), hipMemcpyDeviceToHost);

  // check results
  std::cout << "Verify results between OneMKL & Serial: ";
  VerifyResult(C, C_serial);

  return 0;
}

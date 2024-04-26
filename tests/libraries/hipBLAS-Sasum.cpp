#include <iostream>
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hipblas.h>

#define CHECK_HIP_ERROR(cmd)                                                   \
  do {                                                                         \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      std::cerr << "Encountered HIP error: " << hipGetErrorString(error)       \
                << " at line: " << __LINE__ << std::endl;                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_HIPBLAS_ERROR(cmd)                                               \
  do {                                                                         \
    hipblasStatus_t status = cmd;                                              \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                    \
      std::cerr << "Encountered HIPBLAS error: "                               \
                << hipGetErrorString(static_cast<hipError_t>(status))          \
                << " at line: " << __LINE__ << std::endl;                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main() {
  int n = 1000; // Size of the vector
  int incx = 1;    // Stride between consecutive elements

  // Allocate memory on the host
  std::vector<float> h_x(n);

  // Initialize the vector on the host
  for (int i = 0; i < n; i++) {
    h_x[i] = static_cast<float>(i) - static_cast<float>(n) / 2.0f;
  }

  // Allocate memory on the device
  float *d_x;
  CHECK_HIP_ERROR(hipMalloc(&d_x, n * sizeof(float)));
  hipDeviceSynchronize();

  // Copy data from host to device
  CHECK_HIP_ERROR(
      hipMemcpy(d_x, h_x.data(), n * sizeof(float), hipMemcpyHostToDevice));
  hipDeviceSynchronize();
  // Create a HIPBLAS handle
  hipblasHandle_t handle;
  CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));
  hipDeviceSynchronize();

  // Call hipblasSasum
  float result;
  CHECK_HIPBLAS_ERROR(hipblasSasum(handle, n, d_x, incx, &result));
  hipDeviceSynchronize();

  hipDeviceSynchronize();
  // Destroy the HIPBLAS handle
  CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));

  hipDeviceSynchronize();

  // Compute the expected result on the host
  float expected = 0.0f;
  for (int i = 0; i < n; i++) {
    expected += std::abs(h_x[i]);
  }

  // Compare the results
  float tolerance = 1e-5;
  if (std::abs(result - expected) > tolerance) {
    std::cerr << "Validation failed!" << std::endl;
    std::cerr << "Expected: " << expected << std::endl;
    std::cerr << "Got: " << result << std::endl;
    std::cerr << "Difference: " << result - expected << std::endl;
    std::cerr << "Difference: " << expected - result << std::endl;
    return 1;
  }

  std::cout << "Validation passed!" << std::endl;

  // Free device memory
  CHECK_HIP_ERROR(hipFree(d_x));

  return 0;
}
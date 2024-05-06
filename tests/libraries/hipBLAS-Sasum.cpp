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

void hipblasSasumTest(hipblasHandle_t handle, int n, float *d_x, int incx, const std::vector<float> &h_x) {
  float result;
  CHECK_HIPBLAS_ERROR(hipblasSasum(handle, n, d_x, incx, &result));
  
  float expected = 0.0f;
  for (int i = 0; i < n; i++) {
    expected += std::abs(h_x[i]);
  }

  float tolerance = 1e-5;
  if (std::abs(result - expected) > tolerance) {
    std::cerr << "Validation of Sasum failed!" << std::endl;
    std::cerr << "Expected: " << expected << ", Got: " << result << std::endl;
    std::cerr << "Difference: " << std::abs(result - expected) << std::endl;
  }
}

void hipblasSdotTest(hipblasHandle_t handle, int n, float *d_x, int incx, const std::vector<float> &h_x) {
  float result;
  CHECK_HIPBLAS_ERROR(hipblasSdot(handle, n, d_x, incx, d_x, incx, &result));

  float expected = 0.0f;
  for (int i = 0; i < n; i++) {
    expected += h_x[i] * h_x[i];
  }

  float tolerance = 1e-5;
  if (std::abs(result - expected) > tolerance) {
    std::cerr << "Validation of Sdot failed!" << std::endl;
    std::cerr << "Expected: " << expected << ", Got: " << result << std::endl;
    std::cerr << "Difference: " << std::abs(result - expected) << std::endl;
  }
}

void hipblasSnrm2Test(hipblasHandle_t handle, int n, float *d_x, int incx, const std::vector<float> &h_x) {
  float result;
  CHECK_HIPBLAS_ERROR(hipblasSnrm2(handle, n, d_x, incx, &result));

  float expected = 0.0f;
  for (int i = 0; i < n; i++) {
    expected += h_x[i] * h_x[i];
  }
  expected = std::sqrt(expected);

  float tolerance = 1e-5;
  if (std::abs(result - expected) > tolerance) {
    std::cerr << "Validation of Snrm2 failed!" << std::endl;
    std::cerr << "Expected: " << expected << ", Got: " << result << std::endl;
    std::cerr << "Difference: " << std::abs(result - expected) << std::endl;
  }
}

void hipblasIsamaxTest(hipblasHandle_t handle, int n, float *d_x, int incx, const std::vector<float> &h_x) {
  int result;
  CHECK_HIPBLAS_ERROR(hipblasIsamax(handle, n, d_x, incx, &result));

  int expected = 0;
  float max_val = std::abs(h_x[0]);
  for (int i = 1; i < n; i++) {
    if (std::abs(h_x[i * incx]) > max_val) {
      max_val = std::abs(h_x[i * incx]);
      expected = i;
    }
  }

  if (std::abs(h_x[result * incx]) != max_val) {
    std::cerr << "Validation of Isamax failed!" << std::endl;
    std::cerr << "Expected value: " << max_val << ", Got: " << std::abs(h_x[result * incx]) << std::endl;
    std::cerr << "Expected index: " << expected << ", Got: " << result << std::endl;
  }
}

int main() {
  int n = 1000; // Size of the vector
  int incx = 1; // Stride between consecutive elements

  // Allocate memory on the host
  std::vector<float> h_x(n);

  // Initialize the vector on the host
  for (int i = 0; i < n; i++) {
    h_x[i] = static_cast<float>(i) - static_cast<float>(n) / 2.0f;
  }

  // Allocate memory on the device
  float *d_x;
  CHECK_HIP_ERROR(hipMalloc(&d_x, n * sizeof(float)));

  // Copy data from host to device
  CHECK_HIP_ERROR(
      hipMemcpy(d_x, h_x.data(), n * sizeof(float), hipMemcpyHostToDevice));

  // Create a HIPBLAS handle
  hipblasHandle_t handle;
  CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));

  // Call test functions
  hipblasSasumTest(handle, n, d_x, incx, h_x);
  hipblasSdotTest(handle, n, d_x, incx, h_x);
  hipblasSnrm2Test(handle, n, d_x, incx, h_x);
  hipblasIsamaxTest(handle, n, d_x, incx, h_x);

  // Destroy the HIPBLAS handle
  CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));

  // Free device memory
  CHECK_HIP_ERROR(hipFree(d_x));

  return 0;
}
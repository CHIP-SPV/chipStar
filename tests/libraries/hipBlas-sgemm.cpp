#include <iostream>
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <vector>
#include <algorithm>
#include <cmath>

#define CHECK_HIP_ERROR(cmd) \
    do { \
        hipError_t error  = cmd; \
        if(error != hipSuccess) { \
            std::cerr << "Encountered HIP error at line " << __LINE__ << std::endl; \
            return -1; \
        } \
    } while(0)

#define CHECK_HIPBLAS_ERROR(cmd) \
    do { \
        hipblasStatus_t status  = cmd; \
        if(status != HIPBLAS_STATUS_SUCCESS) { \
            std::cerr << "Encountered HIPBLAS error at line " << __LINE__ << std::endl; \
            return -1; \
        } \
    } while(0)

int main() {
    // Matrix dimensions
    int m = 64, n = 64, k = 64;
    float alpha = 1.0f, beta = 0.0f;

    // Create hipBLAS handle
    hipblasHandle_t handle;
    CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));

    // Allocate and initialize host memory
    std::vector<float> h_A(m * k, 1.0f);
    std::vector<float> h_B(k * n, 1.0f);
    std::vector<float> h_C(m * n, 0.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_HIP_ERROR(hipMalloc(&d_A, m * k * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, k * n * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, m * n * sizeof(float)));

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), m * k * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), k * n * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_C, h_C.data(), m * n * sizeof(float), hipMemcpyHostToDevice));

    // Perform SGEMM operation
    CHECK_HIPBLAS_ERROR(hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k,
                                     &alpha, d_A, m, d_B, k, &beta, d_C, m));

    // Copy result back to host
    CHECK_HIP_ERROR(hipMemcpy(h_C.data(), d_C, m * n * sizeof(float), hipMemcpyDeviceToHost));

    // Verify result on CPU
    float expected = static_cast<float>(k);
    for (const auto& elem : h_C) {
        if (std::abs(elem - expected) > 1e-6) {
            std::cerr << "FAILED!" << std::endl;
            return -1;
        }
    }

    std::cout << "PASSED!" << std::endl;

    // Cleanup
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));

    return 0;
}
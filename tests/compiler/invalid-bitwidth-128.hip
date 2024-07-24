#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

__global__ void rcbrtKernel(double* results) {
    results[12] = rcbrt(8.0);
}

int main() {
    const int numResults = 28;
    double* deviceResults;
    double hostResults[numResults];
    const double epsilon = 1e-6;
    bool all_passed = true;

    // Allocate device memory
    hipMalloc(&deviceResults, numResults * sizeof(double));

    // Launch kernel
    hipLaunchKernelGGL(rcbrtKernel, dim3(1), dim3(1), 0, 0, deviceResults);

    // Copy results back to host
    hipMemcpy(hostResults, deviceResults, numResults * sizeof(double), hipMemcpyDeviceToHost);

    // Compute host results
    double host_rcbrt_result = 1.0 / std::cbrt(8.0);
    float host_rcbrtf_result = 1.0f / std::cbrt(27.0f);

    // Compare results
    if (std::abs(host_rcbrt_result - hostResults[12]) >= epsilon) {
        std::cout << "FAIL: rcbrt test failed - Host: " << host_rcbrt_result << ", Device: " << hostResults[12] << ", Diff: " << std::abs(host_rcbrt_result - hostResults[12]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_rcbrtf_result - hostResults[13]) >= epsilon) {
        std::cout << "FAIL: rcbrtf test failed - Host: " << host_rcbrtf_result << ", Device: " << hostResults[13] << ", Diff: " << std::abs(host_rcbrtf_result - hostResults[13]) << std::endl;
        all_passed = false;
    }

    // Free device memory
    hipFree(deviceResults);

    if (all_passed) {
        std::cout << "PASS: All tests passed!" << std::endl;
    } else {
        std::cout << "FAIL: Some tests failed. Check the output above for details." << std::endl;
    }

    return all_passed ? 0 : 1;
}



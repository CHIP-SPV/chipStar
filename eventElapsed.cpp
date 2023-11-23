#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

// Macro for checking hip errors
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error in " << #call << " at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl; \
        std::abort(); \
    } \
} while (0)

int main() {
    // Initialize the start and stop events
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Record the start event
    HIP_CHECK(hipEventRecord(start, nullptr));
    HIP_CHECK(hipEventSynchronize(start));

    // [Place your kernel or other operations here]

    // Record the stop event
    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));

    // Calculate the elapsed time
    float tElapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&tElapsed, start, stop));
    std::cout << "Elapsed time: " << tElapsed << " ms" << std::endl;

    // Destroy the events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return 0;
}
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    const size_t size = 1024; // Size of the memory to copy

    // Allocate host memory
    int* hostData = new int[size];

    // Allocate device memory
    int* deviceData;
    hipMalloc(&deviceData, size * sizeof(int));

    // Copy data from host to device
    hipMemcpy(deviceData, hostData, size * sizeof(int), hipMemcpyHostToDevice);

    // Free device memory
    hipFree(deviceData);

    // Free host memory
    delete[] hostData;

    return 0;
}

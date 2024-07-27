#include <hip/hip_runtime.h>
#include <iostream>

__global__ void errorKernel() {
    dim3 myDim3 = blockIdx; // This line will cause the error
}

int main() {
    errorKernel<<<1, 1>>>();
    hipDeviceSynchronize();
    
    std::cout << "If you see this, the error wasn't caught at runtime." << std::endl;
    
    return 0;
}

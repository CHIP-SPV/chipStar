#include <hip/hip_runtime.h>

template<typename T>
__global__ void kernelAddOne(T* data) {
    data[0] += 1;
}

void launchKernelAddTwo(int* data) {
  hipLaunchKernelGGL(kernelAddOne<int>, dim3(1), dim3(1), 0, 0, data);
  hipLaunchKernelGGL(kernelAddOne<int>, dim3(1), dim3(1), 0, 0, data);
}
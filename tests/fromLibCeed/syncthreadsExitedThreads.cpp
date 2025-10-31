#include <hip/hip_runtime.h>
#include <iostream>

__global__ void syncTest() { 
 const int tid = threadIdx.x; 

 if (tid > 10)
 return;

  __syncthreads();
}

int main() {
  hipLaunchKernelGGL(syncTest, dim3(1), dim3(100), 0, 0);
  hipDeviceSynchronize();
  std::cout << "PASSED\n";
  return 0;
}
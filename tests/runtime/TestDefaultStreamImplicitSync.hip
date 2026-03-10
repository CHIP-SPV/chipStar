#include <hip/hip_runtime.h>
#include <iostream>

__global__ void setFlag(volatile int *flag) { *flag = 1; }

int main() {
  int managedSupport = 0;
  hipDeviceGetAttribute(&managedSupport, hipDeviceAttributeManagedMemory, 0);
  if (!managedSupport) {
    printf("HIP_SKIP_THIS_TEST\n");
    return 0;
  }

  int *flag = nullptr;
  hipStream_t s = nullptr;

  hipMallocManaged(&flag, sizeof(int));
  *flag = 0;
  hipStreamCreate(&s);
  
  setFlag<<<1, 1, 0, nullptr>>>(flag);
  hipStreamSynchronize(s);
  
  if (*flag != 1) {
    std::cerr << "FAIL" << std::endl;
    hipDeviceSynchronize();
    hipFree(flag);
    hipStreamDestroy(s);
    return 1;
  }
  
  std::cout << "PASS" << std::endl;
  hipFree(flag);
  hipStreamDestroy(s);
  return 0;
}

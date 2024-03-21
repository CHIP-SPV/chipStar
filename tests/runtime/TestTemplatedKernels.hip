#include <hip/hip_runtime.h>

/**
 * Both of these funcitons rely on a templated kernelAddOne<int> function
 * launchKernelAddTwo launches the kernel twice. 
 * Similar pattern is commonly seen in hip-tests
 * 
 * If the templated kernel is replaced by a regular function we get a compilation error:
 * TemplatedKernelAddTwo.hip:(.text+0x0): multiple definition of `__device_stub__kernelAddOne(int*)';
*/
void launchKernelAddOne(int* data);
void launchKernelAddTwo(int* data);

int main() {
  int* data;
  hipHostMalloc((void**)&data, sizeof(int));
  data[0] = 0;
  assert(data[0] == 0);
  launchKernelAddOne(data);
  launchKernelAddTwo(data);

  hipFree(data);
  return 0;
}
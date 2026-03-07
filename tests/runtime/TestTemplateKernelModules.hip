#include <hip/hip_runtime.h>
#include <cstdio>

// Both of these functions rely on a templated kernelAddOne<int> defined
// in separate TUs. Each TU has its own SPIR-V module containing the same
// template instantiation. The runtime must handle this without warning
// about "device function is already registered and mapped to a different
// module" and must dispatch the kernel correctly from both call sites.
void launchKernelAddOne(int* data);
void launchKernelAddTwo(int* data);

int main() {
  int* data;
  hipHostMalloc((void**)&data, sizeof(int));
  data[0] = 0;

  launchKernelAddOne(data);
  hipDeviceSynchronize();
  if (data[0] != 1) {
    printf("FAIL: after launchKernelAddOne expected 1, got %d\n", data[0]);
    return 1;
  }

  launchKernelAddTwo(data);
  hipDeviceSynchronize();
  if (data[0] != 3) {
    printf("FAIL: after launchKernelAddTwo expected 3, got %d\n", data[0]);
    return 1;
  }

  printf("PASS\n");
  hipFree(data);
  return 0;
}

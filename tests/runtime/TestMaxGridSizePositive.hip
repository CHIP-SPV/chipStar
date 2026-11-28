// maxGridSize is an int; a device limit above INT_MAX must not read as negative.
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  hipDeviceProp_t P;
  if (hipGetDeviceProperties(&P, 0) != hipSuccess)
    return 1;
  int Bad = 0;
  for (int D = 0; D < 3; ++D) {
    printf("maxGridSize[%d] = %d\n", D, P.maxGridSize[D]);
    Bad += P.maxGridSize[D] <= 0;
  }
  printf("%s\n", Bad ? "FAIL" : "PASS");
  return Bad != 0;
}

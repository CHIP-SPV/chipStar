// Check hipGetLastError() after failed hipInit() call.
#include <hip/hip_runtime_api.h>
#include <iostream>

int main() {
  // Initial error code should be hipSuccess (derived from CUDA
  // programming manual v12.1 3.2.12. Error Checking).
  if (hipPeekAtLastError() != hipSuccess)
    return 1;
  if (hipGetLastError() != hipSuccess)
    return 2;

  if (hipInit(0xDEADBEEF) != hipErrorInvalidValue)
    return 3;
  if (hipGetLastError() != hipErrorInvalidValue)
    return 4;
  if (hipGetLastError() != hipSuccess)
    return 5;

  std::cout << "PASSED\n";
  return 0;
}

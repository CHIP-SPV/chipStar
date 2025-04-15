// Regression test for a segfault occurring after compile error.
#include <hip/hip_runtime.h>
#include <iostream>

// A device code that fails at runtime due to missing definition.
extern __device__ void foo(int);
__global__ void kernel(int x) { foo(x); }

int main() {
  kernel<<<1, 1>>>(123);
  auto CorrectAnswer = hipGetLastError() != hipSuccess;
  if (CorrectAnswer)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";
  return !CorrectAnswer;
}

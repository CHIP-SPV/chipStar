// Regression test for a segfault occurring after compile error.
#include <hip/hip_runtime.h>

// A device code that fails at runtime due to missing definition.
extern __device__ void foo(int);
__global__ void kernel(int x) { foo(x); }

int main() {
  kernel<<<1, 1>>>(123);
  auto CorrectAnswer = hipGetLastError() != hipSuccess;
  return !CorrectAnswer;
}

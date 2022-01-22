#include <iostream>

#include "hip/hip_runtime.h"

void testf() { std::cout << "Test Function Executed\n"; }

int main() {
  hipDevice_t Dev;
  auto Err = hipGetDevice(&Dev);
  if (Err != hipSuccess)
    std::cout << hipGetErrorName(Err) << std::endl;

  hipStream_t Stream;
  Err = hipStreamCreateWithPriority(&Stream, 0, 0);
  std::cout << hipGetErrorName(Err) << std::endl;
}
#include <iostream>

#include "hip/hip_runtime.h"

void testf() { std::cout << "Test Function Executed\n"; }

int main() {
  hipDevice_t dev;
  auto err = hipGetDevice(&dev);
  if (err != hipSuccess) std::cout << hipGetErrorName(err) << std::endl;

  hipStream_t stream;
  err = hipStreamCreateWithPriority(&stream, 0, 0);
  std::cout << hipGetErrorName(err) << std::endl;
}
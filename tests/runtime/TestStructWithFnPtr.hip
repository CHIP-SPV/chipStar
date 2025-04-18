// A regression test for "SPIR-V Parser: Failed to find size for type id #"
// error which subsequently lead to incorrect struct size calculation.
#include <hip/hip_runtime.h>
#include <iostream>

struct Foo {
  void (*SomeFn)(int) = nullptr; // Not used in device code
  int Val = 123;
};

__global__ void k(int *Out, Foo TheFoo) { *Out = TheFoo.Val; }

int main() {
  int *OutD, OutH = 0;
  (void)hipMalloc(&OutD, sizeof(int));
  k<<<1, 1>>>(OutD, Foo());
  (void)hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost);
  std::cout << "PASSED\n";
  return !(OutH == 123);
}

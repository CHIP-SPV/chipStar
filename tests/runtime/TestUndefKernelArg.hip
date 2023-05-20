#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include <hip/hip_runtime.h>
#include "CHIPBackend.hh"
#include <iostream>
#include <string>

__global__ void undefArgPtr(int *Unused, int A) { printf("%d\n", A); }

int main() {
  undefArgPtr<<<1, 1>>>((int*)0x2, 1);
}

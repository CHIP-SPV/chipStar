// Reproduces CHIP-SPV/chipStar#1555: every hipSetDevice pushes a context.
#include <hip/hip_runtime.h>
#include <cstdio>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

int main() {
  constexpr int Calls = 1000;
  for (int I = 0; I < Calls; ++I)
    if (hipSetDevice(0) != hipSuccess) {
      printf("FAIL: hipSetDevice\n");
      return 1;
    }

  // hipSetDevice replaces the current context, so one pop empties the stack.
  int Pops = 0;
  for (hipCtx_t Ctx = nullptr; Pops <= Calls; ++Pops)
    if (hipCtxPopCurrent(&Ctx) != hipSuccess || Ctx == nullptr)
      break;

  if (Pops != 1) {
    printf("FAIL: %d contexts popped after %d hipSetDevice calls, expected 1\n",
           Pops, Calls);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

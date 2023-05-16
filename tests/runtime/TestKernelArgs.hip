#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include <hip/hip_runtime.h>
#include "CHIPBackend.hh"
#include <iostream>
#include <string>

template <typename T> __global__ void passArgs(T *Dst, T Src) { *Dst = Src; }

template <typename T> T passThroughKernel(T ArgIn) {
  T *ArgOutD;
  assert(hipMalloc(&ArgOutD, sizeof(T)) == hipSuccess);

  // Check source code and CHIP-SPV runtime agreed on the T's size.
  auto KernelPtr = (void *)passArgs<T>;
  auto *ChipKernel = Backend->getActiveDevice()->findKernel(HostPtr(KernelPtr));
  assert(ChipKernel && "Could not find the kernel defined in the source.");
  const auto *FnInfo = ChipKernel->getFuncInfo();
  assert(FnInfo && "Kernel does not have function info!");
  assert(FnInfo->getNumClientArgs() == 2 &&
         "Disagreement on the kernel argument count.");
  FnInfo->visitClientArgs([&](const SPVFuncInfo::ClientArg &Arg) -> void {
    assert(Arg.Index != 1 ||
           Arg.Size == sizeof(T) && "Disagreement on the argument's size.");
  });

  passArgs<T><<<1, 1>>>(ArgOutD, ArgIn);
  T ArgOutH;
  assert(hipMemcpy(&ArgOutH, ArgOutD, sizeof(T), hipMemcpyDeviceToHost) ==
         hipSuccess);
  assert(hipFree(ArgOutD) == hipSuccess);

  return ArgOutH;
}

__global__ void undefArgPtr(int *Unused, int A) { printf("%d\n", A); }

int main() {

  auto IntOut = passThroughKernel(123);
  assert(IntOut == 123);

  struct S0_t {
    int a = 0x80706050;
    char b = 123;
    int c = 0x40302010;
  } S0;
  auto S0Out = passThroughKernel(S0);
  assert(S0Out.a == S0.a);
  assert(S0Out.b == S0.b);
  assert(S0Out.c == S0.c);

  struct S1_t {
    char a = 12;
    struct B_t {
      int a = 0x01020304;
      double b = 12.3;
    } b;
    int c = 0x05060708;
  } S1;
  auto S1Out = passThroughKernel(S1);
  assert(S1Out.a == S1.a);
  assert(S1Out.b.a == S1.b.a);
  assert(S1Out.b.b == S1.b.b);
  assert(S1Out.c == S1.c);

  struct S2_t {
    struct B_t {
      short a;
      char b;
    };
    int a = 0x55;
    B_t b[3] = {{0x1091, 0x11}, {0x2092, 0x22}, {0x3093, 0x33} };
    char c = 0x66;
  } S2;
  auto S2Out = passThroughKernel(S2);
  assert(S2Out.a == S2.a);
  for (int i = 0; i < 3; i++) {
    assert(S2Out.b[i].a == S2.b[i].a);
    assert(S2Out.b[i].b == S2.b[i].b);
  }
  assert(S2Out.c == S2.c);

  struct S3_t {
    char a[7] = {1, 2, 3, 4, 5, 6, 7};
    char b = 8;
  } S3;
  auto S3Out = passThroughKernel(S3);
  for (int i = 0; i < 7; i++)
    assert(S3Out.a[i] == S3.a[i]);
  assert(S3Out.b == S3.b);

  undefArgPtr<<<1, 1>>>((int*)0x2, 1);

  return 0;
}

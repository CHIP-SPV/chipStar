// Test kernel with an indirect call.
//
// NOTES:
//
// * Core SPIR-V does not support indirect calls without an extension. In this
//   case the llvm-spirv inserts SPV_INTEL_function_pointers extension so this
//   program is only runnable in environments that supports it.
//
// * Invalid SPIR-V binary is known to be generated.
//   OpConstantFunctionPointerINTEL instruction is generated with a reference to
//   an undeclared OpFunction (missing forward declaration).
//   https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/2043.

#include <hip/hip_runtime.h>
#include <iostream>
static __device__ int add123(int Val) { return Val + 123; }
static __device__ int sub321(int Val) { return Val - 321; }

// NOTE: it's possible that Clang's optimizer could optimize the indirect
//       calls away.  So far it does not happen on this kernel with
//       Clang-16 + -O3.
typedef int (*ICall)(int);
__global__ void k(int *Data, int Sel) {
  ICall Fns[] = {add123, sub321};
  *Data = Fns[Sel](*Data);
}

// A warp primitive anywhere in the module makes HipWarpsPass stamp
// intel_reqd_sub_group_size. A stamped kernel loses the arguments of an
// indirect call after the first one, so the callees below take two arguments
// and the kernel that reaches them must come out unstamped. touchShfl is never
// launched; it only has to exist so the shuffle declaration is in the module.
// One #include of Kokkos_Core.hpp was enough to trigger this, which broke
// device-side dispatch for every Kokkos application (CHIP-SPV/chipStar#1380).
__device__ int Sink;
__global__ void touchShfl(int V) { Sink = __shfl(V, 0); }

static __device__ double implA(double X, int I) {
  return X * 1000.0 + I * 10.0 + 6.0;
}
static __device__ double implB(double X, int I) { return -1.0; }

typedef double (*Callee)(double, int);

// Select through a kernel argument so the indirect call survives the optimizer.
__global__ void dispatch(double *Out, int Sel) {
  Callee Fns[] = {implA, implB};
  Out[0] = Fns[Sel](7.0, 3);
}

int main() {
  int *DataD = nullptr, DataH = 1000;
  (void)hipMalloc(&DataD, sizeof(int));
  (void)hipMemcpy(DataD, &DataH, sizeof(int), hipMemcpyHostToDevice);
  k<<<1,1>>>(DataD, 0);
  k<<<1,1>>>(DataD, 1);
  (void)hipMemcpy(&DataH, DataD, sizeof(int), hipMemcpyDeviceToHost);
  if (DataH != 802) {
    std::cout << "FAILED: indirect f(1000) chain = " << DataH
              << ", want 802\n";
    return 1;
  }

  double *OutD = nullptr, OutH = -1.0;
  (void)hipMalloc(&OutD, sizeof(double));
  dispatch<<<1,1>>>(OutD, 0);
  (void)hipMemcpy(&OutH, OutD, sizeof(double), hipMemcpyDeviceToHost);
  // 7.0 * 1000 + 3 * 10 + 6.0. A dropped second argument gives 7006.0.
  if (OutH != 7036.0) {
    std::cout << "FAILED: indirect f(7.0, 3) = " << OutH
              << ", want 7036\n";
    return 1;
  }

  std::cout << "PASSED\n";
  return 0;
}

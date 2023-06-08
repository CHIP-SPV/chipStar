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

int main() {
  int *DataD = nullptr, DataH = 1000;
  (void)hipMalloc(&DataD, sizeof(int));
  (void)hipMemcpy(DataD, &DataH, sizeof(int), hipMemcpyHostToDevice);
  k<<<1,1>>>(DataD, 0);
  k<<<1,1>>>(DataD, 1);
  (void)hipMemcpy(&DataH, DataD, sizeof(int), hipMemcpyDeviceToHost);
  return !(DataH == 802);
}

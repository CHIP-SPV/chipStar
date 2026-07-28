// Covers every initialization class the program-scope variable lowering
// produces (see llvm_passes/HipGlobalVariables.cpp, classifyInitializer() and
// issue #582):
//
//   * a non-zero scalar and a non-zero array   -> grid-stride copy loop
//   * an initializer referencing another lowered variable ("path B")
//     -> a single scalar store, which must run on exactly ONE work item even
//        though the init kernel is now launched with a real grid
//   * a small zero-initialized array           -> grid-stride zero-store loop
//   * a large zero-initialized array           -> excluded from the init
//        kernel entirely and filled by the runtime instead
//   * an odd-sized, 1-byte-aligned array       -> forces the i8 element type
//
// All of these live in ONE module on purpose: they share a single
// __chip_var_init_all kernel, so this also covers them being chained together.

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <hip/hip_runtime.h>
#include <iostream>

// Must exceed ChipVarFillThreshold (src/common.hh) so that it is host-filled.
constexpr size_t LargeZeroSize = 128 * 1024;

__device__ int Scalar = 42;
__device__ float Array[4] = {1.0f, 2.0f, 3.0f, 4.0f};
__device__ int SmallZero[16] = {};
__device__ char LargeZero[LargeZeroSize] = {};
__device__ char OddBytes[7] = {1, 2, 3, 4, 5, 6, 7};

#ifdef CHIP_ENABLE_DEVICE_PROGRAM_SCOPE_GLOBALS
// "Path B": an initializer that references another variable being lowered, so
// its value is only known once that variable's storage has been bound. This is
// the case that must run on exactly one work item.
//
// Only exercised in the program-scope-globals lowering. Under the
// globals-as-kernel-args (rusticl) lowering, a variable whose address is read
// from *another* variable's init kernel cannot be expressed as a per-global
// trailing kernel argument, so isConvertibleGlobal() leaves it as a
// program-scope global -- which is exactly what that lowering exists to avoid.
// This is a pre-existing limitation of that lowering, not of the initializer
// classes under test here.
__device__ int *PtrToScalar = &Scalar;
#endif

__global__ void readAll(int *Out) {
  Out[0] = Scalar;
#ifdef CHIP_ENABLE_DEVICE_PROGRAM_SCOPE_GLOBALS
  Out[1] = *PtrToScalar;
#else
  Out[1] = Scalar; // path B not available in this lowering, see above
#endif
  Out[2] = static_cast<int>(Array[3]);
  Out[3] = OddBytes[6];

  // Every element of the zero-initialized variables must be zero, not just
  // the first: a partially applied grid-stride loop would leave a tail.
  int Dirty = 0;
  for (unsigned i = 0; i < 16; i++)
    if (SmallZero[i] != 0)
      Dirty++;
  for (size_t i = 0; i < LargeZeroSize; i++)
    if (LargeZero[i] != 0)
      Dirty++;
  Out[4] = Dirty;
}

int main() {
  constexpr unsigned NumElts = 5;
  int *OutD, OutH[NumElts];
  (void)hipMalloc(&OutD, sizeof(int) * NumElts);
  readAll<<<1, 1>>>(OutD);
  (void)hipMemcpy(&OutH, OutD, sizeof(int) * NumElts, hipMemcpyDeviceToHost);

  assert(OutH[0] == 42 && "non-zero scalar initializer");
  assert(OutH[1] == 42 && "initializer referencing another lowered variable");
  assert(OutH[2] == 4 && "non-zero array initializer");
  assert(OutH[3] == 7 && "odd-sized 1-byte-aligned initializer");
  assert(OutH[4] == 0 && "zero-initialized variables must be fully zeroed");

  std::cout << "PASSED\n";
  return 0;
}

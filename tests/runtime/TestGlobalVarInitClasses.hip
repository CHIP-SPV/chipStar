// Every program-scope variable initializer class in one module (#582).

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <hip/hip_runtime.h>
#include <iostream>

// More elements than the init launch has work items.
constexpr size_t LargeZeroSize = 128 * 1024;
// At least ChipVarFillThreshold (src/common.hh).
constexpr size_t HugeZeroSize = 64 * 1024 * 1024;

__device__ int Scalar = 42;
__device__ float Array[4] = {1.0f, 2.0f, 3.0f, 4.0f};
__device__ int SmallZero[16] = {};
__device__ char LargeZero[LargeZeroSize] = {};
__device__ char OddBytes[7] = {1, 2, 3, 4, 5, 6, 7};
__device__ char HugeZero[HugeZeroSize] = {};

#ifdef CHIP_ENABLE_DEVICE_PROGRAM_SCOPE_GLOBALS
// Without program-scope globals the pointee could not become a kernel arg.
__device__ int *PtrToScalar = &Scalar;
#endif

__global__ void readAll(int *Out) {
  Out[0] = Scalar;
#ifdef CHIP_ENABLE_DEVICE_PROGRAM_SCOPE_GLOBALS
  Out[1] = *PtrToScalar;
#endif
  Out[2] = static_cast<int>(Array[3]);
  Out[3] = OddBytes[6];
  int Dirty = 0;
  for (unsigned i = 0; i < 16; i++)
    if (SmallZero[i] != 0)
      Dirty++;
  for (size_t i = 0; i < LargeZeroSize; i++)
    if (LargeZero[i] != 0)
      Dirty++;
  for (size_t i = 0; i < HugeZeroSize; i += 4096)
    if (HugeZero[i] != 0)
      Dirty++;
  Out[4] = Dirty;
}

int main() {
  constexpr unsigned NumElts = 5;
  int *OutD, OutH[NumElts];
  // Freed dirty device memory must not pass for zero initialization.
  char *Dirt[18];
  for (int i = 0; i < 18; i++) {
    size_t N = i == 17 ? HugeZeroSize : i ? 64 : LargeZeroSize;
    (void)hipMalloc(&Dirt[i], N);
    (void)hipMemset(Dirt[i], 1, N);
  }
  for (char *D : Dirt)
    (void)hipFree(D);
  (void)hipMalloc(&OutD, sizeof(int) * NumElts);
  readAll<<<1, 1>>>(OutD);
  (void)hipMemcpy(&OutH, OutD, sizeof(int) * NumElts, hipMemcpyDeviceToHost);

  assert(OutH[0] == 42 && "non-zero scalar initializer");
#ifdef CHIP_ENABLE_DEVICE_PROGRAM_SCOPE_GLOBALS
  assert(OutH[1] == 42 && "initializer referencing another lowered variable");
#endif
  assert(OutH[2] == 4 && "non-zero array initializer");
  assert(OutH[3] == 7 && "odd-sized 1-byte-aligned initializer");
  assert(OutH[4] == 0 && "zero-initialized variables must be fully zeroed");

  std::cout << "PASSED\n";
  return 0;
}

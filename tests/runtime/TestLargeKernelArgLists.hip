// Test kernel argument lists as large as 4KB are supported as
// specified by CUDA [CUDA C++ Programming Guide 14.5.9.3. Function
// Parameters], even if the backend's driver does not support.
#include <hip/hip_runtime.h>
#include <cstring>
#include <set>
#include <iostream>

constexpr unsigned NumThreads = 10;

struct LargeStruct {
  char Arr[4 * 1024 - 16];
};

struct MediumStruct {
  int Arr[512 - 2];
};

static_assert(sizeof(LargeStruct) + sizeof(char *) + 2 * sizeof(unsigned) ==
                  4 * 1024,
              "test1: Parameter list needs to be 4KB.");
__global__ void test1(LargeStruct LS, char *ValOut, unsigned Idx0,
                      unsigned Idx1) {
  auto TID = threadIdx.x;
  // Modify the argument to see it's thread private. Runtime indeces
  // intends to discourage the LS being promoted to registers.
  LS.Arr[Idx0] += TID;
  ValOut[TID] = LS.Arr[Idx1];
}

static_assert(2 * sizeof(MediumStruct) + sizeof(int *) + 2 * sizeof(int) ==
                  4 * 1024,
              "test2: Parameter list needs to be 4KB.");
__global__ void test2(int *Out, MediumStruct Lhs, int LIdx, MediumStruct Rhs,
                      int RIdx) {
  auto TID = threadIdx.x;
  Out[TID] = Lhs.Arr[TID + LIdx] + Rhs.Arr[TID + RIdx];
}

int main() {
  // Test 1
  LargeStruct LS;
  std::memset(&LS, 0, sizeof(LS));
  LS.Arr[321] = 100;
  char *Out1D, Out1H[NumThreads];
  (void)hipMalloc(&Out1D, sizeof(char) * NumThreads);
  test1<<<1, NumThreads>>>(LS, Out1D, 321, 321);
  (void)hipMemcpy(&Out1H, Out1D, sizeof(char) * NumThreads,
                  hipMemcpyDeviceToHost);
  (void)hipFree(Out1D);
  for (unsigned Idx = 0; Idx < NumThreads; Idx++)
    if (Out1H[Idx] != 100 + Idx)
      return 1;

  // Test 2
  MediumStruct Lhs, Rhs;
  constexpr auto NumElts = sizeof(Lhs.Arr) / sizeof(int);
  for (unsigned i = 0; i < NumElts; i++) {
    Lhs.Arr[i] = (i + 100);
    Rhs.Arr[i] = i + 1;
  }

  int *Out2D, Out2H[NumThreads];
  (void)hipMalloc(&Out2D, sizeof(int) * NumThreads);
  test2<<<1, NumThreads>>>(Out2D, Lhs, 0, Rhs, NumElts - NumThreads);
  (void)hipMemcpy(&Out2H, Out2D, sizeof(int) * NumThreads,
                  hipMemcpyDeviceToHost);
  (void)hipFree(Out2D);

  for (unsigned i = 0; i < NumThreads; i++) {
    auto LhsElt = Lhs.Arr[i];
    auto RhsElt = Rhs.Arr[i + NumElts - NumThreads];
    if (Out2H[i] != LhsElt + RhsElt)
      return 1;
  }

  std::cout << "PASSED\n";
  return 0;
}

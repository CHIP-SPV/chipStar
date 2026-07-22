// Reproducer for chipStar issue #1359.
//
// __syncthreads_and/__syncthreads_or/__syncthreads_count lower to
// __chip_group_all/__chip_group_any/__chip_group_ballot, which are declared
// in include/hip/devicelib/sync_and_util.hh but have no definition in the
// device library. On the OpenCL backend the kernel fails to JIT with
// "error: unresolved external symbol __chip_group_all" and the kernel never
// runs.

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include <hip/hip_runtime.h>

#include <cstdio>

#define HIPCHECK(Expr)                                                         \
  do {                                                                         \
    hipError_t Err = (Expr);                                                   \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %s\n", #Expr, hipGetErrorString(Err));         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void syncthreadsCollectives(int *Out) {
  int Tid = threadIdx.x;
  // All threads pass a true predicate -> nonzero.
  int AllTrue = __syncthreads_and(1);
  // Thread 0 passes false -> zero.
  int AllButOne = __syncthreads_and(Tid != 0);
  // Only thread 3 passes true -> nonzero.
  int AnyOne = __syncthreads_or(Tid == 3);
  // All threads pass false -> zero.
  int AnyNone = __syncthreads_or(0);
  // Exactly 10 threads pass true.
  int CountTen = __syncthreads_count(Tid < 10);
  if (Tid == 0) {
    Out[0] = AllTrue;
    Out[1] = AllButOne;
    Out[2] = AnyOne;
    Out[3] = AnyNone;
    Out[4] = CountTen;
  }
}

int main() {
  constexpr int NumResults = 5;
  int *OutD;
  int OutH[NumResults] = {-1, -1, -1, -1, -1};
  HIPCHECK(hipMalloc(&OutD, sizeof(OutH)));
  HIPCHECK(hipMemcpy(OutD, OutH, sizeof(OutH), hipMemcpyHostToDevice));
  syncthreadsCollectives<<<1, 64>>>(OutD);
  HIPCHECK(hipGetLastError());
  HIPCHECK(hipDeviceSynchronize());
  HIPCHECK(hipMemcpy(OutH, OutD, sizeof(OutH), hipMemcpyDeviceToHost));
  HIPCHECK(hipFree(OutD));

  assert(OutH[0] != 0 && "__syncthreads_and(1) must be true");
  assert(OutH[1] == 0 && "__syncthreads_and(tid != 0) must be false");
  assert(OutH[2] != 0 && "__syncthreads_or(tid == 3) must be true");
  assert(OutH[3] == 0 && "__syncthreads_or(0) must be false");
  assert(OutH[4] == 10 && "__syncthreads_count(tid < 10) must be 10");

  printf("PASSED\n");
  return 0;
}

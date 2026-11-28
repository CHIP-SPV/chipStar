// Clang asserted emitting debug info for a static local emitted outside its function.
#include <hip/hip_runtime.h>

inline __host__ __device__ int *localStatic() {
  static __device__ int Var = 7;
  return &Var;
}
__device__ inline int inner() { return 1; }
__device__ inline int outer() { return inner(); }
__global__ void k(int *Out) { *Out = outer() + *localStatic(); }

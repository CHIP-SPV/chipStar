// Device variables shared by the two translation units of
// TestFixFunctionLocalStaticDeviceVar. Each TU that includes this header gets
// its own linkonce_odr comdat copy of every variable below in its own device
// module, and clang registers none of them with __hipRegisterVar (a
// function-local static is never registered; an inline variable and a
// template static data member are registered only when host code ODR-uses
// them).
#ifndef FUNCTION_LOCAL_STATIC_DEVICE_VAR_HH
#define FUNCTION_LOCAL_STATIC_DEVICE_VAR_HH

#include <hip/hip_runtime.h>

struct Reporter {
  int Count;
  int Pad[3];
};

// (1) A function-local static __device__ variable in an inline
// __host__ __device__ function: the shape of Trilinos STK
// stk_ngp_test/GlobalReporter.hpp getDeviceReporterOnDevice().
inline __host__ __device__ Reporter *localStaticReporter() {
  static __device__ Reporter Rep = {7, {0, 0, 0}};
  return &Rep;
}

// (2) A C++17 inline __device__ namespace-scope variable.
namespace shapes {
inline __device__ int InlineVar = 7;
}

// (3) A __device__ static data member of a class template.
template <typename T> struct Holder {
  static __device__ T Member;
};
template <typename T> __device__ T Holder<T>::Member = T(7);

__device__ inline void readShapes(int *Out) {
  Out[0] = localStaticReporter()->Count;
  Out[1] = shapes::InlineVar;
  Out[2] = Holder<int>::Member;
}

__device__ inline void writeShapes(int V) {
  localStaticReporter()->Count = V;
  shapes::InlineVar = V;
  Holder<int>::Member = V;
}

#endif

// Check C++ standard library calls.
//
// Note that this test only checks that the library calls get
// linked. Most of the calls are mapped to corresponding OpenCL defined
// functions.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>

template <typename T, typename FnT>
__global__ void unaryKernel(FnT Fn, T *Data) {
  *Data = Fn(*Data);
}

template <typename T, typename FnT>
__global__ void binaryKernel(FnT Fn, T *Lhs, T *Rhs) {
  *Lhs = Fn(*Lhs, *Rhs);
}

template <typename T, typename FnT> void launchUnaryFn(FnT Fn, T Arg = T()) {
  T *DataD;
  T DataH = Arg;
  (void)hipMalloc(&DataD, sizeof(T));
  (void)hipMemcpy(DataD, &DataH, sizeof(T), hipMemcpyHostToDevice);
  unaryKernel<<<1, 1>>>(Fn, DataD);
  (void)hipDeviceSynchronize();
  assert(hipGetLastError() == hipSuccess);
  (void)hipFree(DataD);
}

template <typename T, typename FnT> void launchBinaryFn(FnT Fn) {
  T *LhsD, *RhsD;
  T DataH;
  std::memset(&DataH, 0, sizeof(T));
  (void)hipMalloc(&LhsD, sizeof(T));
  (void)hipMalloc(&RhsD, sizeof(T));
  (void)hipMemcpy(LhsD, &DataH, sizeof(T), hipMemcpyHostToDevice);
  (void)hipMemcpy(RhsD, &DataH, sizeof(T), hipMemcpyHostToDevice);
  binaryKernel<<<1, 1>>>(Fn, LhsD, RhsD);
  (void)hipDeviceSynchronize();
  assert(hipGetLastError() == hipSuccess);
  (void)hipFree(LhsD);
  (void)hipFree(RhsD);
}

int main() {

  launchUnaryFn<float>([] __device__(auto x) { return std::abs(x); });
  launchUnaryFn<float>([] __device__(auto x) { return std::expm1(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::log1p(x); }, 2);

  launchUnaryFn<float>([] __device__(auto x) { return std::trunc(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::sin(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::cos(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::tan(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::asin(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::acos(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::atan(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::sinh(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::cosh(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::tanh(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::floor(x); });
  launchUnaryFn<float>([] __device__(auto x) { return std::floorf(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::ceil(x); });
  launchUnaryFn<float>([] __device__(auto x) { return std::ceilf(x); });

  launchUnaryFn<int>([] __device__(int x) { return std::log(x); }, 2);
  launchUnaryFn<float>([] __device__(auto x) { return std::log(x); }, 2);
  launchUnaryFn<float>([] __device__(auto x) { return std::logf(x); }, 2);

  launchUnaryFn<float>([] __device__(auto x) { return std::log2(x); }, 2);
  launchUnaryFn<float>([] __device__(auto x) { return std::log2f(x); }, 2);

  launchUnaryFn<float>([] __device__(auto x) { return std::log10(x); }, 2);
  launchUnaryFn<float>([] __device__(auto x) { return std::log10f(x); }, 2);

  launchUnaryFn<float>([] __device__(auto x) { return std::erf(x); });
  launchUnaryFn<float>([] __device__(auto x) { return std::erff(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::erfc(x); });
  launchUnaryFn<float>([] __device__(auto x) { return std::erfcf(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::sqrt(x); });
  launchUnaryFn<float>([] __device__(auto x) { return std::sqrtf(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::lgamma(x); }, 1);
  launchUnaryFn<float>([] __device__(auto x) { return std::lgammaf(x); }, 1);

  launchUnaryFn<float>([] __device__(auto x) { return std::nearbyint(x); });
  launchUnaryFn<float>([] __device__(auto x) { return std::nearbyintf(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::exp(x); });
  launchUnaryFn<float>([] __device__(auto x) { return std::expf(x); });

  launchUnaryFn<float>([] __device__(auto x) { return std::lrint(x); });
  launchUnaryFn<float>([] __device__(auto x) { return std::lrintf(x); });

  launchBinaryFn<float>(
      [] __device__(auto x, auto y) { return std::copysign(x, y); });
  launchBinaryFn<float>(
      [] __device__(auto x, auto y) { return std::copysignf(x, y); });

  launchBinaryFn<float>(
      [] __device__(auto x, auto y) { return std::nextafter(x, y); });
  launchBinaryFn<float>(
      [] __device__(auto x, auto y) { return std::nextafterf(x, y); });

  std::cout << "PASSED\n";
  return 0;
}

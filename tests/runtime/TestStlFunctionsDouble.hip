// Check C++ standard library calls.
//
// Note that this test only checks that the library calls get
// linked. Most of the calls are mapped to corresponding OpenCL defined
// functions.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

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

template <typename T, typename FnT>
void launchBinaryFn(FnT Fn, T LhsH = T(), T RhsH = T()) {
  T *LhsD, *RhsD;
  (void)hipMalloc(&LhsD, sizeof(T));
  (void)hipMalloc(&RhsD, sizeof(T));
  (void)hipMemcpy(LhsD, &LhsH, sizeof(T), hipMemcpyHostToDevice);
  (void)hipMemcpy(RhsD, &RhsH, sizeof(T), hipMemcpyHostToDevice);
  binaryKernel<<<1, 1>>>(Fn, LhsD, RhsD);
  (void)hipDeviceSynchronize();
  assert(hipGetLastError() == hipSuccess);
  (void)hipFree(LhsD);
  (void)hipFree(RhsD);
}

int main() {

  launchUnaryFn<double>([] __device__(auto x) { return std::expm1(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::log1p(x); }, 2);
  launchUnaryFn<double>([] __device__(auto x) { return std::trunc(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::sin(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::cos(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::tan(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::asin(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::acos(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::atan(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::sinh(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::cosh(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::tanh(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::floor(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::ceil(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::log(x); }, 4);
  launchUnaryFn<double>([] __device__(auto x) { return std::log2(x); }, 2);
  launchUnaryFn<double>([] __device__(auto x) { return std::log10(x); }, 2);
  launchUnaryFn<double>([] __device__(auto x) { return std::erf(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::erfc(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::sqrt(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::lgamma(x); }, 1);
  launchUnaryFn<double>([] __device__(auto x) { return std::nearbyint(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::exp(x); });
  launchUnaryFn<double>([] __device__(auto x) { return std::lrint(x); });
  launchBinaryFn<double>(
      [] __device__(auto x, auto y) { return std::copysign(x, y); });
  launchBinaryFn<double>(
      [] __device__(auto x, auto y) { return std::nextafter(x, y); });
  launchBinaryFn<double>(
      [] __device__(auto x, auto y) { return std::pow(x, y); }, 2.0, 3.0);

  return 0;
}

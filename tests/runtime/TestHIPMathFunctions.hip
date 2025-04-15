// Hosts checks for cmath functions made available by hip_runtime.h
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <hip/hip_runtime.h>
#include <iostream>

template <typename T, typename FnT>
__global__ void unaryKernel(FnT Fn, T *Data) {
  *Data = Fn(*Data);
}

template <typename T, typename FnT> T launchUnaryFn(FnT Fn, T Arg) {
  T *DataD;
  T DataH = Arg;
  (void)hipMalloc(&DataD, sizeof(T));
  (void)hipMemcpy(DataD, &DataH, sizeof(T), hipMemcpyHostToDevice);
  unaryKernel<<<1, 1>>>(Fn, DataD);
  (void)hipMemcpy(&DataH, DataD, sizeof(T), hipMemcpyDeviceToHost);
  assert(hipGetLastError() == hipSuccess);
  (void)hipFree(DataD);
  return DataH;
}

int main() {
  assert(::min(1, 2) == 1);
  assert(::max(1, 2) == 2);
  assert(::log1pf(0) == 0.0f);
  auto Answer =
      launchUnaryFn([] __device__(float f) { return ::log1pf(f); }, 0.0f);
  assert(Answer == 0.0f);
  std::cout << "PASSED\n";
  return 0;
}

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include <hip/hip_runtime.h>

#include <iostream>

template <typename T, typename AtomicOp>
__global__ void applyAtomicOp(T *InOut, T Mod, AtomicOp AOp) {
  AOp(InOut, Mod);
}

template <unsigned NumThreads, typename T, typename AtomicOp>
T launchAtomicOp(T Init, T Mod, AtomicOp AOp) {
  T *DataD;
  T DataH = Init;
  (void)hipMalloc(&DataD, sizeof(T));
  (void)hipMemcpy(DataD, &DataH, sizeof(T), hipMemcpyHostToDevice);
  applyAtomicOp<<<1, NumThreads>>>(DataD, Mod, AOp);
  (void)hipDeviceSynchronize();
  (void)hipMemcpy(&DataH, DataD, sizeof(T), hipMemcpyDeviceToHost);
  (void)hipFree(DataD);
  return DataH;
}

int main() {
  auto Res0 = launchAtomicOp<10, float>(
      0, 1, [] __device__(auto *x, auto y) { return atomicAdd(x, y); });
  assert(Res0 == 10);

  auto Res1 = launchAtomicOp<10, double>(
      10, 2, [] __device__(auto *x, auto y) { return atomicAdd(x, y); });
  assert(Res1 == 30);

  auto Res2 = launchAtomicOp<23, int>(
      100, 1, [] __device__(auto *x, auto y) { return atomicAdd(x, y); });
  assert(Res2 == 123);

  auto Res3 = launchAtomicOp<4, unsigned int>(
      10, 2, [] __device__(auto *x, auto y) { return atomicAdd(x, y); });
  assert(Res3 == 18);

  // not part of either HIP or CUDA APIs
  // auto Res4 = launchAtomicOp<4, unsigned long>(
  //     0x10000000, 2,
  //     [] __device__(auto *x, auto y) { return atomicAdd(x, y); });
  // assert(Res4 == 0x10000000 + 8);

  auto Res5 = launchAtomicOp<23, unsigned long long>(
      0x10000000000ll, 1,
      [] __device__(auto *x, auto y) { return atomicAdd(x, y); });
  assert(Res5 == (0x10000000000ll + 23));

  auto Res6 = launchAtomicOp<10, float>(
      0.0f, 1.0f, [] __device__(float *x, float y) { return atomicAdd(x, y); });
  float Ref6 = 0.0f;
  for (int i = 0; i < 10; i++)
    Ref6 += 1.0f;
  assert(Res6 == Ref6);

  auto Res7 =
      launchAtomicOp<1, float>(1.23f, 3.21f, [] __device__(auto *x, auto y) {
        return atomicExch(x, y);
      });
  assert(Res7 == 3.21f);

  return 0;
}

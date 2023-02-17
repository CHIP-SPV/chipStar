// Regression test for global variable initialization code-gen
// compiling or running very slowly on some drivers. E.g. on an Intel
// iGPU this case may run/compile for hours.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <hip/hip_runtime.h>

constexpr size_t NumElts = 1000000;
__device__ float A[NumElts];
__global__ void readA(float *Out, size_t Index) { *Out = A[Index]; }

int main() {
  float *OutD, OutH = 12.3f;
  (void)hipMalloc(&OutD, sizeof(float));
  (void)hipMemcpy(OutD, &OutH, sizeof(float), hipMemcpyHostToDevice);

  readA<<<1, 1>>>(OutD, 123);

  (void)hipMemcpy(&OutH, OutD, sizeof(float), hipMemcpyDeviceToHost);

  // Global variables without initializer should be zero initialized.
  assert(OutH == 0.0f);

  return 0;
}

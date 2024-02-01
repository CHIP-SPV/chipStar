// Regression test for https://github.com/CHIP-SPV/chipStar/issues/211
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(X)                                                           \
  do {                                                                         \
    if (X != hipSuccess)                                                       \
      exit(2);                                                                 \
  } while (0)

struct Foo {
  int Bar;
  int *Data;
};

__global__ void kernel(Foo Out, const Foo In) { *Out.Data = *In.Data + In.Bar; }

int main() {
  hipDeviceProp_t Prop;
  int Device;
  HIP_CHECK(hipGetDevice(&Device));
  HIP_CHECK(hipGetDeviceProperties(&Prop, Device));
  if (!Prop.canMapHostMemory) {
    printf("SKIP: Test requires canMapHostMemory == 1\n");
    return CHIP_SKIP_TEST;
  }

  int *InH, *OutH;
  HIP_CHECK(hipHostMalloc((void **)&InH, sizeof(int), hipHostMallocMapped));
  HIP_CHECK(hipHostMalloc((void **)&OutH, sizeof(int), hipHostMallocMapped));
  *InH = 23;
  *OutH = 0;

  int *InD, *OutD;
  HIP_CHECK(hipHostGetDevicePointer((void**)&InD, InH, 0));
  HIP_CHECK(hipHostGetDevicePointer((void**)&OutD, OutH, 0));

  Foo In = {100, InD}, Out = {0, OutD};

  kernel<<<dim3(1), dim3(1)>>>(Out, In);
  HIP_CHECK(hipDeviceSynchronize());

  printf("*OutH = %d\n", *OutH);
  bool Failed = *OutH != 123;

  HIP_CHECK(hipHostFree(InH));
  HIP_CHECK(hipHostFree(OutH));
  return Failed;
}

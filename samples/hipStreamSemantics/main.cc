
#include "hip/hip_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK(cmd)                                                            \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(1);                                                                \
    }                                                                         \
  }

#define LEN 512
#define SIZE LEN << 4

#define SINCOS_N 32

#define ATOM_ADD 8192
#define ATOM_ADD_S "8192"
#define MIN_VAL 1923
#define MIN_VAL_S "1923"

int main() {
  float *Ind, *Inh;
  float *Buf1Outd, *Buf1Outh;
  float *Buf2Outd, *Buf2Outh;

  static int device = 0;
  CHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
  printf("info: running on device %s\n", props.name);

  Inh = (float *)malloc(SIZE);
  CHECK(Inh == 0 ? hipErrorMemoryAllocation : hipSuccess);
  Buf1Outh = (float *)malloc(SIZE);
  CHECK(Buf1Outh == 0 ? hipErrorMemoryAllocation : hipSuccess);
  Buf2Outh = (float *)malloc(SIZE);
  CHECK(Buf2Outh == 0 ? hipErrorMemoryAllocation : hipSuccess);

  hipMalloc((void **)&Ind, SIZE);
  hipMalloc((void **)&Buf1Outd, SIZE);
  hipMalloc((void **)&Buf2Outd, SIZE);

  printf("info: copy Host2Device\n");
  CHECK(hipMemcpy(Ind, Inh, SIZE, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(floatMath, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, Ind,
                     Buf1Outd, Buf2Outd);

  printf("info: copy Device2Host\n");
  CHECK(hipMemcpy(Inh, Ind, SIZE, hipMemcpyDeviceToHost));

  printf("info: copy Device2Host\n");
  CHECK(hipMemcpy(Buf1Outh, Buf1Outd, SIZE, hipMemcpyDeviceToHost));

  printf("info: copy Device2Host\n");
  CHECK(hipMemcpy(Buf2Outh, Buf2Outd, SIZE, hipMemcpyDeviceToHost));
}

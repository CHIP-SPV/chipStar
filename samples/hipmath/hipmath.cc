
#include "hip/hip_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(1);                                                                    \
        }                                                                                          \
    }

#define LEN 512
#define SIZE LEN << 4

#define SINCOS_N 32

#define ATOM_ADD 8192
#define ATOM_ADD_S "8192"
#define MIN_VAL 1923
#define MIN_VAL_S "1923"

__device__ __host__
unsigned bitcast_u(float j) {
  union {
    float f;
    unsigned u;
  } a;
  a.f = j;
  return a.u;
}

__device__ __host__
float bitcast_f(unsigned j) {
  union {
    float f;
    unsigned u;
  } a;
  a.u = j;
  return a.f;
}


__global__ void floatMath(float *In, float *Buf1Out, float *Buf2Out) {
  for (size_t i = 0; i < SINCOS_N; ++i)
    sincosf(In[i], Buf1Out + i, Buf2Out + i);

  Buf1Out += SINCOS_N;
  Buf2Out += SINCOS_N;
  In += SINCOS_N;

  Buf2Out[0] = frexpf(In[0], (int *)Buf1Out);
  Buf2Out[1] = remquof(In[1], 2.0f, (int *)(Buf1Out + 1));
  Buf2Out[2] = (float)atomicAdd((int *)(In + 2), ATOM_ADD);
  Buf2Out[3] = (float)atomicSub((int *)(In + 3), ATOM_ADD);
  Buf2Out[4] = (float)atomicMin((int *)(In + 4), MIN_VAL);
  Buf2Out[5] = (float)atomicMax((int *)(In + 5), MIN_VAL);
  Buf2Out[6] = acosf(In[6]);
  Buf2Out[7] = acoshf(In[7]);
  Buf2Out[8] = bitcast_f(__brev(bitcast_u(In[8])));
}



int main() {
    float *Ind, *Inh;
    float *Buf1Outd, *Buf1Outh;
    float *Buf2Outd, *Buf2Outh;

    static int device = 0;
    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    printf("info: running on device %s\n", props.name);

    Inh = (float*)malloc(SIZE);
    CHECK(Inh == 0 ? hipErrorMemoryAllocation : hipSuccess);
    Buf1Outh = (float *)malloc(SIZE);
    CHECK(Buf1Outh == 0 ? hipErrorMemoryAllocation : hipSuccess);
    Buf2Outh = (float *)malloc(SIZE);
    CHECK(Buf2Outh == 0 ? hipErrorMemoryAllocation : hipSuccess);

    // +2 for remquo + frexp
    for (size_t i = 0; i < SINCOS_N + 2; i++) {
      Inh[i] = 0.618f + ((float)i) / 7.0;
    }

    for (size_t i = SINCOS_N + 2; i < LEN; i++) {
      // 0x00a8e392 , 11068306
      Inh[i] = 1.551E-38;
    }

    hipMalloc((void**)&Ind, SIZE);
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

    float diff = 0.0f;
    float eps = 1.0E-6;

    for (size_t i = 0; i < SINCOS_N; i++) {
      // Compare to C library values
      float msin = std::sin(Inh[i]);
      float sindiff = std::fabs(Buf1Outh[i] - msin);
      float mcos = std::cos(Inh[i]);
      float cosdiff = std::fabs(Buf2Outh[i] - mcos);
      printf("IN: %1.5f Sin %1.5f Diff %1.5f Cos %1.5f Diff %1.5f\n", Inh[i],
             Buf1Outh[i], sindiff, Buf2Outh[i], cosdiff);
    }

    if (diff < (eps * SINCOS_N))
      printf("SINCOS PASSED\n");
    else
      printf("SINCOS FAILED\n");

    float *InN = Inh + SINCOS_N;
    float *Out1N = Buf1Outh + SINCOS_N;
    float *Out2N = Buf2Outh + SINCOS_N;

    //    Buf2Out[0] = frexp(In[0], (int*)Buf1Out);
    int tmp;
    float fr = std::frexp(InN[0], &tmp);
    printf("FREXP CLIB: %2.5f %2.5f %i \n"
           "FREXP DEV: %2.5f %2.5f %i \n",
           InN[0], fr, tmp, InN[0], Out2N[0], bitcast_u(Out1N[0]));

    //    Buf2Out[1] = remquo(In[1], 2.0f, (int*)(Buf1Out+1));
    float rmq = std::remquo(InN[1], 2.0f, &tmp);
    printf("REMQUO CLIB: %2.5f %2.5f %i \n"
           "REMQUO DEV: %2.5f %2.5f %i \n",
           InN[1], rmq, tmp, InN[1], Out2N[1], bitcast_u(Out1N[1]));

    //    Buf2Out[2] = (int)atomicAdd((int*)(In+2), 8);
    printf("ATOM ADD: %f + " ATOM_ADD_S " = %u \n", Out2N[2],
           bitcast_u(InN[2]));

    //    Buf2Out[3] = bitcast_uatomicSub((int*)(In+3), 8);
    printf("ATOM SUB: %f - " ATOM_ADD_S " = %u \n", Out2N[3],
           bitcast_u(InN[3]));

    //    Buf2Out[4] = bitcast_uatomicMin((int*)(In+4), 1923);
    printf("ATOM MIN: %f, " MIN_VAL_S " =  %u \n", Out2N[4], bitcast_u(InN[4]));

    //    Buf2Out[5] = bitcast_uatomicMax((int*)(In+5), 1923);
    printf("ATOM MAX: %f, " MIN_VAL_S " =  %u \n", Out2N[5], bitcast_u(InN[5]));

    printf("BREV in: %x out: %x\n", bitcast_u(InN[8]), bitcast_u(Out2N[8]) );

}


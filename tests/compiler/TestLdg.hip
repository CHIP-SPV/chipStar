// Regression test for https://github.com/CHIP-SPV/chip-spv/issues/471.
#include <hip/hip_runtime.h>

#define CHECK(TYPE_)                                                           \
  __global__ void k(TYPE_ *Out, const TYPE_ *In) { *Out = __ldg(In); }

CHECK(char);
CHECK(signed char);
CHECK(unsigned char);
CHECK(short);
CHECK(unsigned short);
CHECK(int);
CHECK(unsigned int);
CHECK(long);
CHECK(unsigned long);
CHECK(long long);
CHECK(unsigned long long);
CHECK(float);
CHECK(double);

CHECK(char2);
CHECK(char4);
CHECK(uchar2);
CHECK(uchar4);
CHECK(short2);
CHECK(short4);
CHECK(ushort2);
CHECK(ushort4);
CHECK(int2);
CHECK(int4);
CHECK(uint2);
CHECK(uint4);
// Skip (u)long# overloads they are not defined in CUDA 12.0
// Programming Manual [7.10].
CHECK(longlong2);
CHECK(ulonglong2);
CHECK(float2);
CHECK(float4);
CHECK(double2);

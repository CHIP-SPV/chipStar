/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef HIP_INCLUDE_DEVICELIB_ATOMICS
#define HIP_INCLUDE_DEVICELIB_ATOMICS

#include <hip/devicelib/macros.hh>

#define DEFOPENCL_ATOMIC2(HIPNAME, CLNAME)                                     \
  extern "C" {                                                                 \
  __device__ int __chip_atomic_##CLNAME##_i(int *address, int i);             \
  __device__ unsigned int __chip_atomic_##CLNAME##_u(unsigned int *address,   \
                                                      unsigned int ui);        \
  __device__ unsigned long long __chip_atomic_##CLNAME##_l(unsigned long long *address,        \
                                           unsigned long long ull);            \
  }                                                                            \
  EXPORT OVLD int atomic##HIPNAME(int *address, int val) {                     \
    return __chip_atomic_##CLNAME##_i(address, val);                        \
  }                                                                            \
  EXPORT OVLD unsigned int atomic##HIPNAME(unsigned int *address,              \
                                           unsigned int val) {                 \
    return __chip_atomic_##CLNAME##_u(address, val);                        \
  }                                                                            \
  EXPORT OVLD unsigned long atomic##HIPNAME(unsigned long *address,            \
                                            unsigned long val) {               \
    return __chip_atomic_##CLNAME##_l((unsigned long long *)address,        \
                                         (unsigned long long)val);             \
  }                                                                            \
  EXPORT OVLD unsigned long long atomic##HIPNAME(unsigned long long *address,  \
                                                 unsigned long long val) {     \
    return __chip_atomic_##CLNAME##_l(address, val);                        \
  }

#define DEFOPENCL_ATOMIC1(HIPNAME, CLNAME)                                     \
  extern "C" {                                                                 \
  __device__ int __chip_atomic_##CLNAME##_i(int *address);                    \
  __device__ unsigned int __chip_atomic_##CLNAME##_u(unsigned int *address);  \
  __device__ unsigned long long __chip_atomic_##CLNAME##_l(unsigned long long *address);       \
  }                                                                            \
  EXPORT OVLD int atomic##HIPNAME(int *address) {                              \
    return __chip_atomic_##CLNAME##_i(address);                             \
  }                                                                            \
  EXPORT OVLD unsigned int atomic##HIPNAME(unsigned int *address) {            \
    return __chip_atomic_##CLNAME##_u(address);                             \
  }                                                                            \
  EXPORT OVLD unsigned long long atomic##HIPNAME(                              \
      unsigned long long *address) {                                           \
    return __chip_atomic_##CLNAME##_l(address);                             \
  }

#define DEFOPENCL_ATOMIC3(HIPNAME, CLNAME)                                     \
  extern "C" {                                                                 \
  __device__ int __chip_atomic_##CLNAME##_i(int *address, int cmp, int val);  \
  __device__ unsigned int __chip_atomic_##CLNAME##_u(unsigned int *address,   \
                                                      unsigned int cmp,        \
                                                      unsigned int val);       \
  __device__ unsigned long long __chip_atomic_##CLNAME##_l(unsigned long long *address,        \
                                           unsigned long long cmp,             \
                                           unsigned long long val);            \
  }                                                                            \
  EXPORT OVLD int atomic##HIPNAME(int *address, int cmp, int val) {            \
    return __chip_atomic_##CLNAME##_i(address, cmp, val);                   \
  }                                                                            \
  EXPORT OVLD unsigned int atomic##HIPNAME(                                    \
      unsigned int *address, unsigned int cmp, unsigned int val) {             \
    return __chip_atomic_##CLNAME##_u(address, cmp, val);                   \
  }                                                                            \
  EXPORT OVLD unsigned long long atomic##HIPNAME(unsigned long long *address,  \
                                                 unsigned long long cmp,       \
                                                 unsigned long long val) {     \
    return __chip_atomic_##CLNAME##_l(address, cmp, val);                   \
  }

DEFOPENCL_ATOMIC2(Add, add);
DEFOPENCL_ATOMIC2(Sub, sub);
DEFOPENCL_ATOMIC2(Exch, xchg);
DEFOPENCL_ATOMIC2(Min, min);
DEFOPENCL_ATOMIC2(Max, max);
DEFOPENCL_ATOMIC2(And, and);
DEFOPENCL_ATOMIC2(Or, or);
DEFOPENCL_ATOMIC2(Xor, xor);

DEFOPENCL_ATOMIC1(Inc, inc);
DEFOPENCL_ATOMIC1(Dec, dec);

DEFOPENCL_ATOMIC3(CAS, cmpxchg)

extern "C" __device__ float    __chip_atomic_add_f32(float *address, float val);
extern "C" __device__ double   __chip_atomic_add_f64(double *address, double val);
extern "C" __device__ float    __chip_atomic_exch_f32(float *address, float val);
extern "C" __device__ unsigned __chip_atomic_inc2_u(unsigned *address, unsigned val);
extern "C" __device__ unsigned __chip_atomic_dec2_u(unsigned *address, unsigned val);

EXPORT float atomicAdd(float *address, float val) {
  return __chip_atomic_add_f32(address, val);
}
// Undocumented atomicAdd variant without return value.
EXPORT void atomicAddNoRet(float *address, float val) {
  (void)__chip_atomic_add_f32(address, val);
}
EXPORT double atomicAdd(double *address, double val) {
  return __chip_atomic_add_f64(address, val);
}
EXPORT float atomicExch(float *address, float val) {
  return __chip_atomic_exch_f32(address, val);
}
EXPORT unsigned atomicInc(unsigned *address, unsigned val) {
  return __chip_atomic_inc2_u(address, val);
}
EXPORT unsigned atomicDec(unsigned *address, unsigned val) {
  return __chip_atomic_dec2_u(address, val);
}

#endif // HIP_INCLUDE_DEVICELIB_ATOMICS 
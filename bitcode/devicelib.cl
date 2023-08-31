/*
 * Copyright (c) 2021-23 chipStar developers
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

#include "ROCm-Device-Libs/ocml/inc/ocml.h"
#define NON_OVLD
#define OVLD __attribute__((overloadable))
// #define AI __attribute__((always_inline))
#define EXPORT NON_OVLD

#define DEFAULT_AS __generic

#define NOOPT __attribute__((optnone))

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#ifndef __opencl_c_generic_address_space
#error __opencl_c_generic_address_space needed!
#endif

EXPORT unsigned /* long */ long int
__chip_umul64hi(unsigned /* long */ long int x,
                unsigned /* long */ long int y) {
  unsigned /* long */ long int mul =
      (unsigned /* long */ long int)x * (unsigned /* long */ long int)y;
  return (unsigned /* long */ long int)(mul >> 64);
}

EXPORT /* long */ long int __chip_mul64hi(/* long */ long int x,
                                          /* long */ long int y) {
  unsigned /* long */ long int mul =
      (unsigned /* long */ long int)x * (unsigned /* long */ long int)y;
  return (/* long */ long int)(mul >> 64);
}

EXPORT unsigned int __chip_sad(int x, int y, unsigned int z) {
  unsigned int result = 0;
  for (int i = 0; i < sizeof(int) * 8; i++) {
    int x_bit = (x >> i) & 1;
    int y_bit = (y >> i) & 1;
    unsigned int diff = abs(x_bit - y_bit);
    result += (diff << i);
  }
  return result + z;
}

EXPORT unsigned int __chip_usad(unsigned int x, unsigned int y,
                                unsigned int z) {
  unsigned int result = 0;
  for (int i = 0; i < sizeof(unsigned int) * 8; i++) {
    unsigned int x_bit = (x >> i) & 1;
    unsigned int y_bit = (y >> i) & 1;
    unsigned int diff = abs((int)x_bit - (int)y_bit);
    result += (diff << i);
  }
  return result + z;
}

// optimization tries to use llvm intrinsics here, but we don't want that
EXPORT NOOPT unsigned int __chip_brev(unsigned int a) {
  unsigned int m;
  a = (a >> 16) | (a << 16); // swap halfwords
  m = 0x00FF00FFU;
  a = ((a >> 8) & m) | ((a << 8) & ~m); // swap bytes
  m = m ^ (m << 4);
  a = ((a >> 4) & m) | ((a << 4) & ~m); // swap nibbles
  m = m ^ (m << 2);
  a = ((a >> 2) & m) | ((a << 2) & ~m);
  m = m ^ (m << 1);
  a = ((a >> 1) & m) | ((a << 1) & ~m);
  return a;
}

EXPORT NOOPT unsigned /* long */ long int
__chip_brevll(unsigned /* long */ long int a) {
  unsigned /* long */ long int m;
  a = (a >> 32) | (a << 32); // swap words
  m = 0x0000FFFF0000FFFFUL;
  a = ((a >> 16) & m) | ((a << 16) & ~m); // swap halfwords
  m = m ^ (m << 8);
  a = ((a >> 8) & m) | ((a << 8) & ~m); // swap bytes
  m = m ^ (m << 4);
  a = ((a >> 4) & m) | ((a << 4) & ~m); // swap nibbles
  m = m ^ (m << 2);
  a = ((a >> 2) & m) | ((a << 2) & ~m);
  m = m ^ (m << 1);
  a = ((a >> 1) & m) | ((a << 1) & ~m);
  return a;
}

struct ucharHolder {
  union {
    unsigned char c[4];
    unsigned int ui;
  };
};

struct uchar2Holder {
  union {
    unsigned int ui[2];
    unsigned char c[8];
  };
};

EXPORT unsigned int __chip_byte_perm(unsigned int x, unsigned int y,
                                     unsigned int s) {
  struct uchar2Holder cHoldVal;
  struct ucharHolder cHoldKey;
  struct ucharHolder cHoldOut;
  cHoldKey.ui = s;
  cHoldVal.ui[0] = x;
  cHoldVal.ui[1] = y;
  cHoldOut.c[0] = cHoldVal.c[cHoldKey.c[0]];
  cHoldOut.c[1] = cHoldVal.c[cHoldKey.c[1]];
  cHoldOut.c[2] = cHoldVal.c[cHoldKey.c[2]];
  cHoldOut.c[3] = cHoldVal.c[cHoldKey.c[3]];
  return cHoldOut.ui;
}

EXPORT unsigned int __chip_ffs(unsigned int input) {
  return (input == 0 ? -1 : ctz(input)) + 1;
}

EXPORT int __chip_ctzll(/* long */ long int x) {
  if (x == 0) {
    return sizeof(/* long */ long int) * 8;
  }
  int count = 0;
  while ((x & 1LL) == 0) {
    x >>= 1;
    count++;
  }
  return count;
}

EXPORT unsigned int __chip_ffsll(/* long */ long int input) {
  return (input == 0 ? -1 : __chip_ctzll(input)) + 1;
}

EXPORT unsigned int __lastbit_u32_u64(unsigned /* long */ long input) {
  return input == 0 ? -1 : __chip_ctzll(input);
}

EXPORT unsigned int __bitextract_u32(unsigned int src0, unsigned int src1,
                                     unsigned int src2) {
  unsigned long offset = src1 & 31;
  unsigned long width = src2 & 31;
  return width == 0 ? 0 : (src0 << (32 - offset - width)) >> (32 - width);
}

EXPORT unsigned /* long */ long __bitextract_u64(unsigned /* long */ long src0,
                                                 unsigned int src1,
                                                 unsigned int src2) {
  unsigned /* long */ long offset = src1 & 63;
  unsigned /* long */ long width = src2 & 63;
  return width == 0 ? 0 : (src0 << (64 - offset - width)) >> (64 - width);
}

EXPORT unsigned int __bitinsert_u32(unsigned int src0, unsigned int src1,
                                    unsigned int src2, unsigned int src3) {
  unsigned long offset = src2 & 31;
  unsigned long width = src3 & 31;
  unsigned long mask = (1 << width) - 1;
  return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

EXPORT unsigned int __chip_funnelshift_l(unsigned int lo, unsigned int hi,
                                         unsigned int shift) {
  unsigned /* long */ long concat = ((unsigned /* long */ long)hi << 32) | lo;
  unsigned int shifted = concat << (shift & 31);
  return shifted >> 32;
}

EXPORT unsigned int __chip_funnelshift_lc(unsigned int lo, unsigned int hi,
                                          unsigned int shift) {
  unsigned /* long */ long concat = ((unsigned /* long */ long)hi << 32) | lo;
  unsigned int shifted = concat << (shift & 31);
  unsigned int clamped_shift = shift < 32 ? shift : 32;
  return shifted >> (32 - clamped_shift);
}

EXPORT unsigned int __chip_funnelshift_r(unsigned int lo, unsigned int hi,
                                         unsigned int shift) {
  unsigned /* long */ long concat = ((unsigned /* long */ long)hi << 32) | lo;
  unsigned int shifted = concat >> (shift & 31);
  return shifted;
}

EXPORT unsigned int __chip_funnelshift_rc(unsigned int lo, unsigned int hi,
                                          unsigned int shift) {
  unsigned /* long */ long concat = ((unsigned /* long */ long)hi << 32) | lo;
  unsigned int shifted = concat >> (shift & 31);
  unsigned int clamped_shift = shift < 32 ? shift : 32;
  return shifted << (32 - clamped_shift);
}

EXPORT float __chip_saturate_f32(float x) {
  return (x < 0.0f) ? 0.0f : ((x > 1.0f) ? 1.0f : x);
}

EXPORT float __chip_jn_f32(int n, float x) {
  if (n == 0)
    return __ocml_j0_f32(x);
  if (n == 1)
    return __ocml_j1_f32(x);

  float x0 = __ocml_j0_f32(x);
  float x1 = __ocml_j1_f32(x);
  for (int i = 1; i < n; ++i) {
    float x2 = (2 * i) / x * x1 - x0;
    x0 = x1;
    x1 = x2;
  }

  return x1;
}

EXPORT double __chip_jn_f64(int n, double x) {
  if (n == 0)
    return __ocml_j0_f64(x);
  if (n == 1)
    return __ocml_j1_f64(x);

  double x0 = __ocml_j0_f64(x);
  double x1 = __ocml_j1_f64(x);
  for (int i = 1; i < n; ++i) {
    double x2 = (2 * i) / x * x1 - x0;
    x0 = x1;
    x1 = x2;
  }

  return x1;
}

EXPORT float __chip_yn_f32(int n, float x) {
  if (n == 0)
    return __ocml_y0_f32(x);
  if (n == 1)
    return __ocml_y1_f32(x);

  float x0 = __ocml_y0_f32(x);
  float x1 = __ocml_y1_f32(x);
  for (int i = 1; i < n; ++i) {
    float x2 = (2 * i) / x * x1 - x0;
    x0 = x1;
    x1 = x2;
  }

  return x1;
}

EXPORT double __chip_yn_f64(int n, double x) {
  if (n == 0)
    return __ocml_y0_f64(x);
  if (n == 1)
    return __ocml_y1_f64(x);

  double x0 = __ocml_y0_f64(x);
  double x1 = __ocml_y1_f64(x);
  for (int i = 1; i < n; ++i) {
    double x2 = (2 * i) / x * x1 - x0;
    x0 = x1;
    x1 = x2;
  }

  return x1;
}

EXPORT /* long */ long int __chip_llrint_f32(float x) {
  return (/* long */ long int)(rint(x));
}
EXPORT /* long */ long int __chip_llrint_f64(double x) {
  return (/* long */ long int)(rint(x));
}

EXPORT /* long */ long int __chip_llround_f32(float x) {
  return (/* long */ long int)(round(x));
}
EXPORT /* long */ long int __chip_llround_f64(double x) {
  return (/* long */ long int)(round(x));
}

EXPORT long int __chip_lrint_f32(float x) { return (long int)(rint(x)); }
EXPORT long int __chip_lrint_f64(double x) { return (long int)(rint(x)); }

EXPORT long int __chip_lround_f32(float x) { return (long int)(round(x)); }
EXPORT long int __chip_lround_f64(double x) { return (long int)(round(x)); }

OVLD float length(float4 f);
OVLD double length(double4 f);
EXPORT float __chip_norm4d_f32(float x, float y, float z, float w) {
  float4 temp = (float4)(x, y, z, w);
  return length(temp);
}
EXPORT double __chip_norm4d_f64(double x, double y, double z, double w) {
  double4 temp = (double4)(x, y, z, w);
  return length(temp);
}
EXPORT float __chip_norm3d_f32(float x, float y, float z) {
  float4 temp = (float4)(x, y, z, 0.0f);
  return length(temp);
}
EXPORT double __chip_norm3d_f64(double x, double y, double z) {
  double4 temp = (double4)(x, y, z, 0.0);
  return length(temp);
}

EXPORT float __chip_norm_f32(int dim, const float *a) {
  float r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return sqrt(r);
}

EXPORT double __chip_norm_f64(int dim, const double *a) {
  float r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return sqrt(r);
}

EXPORT float __chip_rnorm_f32(int dim, const float *a) {
  float r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return sqrt(r);
}

EXPORT double __chip_rnorm_f64(int dim, const double *a) {
  double r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return sqrt(r);
}

EXPORT void __chip_sincospi_f32(float x, float *sptr, float *cptr) {
  *sptr = sinpi(x);
  *cptr = cospi(x);
}

EXPORT void __chip_sincospi_f64(double x, double *sptr, double *cptr) {
  *sptr = sinpi(x);
  *cptr = cospi(x);
}

EXPORT float __chip_frexp_f32(float x, DEFAULT_AS int *i) {
  int tmp;
  float ret = frexp(x, &tmp);
  *i = tmp;
  return ret;
}
EXPORT double __chip_frexp_f64(double x, DEFAULT_AS int *i) {
  int tmp;
  double ret = frexp(x, &tmp);
  *i = tmp;
  return ret;
}

EXPORT float __chip_ldexp_f32(float x, int k) { return ldexp(x, k); }
EXPORT double __chip_ldexp_f64(double x, int k) { return ldexp(x, k); }

EXPORT float __chip_modf_f32(float x, DEFAULT_AS float *i) {
  float tmp;
  float ret = modf(x, &tmp);
  *i = tmp;
  return ret;
}
EXPORT double __chip_modf_f64(double x, DEFAULT_AS double *i) {
  double tmp;
  double ret = modf(x, &tmp);
  *i = tmp;
  return ret;
}

EXPORT float __chip_remquo_f32(float x, float y, DEFAULT_AS int *quo) {
  int tmp;
  float rem = remquo(x, y, &tmp);
  *quo = tmp;
  return rem;
}
EXPORT double __chip_remquo_f64(double x, double y, DEFAULT_AS int *quo) {
  int tmp;
  double rem = remquo(x, y, &tmp);
  *quo = tmp;
  return rem;
}

EXPORT float __chip_sincos_f32(float x, DEFAULT_AS float *cos) {
  float tmp;
  float sin = sincos(x, &tmp);
  *cos = tmp;
  return sin;
}

EXPORT double __chip_sincos_f64(double x, DEFAULT_AS double *cos) {
  double tmp;
  double sin = sincos(x, &tmp);
  *cos = tmp;
  return sin;
}

/* other */

// local_barrier
EXPORT void __chip_syncthreads() { barrier(CLK_LOCAL_MEM_FENCE); }

// local_fence
EXPORT void __chip_threadfence_block() { mem_fence(CLK_LOCAL_MEM_FENCE); }

// global_fence
EXPORT void __chip_threadfence() {
  mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

// system_fence
EXPORT void __chip_threadfence_system() {
  mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
/* memory routines */

// sets size bytes of the memory pointed to by ptr to value
// interpret ptr as a unsigned char so that it writes as bytes
EXPORT void *__chip_memset(DEFAULT_AS void *ptr, int value, size_t size) {
  volatile unsigned char *temporary = ptr;

  for (int i = 0; i < size; i++)
    temporary[i] = value;

  return ptr;
}

EXPORT void *__chip_memcpy(DEFAULT_AS void *dest, DEFAULT_AS const void *src,
                           size_t n) {
  volatile unsigned char *temporary_dest = dest;
  volatile const unsigned char *temporary_src = src;

  for (int i = 0; i < n; i++)
    temporary_dest[i] = temporary_src[i];

  return dest;
}

/**********************************************************************/

EXPORT int __chip_clz_i(int var) { return clz(var); }

EXPORT long __chip_clz_li(long var) { return clz(var); }

EXPORT int __chip_ctz_i(int var) { return ctz(var); }

EXPORT long __chip_ctz_li(long var) { return ctz(var); }

/**********************************************************************/

/* We should be able to map the _system-variations to the same
   OpenCL atomics as long as we ensure the target has fine grain
   SVM atomics supported. This should be included in the device
   returning >= 6.0 compute capability.

   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
*/

#define DEF_CHIP_ATOMIC2(NAME)                                                 \
  int OVLD atomic_##NAME(volatile __generic int *, int);                       \
  uint OVLD atomic_##NAME(volatile __generic uint *, uint);                    \
  int __chip_atomic_##NAME##_i(DEFAULT_AS int *address, int i) {               \
    return atomic_##NAME((volatile __generic int *)address, i);                \
  }                                                                            \
  uint __chip_atomic_##NAME##_u(DEFAULT_AS uint *address, uint ui) {           \
    return atomic_##NAME((volatile __generic int *)address, ui);               \
  }                                                                            \
  ulong __chip_atomic_##NAME##_l(DEFAULT_AS ulong *address, ulong ull) {       \
    return atomic_##NAME((volatile __generic int *)address, ull);              \
  }                                                                            \
  int __chip_atomic_##NAME##_system_i(DEFAULT_AS int *address, int i) {        \
    return atomic_##NAME((volatile __generic int *)address, i);                \
  }                                                                            \
  uint __chip_atomic_##NAME##_system_u(DEFAULT_AS uint *address, uint ui) {    \
    return atomic_##NAME((volatile __generic int *)address, ui);               \
  }                                                                            \
  ulong __chip_atomic_##NAME##_system_l(DEFAULT_AS ulong *address,             \
                                        ulong ull) {                           \
    return atomic_##NAME((volatile __generic int *)address, ull);              \
  }

// __chip_atomic_add_i, __chip_atomic_add_u, __chip_atomic_add_l
DEF_CHIP_ATOMIC2(add)
// __chip_atomic_sub_i, __chip_atomic_sub_u, __chip_atomic_sub_l
DEF_CHIP_ATOMIC2(sub)
// __chip_atomic_xchg_i, __chip_atomic_xchg_u, __chip_atomic_xchg_l
DEF_CHIP_ATOMIC2(xchg)
// __chip_atomic_min_i, __chip_atomic_min_u, __chip_atomic_min_l
DEF_CHIP_ATOMIC2(min)
// __chip_atomic_max_i, __chip_atomic_max_u, __chip_atomic_max_l
DEF_CHIP_ATOMIC2(max)
// __chip_atomic_and_i, __chip_atomic_and_u, __chip_atomic_and_l
DEF_CHIP_ATOMIC2(and)
// __chip_atomic_or_i, __chip_atomic_or_u, __chip_atomic_or_l
DEF_CHIP_ATOMIC2(or)
// __chip_atomic_xor_i, __chip_atomic_xor_u, __chip_atomic_xor_l
DEF_CHIP_ATOMIC2(xor)

#define DEF_CHIP_ATOMIC1(NAME)                                                 \
  int OVLD atomic_##NAME(volatile __generic int *);                            \
  uint OVLD atomic_##NAME(volatile __generic uint *);                          \
  int __chip_atomic_##NAME##_i(DEFAULT_AS int *address) {                      \
    return atomic_##NAME((volatile __generic int *)address);                   \
  }                                                                            \
  uint __chip_atomic_##NAME##_u(DEFAULT_AS uint *address) {                    \
    return atomic_##NAME((volatile __generic int *)address);                   \
  }                                                                            \
  ulong __chip_atomic_##NAME##_l(DEFAULT_AS ulong *address) {                  \
    return atomic_##NAME((volatile __generic int *)address);                   \
  }

// __chip_atomic_inc_i, __chip_atomic_inc_u, __chip_atomic_inc_l
DEF_CHIP_ATOMIC1(inc)
// __chip_atomic_dec_i, __chip_atomic_dec_u, __chip_atomic_dec_l
DEF_CHIP_ATOMIC1(dec)

#define DEF_CHIP_ATOMIC3(NAME)                                                 \
  int OVLD atomic_##NAME(volatile __generic int *, int, int);                  \
  uint OVLD atomic_##NAME(volatile __generic uint *, uint, uint);              \
  ulong OVLD atomic_##NAME(volatile __generic ulong *, ulong, ulong);          \
  int __chip_atomic_##NAME##_i(DEFAULT_AS int *address, int cmp, int val) {    \
    return atomic_##NAME((volatile __global int *)address, cmp, val);          \
  }                                                                            \
  uint __chip_atomic_##NAME##_u(DEFAULT_AS uint *address, uint cmp,            \
                                uint val) {                                    \
    return atomic_##NAME((volatile __global uint *)address, cmp, val);         \
  }                                                                            \
  ulong __chip_atomic_##NAME##_l(DEFAULT_AS ulong *address, ulong cmp,         \
                                 ulong val) {                                  \
    return atomic_##NAME((volatile __global ulong *)address, cmp, val);        \
  }                                                                            \
  int __chip_atomic_##NAME##_system_i(DEFAULT_AS int *address, int cmp,        \
                                      int val) {                               \
    return atomic_##NAME((volatile __global int *)address, cmp, val);          \
  }                                                                            \
  uint __chip_atomic_##NAME##_system_u(DEFAULT_AS uint *address, uint cmp,     \
                                       uint val) {                             \
    return atomic_##NAME((volatile __global uint *)address, cmp, val);         \
  }                                                                            \
  ulong __chip_atomic_##NAME##_system_l(DEFAULT_AS ulong *address, ulong cmp,  \
                                        ulong val) {                           \
    return atomic_##NAME((volatile __global ulong *)address, cmp, val);        \
  }

// __chip_atomic_cmpxchg_i, __chip_atomic_cmpxchg_u, __chip_atomic_cmpxchg_l
DEF_CHIP_ATOMIC3(cmpxchg)

#ifndef CHIP_EXT_FLOAT_ATOMICS

/* This code is adapted from AMD's HIP sources */
static OVLD float __chip_atomic_add_f32(volatile local float *address,
                                        float val) {
  volatile local uint *uaddr = (volatile local uint *)address;
  uint old = *uaddr;
  uint r;

  do {
    r = old;
    old = atomic_cmpxchg(uaddr, r, as_uint(val + as_float(r)));
  } while (r != old);

  return as_float(r);
}

static OVLD float __chip_atomic_add_f32(volatile global float *address,
                                        float val) {
  volatile global uint *uaddr = (volatile global uint *)address;
  uint old = *uaddr;
  uint r;

  do {
    r = old;
    old = atomic_cmpxchg(uaddr, r, as_uint(val + as_float(r)));
  } while (r != old);

  return as_float(r);
}

EXPORT float __chip_atomic_add_f32(DEFAULT_AS float *address, float val) {
  volatile global float *gi = to_global(address);
  if (gi)
    return __chip_atomic_add_f32(gi, val);
  volatile local float *li = to_local(address);
  if (li)
    return __chip_atomic_add_f32(li, val);
  return 0;
}

static OVLD double __chip_atom_add_f64(volatile global double *address,
                                       double val) {
  volatile global ulong *uaddr = (volatile global ulong *)address;
  ulong old = *uaddr;
  ulong r;

  do {
    r = old;
    old = atom_cmpxchg(uaddr, r, as_ulong(val + as_double(r)));
  } while (r != old);

  return as_double(r);
}

static OVLD double __chip_atom_add_f64(volatile local double *address,
                                       double val) {
  volatile local ulong *uaddr = (volatile local ulong *)address;
  ulong old = *uaddr;
  ulong r;

  do {
    r = old;
    old = atom_cmpxchg(uaddr, r, as_ulong(val + as_double(r)));
  } while (r != old);

  return as_double(r);
}

EXPORT double __chip_atomic_add_f64(DEFAULT_AS double *address, double val) {
  volatile global double *gi = to_global((DEFAULT_AS double *)address);
  if (gi)
    return __chip_atom_add_f64(gi, val);
  volatile local double *li = to_local((DEFAULT_AS double *)address);
  if (li)
    return __chip_atom_add_f64(li, val);
  return 0;
}

#else

/* https://registry.khronos.org/OpenCL/extensions/ext/cl_ext_float_atomics.html
 */

#if !defined(__opencl_c_ext_fp32_global_atomic_add) ||                         \
    !defined(__opencl_c_ext_fp64_global_atomic_add) ||                         \
    !defined(__opencl_c_ext_fp32_local_atomic_add) ||                          \
    !defined(__opencl_c_ext_fp64_local_atomic_add)
#error cl_ext_float_atomics needed!
#endif

float OVLD atomic_fetch_add(volatile __generic float *, float);
double OVLD atomic_fetch_add(volatile __generic double *, double);

static OVLD float __chip_atomic_add_f32(volatile local float *address,
                                        float val) {
  return atomic_fetch_add(address, val);
}

static OVLD float __chip_atomic_add_f32(volatile global float *address,
                                        float val) {
  return atomic_fetch_add(address, val);
}

EXPORT float __chip_atomic_add_f32(DEFAULT_AS float *address, float val) {
  return atomic_fetch_add(address, val);
}

static OVLD double __chip_atom_add_f64(volatile local double *address,
                                       double val) {
  return atomic_fetch_add(address, val);
}

static OVLD double __chip_atom_add_f64(volatile global double *address,
                                       double val) {
  return atomic_fetch_add(address, val);
}

EXPORT double __chip_atomic_add_f64(DEFAULT_AS double *address, double val) {
  return atomic_fetch_add(address, val);
}

#endif

EXPORT float __chip_atomic_add_system_f32(DEFAULT_AS float *address,
                                          float val) {
  return __chip_atomic_add_f32(address, val);
}

EXPORT double __chip_atomic_add_system_f64(DEFAULT_AS double *address,
                                           double val) {
  return __chip_atomic_add_f64(address, val);
}

static OVLD float __chip_atomic_exch_f32(volatile local float *address,
                                         float val) {
  return as_float(atomic_xchg((volatile local uint *)(address), as_uint(val)));
}
static OVLD float __chip_atomic_exch_f32(volatile global float *address,
                                         float val) {
  return as_float(atomic_xchg((volatile global uint *)(address), as_uint(val)));
}
EXPORT float __chip_atomic_exch_f32(DEFAULT_AS float *address, float val) {
  volatile global float *gi = to_global(address);
  if (gi)
    return __chip_atomic_exch_f32(gi, val);
  volatile local float *li = to_local(address);
  if (li)
    return __chip_atomic_exch_f32(li, val);
  return 0;
}
EXPORT float __chip_atomic_exch_system_f32(DEFAULT_AS float *address,
                                           float val) {
  return __chip_atomic_exch_f32(address, val);
}

static OVLD int __chip_atomic_exch_i(volatile local int *address, int val) {
  return as_int(atomic_xchg((volatile local uint *)(address), as_uint(val)));
}
static OVLD int __chip_atomic_exch_i(volatile global int *address, int val) {
  return as_int(atomic_xchg((volatile global uint *)(address), as_uint(val)));
}
EXPORT int __chip_atomic_exch_i(DEFAULT_AS int *address, int val) {
  volatile global int *gi = to_global(address);
  if (gi)
    return __chip_atomic_exch_i(gi, val);
  volatile local int *li = to_local(address);
  if (li)
    return __chip_atomic_exch_i(li, val);
  return 0;
}
EXPORT int __chip_atomic_exch_system_i(DEFAULT_AS int *address, int val) {
  return __chip_atomic_exch_i(address, val);
}

static OVLD unsigned int
__chip_atomic_exch_u(volatile local unsigned int *address, unsigned int val) {
  return as_uint(atomic_xchg((volatile local uint *)(address), as_uint(val)));
}
static OVLD unsigned int
__chip_atomic_exch_u(volatile global unsigned int *address, unsigned int val) {
  return as_uint(atomic_xchg((volatile global uint *)(address), as_uint(val)));
}
EXPORT unsigned int __chip_atomic_exch_u(DEFAULT_AS unsigned int *address,
                                         unsigned int val) {
  volatile global unsigned int *gi = to_global(address);
  if (gi)
    return __chip_atomic_exch_u(gi, val);
  volatile local unsigned int *li = to_local(address);
  if (li)
    return __chip_atomic_exch_u(li, val);
  return 0;
}
EXPORT unsigned int
__chip_atomic_exch_system_u(DEFAULT_AS unsigned int *address,
                            unsigned int val) {
  return __chip_atomic_exch_u(address, val);
}

static OVLD unsigned long int
__chip_atomic_exch_ul(volatile local unsigned long int *address,
                      unsigned long int val) {
  return as_ulong(
      atom_xchg((volatile local unsigned long *)(address), as_ulong(val)));
}
static OVLD unsigned long int
__chip_atomic_exch_ul(volatile global unsigned long int *address,
                      unsigned long int val) {
  return as_ulong(
      atom_xchg((volatile global unsigned long *)(address), as_ulong(val)));
}
EXPORT unsigned long int
__chip_atomic_exch_ul(DEFAULT_AS unsigned long int *address,
                      unsigned long int val) {
  volatile global unsigned long int *gi = to_global(address);
  if (gi)
    return __chip_atomic_exch_ul(gi, val);
  volatile local unsigned long int *li = to_local(address);
  if (li)
    return __chip_atomic_exch_ul(li, val);
  return 0;
}
EXPORT unsigned long int
__chip_atomic_exch_system_ul(DEFAULT_AS unsigned long int *address,
                             unsigned long int val) {
  return __chip_atomic_exch_ul(address, val);
}

EXPORT long int __chip_atomic_exch_system_l(DEFAULT_AS long int *address,
                                            long int val) {
  return __chip_atomic_exch_ul((DEFAULT_AS unsigned long int *)address, val);
}

static OVLD uint __chip_atomic_inc2_u(volatile local uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, ((r >= val) ? 0 : (r + 1)));
  } while (r != old);

  return r;
}

static OVLD uint __chip_atomic_dec2_u(volatile local uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, (((r == 0) || (r > val)) ? val : (r - 1)));
  } while (r != old);

  return r;
}

static OVLD uint __chip_atomic_inc2_u(volatile global uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, ((r >= val) ? 0 : (r + 1)));
  } while (r != old);

  return r;
}

static OVLD uint __chip_atomic_dec2_u(volatile global uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, (((r == 0) || (r > val)) ? val : (r - 1)));
  } while (r != old);

  return r;
}

EXPORT uint __chip_atomic_inc2_u(DEFAULT_AS uint *address, uint val) {
  volatile global uint *gi = to_global((DEFAULT_AS uint *)address);
  if (gi)
    return __chip_atomic_inc2_u(gi, val);
  volatile local uint *li = to_local((DEFAULT_AS uint *)address);
  if (li)
    return __chip_atomic_inc2_u(li, val);
  return 0;
}

EXPORT uint __chip_atomic_dec2_u(DEFAULT_AS uint *address, uint val) {
  volatile global uint *gi = to_global((DEFAULT_AS uint *)address);
  if (gi)
    return __chip_atomic_dec2_u(gi, val);
  volatile local uint *li = to_local((DEFAULT_AS uint *)address);
  if (li)
    return __chip_atomic_dec2_u(li, val);
  return 0;
}
/**********************************************************************/

// Use the Intel versions for now by default, since the Intel OpenCL CPU
// driver still implements only them, not the KHR versions.
#define sub_group_shuffle intel_sub_group_shuffle
#define sub_group_shuffle_xor intel_sub_group_shuffle_xor

int OVLD sub_group_shuffle(int var, uint srcLane);
float OVLD sub_group_shuffle(float var, uint srcLane);
int OVLD sub_group_shuffle_xor(int var, uint value);
float OVLD sub_group_shuffle_xor(float var, uint value);

// Compute the full warp lane id given a subwarp of size wSize and
// a "logical" lane id within it.
//
// Assumes that each subwarp behaves as a separate entity
// with a starting logical lane ID of 0.
__attribute__((always_inline)) static int warpLaneId(int subWarpLaneId,
                                                     int wSize) {
  if (wSize == DEFAULT_WARP_SIZE)
    return subWarpLaneId;
  unsigned laneId = get_sub_group_local_id();
  unsigned logicalSubWarp = laneId / wSize;
  return logicalSubWarp * wSize + subWarpLaneId;
}

#define __SHFL(T)                                                              \
  EXPORT OVLD T __shfl(T var, int srcLane, int wSize) {                        \
    int laneId = get_sub_group_local_id();                                     \
    return sub_group_shuffle(var, warpLaneId(srcLane, wSize));                 \
  }

__SHFL(int);
__SHFL(uint);
__SHFL(long);
__SHFL(ulong);
__SHFL(float);
__SHFL(double);

#define __SHFL_XOR(T)                                                          \
  EXPORT OVLD T __shfl_xor(T var, int value, int warpSizeOverride) {           \
    return sub_group_shuffle_xor(var, value);                                  \
  }

__SHFL_XOR(int);
__SHFL_XOR(uint);
__SHFL_XOR(long);
__SHFL_XOR(ulong);
__SHFL_XOR(float);
__SHFL_XOR(double);

#define __SHFL_UP(T)                                                           \
  EXPORT OVLD T __shfl_up(T var, uint delta, int wSize) {                      \
    int laneId = get_sub_group_local_id();                                     \
    int logicalSubWarp = laneId / wSize;                                       \
    int logicalSubWarpLaneId = laneId % wSize;                                 \
    int subWarpSrcId = logicalSubWarpLaneId - delta;                           \
    if (subWarpSrcId < 0)                                                      \
      subWarpSrcId = logicalSubWarpLaneId;                                     \
    return sub_group_shuffle(var, logicalSubWarp * wSize + subWarpSrcId);      \
  }

__SHFL_UP(int);
__SHFL_UP(uint);
__SHFL_UP(long);
__SHFL_UP(ulong);
__SHFL_UP(float);
__SHFL_UP(double);

#define __SHFL_DOWN(T)                                                         \
  EXPORT OVLD T __shfl_down(T var, uint delta, int wSize) {                    \
    int laneId = get_sub_group_local_id();                                     \
    int logicalSubWarp = laneId / wSize;                                       \
    int logicalSubWarpLaneId = laneId % wSize;                                 \
    int subWarpSrcId = logicalSubWarpLaneId + delta;                           \
    if (subWarpSrcId >= wSize)                                                 \
      subWarpSrcId = logicalSubWarpLaneId;                                     \
    return sub_group_shuffle(var, logicalSubWarp * wSize + subWarpSrcId);      \
  }

__SHFL_DOWN(int);
__SHFL_DOWN(uint);
__SHFL_DOWN(long);
__SHFL_DOWN(ulong);
__SHFL_DOWN(float);
__SHFL_DOWN(double);

__attribute__((overloadable)) uint4 sub_group_ballot(int predicate);
EXPORT OVLD ulong __chip_ballot(int predicate) {
#if DEFAULT_WARP_SIZE <= 32
  return sub_group_ballot(predicate).x;
#else
  return sub_group_ballot(predicate).x | (sub_group_ballot(predicate).y << 32);
#endif
}

EXPORT OVLD int __chip_all(int predicate) {
  return __chip_ballot(predicate) == ~0;
}

EXPORT OVLD int __chip_any(int predicate) {
  return __chip_ballot(predicate) != 0;
}

EXPORT OVLD unsigned __chip_lane_id() { return get_sub_group_local_id(); }

EXPORT OVLD void __chip_syncwarp() {
  // CUDA docs speaks only about "memory". It's not specifying that it would
  // only flush local memory.
  return sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
}

typedef struct {
  intptr_t image;
  intptr_t sampler;
} *hipTextureObject_t;

EXPORT float __chip_tex2D_f32(hipTextureObject_t textureObject, float x,
                              float y) {
  return read_imagef(
             __builtin_astype(textureObject->image, read_only image2d_t),
             __builtin_astype(textureObject->sampler, sampler_t),
             (float2)(x, y))
      .x;
}

EXPORT float __chip_int_as_float(int x) { return as_float(x); }
EXPORT int __chip_float_as_int(float x) { return as_int(x); }
EXPORT float __chip_uint_as_float(uint x) { return as_float(x); }
EXPORT uint __chip_float_as_uint(float x) { return as_uint(x); }
// In HIP long long is 64-bit integer. In OpenCL it's 128-bit integer.
EXPORT long __chip_double_as_longlong(double x) { return as_long(x); }
EXPORT double __chip_longlong_as_double(long int x) { return as_double(x); }

#ifdef CHIP_ENABLE_NON_COMPLIANT_DEVICELIB_CODE

// See c_to_opencl.def for details.
#define DEF_UNARY_FN_MAP(NAME_, TYPE_)                                         \
  TYPE_ MAP_PREFIX##NAME_(TYPE_ x) { return NAME_(x); }
#define DEF_BINARY_FN_MAP(NAME_, TYPE_)                                        \
  TYPE_ MAP_PREFIX##NAME_(TYPE_ x, TYPE_ y) { return NAME_(x, y); }
#include "c_to_opencl.def"
#undef UNARY_FN
#undef BINARY_FN
#endif

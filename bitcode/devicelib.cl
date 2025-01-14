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

NOOPT void* device_malloc(unsigned int size) {return (void*)0;};
NOOPT void device_free(void* ptr) {};

// Given a 32/64-bit value exec mask and an integer value base (between 0 and WAVEFRONT_SIZE),
// find the n-th (given by offset) set bit in the exec mask from the base bit, and return the bit position.
// If not found, return -1.
// In HIP long long is 64-bit integer. In OpenCL it's 128-bit integer.
EXPORT int __chip__fns64(unsigned long int mask, unsigned int base, int offset) {
  unsigned long int temp_mask = mask;
  int temp_offset = offset;

  if (offset == 0) {
    temp_mask &= (1 << base);
    temp_offset = 1;
  }
  else if (offset < 0) {
    temp_mask = __builtin_bitreverse64(mask);
    base = 63 - base;
    temp_offset = -offset;
  }

  temp_mask = temp_mask & ((~0ULL) << base);
  if (__builtin_popcountll(temp_mask) < temp_offset)
    return -1;
  int total = 0;
  for (int i = 0x20; i > 0; i >>= 1) {
    unsigned long int temp_mask_lo = temp_mask & ((1ULL << i) - 1);
    int pcnt = __builtin_popcountll(temp_mask_lo);
    if (pcnt < temp_offset) {
      temp_mask = temp_mask >> i;
      temp_offset -= pcnt;
      total += i;
    }
    else {
      temp_mask = temp_mask_lo;
    }
  }
  if (offset < 0)
    return 63 - total;
  else
    return total;
}

EXPORT int __chip__fns32(unsigned long int mask, unsigned int base, int offset) {
  unsigned long int temp_mask = mask;
  int temp_offset = offset;
  if (offset == 0) {
    temp_mask &= (1 << base);
    temp_offset = 1;
  }
  else if (offset < 0) {
    temp_mask = __builtin_bitreverse64(mask);
    base = 63 - base;
    temp_offset = -offset;
  }
  temp_mask = temp_mask & ((~0ULL) << base);
  if (__builtin_popcountll(temp_mask) < temp_offset)
    return -1;
  int total = 0;
  for (int i = 0x20; i > 0; i >>= 1) {
    unsigned long int temp_mask_lo = temp_mask & ((1ULL << i) - 1);
    int pcnt = __builtin_popcountll(temp_mask_lo);
    if (pcnt < temp_offset) {
      temp_mask = temp_mask >> i;
      temp_offset -= pcnt;
      total += i;
    }
    else {
      temp_mask = temp_mask_lo;
    }
  }
  if (offset < 0)
    return 63 - total;
  else
    return total;
}


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

EXPORT unsigned int __chip_bitinsert_u32(uint src0, uint src1, uint raw_offset,
                                         uint raw_width) {
  uint offset = raw_offset & 31u;
  uint width = raw_width & 31u;
  uint mask = (1u << width) - 1u;
  return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

EXPORT ulong __chip_bitinsert_u64(ulong src0, ulong src1, ulong raw_offset,
                                  ulong raw_width) {
  ulong offset = raw_offset & 63ul;
  ulong width = raw_width & 63ul;
  ulong mask = (1ul << width) - 1ul;
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

EXPORT int __chip_syncthreads_count(int predicate) {
  // Get thread info
  int lid = get_local_id(0);
  int sub_group_size = get_sub_group_size();
  int sub_group_id = lid / sub_group_size;
  int local_size = get_local_size(0);
    
  // First sum within each sub-group
  int sub_group_sum = sub_group_reduce_add(predicate);
    
  // Only first thread in each sub-group participates in final sum
  int my_contribution = (lid % sub_group_size == 0) ? sub_group_sum : 0;
    
  // Sum up all sub-group sums using work-group reduction
  int total = work_group_reduce_add(my_contribution);

  // All threads get the same result
  return total;
}

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
  unsigned char *temporary = ptr;

  for (int i = 0; i < size; i++)
    temporary[i] = value;

  return ptr;
}

EXPORT void *__chip_memcpy(DEFAULT_AS void *dest, DEFAULT_AS const void *src,
                           size_t n) {
  unsigned char *temporary_dest = dest;
  const unsigned char *temporary_src = src;

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

#define DEF_CHIP_ATOMIC2_ORDER_SCOPE(NAME, OP, ORDER, SCOPE)                  \
  int OVLD atomic_##OP##_explicit (volatile __generic int *, int,             \
                                   memory_order order, memory_scope scope);   \
  uint OVLD atomic_##OP##_explicit (volatile __generic uint *, uint,          \
                                    memory_order order, memory_scope scope);  \
  ulong OVLD atomic_##OP##_explicit (volatile __generic ulong *, ulong,       \
                                     memory_order order, memory_scope scope); \
  int __chip_atomic_##NAME##_i (DEFAULT_AS int *address, int i)               \
  {                                                                           \
    return atomic_##OP##_explicit ((volatile __generic int *)address, i,      \
                                   memory_order_##ORDER,                      \
                                   memory_scope_##SCOPE);                     \
  }                                                                           \
  uint __chip_atomic_##NAME##_u (DEFAULT_AS uint *address, uint ui)           \
  {                                                                           \
    return atomic_##OP##_explicit ((volatile __generic uint *)address, ui,    \
                                   memory_order_##ORDER,                      \
                                   memory_scope_##SCOPE);                     \
  }                                                                           \
  ulong __chip_atomic_##NAME##_l (DEFAULT_AS ulong *address, ulong ull)       \
  {                                                                           \
    return atomic_##OP##_explicit ((volatile __generic ulong *)address, ull,  \
                                   memory_order_##ORDER,                      \
                                   memory_scope_##SCOPE);                     \
  }

#define DEF_CHIP_ATOMIC2(NAME, OP)                                            \
  DEF_CHIP_ATOMIC2_ORDER_SCOPE (NAME, OP, relaxed, device)                    \
  DEF_CHIP_ATOMIC2_ORDER_SCOPE (NAME##_system, OP, relaxed, all_svm_devices)  \
  DEF_CHIP_ATOMIC2_ORDER_SCOPE (NAME##_block, OP, relaxed, work_group)

// __chip_atomic_add_i, __chip_atomic_add_u, __chip_atomic_add_l
DEF_CHIP_ATOMIC2 (add, fetch_add)

// __chip_atomic_sub_i, __chip_atomic_sub_u, __chip_atomic_sub_l
DEF_CHIP_ATOMIC2 (sub, fetch_sub)

// __chip_atomic_min_i, __chip_atomic_min_u, __chip_atomic_min_l
DEF_CHIP_ATOMIC2 (min, fetch_min)

// __chip_atomic_max_i, __chip_atomic_max_u, __chip_atomic_max_l
DEF_CHIP_ATOMIC2 (max, fetch_max)

// __chip_atomic_and_i, __chip_atomic_and_u, __chip_atomic_and_l
DEF_CHIP_ATOMIC2 (and, fetch_and)

// __chip_atomic_or_i, __chip_atomic_or_u, __chip_atomic_or_l
DEF_CHIP_ATOMIC2 (or, fetch_or)

// __chip_atomic_xor_i, __chip_atomic_xor_u, __chip_atomic_xor_l
DEF_CHIP_ATOMIC2 (xor, fetch_xor)

// __chip_atomic_xchg_i, __chip_atomic_xchg_u, __chip_atomic_xchg_l
DEF_CHIP_ATOMIC2 (exch, exchange)

#define DEF_CHIP_ATOMIC1_ORDER_SCOPE(NAME, OP, ORDER, SCOPE)                  \
  int OVLD atomic_##OP##_explicit (volatile __generic int *, int,             \
                                   memory_order order, memory_scope scope);   \
  uint OVLD atomic_##OP##_explicit (volatile __generic uint *, uint,          \
                                    memory_order order, memory_scope scope);  \
  ulong OVLD atomic_##OP##_explicit (volatile __generic ulong *, ulong,       \
                                     memory_order order, memory_scope scope); \
  int __chip_atomic_##NAME##_i (DEFAULT_AS int *address)                      \
  {                                                                           \
    return atomic_##OP##_explicit ((volatile __generic int *)address, 1,      \
                                   memory_order_##ORDER,                      \
                                   memory_scope_##SCOPE);                     \
  }                                                                           \
  uint __chip_atomic_##NAME##_u (DEFAULT_AS uint *address)                    \
  {                                                                           \
    return atomic_##OP##_explicit ((volatile __generic uint *)address, 1,     \
                                   memory_order_##ORDER,                      \
                                   memory_scope_##SCOPE);                     \
  }                                                                           \
  ulong __chip_atomic_##NAME##_l (DEFAULT_AS ulong *address)                  \
  {                                                                           \
    return atomic_##OP##_explicit ((volatile __generic ulong *)address, 1,    \
                                   memory_order_##ORDER,                      \
                                   memory_scope_##SCOPE);                     \
  }

// __chip_atomic_inc_i, __chip_atomic_inc_u, __chip_atomic_inc_l
DEF_CHIP_ATOMIC1_ORDER_SCOPE (inc, fetch_add, relaxed, device)

DEF_CHIP_ATOMIC1_ORDER_SCOPE (inc_system, fetch_add, relaxed, all_svm_devices)

DEF_CHIP_ATOMIC1_ORDER_SCOPE (inc_block, fetch_add, relaxed, work_group)

// __chip_atomic_dec_i, __chip_atomic_dec_u, __chip_atomic_dec_l
DEF_CHIP_ATOMIC1_ORDER_SCOPE (dec, fetch_sub, relaxed, device)

DEF_CHIP_ATOMIC1_ORDER_SCOPE (dec_system, fetch_sub, relaxed, all_svm_devices)

DEF_CHIP_ATOMIC1_ORDER_SCOPE (dec_block, fetch_sub, relaxed, work_group)

#define DEF_CHIP_ATOMIC3_ORDER_SCOPE(NAME, OP, ORDER, SCOPE)                  \
  int __chip_atomic_##NAME##_i (DEFAULT_AS int *address, int cmp, int val)    \
  {                                                                           \
    atomic_##OP##_explicit ((volatile __generic atomic_int *)address,         \
                            (__generic int *)&cmp, val, memory_order_##ORDER, \
                            memory_order_##ORDER, memory_scope_##SCOPE);      \
    return cmp;                                                               \
  }                                                                           \
  uint __chip_atomic_##NAME##_u (DEFAULT_AS uint *address, uint cmp,          \
                                 uint val)                                    \
  {                                                                           \
    atomic_##OP##_explicit ((volatile __generic atomic_uint *)address,        \
                            (__generic uint *)&cmp, val,                      \
                            memory_order_##ORDER, memory_order_##ORDER,       \
                            memory_scope_##SCOPE);                            \
    return cmp;                                                               \
  }                                                                           \
  ulong __chip_atomic_##NAME##_l (DEFAULT_AS ulong *address, ulong cmp,       \
                                  ulong val)                                  \
  {                                                                           \
    atomic_##OP##_explicit ((volatile __generic atomic_ulong *)address,       \
                            (__generic ulong *)&cmp, val,                     \
                            memory_order_##ORDER, memory_order_##ORDER,       \
                            memory_scope_##SCOPE);                            \
    return cmp;                                                               \
  }

// __chip_atomic_cmpxchg_i, __chip_atomic_cmpxchg_u, __chip_atomic_cmpxchg_l
DEF_CHIP_ATOMIC3_ORDER_SCOPE (cmpxchg, compare_exchange_strong, relaxed,
                              device)

DEF_CHIP_ATOMIC3_ORDER_SCOPE (cmpxchg_system, compare_exchange_strong, relaxed,
                              all_svm_devices)

DEF_CHIP_ATOMIC3_ORDER_SCOPE (cmpxchg_block, compare_exchange_strong, relaxed,
                              work_group)

/**************************************************************************************/
/**************************************************************************************/
/**************************************************************************************/
/**************************************************************************************/

#define DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME, OP, ORDER, SCOPE)                  \
  float OVLD atomic_##OP##_explicit(volatile __generic float *, float,         \
                                    memory_order order, memory_scope scope);   \
  double OVLD atomic_##OP##_explicit(volatile __generic double *, double,      \
                                     memory_order order, memory_scope scope);  \
  EXPORT float __chip_atomic_##NAME##_f32(DEFAULT_AS float *address,           \
                                          float i) {                           \
    return atomic_##OP##_explicit((volatile __generic float *)address, i,      \
                                  memory_order_##ORDER, memory_scope_##SCOPE); \
  }                                                                            \
  EXPORT double __chip_atomic_##NAME##_f64(DEFAULT_AS double *address,         \
                                           double ui) {                        \
    return atomic_##OP##_explicit((volatile __generic double *)address, ui,    \
                                  memory_order_##ORDER, memory_scope_##SCOPE); \
  }

#define DEF_CHIP_ATOMIC2F(NAME, OP)                                            \
  DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME, OP, relaxed, device)                     \
  DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME##_system, OP, relaxed, all_svm_devices)   \
  DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME##_block, OP, relaxed, work_group)

DEF_CHIP_ATOMIC2F(exch, exchange)

/**************************************************************************************/
/**************************************************************************************/
/**************************************************************************************/
/**************************************************************************************/

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

#define __SHFL_SYNC(T)                                                         \
  EXPORT OVLD T __shfl_sync(unsigned mask, T var, int srcLane, int width) {    \
    if (mask == 0) {                                                           \
      return 0;                                                                \
    } else if (mask == 0xFFFFFFFF) {                                           \
      return __shfl(var, srcLane, width);                                      \
    } else {                                                                   \
      if (get_sub_group_local_id() == 0) {                                     \
        printf("warning: Partial mask in __shfl_sync is not fully supported\n");\
      }                                                                        \
      return __shfl(var, srcLane, width);                                      \
    }                                                                          \
  }

#define __SHFL_UP_SYNC(T)                                                      \
  EXPORT OVLD T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width) { \
    if (mask == 0) {                                                           \
      return 0;                                                                \
    } else if (mask == 0xFFFFFFFF) {                                           \
      return __shfl_up(var, delta, width);                                     \
    } else {                                                                   \
      if (get_sub_group_local_id() == 0) {                                     \
        printf("warning: Partial mask in __shfl_up_sync is not fully supported\n");\
      }                                                                        \
      return __shfl_up(var, delta, width);                                     \
    }                                                                          \
  }

#define __SHFL_DOWN_SYNC(T)                                                    \
  EXPORT OVLD T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width) { \
    if (mask == 0) {                                                           \
      return 0;                                                                \
    } else if (mask == 0xFFFFFFFF) {                                           \
      return __shfl_down(var, delta, width);                                   \
    } else {                                                                   \
      if (get_sub_group_local_id() == 0) {                                     \
        printf("warning: Partial mask in __shfl_down_sync is not fully supported\n");\
      }                                                                        \
      return __shfl_down(var, delta, width);                                   \
    }                                                                          \
  }

#define __SHFL_XOR_SYNC(T)                                                     \
  EXPORT OVLD T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width) { \
    if (mask == 0) {                                                           \
      return 0;                                                                \
    } else if (mask == 0xFFFFFFFF) {                                           \
      return __shfl_xor(var, laneMask, width);                                 \
    } else {                                                                   \
      if (get_sub_group_local_id() == 0) {                                     \
        printf("warning: Partial mask in __shfl_xor_sync is not fully supported\n");\
      }                                                                        \
      return __shfl_xor(var, laneMask, width);                                 \
    }                                                                          \
  }

__SHFL_SYNC(int);
__SHFL_SYNC(uint);
__SHFL_SYNC(long);
__SHFL_SYNC(ulong);
__SHFL_SYNC(float);
__SHFL_SYNC(double);

__SHFL_UP_SYNC(int);
__SHFL_UP_SYNC(uint);
__SHFL_UP_SYNC(long);
__SHFL_UP_SYNC(ulong);
__SHFL_UP_SYNC(float);
__SHFL_UP_SYNC(double);

__SHFL_DOWN_SYNC(int);
__SHFL_DOWN_SYNC(uint);
__SHFL_DOWN_SYNC(long);
__SHFL_DOWN_SYNC(ulong);
__SHFL_DOWN_SYNC(float);
__SHFL_DOWN_SYNC(double);

__SHFL_XOR_SYNC(int);
__SHFL_XOR_SYNC(uint);
__SHFL_XOR_SYNC(long);
__SHFL_XOR_SYNC(ulong);
__SHFL_XOR_SYNC(float);
__SHFL_XOR_SYNC(double);


// The definition is linked at runtime from one of the ballot*.cl files.
EXPORT OVLD ulong __chip_ballot(int predicate);

EXPORT OVLD int __chip_all(int predicate) {
  return __chip_ballot(predicate) == ~0;
}

EXPORT OVLD int __chip_any(int predicate) {
  return __chip_ballot(predicate) != 0;
}

EXPORT OVLD unsigned __chip_ballot_sync(unsigned mask, int predicate) {
  if (mask == 0) {
    return 0;
  } else if (mask == 0xFFFFFFFF) {
    return __chip_ballot(predicate);
  } else {
    if (get_sub_group_local_id() == 0) {
      printf("warning: Partial mask in __ballot_sync is not fully supported\n");
    }
    return __chip_ballot(predicate) & mask;
  }
}

EXPORT OVLD int __chip_any_sync(unsigned mask, int predicate) {
  if (mask == 0) {
    return 0;
  } else if (mask == 0xFFFFFFFF) {
    return __chip_any(predicate);
  } else {
    unsigned ballot = __chip_ballot(predicate) & mask;
    return ballot != 0;
  }
}

EXPORT OVLD int __chip_all_sync(unsigned mask, int predicate) {
  if (mask == 0) {
    return 1;
  } else if (mask == 0xFFFFFFFF) {
    return __chip_all(predicate);
  } else {
    unsigned ballot = __chip_ballot(predicate);
    return (ballot & mask) == mask;
  }
}


EXPORT OVLD unsigned __chip_lane_id() { return get_sub_group_local_id(); }

EXPORT OVLD void __chip_syncwarp() {
  // CUDA docs speaks only about "memory". It's not specifying that it would
  // only flush local memory.
  return sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
}

// See c_to_opencl.def for details.
#define DEF_UNARY_FN_MAP(FROM_FN_, TO_FN_, TYPE_)                              \
  TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x) { return TO_FN_(x); }
#define DEF_BINARY_FN_MAP(FROM_FN_, TO_FN_, TYPE_)                             \
  TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_ x, TYPE_ y) { return TO_FN_(x, y); }
#include "c_to_opencl.def"
#undef UNARY_FN
#undef BINARY_FN


typedef unsigned long __chip_obfuscated_ptr_t;

// Helper union for double-uint2 conversion (if not already defined)
union double_uint2 {
    double d;
    uint2 u2;
};

// Atomic max for double
EXPORT double __chip_atomic_max_f64(__chip_obfuscated_ptr_t address, double val) {
  volatile global double *gi = to_global((DEFAULT_AS double *)address);
  if (gi) {
    double old = *gi;
    double assumed;
    do {
      assumed = old;
      if (val > assumed) {
        union double_uint2 old_u, assumed_u, val_u;
        old_u.d = old;
        assumed_u.d = assumed;
        val_u.d = val;
        
        uint2 temp;
        temp.x = atomic_cmpxchg((volatile global uint *)gi,
                               assumed_u.u2.x,
                               val_u.u2.x);
        temp.y = atomic_cmpxchg((volatile global uint *)((global uint *)gi + 1),
                               assumed_u.u2.y,
                               val_u.u2.y);
        
        union double_uint2 result;
        result.u2 = temp;
        old = result.d;
      } else {
        break;
      }
    } while (assumed != old);
    return old;
  }
  volatile local double *li = to_local((DEFAULT_AS double *)address);
  if (li) {
    double old = *li;
    double assumed;
    do {
      assumed = old;
      if (val > assumed) {
        union double_uint2 old_u, assumed_u, val_u;
        old_u.d = old;
        assumed_u.d = assumed;
        val_u.d = val;
        
        uint2 temp;
        temp.x = atomic_cmpxchg((volatile local uint *)li,
                               assumed_u.u2.x,
                               val_u.u2.x);
        temp.y = atomic_cmpxchg((volatile local uint *)((local uint *)li + 1),
                               assumed_u.u2.y,
                               val_u.u2.y);
        
        union double_uint2 result;
        result.u2 = temp;
        old = result.d;
      } else {
        break;
      }
    } while (assumed != old);
    return old;
  }
  return 0;
}

// Atomic min for double
EXPORT double __chip_atomic_min_f64(__chip_obfuscated_ptr_t address, double val) {
  volatile global double *gi = to_global((DEFAULT_AS double *)address);
  if (gi) {
    double old = *gi;
    double assumed;
    do {
      assumed = old;
      if (val < assumed) {
        union double_uint2 old_u, assumed_u, val_u;
        old_u.d = old;
        assumed_u.d = assumed;
        val_u.d = val;
        
        uint2 temp;
        temp.x = atomic_cmpxchg((volatile global uint *)gi,
                               assumed_u.u2.x,
                               val_u.u2.x);
        temp.y = atomic_cmpxchg((volatile global uint *)((global uint *)gi + 1),
                               assumed_u.u2.y,
                               val_u.u2.y);
        
        union double_uint2 result;
        result.u2 = temp;
        old = result.d;
      } else {
        break;
      }
    } while (assumed != old);
    return old;
  }
  volatile local double *li = to_local((DEFAULT_AS double *)address);
  if (li) {
    double old = *li;
    double assumed;
    do {
      assumed = old;
      if (val < assumed) {
        union double_uint2 old_u, assumed_u, val_u;
        old_u.d = old;
        assumed_u.d = assumed;
        val_u.d = val;
        
        uint2 temp;
        temp.x = atomic_cmpxchg((volatile local uint *)li,
                               assumed_u.u2.x,
                               val_u.u2.x);
        temp.y = atomic_cmpxchg((volatile local uint *)((local uint *)li + 1),
                               assumed_u.u2.y,
                               val_u.u2.y);
        
        union double_uint2 result;
        result.u2 = temp;
        old = result.d;
      } else {
        break;
      }
    } while (assumed != old);
    return old;
  }
  return 0;
}

// Helper union for float-uint conversion
union float_uint {
    float f;
    uint u;
};

// Atomic max for float
EXPORT float __chip_atomic_max_f32(__chip_obfuscated_ptr_t address, float val) {
  volatile global float *gi = to_global((DEFAULT_AS float *)address);
  if (gi) {
    float old = *gi;
    float assumed;
    do {
      assumed = old;
      if (val > assumed) {
        union float_uint old_u, assumed_u, val_u;
        old_u.f = old;
        assumed_u.f = assumed;
        val_u.f = val;
        
        uint temp = atomic_cmpxchg((volatile global uint *)gi,
                                 assumed_u.u,
                                 val_u.u);
        
        union float_uint result;
        result.u = temp;
        old = result.f;
      } else {
        break;
      }
    } while (assumed != old);
    return old;
  }
  volatile local float *li = to_local((DEFAULT_AS float *)address);
  if (li) {
    float old = *li;
    float assumed;
    do {
      assumed = old;
      if (val > assumed) {
        union float_uint old_u, assumed_u, val_u;
        old_u.f = old;
        assumed_u.f = assumed;
        val_u.f = val;
        
        uint temp = atomic_cmpxchg((volatile local uint *)li,
                                 assumed_u.u,
                                 val_u.u);
        
        union float_uint result;
        result.u = temp;
        old = result.f;
      } else {
        break;
      }
    } while (assumed != old);
    return old;
  }
  return 0;
}

// Atomic min for float
EXPORT float __chip_atomic_min_f32(__chip_obfuscated_ptr_t address, float val) {
  volatile global float *gi = to_global((DEFAULT_AS float *)address);
  if (gi) {
    float old = *gi;
    float assumed;
    do {
      assumed = old;
      if (val < assumed) {
        union float_uint old_u, assumed_u, val_u;
        old_u.f = old;
        assumed_u.f = assumed;
        val_u.f = val;
        
        uint temp = atomic_cmpxchg((volatile global uint *)gi,
                                 assumed_u.u,
                                 val_u.u);
        
        union float_uint result;
        result.u = temp;
        old = result.f;
      } else {
        break;
      }
    } while (assumed != old);
    return old;
  }
  volatile local float *li = to_local((DEFAULT_AS float *)address);
  if (li) {
    float old = *li;
    float assumed;
    do {
      assumed = old;
      if (val < assumed) {
        union float_uint old_u, assumed_u, val_u;
        old_u.f = old;
        assumed_u.f = assumed;
        val_u.f = val;
        
        uint temp = atomic_cmpxchg((volatile local uint *)li,
                                 assumed_u.u,
                                 val_u.u);
        
        union float_uint result;
        result.u = temp;
        old = result.f;
      } else {
        break;
      }
    } while (assumed != old);
    return old;
  }
  return 0;
}

// Returns the high 32 bits of a double as an integer
EXPORT int __chip_double2hiint(double x) {
  union {
    double d;
    ulong i;
  } bits;
  bits.d = x;
  return (int)(bits.i >> 32); // Always gets the high 32 bits regardless of endianness
}
EXPORT void __builtin_amdgcn_wave_barrier() {
  barrier(CLK_LOCAL_MEM_FENCE);
}

EXPORT void __builtin_amdgcn_fence(int scope, const char* order) {
  // Implement memory fence using OpenCL mem_fence()
  if (scope == 1) { // System scope
    mem_fence(CLK_GLOBAL_MEM_FENCE);
  }
  else if (scope == 2) { // Device scope  
    mem_fence(CLK_LOCAL_MEM_FENCE);
  }
  // Other scopes map to no-op
}

EXPORT int __builtin_amdgcn_ds_bpermute(/*__local int* lmem,*/ int byte_offset, int src_data) {
    // Write source data to local memory at this thread's position
    int lid = get_local_id(0);
    extern __local int* lmem;
    lmem[lid] = src_data;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Convert byte offset to lane index (divide by 4 since we're dealing with ints)
    int target_lane = (byte_offset >> 2) & 63;  // 63 is wavefront size - 1
    
    // Read from the target lane
    return lmem[target_lane];
}


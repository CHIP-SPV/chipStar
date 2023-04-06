/*
 * This file provides math library prototypes for HIP device code,
 * which indirectly call OpenCL math library.
 * The reasons we can't directly call OpenCL here are
 * 1) This file is compiled in C++ mode, which results in different mangling
 *    than files compiled in OpenCL mode
 * 2) some functions have the same name in HIP as in OpenCL but different
 *    signature
 * 3) some OpenCL functions (e.g. geometric) take vector arguments
 *    but HIP/CUDA do not have vectors.
 *
 * the counterpart to this file, compiled in OpenCL mode, is devicelib.cl
 *
 * portions copyright:
 *
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
 */

#ifndef HIP_INCLUDE_HIP_SPIRV_MATHLIB_H
#define HIP_INCLUDE_HIP_SPIRV_MATHLIB_H

#include <hip/devicelib/sync_and_util.hh>
#include <hip/devicelib/type_casting_intrinsics.hh>

#include <hip/devicelib/bfloat16/bfloat162_math.hh>
#include <hip/devicelib/bfloat16/bfloat16_comparison.hh>
#include <hip/devicelib/bfloat16/bfloat16_math.hh>
#include <hip/devicelib/bfloat16/bfloat162_comparison.hh>
#include <hip/devicelib/bfloat16/bfloat16_arithemtic.hh>
#include <hip/devicelib/bfloat16/bfloat16_conversion_and_movement.hh>

#include <hip/devicelib/half/half2_arithmetic.hh>
#include <hip/devicelib/half/half2_math.hh>
#include <hip/devicelib/half/half_comparison.hh>
#include <hip/devicelib/half/half_math.hh>
#include <hip/devicelib/half/half2_comparison.hh>
#include <hip/devicelib/half/half_arithmetic.hh>
#include <hip/devicelib/half/half_conversion_and_movement.hh>

#include <hip/devicelib/double_precision/dp_intrinsics.hh>
#include <hip/devicelib/double_precision/dp_math.hh>

#include <hip/devicelib/single_precision/sp_intrinsics.hh>
#include <hip/devicelib/single_precision/sp_math.hh>

#include <hip/devicelib/integer/int_intrinsics.hh>
#include <hip/devicelib/integer/int_math.hh>

EXPORT unsigned int __funnelshift_l(unsigned int lo, unsigned int hi,
                                    unsigned int shift);
EXPORT unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi,
                                     unsigned int shift);
EXPORT unsigned int __funnelshift_r(unsigned int lo, unsigned int hi,
                                    unsigned int shift);
EXPORT unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi,
                                     unsigned int shift);



/**********************************************************************/

#if defined(__HIP_DEVICE_COMPILE__)

// EXPORT void __sincosf(float x, float *sptr, float *cptr) {
//   *sptr = sin(x);
//   *cptr = cos(x);
// }

/**********************************************************************/

extern "C" {
NON_OVLD void GEN_NAME(local_barrier)();
NON_OVLD int GEN_NAME(group_all)(int predicate);
NON_OVLD int GEN_NAME(group_any)(int predicate);
NON_OVLD ulong GEN_NAME(group_ballot)(int predicate);
}

unsigned __activemask()
    __attribute__((unavailable("unsupported in CHIP-SPV.")));

// memory routines

/**********************************************************************/

#else
// EXPORT void __sincosf(float x, float *sptr, float *cptr);

EXPORT unsigned __activemask()
    __attribute__((unavailable("unsupported in CHIP-SPV.")));

#endif

EXPORT
uint64_t __make_mantissa_base8(const char *tagp) {
  uint64_t r = 0;
  while (tagp) {
    char tmp = *tagp;

    if (tmp >= '0' && tmp <= '7')
      r = (r * 8u) + tmp - '0';
    else
      return 0;

    ++tagp;
  }

  return r;
}

EXPORT
uint64_t __make_mantissa_base10(const char *tagp) {
  uint64_t r = 0;
  while (tagp) {
    char tmp = *tagp;

    if (tmp >= '0' && tmp <= '9')
      r = (r * 10u) + tmp - '0';
    else
      return 0;

    ++tagp;
  }

  return r;
}

EXPORT
uint64_t __make_mantissa_base16(const char *tagp) {
  uint64_t r = 0;
  while (tagp) {
    char tmp = *tagp;

    if (tmp >= '0' && tmp <= '9')
      r = (r * 16u) + tmp - '0';
    else if (tmp >= 'a' && tmp <= 'f')
      r = (r * 16u) + tmp - 'a' + 10;
    else if (tmp >= 'A' && tmp <= 'F')
      r = (r * 16u) + tmp - 'A' + 10;
    else
      return 0;

    ++tagp;
  }

  return r;
}

EXPORT
uint64_t __make_mantissa(const char *tagp) {
  if (!tagp)
    return 0u;

  if (*tagp == '0') {
    ++tagp;

    if (*tagp == 'x' || *tagp == 'X')
      return __make_mantissa_base16(tagp);
    else
      return __make_mantissa_base8(tagp);
  }

  return __make_mantissa_base10(tagp);
}

// EXPORT
// float nanf(const char *tagp) {
//   union {
//     float val;
//     struct ieee_float {
//       uint32_t mantissa : 22;
//       uint32_t quiet : 1;
//       uint32_t exponent : 8;
//       uint32_t sign : 1;
//     } bits;

//     static_assert(sizeof(float) == sizeof(ieee_float), "");
//   } tmp;

//   tmp.bits.sign = 0u;
//   tmp.bits.exponent = ~0u;
//   tmp.bits.quiet = 1u;
//   tmp.bits.mantissa = __make_mantissa(tagp);

//   return tmp.val;
// }

// EXPORT
// double nan(const char *tagp) {
//   union {
//     double val;
//     struct ieee_double {
//       uint64_t mantissa : 51;
//       uint32_t quiet : 1;
//       uint32_t exponent : 11;
//       uint32_t sign : 1;
//     } bits;
//     static_assert(sizeof(double) == sizeof(ieee_double), "");
//   } tmp;

//   tmp.bits.sign = 0u;
//   tmp.bits.exponent = ~0u;
//   tmp.bits.quiet = 1u;
//   tmp.bits.mantissa = __make_mantissa(tagp);

//   return tmp.val;
// }

/**********************************************************************/

// integer intrinsic function __poc __clz __ffs __brev

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
NON_OVLD unsigned GEN_NAME2(popcount, ui)(unsigned var);
NON_OVLD unsigned long GEN_NAME2(popcount, ul)(unsigned long var);
NON_OVLD int GEN_NAME2(clz, i)(int var);
NON_OVLD long int GEN_NAME2(clz, li)(long int var);
NON_OVLD int GEN_NAME2(ctz, i)(int var);
NON_OVLD long int GEN_NAME2(ctz, li)(long int var);
}

EXPORT unsigned __popc(unsigned input) {
  return GEN_NAME2(popcount, ui)(input);
}
EXPORT unsigned __popcll(unsigned long int input) {
  return (unsigned)GEN_NAME2(popcount, ul)(input);
}
EXPORT int __clz(int input) { return GEN_NAME2(clz, i)(input); }
EXPORT int __clzll(long int input) { return (int)GEN_NAME2(clz, li)(input); }
EXPORT int __ctz(int input) { return GEN_NAME2(ctz, i)(input); }
EXPORT int __ctzll(long int input) { return (int)GEN_NAME2(ctz, li)(input); }
#else
EXPORT unsigned __popc(unsigned input);
EXPORT unsigned __popcll(unsigned long int input);
EXPORT int __clz(int input);
EXPORT int __clzll(long int input);
EXPORT int __ctz(int input);
EXPORT int __ctzll(long int input);
#endif

EXPORT unsigned int __ffs(unsigned int input) {
  return (input == 0 ? -1 : __ctz(input)) + 1;
}

EXPORT unsigned int __ffsll(unsigned long long int input) {
  return (input == 0 ? -1 : __ctzll(input)) + 1;
}

EXPORT unsigned int __ffs(int input) {
  return (input == 0 ? -1 : __ctz(input)) + 1;
}

EXPORT unsigned int __ffsll(long long int input) {
  return (input == 0 ? -1 : __ctzll(input)) + 1;
}

// optimization tries to use llvm intrinsics here, but we don't want that
EXPORT NOOPT unsigned int __brev(unsigned int a) {
  uint32_t m;
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

EXPORT NOOPT unsigned long long int __brevll(unsigned long long int a) {
  uint64_t m;
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

EXPORT unsigned int __lastbit_u32_u64(uint64_t input) {
  return input == 0 ? -1 : __ctzll(input);
}

EXPORT unsigned int __bitextract_u32(unsigned int src0, unsigned int src1,
                                     unsigned int src2) {
  uint32_t offset = src1 & 31;
  uint32_t width = src2 & 31;
  return width == 0 ? 0 : (src0 << (32 - offset - width)) >> (32 - width);
}

EXPORT uint64_t __bitextract_u64(uint64_t src0, unsigned int src1,
                                 unsigned int src2) {
  uint64_t offset = src1 & 63;
  uint64_t width = src2 & 63;
  return width == 0 ? 0 : (src0 << (64 - offset - width)) >> (64 - width);
}

EXPORT unsigned int __bitinsert_u32(unsigned int src0, unsigned int src1,
                                    unsigned int src2, unsigned int src3) {
  uint32_t offset = src2 & 31;
  uint32_t width = src3 & 31;
  uint32_t mask = (1 << width) - 1;
  return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

#if !defined(__HIP_DEVICE_COMPILE__)
EXPORT unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);

EXPORT unsigned int __hadd(int x, int y);
EXPORT int __rhadd(int x, int y);

EXPORT int __mul24(int x, int y);
EXPORT long long int __mul64hi(long long int x, long long int y);
EXPORT int __mulhi(int x, int y);

EXPORT int __umul24(unsigned int x, unsigned int y);
EXPORT unsigned long long int __umul64hi(unsigned long long int x,
                                         unsigned long long int y);
EXPORT unsigned int __umulhi(unsigned int x, unsigned int y);

EXPORT unsigned int __sad(int x, int y, int z);
EXPORT unsigned int __usad(unsigned int x, unsigned int y, unsigned int z);

EXPORT unsigned int __uhadd(unsigned int x, unsigned int y);
EXPORT unsigned int __urhadd(unsigned int x, unsigned int y);
#else

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

EXPORT unsigned int __byte_perm(unsigned int x, unsigned int y,
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

extern "C" {
NON_OVLD int GEN_NAME2(hadd, i)(int x, int y);
NON_OVLD int GEN_NAME2(rhadd, i)(int x, int y);
NON_OVLD unsigned GEN_NAME2(uhadd, ui)(unsigned x, unsigned y);
NON_OVLD unsigned GEN_NAME2(urhadd, ui)(unsigned x, unsigned y);
}

EXPORT int __hadd(int x, int y) { return GEN_NAME2(hadd, i)(x, y); }
EXPORT int __rhadd(int x, int y) { return GEN_NAME2(rhadd, i)(x, y); }
EXPORT unsigned __uhadd(unsigned x, unsigned y) {
  return GEN_NAME2(uhadd, ui)(x, y);
}
EXPORT unsigned __urhadd(unsigned x, unsigned y) {
  return GEN_NAME2(urhadd, ui)(x, y);
}

extern "C" {
NON_OVLD int GEN_NAME2(mul24, i)(int x, int y);
NON_OVLD int GEN_NAME2(mulhi, i)(int x, int y);
NON_OVLD long int GEN_NAME2(mul64hi, li)(long int x, long int y);
}

EXPORT int __mul24(int x, int y) { return GEN_NAME2(mul24, i)(x, y); }
EXPORT long long __mul64hi(long int x, long int y) {
  return GEN_NAME2(mul64hi, li)(x, y);
}
EXPORT int __mulhi(int x, int y) { return GEN_NAME2(mulhi, i)(x, y); }

extern "C" {
NON_OVLD unsigned GEN_NAME2(umul24, ui)(unsigned x, unsigned y);
NON_OVLD unsigned GEN_NAME2(umulhi, ui)(unsigned x, unsigned y);
NON_OVLD unsigned long GEN_NAME2(umul64hi, uli)(unsigned long x,
                                                unsigned long y);
}

EXPORT unsigned __umul24(unsigned x, unsigned y) {
  return GEN_NAME2(umul24, ui)(x, y);
}
EXPORT unsigned long __umul64hi(unsigned long x, unsigned long y) {
  return GEN_NAME2(umul64hi, uli)(x, y);
}
EXPORT unsigned __umulhi(unsigned x, unsigned y) {
  return GEN_NAME2(umulhi, ui)(x, y);
}

EXPORT unsigned int __sad(int x, int y, int z) {
  return x > y ? x - y + z : y - x + z;
}
EXPORT unsigned int __usad(unsigned int x, unsigned int y, unsigned int z) {
  return x > y ? x - y + z : y - x + z;
}

#endif

/**********************************************************************/

#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif

// TODO
// EXPORT api_half fma(api_half x, api_half y, api_half z) {
//   // TODO
//   // return fma_h(x, y, z);
// }

#pragma push_macro("__DEF_FLOAT_FUN")
#pragma push_macro("__DEF_FLOAT_FUN2")
#pragma push_macro("__HIP_OVERLOAD")
#pragma push_macro("__HIP_OVERLOAD2")

// __hip_enable_if::type is a type function which returns __T if __B is true.
template <bool __B, class __T = void> struct __hip_enable_if {};

template <class __T> struct __hip_enable_if<true, __T> { typedef __T type; };

// __HIP_OVERLOAD1 is used to resolve function calls with integer argument to
// avoid compilation error due to ambibuity. e.g. floor(5) is resolved with
// floor(double).
#define __HIP_OVERLOAD1(__retty, __fn)                                         \
  template <typename __T>                                                      \
  __DEVICE__ typename __hip_enable_if<std::numeric_limits<__T>::is_integer,    \
                                      __retty>::type                           \
  __fn(__T __x) {                                                              \
    return ::__fn((double)__x);                                                \
  }

// __HIP_OVERLOAD2 is used to resolve function calls with mixed float/double
// or integer argument to avoid compilation error due to ambibuity. e.g.
// max(5.0f, 6.0) is resolved with max(double, double).
#define __HIP_OVERLOAD2(__retty, __fn)                                         \
  template <typename __T1, typename __T2>                                      \
  __DEVICE__                                                                   \
      typename __hip_enable_if<std::numeric_limits<__T1>::is_specialized &&    \
                                   std::numeric_limits<__T2>::is_specialized,  \
                               __retty>::type                                  \
      __fn(__T1 __x, __T2 __y) {                                               \
    return __fn((double)__x, (double)__y);                                     \
  }

// Define cmath functions with float argument and returns float.
#define __DEF_FUN1(retty, func)                                                \
  EXPORT                                                                       \
  float func(float x) { return func##f(x); }                                   \
  __HIP_OVERLOAD1(retty, func)

// Define cmath functions with float argument and returns retty.
#define __DEF_FUNI(retty, func)                                                \
  EXPORT                                                                       \
  retty func(float x) { return func##f(x); }                                   \
  __HIP_OVERLOAD1(retty, func)

// define cmath functions with two float arguments.
#define __DEF_FUN2(retty, func)                                                \
  EXPORT                                                                       \
  float func(float x, float y) { return func##f(x, y); }                       \
  __HIP_OVERLOAD2(retty, func)

__HIP_OVERLOAD2(bool, isunordered);

__HIP_OVERLOAD2(double, max)
__HIP_OVERLOAD2(double, min)

namespace std {
__HIP_OVERLOAD1(long, lrint);
}

#pragma pop_macro("__DEF_FLOAT_FUN")
#pragma pop_macro("__DEF_FLOAT_FUN2")
#pragma pop_macro("__HIP_OVERLOAD")
#pragma pop_macro("__HIP_OVERLOAD2")

/**********************************************************************/

#if defined(__HIP_DEVICE_COMPILE__)

#define DEFOPENCL_ATOMIC2(HIPNAME, CLNAME)                                     \
  extern "C" {                                                                 \
  NON_OVLD int GEN_NAME2(atomic_##CLNAME, i)(int *address, int i);             \
  NON_OVLD unsigned int GEN_NAME2(atomic_##CLNAME, u)(unsigned int *address,   \
                                                      unsigned int ui);        \
  NON_OVLD unsigned long long GEN_NAME2(atomic_##CLNAME,                       \
                                        l)(unsigned long long *address,        \
                                           unsigned long long ull);            \
  }                                                                            \
  EXPORT OVLD int atomic##HIPNAME(int *address, int val) {                     \
    return GEN_NAME2(atomic_##CLNAME, i)(address, val);                        \
  }                                                                            \
  EXPORT OVLD unsigned int atomic##HIPNAME(unsigned int *address,              \
                                           unsigned int val) {                 \
    return GEN_NAME2(atomic_##CLNAME, u)(address, val);                        \
  }                                                                            \
  EXPORT OVLD unsigned long atomic##HIPNAME(unsigned long *address,            \
                                            unsigned long val) {               \
    return GEN_NAME2(atomic_##CLNAME, l)((unsigned long long *)address,        \
                                         (unsigned long long)val);             \
  }                                                                            \
  EXPORT OVLD unsigned long long atomic##HIPNAME(unsigned long long *address,  \
                                                 unsigned long long val) {     \
    return GEN_NAME2(atomic_##CLNAME, l)(address, val);                        \
  }

#define DEFOPENCL_ATOMIC1(HIPNAME, CLNAME)                                     \
  extern "C" {                                                                 \
  NON_OVLD int GEN_NAME2(atomic_##CLNAME, i)(int *address);                    \
  NON_OVLD unsigned int GEN_NAME2(atomic_##CLNAME, u)(unsigned int *address);  \
  NON_OVLD unsigned long long GEN_NAME2(atomic_##CLNAME,                       \
                                        l)(unsigned long long *address);       \
  }                                                                            \
  EXPORT OVLD int atomic##HIPNAME(int *address) {                              \
    return GEN_NAME2(atomic_##CLNAME, i)(address);                             \
  }                                                                            \
  EXPORT OVLD unsigned int atomic##HIPNAME(unsigned int *address) {            \
    return GEN_NAME2(atomic_##CLNAME, u)(address);                             \
  }                                                                            \
  EXPORT OVLD unsigned long long atomic##HIPNAME(                              \
      unsigned long long *address) {                                           \
    return GEN_NAME2(atomic_##CLNAME, l)(address);                             \
  }

#define DEFOPENCL_ATOMIC3(HIPNAME, CLNAME)                                     \
  extern "C" {                                                                 \
  NON_OVLD int GEN_NAME2(atomic_##CLNAME, i)(int *address, int cmp, int val);  \
  NON_OVLD unsigned int GEN_NAME2(atomic_##CLNAME, u)(unsigned int *address,   \
                                                      unsigned int cmp,        \
                                                      unsigned int val);       \
  NON_OVLD unsigned long long GEN_NAME2(atomic_##CLNAME,                       \
                                        l)(unsigned long long *address,        \
                                           unsigned long long cmp,             \
                                           unsigned long long val);            \
  }                                                                            \
  EXPORT OVLD int atomic##HIPNAME(int *address, int cmp, int val) {            \
    return GEN_NAME2(atomic_##CLNAME, i)(address, cmp, val);                   \
  }                                                                            \
  EXPORT OVLD unsigned int atomic##HIPNAME(                                    \
      unsigned int *address, unsigned int cmp, unsigned int val) {             \
    return GEN_NAME2(atomic_##CLNAME, u)(address, cmp, val);                   \
  }                                                                            \
  EXPORT OVLD unsigned long long atomic##HIPNAME(unsigned long long *address,  \
                                                 unsigned long long cmp,       \
                                                 unsigned long long val) {     \
    return GEN_NAME2(atomic_##CLNAME, l)(address, cmp, val);                   \
  }

#else

#define DEFOPENCL_ATOMIC2(HIPNAME, CLNAME)                                     \
  EXPORT OVLD int atomic##HIPNAME(int *address, int val);                      \
  EXPORT OVLD unsigned int atomic##HIPNAME(unsigned int *address,              \
                                           unsigned int val);                  \
  EXPORT OVLD unsigned long atomic##HIPNAME(unsigned long *address,            \
                                            unsigned long val);                \
  EXPORT OVLD unsigned long long atomic##HIPNAME(unsigned long long *address,  \
                                                 unsigned long long val);

#define DEFOPENCL_ATOMIC1(HIPNAME, CLNAME)                                     \
  EXPORT OVLD int atomic##HIPNAME(int *address);                               \
  EXPORT OVLD unsigned int atomic##HIPNAME(unsigned int *address);             \
  EXPORT OVLD unsigned long long atomic##HIPNAME(unsigned long long *address);

#define DEFOPENCL_ATOMIC3(HIPNAME, CLNAME)                                     \
  EXPORT OVLD int atomic##HIPNAME(int *address, int cmp, int val);             \
  EXPORT OVLD unsigned int atomic##HIPNAME(                                    \
      unsigned int *address, unsigned int cmp, unsigned int val);              \
  EXPORT OVLD unsigned long long atomic##HIPNAME(unsigned long long *address,  \
                                                 unsigned long long cmp,       \
                                                 unsigned long long val);

#endif

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

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
NON_OVLD float GEN_NAME2(atomic_add, f)(float *address, float val);
NON_OVLD double GEN_NAME2(atomic_add, d)(double *address, double val);
NON_OVLD float GEN_NAME2(atomic_exch, f)(float *address, float val);
NON_OVLD unsigned GEN_NAME2(atomic_inc2, u)(unsigned *address, unsigned val);
NON_OVLD unsigned GEN_NAME2(atomic_dec2, u)(unsigned *address, unsigned val);
}
EXPORT float atomicAdd(float *address, float val) {
  return GEN_NAME2(atomic_add, f)(address, val);
}
// Undocumented atomicAdd variant without return value.
EXPORT void atomicAddNoRet(float *address, float val) {
  (void)GEN_NAME2(atomic_add, f)(address, val);
}
EXPORT double atomicAdd(double *address, double val) {
  return GEN_NAME2(atomic_add, d)(address, val);
}
EXPORT float atomicExch(float *address, float val) {
  return GEN_NAME2(atomic_exch, f)(address, val);
}
EXPORT unsigned atomicInc(unsigned *address, unsigned val) {
  return GEN_NAME2(atomic_inc2, u)(address, val);
}
EXPORT unsigned atomicDec(unsigned *address, unsigned val) {
  return GEN_NAME2(atomic_dec2, u)(address, val);
}
#else
EXPORT void atomicAddNoRet(float *address, float val);
EXPORT float atomicAdd(float *address, float val);
EXPORT double atomicAdd(double *address, double val);
EXPORT float atomicExch(float *address, float val);
EXPORT unsigned atomicInc(unsigned *address, unsigned val);
EXPORT unsigned atomicDec(unsigned *address, unsigned val);
#endif

// TODO: This is a temporary implementation of clock64(),
//       in future it will be changed with more reliable implementation.
__device__ static unsigned long long __chip_clk_counter = 0;
EXPORT unsigned long long clock64() {
  atomicAdd(&__chip_clk_counter, 1);
  return __chip_clk_counter;
}
// TODO: This is a temporary implementation of clock(),
//       in future it will be changed with more reliable implementation.
//       It is encouraged to use clock64() over clock() so that chance of data
//       loss can be avoided.
EXPORT clock_t clock() { return (clock_t)clock64(); }

/**********************************************************************/

#if defined(__HIP_DEVICE_COMPILE__)
extern "C" {
NON_OVLD int GEN_NAME(group_all)(int pred);
NON_OVLD int GEN_NAME(group_any)(int pred);
NON_OVLD uint64_t GEN_NAME(group_ballot)(int pred);
}

#else

#endif

#include <hip/spirv_hip_runtime.h>

#endif

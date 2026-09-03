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

#include <hip/devicelib/atomics.hh>

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

#include <hip/devicelib/host_math_funcs.hh>

#pragma push_macro("__DEF_FLOAT_FUN")
#pragma push_macro("__DEF_FLOAT_FUN2")
#pragma push_macro("__HIP_OVERLOAD")
#pragma push_macro("__HIP_OVERLOAD2")
__device__ inline void* operator new(size_t, void* ptr) noexcept { return ptr; }
__device__ inline void* operator new[](size_t, void* ptr) noexcept { return ptr; }
__device__ inline void operator delete(void*, void*) noexcept {}
__device__ inline void operator delete[](void*, void*) noexcept {}



// Device-side dynamic allocation. Gated behind CHIP_ENABLE_DEVICE_MALLOC
// because the underlying heap is a program-scope global that some OpenCL
// drivers (e.g. rusticl/radeonsi) cannot consume (issue #1279).
#ifdef CHIP_ENABLE_DEVICE_PROGRAM_SCOPE_GLOBALS
extern "C" __device__ void * __chip_malloc(unsigned int size);
extern "C" __device__ void __chip_free(void *ptr);
extern "C" __device__ void __chip_init_device_heap(void* device_heap);

EXPORT void * malloc(unsigned int size) {
    return __chip_malloc(size);
}
EXPORT void free(void *ptr) {
    __chip_free(ptr);
}
#endif

// __hip_enable_if::type is a type function which returns __T if __B is true.
template <bool __B, class __T = void> struct __hip_enable_if {};

template <class __T> struct __hip_enable_if<true, __T> { typedef __T type; };

// Device-compatible numeric_limits for basic types
template<typename T> struct __hip_numeric_limits {
  static constexpr bool is_integer = false;
  static constexpr bool is_specialized = false;
};

template<> struct __hip_numeric_limits<bool> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<char> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<signed char> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<unsigned char> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<short> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<unsigned short> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<int> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<unsigned int> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<long> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<unsigned long> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<long long> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<unsigned long long> {
  static constexpr bool is_integer = true;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<float> {
  static constexpr bool is_integer = false;
  static constexpr bool is_specialized = true;
};
template<> struct __hip_numeric_limits<double> {
  static constexpr bool is_integer = false;
  static constexpr bool is_specialized = true;
};

// __HIP_OVERLOAD1 is used to resolve function calls with integer argument to
// avoid compilation error due to ambibuity. e.g. floor(5) is resolved with
// floor(double).
#define __HIP_OVERLOAD1(__retty, __fn)                                         \
  template <typename __T>                                                      \
  __DEVICE__ typename __hip_enable_if<__hip_numeric_limits<__T>::is_integer,    \
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
      typename __hip_enable_if<__hip_numeric_limits<__T1>::is_specialized &&    \
                                   __hip_numeric_limits<__T2>::is_specialized,  \
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
__HIP_OVERLOAD2(double, pow)

// Integer arguments must behave as if converted to double ([cmath.syn]).
// devicelib declares float, double and __half flavours of these names, so
// without an integral overload an integer argument converts equally well to
// more than one of them and the call is ambiguous. The host standard library
// supplies these overloads itself, but under libc++ they are not constexpr,
// so they are host-only and gone from the device pass. See
// CHIP-SPV/chipStar#1586.

__HIP_OVERLOAD1(double, acosh)
__HIP_OVERLOAD1(double, asinh)
__HIP_OVERLOAD1(double, atanh)
__HIP_OVERLOAD1(double, ceil)
__HIP_OVERLOAD1(double, cos)
__HIP_OVERLOAD1(double, erf)
__HIP_OVERLOAD1(double, exp)
__HIP_OVERLOAD1(double, exp2)
__HIP_OVERLOAD1(double, floor)
__HIP_OVERLOAD1(double, lgamma)
__HIP_OVERLOAD1(double, log)
__HIP_OVERLOAD1(double, log10)
__HIP_OVERLOAD1(double, log1p)
__HIP_OVERLOAD1(double, log2)
__HIP_OVERLOAD1(double, logb)
__HIP_OVERLOAD1(double, nearbyint)
__HIP_OVERLOAD1(double, rint)
__HIP_OVERLOAD1(double, sin)
__HIP_OVERLOAD1(double, sqrt)
__HIP_OVERLOAD1(double, tan)
__HIP_OVERLOAD1(double, tgamma)
__HIP_OVERLOAD1(double, trunc)
__HIP_OVERLOAD1(double, cospi)
__HIP_OVERLOAD1(double, exp10)
__HIP_OVERLOAD1(double, rsqrt)
__HIP_OVERLOAD1(double, sinpi)

__HIP_OVERLOAD2(double, atan2)
__HIP_OVERLOAD2(double, copysign)
__HIP_OVERLOAD2(double, fdim)
__HIP_OVERLOAD2(double, fmin)
__HIP_OVERLOAD2(double, nextafter)
__HIP_OVERLOAD2(double, remainder)
namespace std {
__HIP_OVERLOAD1(long, lrint);
__HIP_OVERLOAD1(double, erfc);
__HIP_OVERLOAD1(double, tanh);
__HIP_OVERLOAD1(double, cosh);
__HIP_OVERLOAD1(double, sinh);
__HIP_OVERLOAD1(double, atan);
__HIP_OVERLOAD1(double, acos);
__HIP_OVERLOAD1(double, asin);

// CHIP-SPV/chipStar#1586: re-export the global integral overloads above.
using ::acosh;
using ::asinh;
using ::atanh;
using ::ceil;
using ::cos;
using ::erf;
using ::exp;
using ::exp2;
using ::floor;
using ::lgamma;
using ::log;
using ::log10;
using ::log1p;
using ::log2;
using ::logb;
using ::nearbyint;
using ::rint;
using ::sin;
using ::sqrt;
using ::tan;
using ::tgamma;
using ::trunc;
using ::atan2;
using ::copysign;
using ::fdim;
using ::fmin;
using ::nextafter;
using ::remainder;

// libstdc++ pulls the C `modf` into std with a using-declaration, which covers
// the double overload, but its float overload is a host-only inline. The device
// `float modf(float, float *)` lives in the global namespace only, so a device
// call written as `using std::modf; modf(x, &integral)` has no viable candidate
// (float * does not convert to double *). Kokkos writes every math call that
// way.
//
// Re-export the global overload set rather than declaring a new function here.
// libstdc++'s <math.h> does `using std::modf;` at global scope, so a distinct
// std::modf(float, float *) would collide with the global one declared in
// devicelib/single_precision/sp_math.hh:
//
//   math.h:54:12: error: target of using declaration conflicts with
//   declaration already in scope
//
// and every translation unit that includes <math.h> would stop compiling. A
// using-declaration names the same entity, so bouncing it back out to the
// global namespace is a no-op.
using ::modf;
} // namespace std

#pragma pop_macro("__DEF_FLOAT_FUN")
#pragma pop_macro("__DEF_FLOAT_FUN2")
#pragma pop_macro("__HIP_OVERLOAD")
#pragma pop_macro("__HIP_OVERLOAD2")

/**********************************************************************/

// Device-side clock. Gated behind CHIP_ENABLE_DEVICE_CLOCK because the
// monotonic counter is a program-scope global that some OpenCL drivers
// (e.g. rusticl/radeonsi) cannot consume (issue #1279). When disabled, the
// clock* functions remain callable but return 0 (no global is emitted).
#ifdef CHIP_ENABLE_DEVICE_PROGRAM_SCOPE_GLOBALS
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
EXPORT long clock() { return (long)clock64(); }

EXPORT unsigned long long wall_clock64() {
  atomicAdd(&__chip_clk_counter, 1);
  return __chip_clk_counter;
}
// TODO: This is a temporary implementation of clock(),
//       in future it will be changed with more reliable implementation.
//       It is encouraged to use clock64() over clock() so that chance of data
//       loss can be avoided.
EXPORT long wall_clock() { return (long)wall_clock64(); }
#else
EXPORT unsigned long long clock64() { return 0; }
EXPORT long clock() { return 0; }
EXPORT unsigned long long wall_clock64() { return 0; }
EXPORT long wall_clock() { return 0; }
#endif

#include <hip/spirv_hip_runtime.h>

#endif

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
__HIP_OVERLOAD2(double, pow)

namespace std {
__HIP_OVERLOAD1(long, lrint);
__HIP_OVERLOAD1(double, lgamma);
__HIP_OVERLOAD1(double, erfc);
__HIP_OVERLOAD1(double, erf);
__HIP_OVERLOAD1(double, tanh);
__HIP_OVERLOAD1(double, cosh);
__HIP_OVERLOAD1(double, sinh);
__HIP_OVERLOAD1(double, atan);
__HIP_OVERLOAD1(double, acos);
__HIP_OVERLOAD1(double, asin);
__HIP_OVERLOAD1(double, tan);
} // namespace std

#pragma pop_macro("__DEF_FLOAT_FUN")
#pragma pop_macro("__DEF_FLOAT_FUN2")
#pragma pop_macro("__HIP_OVERLOAD")
#pragma pop_macro("__HIP_OVERLOAD2")

/**********************************************************************/

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

#include <hip/spirv_hip_runtime.h>

#endif

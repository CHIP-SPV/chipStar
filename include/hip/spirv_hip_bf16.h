/**
 * MIT License
 *
 * Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * \file
 * \brief hip_bf16.h provides struct for __hip_bfloat16 types
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16 bfloat16 Precision Intrinsics
 * This section describes hip_bfloat16 precision intrinsic functions.
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_ARITH Bfloat16 Arithmetic Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_COMP Bfloat16 Comparision Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_COMP Bfloat162 Comparision Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_ARITH Bfloat162 Arithmetic Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_CONV Bfloat16 Conversion Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_CONV Bfloat162 Conversion Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_MATH Bfloat16 Math Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_MATH Bfloat162 Math Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */


#ifndef _HIP_INCLUDE_HIP_SPIRV_DETAIL_HIP_BF16_H_
#define _HIP_INCLUDE_HIP_SPIRV_DETAIL_HIP_BF16_H_

#include "spirv_hip_vector_types.h"  // float2 etc
#include "hip/spirv_hip_devicelib.hh" // device functions

#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__
#else
#include <climits>
#define __HOST_DEVICE__ __host__ __device__
#endif

// Since we are using unsigned short to represent data in bfloat16, it can be of different sizes on
// different machines. These naive checks should prevent some undefined behavior on systems which
// have different sizes for basic types.
#if !defined(__HIPCC_RTC__)
static_assert(CHAR_BIT == 8, "byte size should be of 8 bits");
#endif
static_assert(sizeof(unsigned short) == 2, "size of unsigned short should be 2 bytes");

/*! \brief Struct to represent a 16 bit brain floating point number. */
struct __hip_bfloat16 {
  unsigned short data;

  // Default constructor
  __HOST_DEVICE__ __hip_bfloat16() : data(0) {}

  // Add implicit conversion constructor from float
  __HOST_DEVICE__ __hip_bfloat16(float f) {
    union {
      float fp32;
      unsigned int u32;
    } u = {f};
    if (~u.u32 & 0x7f800000) {
      u.u32 += 0x7fff + ((u.u32 >> 16) & 1);
    }
    this->data = u.u32 >> 16;
  }

  // Add implicit conversion operator to float
  __HOST_DEVICE__ operator float() const {
    unsigned int uval = 0;
    uval = this->data << 16;
    union {
      unsigned int u32;
      float fp32;
    } u = {uval};
    return u.fp32;
  }
};

/*! \brief Struct to represent two 16 bit brain floating point numbers. */
struct __hip_bfloat162 {
  __hip_bfloat16 x;
  __hip_bfloat16 y;
};

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Converts bfloat16 to float
 */
__HOST_DEVICE__ inline float __bfloat162float(__hip_bfloat16 a) {
  unsigned int uval = 0;
  uval = a.data << 16;
  union {
    unsigned int u32;
    float fp32;
  } u = {uval};
  return u.fp32;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Converts float to bfloat16
 */
__HOST_DEVICE__ __hip_bfloat16 __float2bfloat16(float f) {
  __hip_bfloat16 ret;
  union {
    float fp32;
    unsigned int u32;
  } u = {f};
  if (~u.u32 & 0x7f800000) {
    // When the exponent bits are not all 1s, then the value is zero, normal,
    // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
    // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
    // This causes the bfloat16's mantissa to be incremented by 1 if the 16
    // least significant bits of the float mantissa are greater than 0x8000,
    // or if they are equal to 0x8000 and the least significant bit of the
    // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
    // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
    // has the value 0x7f, then incrementing it causes it to become 0x00 and
    // the exponent is incremented by one, which is the next higher FP value
    // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
    // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
    // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
    // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
    // incrementing it causes it to become an exponent of 0xFF and a mantissa
    // of 0x00, which is Inf, the next higher value to the unrounded value.
    u.u32 += 0x7fff + ((u.u32 >> 16) & 1);  // Round to nearest, round to even
  } else if (u.u32 & 0xffff) {
    // When all of the exponent bits are 1, the value is Inf or NaN.
    // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
    // mantissa bit. Quiet NaN is indicated by the most significant mantissa
    // bit being 1. Signaling NaN is indicated by the most significant
    // mantissa bit being 0 but some other bit(s) being 1. If any of the
    // lower 16 bits of the mantissa are 1, we set the least significant bit
    // of the bfloat16 mantissa, in order to preserve signaling NaN in case
    // the bloat16's mantissa bits are all 0.
    u.u32 |= 0x10000;  // Preserve signaling NaN
  }

  ret.data = (u.u32 >> 16);
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Converts and moves bfloat162 to float2
 */
__HOST_DEVICE__ float2 __bfloat1622float2(const __hip_bfloat162 a) {
  return float2{__bfloat162float(a.x), __bfloat162float(a.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Moves bfloat16 value to bfloat162
 */
__device__ __hip_bfloat162 __bfloat162bfloat162(const __hip_bfloat16 a) {
  return __hip_bfloat162{a, a};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets bits in a __hip_bfloat16 as a signed short integer
 */
__device__ short int __bfloat16_as_short(const __hip_bfloat16 h) { return (short)h.data; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets bits in a __hip_bfloat16 as an unsigned signed short integer
 */
__device__ unsigned short int __bfloat16_as_ushort(const __hip_bfloat16 h) { return h.data; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Convert double to __hip_bfloat16
 */
__HOST_DEVICE__ __hip_bfloat16 __double2bfloat16(const double a) {
  return __float2bfloat16((float)a);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Convert float2 to __hip_bfloat162
 */
__HOST_DEVICE__ __hip_bfloat162 __float22bfloat162_rn(const float2 a) {
  return __hip_bfloat162{__float2bfloat16(a.x), __float2bfloat16(a.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Combine two __hip_bfloat16 to __hip_bfloat162
 */
__device__ __hip_bfloat162 __halves2bfloat162(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __hip_bfloat162{a, b};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns high 16 bits of __hip_bfloat162
 */
__device__ __hip_bfloat16 __high2bfloat16(const __hip_bfloat162 a) { return a.y; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns high 16 bits of __hip_bfloat162
 */
__device__ __hip_bfloat162 __high2bfloat162(const __hip_bfloat162 a) {
  return __hip_bfloat162{a.y, a.y};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Converts high 16 bits of __hip_bfloat162 to float and returns the result
 */
__HOST_DEVICE__ float __high2float(const __hip_bfloat162 a) { return __bfloat162float(a.y); }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Extracts high 16 bits from each and combines them
 */
__device__ __hip_bfloat162 __highs2bfloat162(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{a.y, b.y};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns low 16 bits of __hip_bfloat162
 */
__device__ __hip_bfloat16 __low2bfloat16(const __hip_bfloat162 a) { return a.x; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns low 16 bits of __hip_bfloat162
 */
__device__ __hip_bfloat162 __low2bfloat162(const __hip_bfloat162 a) {
  return __hip_bfloat162{a.x, a.x};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Converts low 16 bits of __hip_bfloat162 to float and returns the result
 */
__HOST_DEVICE__ float __low2float(const __hip_bfloat162 a) { return __bfloat162float(a.x); }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Swaps both halves
 */
__device__ __hip_bfloat162 __lowhigh2highlow(const __hip_bfloat162 a) {
  return __hip_bfloat162{a.y, a.x};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Extracts low 16 bits from each and combines them
 */
__device__ __hip_bfloat162 __lows2bfloat162(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{a.x, b.x};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets short int into a bfloat16
 */
__device__ __hip_bfloat16 __short_as_bfloat16(const short int a) {
  return __hip_bfloat16{(unsigned short)a};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets unsigned short int into a bfloat16
 */
__device__ __hip_bfloat16 __ushort_as_bfloat16(const unsigned short int a) {
  return __hip_bfloat16{a};
}


/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Adds two bfloat16 values
 */
__device__ __hip_bfloat16 __hadd(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Subtracts two bfloat16 values
 */
__device__ __hip_bfloat16 __hsub(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) - __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Divides two bfloat16 values
 */
__device__ __hip_bfloat16 __hdiv(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) / __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Performs FMA of given bfloat16 values
 */
__device__ __hip_bfloat16 __hfma(const __hip_bfloat16 a, const __hip_bfloat16 b,
                                 const __hip_bfloat16 c) {
  return __float2bfloat16(
      fmaf(__bfloat162float(a), __bfloat162float(b), __bfloat162float(c)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Multiplies two bfloat16 values
 */
__device__ __hip_bfloat16 __hmul(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Negate a bfloat16 value
 */
__device__ __hip_bfloat16 __hneg(const __hip_bfloat16 a) {
  auto ret = a;
  ret.data ^= 0x8000;
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Returns absolute of a bfloat16
 */
__device__ __hip_bfloat16 __habs(const __hip_bfloat16 a) {
  auto ret = a;
  ret.data &= 0x7FFF;
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Divides bfloat162 values
 */
__device__ __hip_bfloat162 __h2div(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{__float2bfloat16(__bfloat162float(a.x) / __bfloat162float(b.x)),
                         __float2bfloat16(__bfloat162float(a.y) / __bfloat162float(b.y))};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Returns absolute of a bfloat162
 */
__device__ __hip_bfloat162 __habs2(const __hip_bfloat162 a) {
  return __hip_bfloat162{__habs(a.x), __habs(a.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Adds two bfloat162 values
 */
__device__ __hip_bfloat162 __hadd2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{__hadd(a.x, b.x), __hadd(a.y, b.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Performs FMA of given bfloat162 values
 */
__device__ __hip_bfloat162 __hfma2(const __hip_bfloat162 a, const __hip_bfloat162 b,
                                   const __hip_bfloat162 c) {
  return __hip_bfloat162{__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Multiplies two bfloat162 values
 */
__device__ __hip_bfloat162 __hmul2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{__hmul(a.x, b.x), __hmul(a.y, b.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Converts a bfloat162 into negative
 */
__device__ __hip_bfloat162 __hneg2(const __hip_bfloat162 a) {
  return __hip_bfloat162{__hneg(a.x), __hneg(a.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Subtracts two bfloat162 values
 */
__device__ __hip_bfloat162 __hsub2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{__hsub(a.x, b.x), __hsub(a.y, b.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values
 */
__device__ bool __heq(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) == __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered equal
 */
__device__ bool __hequ(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) < __bfloat162float(b)) &&
      !(__bfloat162float(a) > __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - greater than
 */
__device__ bool __hgt(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) > __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered greater than
 */
__device__ bool __hgtu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) <= __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - greater than equal
 */
__device__ bool __hge(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) >= __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered greater than equal
 */
__device__ bool __hgeu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) < __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - not equal
 */
__device__ bool __hne(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) != __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered not equal
 */
__device__ bool __hneu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) == __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - return max
 */
__device__ __hip_bfloat16 __hmax(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - return min
 */
__device__ __hip_bfloat16 __hmin(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __float2bfloat16(fminf(__bfloat162float(a), __bfloat162float(b)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - less than operator
 */
__device__ bool __hlt(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) < __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered less than
 */
__device__ bool __hltu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) >= __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - less than
 */
__device__ bool __hle(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return __bfloat162float(a) <= __bfloat162float(b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered less than equal
 */
__device__ bool __hleu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !(__bfloat162float(a) > __bfloat162float(b));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Checks if number is inf
 */
__device__ int __hisinf(const __hip_bfloat16 a) { return isinf(__bfloat162float(a)); }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Checks if number is nan
 */
__device__ bool __hisnan(const __hip_bfloat16 a) { return isnan(__bfloat162float(a)); }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks if two numbers are equal
 */
__device__ bool __hbeq2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __heq(a.x, b.x) && __heq(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks if two numbers are equal - unordered
 */
__device__ bool __hbequ2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hequ(a.x, b.x) && __hequ(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b
 */
__device__ bool __hbge2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hge(a.x, b.x) && __hge(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b - unordered
 */
__device__ bool __hbgeu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hgeu(a.x, b.x) && __hgeu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b
 */
__device__ bool __hbgt2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hgt(a.x, b.x) && __hgt(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b - unordered
 */
__device__ bool __hbgtu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hgtu(a.x, b.x) && __hgtu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b
 */
__device__ bool __hble2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hle(a.x, b.x) && __hle(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b - unordered
 */
__device__ bool __hbleu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hleu(a.x, b.x) && __hleu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b
 */
__device__ bool __hblt2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hlt(a.x, b.x) && __hlt(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b - unordered
 */
__device__ bool __hbltu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hltu(a.x, b.x) && __hltu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b
 */
__device__ bool __hbne2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hne(a.x, b.x) && __hne(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b
 */
__device__ bool __hbneu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hneu(a.x, b.x) && __hneu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b, returns 1.0 if equal, otherwise 0.0
 */
__device__ __hip_bfloat162 __heq2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{{__heq(a.x, b.x) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)},
                         {__heq(a.y, b.y) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b, returns 1.0 if greater than equal, otherwise 0.0
 */
__device__ __hip_bfloat162 __hge2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hge(a.x, b.x) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)},
                         {__hge(a.y, b.y) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b, returns 1.0 if greater than equal, otherwise 0.0
 */
__device__ __hip_bfloat162 __hgt2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hgt(a.x, b.x) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)},
                         {__hgt(a.y, b.y) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a is NaN, returns 1.0 if NaN, otherwise 0.0
 */
__device__ __hip_bfloat162 __hisnan2(const __hip_bfloat162 a) {
  return __hip_bfloat162{
      {isnan(__bfloat162float(a.x)) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)},
      {isnan(__bfloat162float(a.y)) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b, returns 1.0 if greater than equal, otherwise 0.0
 */
__device__ __hip_bfloat162 __hle2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hle(a.x, b.x) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)},
                         {__hle(a.y, b.y) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b, returns 1.0 if greater than equal, otherwise 0.0
 */
__device__ __hip_bfloat162 __hlt2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hlt(a.x, b.x) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)},
                         {__hlt(a.y, b.y) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Returns max of two elements
 */
__device__ __hip_bfloat162 __hmax2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{
      __float2bfloat16(fmaxf(__bfloat162float(a.x), __bfloat162float(b.x))),
      __float2bfloat16(fmaxf(__bfloat162float(a.y), __bfloat162float(b.y)))};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Returns min of two elements
 */
__device__ __hip_bfloat162 __hmin2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{
      __float2bfloat16(fminf(__bfloat162float(a.x), __bfloat162float(b.x))),
      __float2bfloat16(fminf(__bfloat162float(a.y), __bfloat162float(b.y)))};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks for not equal to
 */
__device__ __hip_bfloat162 __hne2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hne(a.x, b.x) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)},
                         {__hne(a.y, b.y) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f)}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate ceil of bfloat16
 */
__device__ __hip_bfloat16 hceil(const __hip_bfloat16 h) {
  return __float2bfloat16(ceilf(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate cosine of bfloat16
 */
__device__ __hip_bfloat16 hcos(const __hip_bfloat16 h) {
  return __float2bfloat16(cosf(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate exponential of bfloat16
 */
__device__ __hip_bfloat16 hexp(const __hip_bfloat16 h) {
  return __float2bfloat16(expf(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate exponential 10 of bfloat16
 */
__device__ __hip_bfloat16 hexp10(const __hip_bfloat16 h) {
  return __float2bfloat16(exp10f(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate exponential 2 of bfloat16
 */
__device__ __hip_bfloat16 hexp2(const __hip_bfloat16 h) {
  return __float2bfloat16(exp2f(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate floor of bfloat16
 */
__device__ __hip_bfloat16 hfloor(const __hip_bfloat16 h) {
  return __float2bfloat16(floorf(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate natural log of bfloat16
 */
__device__ __hip_bfloat16 hlog(const __hip_bfloat16 h) {
  return __float2bfloat16(logf(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate log 10 of bfloat16
 */
__device__ __hip_bfloat16 hlog10(const __hip_bfloat16 h) {
  return __float2bfloat16(log10f(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate log 2 of bfloat16
 */
__device__ __hip_bfloat16 hlog2(const __hip_bfloat16 h) {
  return __float2bfloat16(log2f(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate reciprocal
 */
__device__ __hip_bfloat16 hrcp(const __hip_bfloat16 h) {
  return __float2bfloat16(1.0f / (__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Round to nearest int
 */
__device__ __hip_bfloat16 hrint(const __hip_bfloat16 h) {
  return __float2bfloat16(rintf(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Reciprocal square root
 */
__device__ __hip_bfloat16 hrsqrt(const __hip_bfloat16 h) {
  return __float2bfloat16(rsqrtf(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate sin of bfloat16
 */
__device__ __hip_bfloat16 hsin(const __hip_bfloat16 h) {
  return __float2bfloat16(sinf(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate sqrt of bfloat16
 */
__device__ __hip_bfloat16 hsqrt(const __hip_bfloat16 h) {
  return __float2bfloat16(sqrtf(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate truncate of bfloat16
 */
__device__ __hip_bfloat16 htrunc(const __hip_bfloat16 h) {
  return __float2bfloat16(truncf(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate ceil of bfloat162
 */
__device__ __hip_bfloat162 h2ceil(const __hip_bfloat162 h) {
  return __hip_bfloat162{hceil(h.x), hceil(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate cosine of bfloat162
 */
__device__ __hip_bfloat162 h2cos(const __hip_bfloat162 h) {
  return __hip_bfloat162{hcos(h.x), hcos(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential of bfloat162
 */
__device__ __hip_bfloat162 h2exp(const __hip_bfloat162 h) {
  return __hip_bfloat162{hexp(h.x), hexp(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential 10 of bfloat162
 */
__device__ __hip_bfloat162 h2exp10(const __hip_bfloat162 h) {
  return __hip_bfloat162{hexp10(h.x), hexp10(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential 2 of bfloat162
 */
__device__ __hip_bfloat162 h2exp2(const __hip_bfloat162 h) {
  return __hip_bfloat162{hexp2(h.x), hexp2(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate floor of bfloat162
 */
__device__ __hip_bfloat162 h2floor(const __hip_bfloat162 h) {
  return __hip_bfloat162{hfloor(h.x), hfloor(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate natural log of bfloat162
 */
__device__ __hip_bfloat162 h2log(const __hip_bfloat162 h) {
  return __hip_bfloat162{hlog(h.x), hlog(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate log 10 of bfloat162
 */
__device__ __hip_bfloat162 h2log10(const __hip_bfloat162 h) {
  return __hip_bfloat162{hlog10(h.x), hlog10(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate log 2 of bfloat162
 */
__device__ __hip_bfloat162 h2log2(const __hip_bfloat162 h) {
  return __hip_bfloat162{hlog2(h.x), hlog2(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector reciprocal
 */
__device__ __hip_bfloat162 h2rcp(const __hip_bfloat162 h) {
  return __hip_bfloat162{hrcp(h.x), hrcp(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector round to nearest int
 */
__device__ __hip_bfloat162 h2rint(const __hip_bfloat162 h) {
  return __hip_bfloat162{hrint(h.x), hrint(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector reciprocal square root
 */
__device__ __hip_bfloat162 h2rsqrt(const __hip_bfloat162 h) {
  return __hip_bfloat162{hrsqrt(h.x), hrsqrt(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate sin of bfloat162
 */
__device__ __hip_bfloat162 h2sin(const __hip_bfloat162 h) {
  return __hip_bfloat162{hsin(h.x), hsin(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate sqrt of bfloat162
 */
__device__ __hip_bfloat162 h2sqrt(const __hip_bfloat162 h) {
  return __hip_bfloat162{hsqrt(h.x), hsqrt(h.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate truncate of bfloat162
 */
__device__ __hip_bfloat162 h2trunc(const __hip_bfloat162 h) {
  return __hip_bfloat162{htrunc(h.x), htrunc(h.y)};
}

// Conversion functions with specific rounding modes
__device__ int __bfloat162int_rd(const __hip_bfloat16 h) {
  return static_cast<int>(floorf(__bfloat162float(h)));
}

__device__ int __bfloat162int_rn(const __hip_bfloat16 h) {
  return static_cast<int>(roundf(__bfloat162float(h)));
}

__device__ int __bfloat162int_ru(const __hip_bfloat16 h) {
  return static_cast<int>(ceilf(__bfloat162float(h)));
}

// Conversion functions for 64-bit integers
__device__ long long int __bfloat162ll_rd(const __hip_bfloat16 h) {
  return static_cast<long long int>(floorf(__bfloat162float(h)));
}

__device__ long long int __bfloat162ll_rn(const __hip_bfloat16 h) {
  return static_cast<long long int>(roundf(__bfloat162float(h)));
}

__device__ long long int __bfloat162ll_ru(const __hip_bfloat16 h) {
  return static_cast<long long int>(ceilf(__bfloat162float(h)));
}

// Conversion functions for unsigned types
__device__ unsigned int __bfloat162uint_rd(const __hip_bfloat16 h) {
  return static_cast<unsigned int>(floorf(__bfloat162float(h)));
}

__device__ unsigned int __bfloat162uint_rn(const __hip_bfloat16 h) {
  return static_cast<unsigned int>(roundf(__bfloat162float(h)));
}

__device__ unsigned int __bfloat162uint_ru(const __hip_bfloat16 h) {
  return static_cast<unsigned int>(ceilf(__bfloat162float(h)));
}

__device__ unsigned long long int __bfloat162ull_rd(const __hip_bfloat16 h) {
  return static_cast<unsigned long long int>(floorf(__bfloat162float(h)));
}

__device__ unsigned long long int __bfloat162ull_rn(const __hip_bfloat16 h) {
  return static_cast<unsigned long long int>(roundf(__bfloat162float(h)));
}

__device__ unsigned long long int __bfloat162ull_ru(const __hip_bfloat16 h) {
  return static_cast<unsigned long long int>(ceilf(__bfloat162float(h)));
}

// Additional conversion functions
__host__ __device__ __hip_bfloat162 __float2bfloat162_rn(const float a) { return __hip_bfloat162{__float2bfloat16(a), __float2bfloat16(a)}; }
__host__ __device__ __hip_bfloat162 __floats2bfloat162_rn(const float a, const float b) { return __hip_bfloat162{__float2bfloat16(a), __float2bfloat16(b)}; }

// Load/store instructions
__device__ __hip_bfloat16 __ldca(const __hip_bfloat16* ptr) { 
    return *ptr;
}
__device__ __hip_bfloat162 __ldca(const __hip_bfloat162* ptr) { 
    return *ptr;
}
__device__ __hip_bfloat16 __ldcg(const __hip_bfloat16* ptr) { 
    return *ptr;
}
__device__ __hip_bfloat162 __ldcg(const __hip_bfloat162* ptr) { 
    return *ptr;
}
__device__ __hip_bfloat16 __ldcs(const __hip_bfloat16* ptr) { 
    return *ptr;
}
__device__ __hip_bfloat162 __ldcs(const __hip_bfloat162* ptr) { 
    return *ptr;
}
__device__ __hip_bfloat16 __ldcv(const __hip_bfloat16* ptr) { 
    return *ptr;
}
__device__ __hip_bfloat162 __ldcv(const __hip_bfloat162* ptr) { 
    return *ptr;
}
__device__ __hip_bfloat16 __ldg(const __hip_bfloat16* ptr) { 
    return *ptr;
}
__device__ __hip_bfloat162 __ldg(const __hip_bfloat162* ptr) { 
    return *ptr;
}
__device__ __hip_bfloat16 __ldlu(const __hip_bfloat16* ptr) { 
    return *ptr;
}
__device__ __hip_bfloat162 __ldlu(const __hip_bfloat162* ptr) { 
    return *ptr;
}

__device__ void __stcg(__hip_bfloat16* ptr, __hip_bfloat16 value) { 
    *ptr = value;
}
__device__ void __stcg(__hip_bfloat162* ptr, __hip_bfloat162 value) { 
    *ptr = value;
}
__device__ void __stcs(__hip_bfloat16* ptr, __hip_bfloat16 value) { 
    *ptr = value;
}
__device__ void __stcs(__hip_bfloat162* ptr, __hip_bfloat162 value) { 
    *ptr = value;
}
__device__ void __stwb(__hip_bfloat16* ptr, __hip_bfloat16 value) { 
    *ptr = value;
}
__device__ void __stwb(__hip_bfloat162* ptr, __hip_bfloat162 value) { 
    *ptr = value;
}
__device__ void __stwt(__hip_bfloat16* ptr, __hip_bfloat16 value) { 
    *ptr = value;
}
__device__ void __stwt(__hip_bfloat162* ptr, __hip_bfloat162 value) { 
    *ptr = value;
}

// Shuffle operations
__device__ __hip_bfloat16 __shfl_down_sync(unsigned mask, __hip_bfloat16 var, unsigned int delta, int width = 32) {
    float f = __bfloat162float(var);
    float shuffled = __shfl_down_sync(mask, f, delta, width);
    return __float2bfloat16(shuffled);
}

__device__ __hip_bfloat162 __shfl_down_sync(unsigned mask, __hip_bfloat162 var, unsigned int delta, int width = 32) {
    float x = __bfloat162float(var.x);
    float y = __bfloat162float(var.y);
    float shuffled_x = __shfl_down_sync(mask, x, delta, width);
    float shuffled_y = __shfl_down_sync(mask, y, delta, width);
    return __hip_bfloat162{__float2bfloat16(shuffled_x), __float2bfloat16(shuffled_y)};
}

__device__ __hip_bfloat16 __shfl_sync(unsigned mask, __hip_bfloat16 var, int delta, int width = 32) {
    float f = __bfloat162float(var);
    float shuffled = __shfl_sync(mask, f, delta, width);
    return __float2bfloat16(shuffled);
}

__device__ __hip_bfloat162 __shfl_sync(unsigned mask, __hip_bfloat162 var, int delta, int width = 32) {
    float x = __bfloat162float(var.x);
    float y = __bfloat162float(var.y);
    float shuffled_x = __shfl_sync(mask, x, delta, width);
    float shuffled_y = __shfl_sync(mask, y, delta, width);
    return __hip_bfloat162{__float2bfloat16(shuffled_x), __float2bfloat16(shuffled_y)};
}

__device__ __hip_bfloat16 __shfl_up_sync(unsigned mask, __hip_bfloat16 var, unsigned int delta, int width = 32) {
    float f = __bfloat162float(var);
    float shuffled = __shfl_up_sync(mask, f, delta, width);
    return __float2bfloat16(shuffled);
}

__device__ __hip_bfloat162 __shfl_up_sync(unsigned mask, __hip_bfloat162 var, unsigned int delta, int width = 32) {
    float x = __bfloat162float(var.x);
    float y = __bfloat162float(var.y);
    float shuffled_x = __shfl_up_sync(mask, x, delta, width);
    float shuffled_y = __shfl_up_sync(mask, y, delta, width);
    return __hip_bfloat162{__float2bfloat16(shuffled_x), __float2bfloat16(shuffled_y)};
}

__device__ __hip_bfloat16 __shfl_xor_sync(unsigned mask, __hip_bfloat16 var, int delta, int width = 32) {
    float f = __bfloat162float(var);
    float shuffled = __shfl_xor_sync(mask, f, delta, width);
    return __float2bfloat16(shuffled);
}

__device__ __hip_bfloat162 __shfl_xor_sync(unsigned mask, __hip_bfloat162 var, int delta, int width = 32) {
    float x = __bfloat162float(var.x);
    float y = __bfloat162float(var.y);
    float shuffled_x = __shfl_xor_sync(mask, x, delta, width);
    float shuffled_y = __shfl_xor_sync(mask, y, delta, width);
    return __hip_bfloat162{__float2bfloat16(shuffled_x), __float2bfloat16(shuffled_y)};
}

// make_bfloat162
__host__ __device__ __hip_bfloat162 make_bfloat162(__hip_bfloat16 x, __hip_bfloat16 y) {
    return __hip_bfloat162{x, y};
}



#endif

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

#ifndef HIP_INCLUDE_DEVICELIB_SP_INTRINSICS_H
#define HIP_INCLUDE_DEVICELIB_SP_INTRINSICS_H

#include <hip/devicelib/macros.hh>
#include <hip/devicelib/single_precision/sp_math.hh>

static inline __device__ float __cosf(float x) { return ::cos(x); }
static inline __device__ float __expf(float x) { return ::expf(x); }
static inline __device__ float __log10f(float x) { return ::log10(x); }
static inline __device__ float __log2f(float x) { return ::log2(x); }
static inline __device__ float __logf(float x) { return ::log(x); }
static inline __device__ float __sinf(float x) { return ::sin(x); }
static inline __device__ float __tanf(float x) { return ::tan(x); }

// __device__​ float __exp10f ( float  x )
// __device__​ float __fadd_rd ( float  x, float  y )
// __device__​ float __fadd_rn ( float  x, float  y )
// __device__​ float __fadd_ru ( float  x, float  y )
// __device__​ float __fadd_rz ( float  x, float  y )
// __device__​ float __fdiv_rd ( float  x, float  y )
// __device__​ float __fdiv_rn ( float  x, float  y )
// __device__​ float __fdiv_ru ( float  x, float  y )
// __device__​ float __fdiv_rz ( float  x, float  y )
__device__ float __fdividef(float x, float y);
// __device__​ float __fmaf_ieee_rd ( float  x, float  y, float  z )
// __device__​ float __fmaf_ieee_rn ( float  x, float  y, float  z )
// __device__​ float __fmaf_ieee_ru ( float  x, float  y, float  z )
// __device__​ float __fmaf_ieee_rz ( float  x, float  y, float  z )
// __device__​ float __fmaf_rd ( float  x, float  y, float  z )
// __device__​ float __fmaf_rn ( float  x, float  y, float  z )
// __device__​ float __fmaf_ru ( float  x, float  y, float  z )
// __device__​ float __fmaf_rz ( float  x, float  y, float  z )
// __device__​ float __fmul_rd ( float  x, float  y )
// __device__​ float __fmul_rn ( float  x, float  y )
// __device__​ float __fmul_ru ( float  x, float  y )
// __device__​ float __fmul_rz ( float  x, float  y )
// __device__​ float __frcp_rd ( float  x )
// __device__​ float __frcp_rn ( float  x )
// __device__​ float __frcp_ru ( float  x )
// __device__​ float __frcp_rz ( float  x )
// __device__​ float __frsqrt_rn ( float  x )
// __device__​ float __fsqrt_rd ( float  x )
// __device__​ float __fsqrt_rn ( float  x )
// __device__​ float __fsqrt_ru ( float  x )
// __device__​ float __fsqrt_rz ( float  x )
// __device__​ float __fsub_rd ( float  x, float  y )
// __device__​ float __fsub_rn ( float  x, float  y )
// __device__​ float __fsub_ru ( float  x, float  y )
// __device__​ float __fsub_rz ( float  x, float  y )
// __device__​ float __powf ( float  x, float  y )
// __device__​ float __saturatef ( float  x )
// __device__​ void __sincosf ( float  x, float* sptr, float* cptr )

#endif // include guard

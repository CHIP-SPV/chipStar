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
#include "CHIPSPVConfig.hh"

/**
 * @brief Declare as extern - we state that these funcitons are implemented and
 * will be found at link time
 *
 * The format is as follows:
 * 1. Declare the external function which will be executed with the appropriate
 * linkage. Inline comment specifying where the implementation is coming from.
 * (OpenCL, OCML, custom) note: some of these declarations are not strictly
 * necesary but are included for completeness.
 * 2. If necessary, define the type specific function and bind it to the
 * function declared in 1. cosf(x) -> cos(x)
 */

extern "C++" __device__ float native_cos(float x); // OpenCL
extern "C++" inline __device__ float __cosf(float x) { return native_cos(x); }

extern "C++" __device__ float native_exp10(float x); // OpenCL
extern "C++" inline __device__ float __exp10f(float x) {
  return native_exp10(x);
}

extern "C++" __device__ float native_exp(float x); // OpenCL
extern "C++" inline __device__ float __expf(float x) { return native_exp(x); }

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ float __ocml_add_rtn_f32(float x, float y);
extern "C++" inline __device__ float __fadd_rd(float x, float y)  {
  return __ocml_add_rtn_f32(x, y);
}

// extern "C++" __device__ float __ocml_add_rte_f32(float x, float y);
extern "C++" inline __device__ float __fadd_rn(float x, float y) {
  return __ocml_add_rte_f32(x, y);
}

// extern "C++" __device__ float __ocml_add_rtp_f32(float x, float y);
extern "C++" inline __device__ float __fadd_ru(float x, float y) {
  return __ocml_add_rtp_f32(x, y);
}

// extern "C++" __device__ float __ocml_add_rtz_f32(float x, float y);
extern "C++" inline __device__ float __fadd_rz(float x, float y) {
  return __ocml_add_rtz_f32(x, y);
}
#else
extern "C++" inline __device__ float __fadd_rd(float x, float y) { return x + y;}

extern "C++" inline __device__ float __fadd_rn(float x, float y) { return x + y;}

extern "C++" inline __device__ float __fadd_ru(float x, float y) { return x + y;}

extern "C++" inline __device__ float __fadd_rz(float x, float y) { return x + y;}
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ float __ocml_div_rtn_f32(float x, float y);
extern "C++" inline __device__ float __fdiv_rd(float x, float y) {
  return __ocml_div_rtn_f32(x, y);
}

// extern "C++" __device__ float __ocml_div_rte_f32(float x, float y);
extern "C++" inline __device__ float __fdiv_rn(float x, float y) { 
  return __ocml_div_rte_f32(x, y);
}

// extern "C++" __device__ float __ocml_div_rtp_f32(float x, float y);
extern "C++" inline __device__ float __fdiv_ru(float x, float y) { 
  return __ocml_div_rtp_f32(x, y);
}

// extern "C++" __device__ float __ocml_div_rtz_f32(float x, float y);
extern "C++" inline __device__ float __fdiv_rz(float x, float y) {
  return __ocml_div_rtz_f32(x, y);
}
#else
extern "C++" inline __device__ float __fdiv_rd(float x, float y) { return x / y;}

extern "C++" inline __device__ float __fdiv_rn(float x, float y) { return x / y;}

extern "C++" inline __device__ float __fdiv_ru(float x, float y) { return x / y;}

extern "C++" inline __device__ float __fdiv_rz(float x, float y) { return x / y;}
#endif

extern "C++" __device__ float native_divide(float x, float y); // OpenCL
extern "C++" inline __device__ float __fdividef(float x, float y) {
  return native_divide(x, y);
}

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
extern "C++" __device__ float __fmaf_ieee_rd(float x, float y, float z);

extern "C++" __device__ float __fmaf_ieee_rn(float x, float y, float z);

extern "C++" __device__ float __fmaf_ieee_ru(float x, float y, float z);

extern "C++" __device__ float __fmaf_ieee_rz(float x, float y, float z);
#else
extern "C++" inline __device__ float __fmaf_ieee_rd(float x, float y, float z) { return fmaf(x, y, z); }

extern "C++" inline __device__ float __fmaf_ieee_rn(float x, float y, float z) { return fmaf(x, y, z); }

extern "C++" inline __device__ float __fmaf_ieee_ru(float x, float y, float z) { return fmaf(x, y, z); }

extern "C++" inline __device__ float __fmaf_ieee_rz(float x, float y, float z) { return fmaf(x, y, z); }
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ float __ocml_fma_rtn_f32(float x, float y, float z);
extern "C++" inline __device__ float __fmaf_rd(float x, float y, float z) {
   return __ocml_fma_rtn_f32(x, y, z); 
}

// extern "C++" __device__ float __ocml_fma_rte_f32(float, float, float);
extern "C++" inline __device__ float __fmaf_rn(float x, float y, float z) {
  return __ocml_fma_rte_f32(x, y, z);
}

// extern "C++" __device__ float __ocml_fma_rtp_f32(float x, float y, float z);
extern "C++" inline __device__ float __fmaf_ru(float x, float y, float z) {
  return __ocml_fma_rtp_f32(x, y, z);
}

// extern "C++" __device__ float __ocml_fma_rtz_f32(float x, float y, float z);
extern "C++" inline __device__ float __fmaf_rz(float x, float y, float z) {
    return __ocml_fma_rtz_f32(x, y, z);
}
#else
extern "C++" inline __device__ float __fmaf_rd(float x, float y, float z) { return fmaf(x, y, z); }

extern "C++" inline __device__ float __fmaf_rn(float x, float y, float z) { return fmaf(x, y, z); }

extern "C++" inline __device__ float __fmaf_ru(float x, float y, float z) { return fmaf(x, y, z); }

extern "C++" inline __device__ float __fmaf_rz(float x, float y, float z) { return fmaf(x, y, z); }
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ float __ocml_mul_rtn_f32(float x, float y);
extern "C++" inline __device__ float __fmul_rd(float x, float y) {
  return __ocml_mul_rtn_f32(x, y);
}

// extern "C++" __device__ float __ocml_mul_rte_f32(float x, float y);
extern "C++" inline __device__ float __fmul_rn(float x, float y) {
  return __ocml_mul_rte_f32(x, y);
}

// extern "C++" __device__ float __ocml_mul_rtp_f32(float x, float y);
extern "C++" inline __device__ float __fmul_ru(float x, float y) {
  return __ocml_mul_rtp_f32(x, y);
}

// extern "C++" __device__ float __ocml_mul_rtz_f32(float x, float y);
extern "C++" inline __device__ float __fmul_rz(float x, float y) {
  return __ocml_mul_rtz_f32(x, y);
}
#else
extern "C++" inline __device__ float __fmul_rd(float x, float y) { return x * y; }

extern "C++" inline __device__ float __fmul_rn(float x, float y) { return x * y; }

extern "C++" inline __device__ float __fmul_ru(float x, float y) { return x * y; }

extern "C++" inline __device__ float __fmul_rz(float x, float y) { return x * y; }
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
extern "C++" __device__ float __frcp_rd(float x);

extern "C++" __device__ float __frcp_rn(float x);

extern "C++" __device__ float __frcp_ru(float x);

extern "C++" __device__ float __frcp_rz(float x);
#else
extern "C++" inline __device__ float __frcp_rd(float x) { return 1.0f / x; }

extern "C++" inline __device__ float __frcp_rn(float x) { return 1.0f / x; }

extern "C++" inline __device__ float __frcp_ru(float x) { return 1.0f / x; }

extern "C++" inline __device__ float __frcp_rz(float x) { return 1.0f / x; }
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
extern "C++" __device__ float __frsqrt_rn(float x);
#else 
extern "C++" inline __device__ float __frsqrt_rn(float x) { return rsqrt(x); }
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ float __ocml_sqrt_rtn_f32(float x);
extern "C++" inline __device__ float __fsqrt_rd(float x) {
  return __ocml_sqrt_rtn_f32(x);
}

// extern "C++" __device__ float __ocml_sqrt_rte_f32(float x);
extern "C++" inline __device__ float __fsqrt_rn(float x) {
  return __ocml_sqrt_rte_f32(x);
}

// extern "C++" __device__ float __ocml_sqrt_rtp_f32(float x);
extern "C++" inline __device__ float __fsqrt_ru(float x) {
  return __ocml_sqrt_rtp_f32(x);
}

// extern "C++" __device__ float __ocml_sqrt_rtz_f32(float x);
extern "C++" inline __device__ float __fsqrt_rz(float x) {
  return __ocml_sqrt_rtz_f32(x); 
}
#else
extern "C++" inline __device__ float __fsqrt_rd(float x) { return sqrt(x); }

extern "C++" inline __device__ float __fsqrt_rn(float x) { return sqrt(x); }

extern "C++" inline __device__ float __fsqrt_ru(float x) { return sqrt(x); }

extern "C++" inline __device__ float __fsqrt_rz(float x) { return sqrt(x); }
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ float __ocml_sub_rtn_f32(float x, float y);
extern "C++" inline __device__ float __fsub_rd(float x, float y) {
  return __ocml_sub_rtn_f32(x, y); 
}

// extern "C++" __device__ float __ocml_sub_rte_f32(float x, float y);
extern "C++" inline __device__ float __fsub_rn(float x, float y) {
  return __ocml_sub_rte_f32(x, y);
}

// extern "C++" __device__ float __ocml_sub_rtp_f32(float x, float y);
extern "C++" inline __device__ float __fsub_ru(float x, float y) {
  return __ocml_sub_rtp_f32(x, y); 
}

// extern "C++" __device__ float __ocml_sub_rtz_f32(float x, float y);
extern "C++" inline __device__ float __fsub_rz(float x, float y) {
  return __ocml_sub_rtz_f32(x, y);
}
#else
extern "C++" inline __device__ float __fsub_rd(float x, float y) { return x - y; }

extern "C++" inline __device__ float __fsub_rn(float x, float y) { return x - y; }

extern "C++" inline __device__ float __fsub_ru(float x, float y) { return x - y; }

extern "C++" inline __device__ float __fsub_rz(float x, float y) { return x - y; }
#endif

extern "C++" __device__ float native_log10(float x); // OpenCL
extern "C++" inline __device__ float __log10f(float x) {
  return native_log10(x);
}

extern "C++" __device__ float native_log2(float x); // OpenCL
extern "C++" inline __device__ float __log2f(float x) { return native_log2(x); }

extern "C++" __device__ float native_log(float x); // OpenCL
extern "C++" inline __device__ float __logf(float x) {
  return native_log(x);
}

extern "C++" __device__ float native_exp2(float x); // OpenCL
// extern "C++" inline __device__ float native_log2 ( float  x, float  y ); //
// OpenCL (already declared)
extern "C++" inline __device__ float __powf(float x, float y) {
  return native_exp2(y * native_log2(x));
}

extern "C" __device__  float __chip_saturate_f32 ( float  x ); // custom
extern "C++" inline __device__ float __saturatef(float x) {
  return __chip_saturate_f32(x);
}

// extern "C++" __device__ float native_cos(float x); // OpenCL (already
// declared)
extern "C++" __device__ float native_sin(float x); // OpenCL
extern "C++" inline __device__ void __sincosf(float x, float *sptr,
                                              float *cptr) {
  *sptr = native_sin(x);
  *cptr = native_cos(x);
}

// extern "C++" __device__ float native_sin(float x); // OpenCL (already declared)
extern "C++" inline __device__ float __sinf(float x) { return native_sin(x); }

extern "C++" __device__ float native_tan(float x); // OpenCL
extern "C++" inline __device__ float __tanf(float x) { return native_tan(x); }

#endif // include guard

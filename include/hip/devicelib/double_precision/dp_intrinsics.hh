/*
 * Copyright (c) 2021-22 chipStar developers
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

#ifndef HIP_INCLUDE_DEVICELIB_DP_INTRINSICS_H
#define HIP_INCLUDE_DEVICELIB_DP_INTRINSICS_H

#include <hip/devicelib/macros.hh>
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

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ double __ocml_add_rtn_f64(double x, double y);
extern "C++" inline __device__ double __dadd_rd(double x, double y) {
  return __ocml_add_rtn_f64(x, y);
}

// extern "C++" __device__ double __ocml_add_rte_f64(double x, double y);
extern "C++" inline __device__ double __dadd_rn(double x, double y) {
  return __ocml_add_rte_f64(x, y);
}

// extern "C++" __device__ double __ocml_add_rtp_f64(double x, double y);
extern "C++" inline __device__ double __dadd_ru(double x, double y) {
  return __ocml_add_rtp_f64(x, y);
}

// extern "C++" __device__ double __ocml_add_rtz_f64(double x, double y);
extern "C++" inline __device__ double __dadd_rz(double x, double y) {
  return __ocml_add_rtz_f64(x, y);
}
#else
extern "C++" inline __device__ double __dadd_rd(double x, double y) {
  return x + y;
}
extern "C++" inline __device__ double __dadd_rn(double x, double y) {
  return x + y;
}
extern "C++" inline __device__ double __dadd_ru(double x, double y) {
  return x + y;
}
extern "C++" inline __device__ double __dadd_rz(double x, double y) {
  return x + y;
}
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ double __ocml_div_rtn_f64(double x, double y);
extern "C++" inline __device__ double __ddiv_rd(double x, double y) {
  return __ocml_div_rtn_f64(x, y);
}

// extern "C++" __device__ double __ocml_div_rte_f64(double x, double y);
extern "C++" inline __device__ double __ddiv_rn(double x, double y) {
  return __ocml_div_rte_f64(x, y);
}

// extern "C++" __device__ double __ocml_div_rtp_f64(double x, double y);
extern "C++" inline __device__ double __ddiv_ru(double x, double y) {
  return __ocml_div_rtp_f64(x, y);
}

// extern "C++" __device__ double __ocml_div_rtz_f64(double x, double y);
extern "C++" inline __device__ double __ddiv_rz(double x, double y) {
  return __ocml_div_rtz_f64(x, y);
}
#else
extern "C++" inline __device__ double __ddiv_rd(double x, double y) {
  return x / y;
}
extern "C++" inline __device__ double __ddiv_rn(double x, double y) {
  return x / y;
}
extern "C++" inline __device__ double __ddiv_ru(double x, double y) {
  return x / y;
}
extern "C++" inline __device__ double __ddiv_rz(double x, double y) {
  return x / y;
}
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ double __ocml_mul_rtn_f64(double x, double y);
extern "C++" inline __device__ double __dmul_rd(double x, double y) {
  return __ocml_mul_rtn_f64(x, y);
}

// extern "C++" __device__ double __ocml_mul_rte_f64(double x, double y);
extern "C++" inline __device__ double __dmul_rn(double x, double y) {
  return __ocml_mul_rte_f64(x, y);
}

// extern "C++" __device__ double __ocml_mul_rtp_f64(double x, double y);
extern "C++" inline __device__ double __dmul_ru(double x, double y) {
  return __ocml_mul_rtp_f64(x, y);
}

// extern "C++" __device__ double __ocml_mul_rtz_f64(double x, double y);
extern "C++" inline __device__ double __dmul_rz(double x, double y) {
  return __ocml_mul_rtz_f64(x, y);
}
#else
extern "C++" inline __device__ double __dmul_rd(double x, double y) {
  return x * y;
}
extern "C++" inline __device__ double __dmul_rn(double x, double y) {
  return x * y;
}
extern "C++" inline __device__ double __dmul_ru(double x, double y) {
  return x * y;
}
extern "C++" inline __device__ double __dmul_rz(double x, double y) {
  return x * y;
}
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
extern "C++"  __device__ double __drcp_rd(double x);
extern "C++"  __device__ double __drcp_rn(double x);
extern "C++"  __device__ double __drcp_ru(double x);
extern "C++"  __device__ double __drcp_rz(double x);
#else
extern "C++" inline __device__ double __drcp_rd(double x) { return 1.0f / x; }
extern "C++" inline __device__ double __drcp_rn(double x) { return 1.0f / x; }
extern "C++" inline __device__ double __drcp_ru(double x) { return 1.0f / x; }
extern "C++" inline __device__ double __drcp_rz(double x) { return 1.0f / x; }
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ double __ocml_sqrt_rtn_f64(double x);
extern "C++" inline __device__ double __dsqrt_rd(double x) {
  return __ocml_sqrt_rtn_f64(x);
}

// extern "C++" __device__ double __ocml_sqrt_rte_f64(double x);
extern "C++" inline __device__ double __dsqrt_rn(double x) {
  return __ocml_sqrt_rte_f64(x);
}

// extern "C++" __device__ double __ocml_sqrt_rtp_f64(double x);
extern "C++" inline __device__ double __dsqrt_ru(double x) {
  return __ocml_sqrt_rtp_f64(x);
}

// extern "C++" __device__ double __ocml_sqrt_rtz_f64(double x);
extern "C++" inline __device__ double __dsqrt_rz(double x) {
  return __ocml_sqrt_rtz_f64(x);
}
#else
extern "C++" inline __device__ double __dsqrt_rd(double x) { return sqrt(x); }
extern "C++" inline __device__ double __dsqrt_rn(double x) { return sqrt(x); }
extern "C++" inline __device__ double __dsqrt_ru(double x) { return sqrt(x); }
extern "C++" inline __device__ double __dsqrt_rz(double x) { return sqrt(x); }
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ float __ocml_sub_rtn_f64(float x, float y);
extern "C++" inline __device__ double __dsub_rd(double x, double y) {
  return __ocml_sub_rtn_f64(x, y);
}

// extern "C++" __device__ float __ocml_sub_rte_f64(float x, float y);
extern "C++" inline __device__ double __dsub_rn(double x, double y) {
  return __ocml_sub_rte_f64(x, y);
}

// extern "C++" __device__ float __ocml_sub_rtp_f64(float x, float y);
extern "C++" inline __device__ double __dsub_ru(double x, double y) {
  return __ocml_sub_rtp_f64(x, y);
}

// extern "C++" __device__ float __ocml_sub_rtz_f64(float x, float y);
extern "C++" inline __device__ double __dsub_rz(double x, double y) {
  return __ocml_sub_rtz_f64(x, y);
}
#else
extern "C++" inline __device__ double __dsub_rd(double x, double y) {
  return x - y;
}
extern "C++" inline __device__ double __dsub_rn(double x, double y) {
  return x - y;
}
extern "C++" inline __device__ double __dsub_ru(double x, double y) {
  return x - y;
}
extern "C++" inline __device__ double __dsub_rz(double x, double y) {
  return x - y;
}
#endif

#if defined(OCML_BASIC_ROUNDED_OPERATIONS)
// extern "C++" __device__ double __ocml_fma_rtn_f64(double x, double y,
                                                        //  double z);
extern "C++" inline __device__ double __fma_rd(double x, double y, double z) {
  return __ocml_fma_rtn_f64(x, y, z);
}

// extern "C++" __device__ double __ocml_fma_rte_f64(double x, double y,
                                                        //  double z);
extern "C++" inline __device__ double __fma_rn(double x, double y, double z) {
  return __ocml_fma_rte_f64(x, y, z);
}

// extern "C++" __device__ double __ocml_fma_rtp_f64(double x, double y,
                                                        //  double z);
extern "C++" inline __device__ double __fma_ru(double x, double y, double z) {
  return __ocml_fma_rtp_f64(x, y, z);
}

// extern "C++" __device__ double __ocml_fma_rtz_f64(double x, double y,
                                                        //  double z);
extern "C++" inline __device__ double __fma_rz(double x, double y, double z) {
  return __ocml_fma_rtz_f64(x, y, z);
}
#else
extern "C++" __device__ double fma(double x, double y, double z);
extern "C++" inline __device__ double __fma_rd(double x, double y, double z) {
  return fma(x, y, z);
}
extern "C++" inline __device__ double __fma_rn(double x, double y, double z) {
  return fma(x, y, z);
}
extern "C++" inline __device__ double __fma_ru(double x, double y, double z) {
  return fma(x, y, z);
}
extern "C++" inline __device__ double __fma_rz(double x, double y, double z) {
  return fma(x, y, z);
}
#endif

#endif // include guard

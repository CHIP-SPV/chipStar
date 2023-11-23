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

#ifndef HIP_INCLUDE_DEVICELIB_SP_MATH_H
#define HIP_INCLUDE_DEVICELIB_SP_MATH_H

#include <hip/devicelib/macros.hh>

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
extern "C++" __device__ float acos(float x); // OpenCL
extern "C++" inline __device__ float acosf(float x) { return ::acos(x); }

extern "C++" __device__ float acosh(float x); // OpenCL
extern "C++" inline __device__ float acoshf(float x) { return ::acosh(x); }

extern "C++" __device__ float asin(float x); // OpenCL
extern "C++" inline __device__ float asinf(float x) { return ::asin(x); }

extern "C++" __device__ float asinh(float x); // OpenCL
extern "C++" inline __device__ float asinhf(float x) { return ::asinh(x); }

extern "C++" __device__ float atan2(float y, float x); // OpenCL
extern "C++" inline __device__ float atan2f(float y, float x) {
  return ::atan2(y, x);
}

extern "C++" __device__ float atan(float x); // OpenCL
extern "C++" inline __device__ float atanf(float x) { return ::atan(x); }

extern "C++" __device__ float atanh(float x); // OpenCL
extern "C++" inline __device__ float atanhf(float x) { return ::atanh(x); }

extern "C++" __device__ float cbrt(float x); // OpenCL
extern "C++" inline __device__ float cbrtf(float x) { return ::cbrt(x); }

extern "C++" __device__ float ceil(float x); // OpenCL
extern "C++" inline __device__ float ceilf(float x) { return ::ceil(x); }

extern "C++" __device__ float copysign(float x, float y); // OpenCL
extern "C++" inline __device__ float copysignf(float x, float y) {
  return ::copysign(x, y);
}

extern "C++" __device__ float cos(float x); // OpenCL
extern "C++" __device__ float native_cos(float x); // OpenCL
extern "C++" inline __device__ float cosf(float x) {
#ifdef __FAST_MATH__
  return ::native_cos(x);
#else
  return ::cos(x);
#endif
}

extern "C++" __device__ float cosh(float x); // OpenCL
extern "C++" inline __device__ float coshf(float x) { return ::cosh(x); }

extern "C++" __device__ float cospi(float x); // OpenCL
extern "C++" inline __device__ float cospif(float x) { return ::cospi(x); }

extern "C" __device__  float __ocml_i0_f32(float x); // OCML
extern "C++" inline __device__ float cyl_bessel_i0f(float x) { return ::__ocml_i0_f32(x); }

extern "C" __device__  float __ocml_i1_f32(float x); // OCML
extern "C++" inline __device__ float cyl_bessel_i1f(float x) { return ::__ocml_i1_f32(x); }

extern "C++" __device__ float erfc(float x); // OpenCL
extern "C++" inline __device__ float erfcf(float x) { return ::erfc(x); }

extern "C" __device__  float __ocml_erfcinv_f32(float x); // OCML
extern "C++" inline __device__ float erfcinvf(float x) { return ::__ocml_erfcinv_f32(x); }

extern "C" __device__  float __ocml_erfcx_f32(float x); // OCML
extern "C++" inline __device__ float erfcxf(float x) { return ::__ocml_erfcx_f32(x); }

extern "C++" __device__ float erf(float x); // OpenCL
extern "C++" inline __device__ float erff(float x) { return ::erf(x); }

extern "C" __device__  float __ocml_erfinv_f32(float x); // OCML
extern "C++" inline __device__ float erfinvf(float x) { return ::__ocml_erfinv_f32(x); }

extern "C++" __device__ float exp10(float x); // OpenCL
extern "C++" __device__ float native_exp10(float x); // OpenCL
extern "C++" inline __device__ float exp10f(float x) {
#ifdef __FAST_MATH__
  return ::native_exp10(x);
#else
  return ::exp10(x);
#endif
}

extern "C++" __device__ float exp2(float x); // OpenCL
extern "C++" __device__ float native_exp2(float x); // OpenCL
extern "C++" inline __device__ float exp2f(float x) {
#ifdef __FAST_MATH__
  return ::native_exp2(x);
#else
  return ::exp2(x);
#endif
}

extern "C++" __device__ float exp(float x); // OpenCL
extern "C++" __device__ float native_exp(float x); // OpenCL
extern "C++" inline __device__ float expf(float x) {
#ifdef __FAST_MATH__
  return ::native_exp(x);
#else
  return ::exp(x);
#endif
}

extern "C++" __device__ float expm1(float x); // OpenCL
extern "C++" inline __device__ float expm1f(float x) { return ::expm1(x); }

extern "C++" __device__ float fabs(float x); // OpenCL
extern "C++" inline __device__ float fabsf(float x) { return ::fabs(x); }

extern "C++" __device__ float fdim(float x, float y); // OpenCL
extern "C++" inline __device__ float fdimf(float x, float y) {
  return ::fdim(x, y);
}

extern "C++" __device__ float native_divide(float x, float y); // OpenCL
extern "C++" inline __device__ float fdividef(float x, float y) {
#ifdef __FAST_MATH__
  return native_divide(x, y);
#else
  return x / y;
#endif
}

extern "C++" __device__ float floor(float x); // OpenCL
extern "C++" inline __device__ float floorf(float x) { return ::floor(x); }

extern "C++" __device__ float fma(float x, float y, float z); // OpenCL
extern "C++" inline __device__ float fmaf(float x, float y, float z) {
  return ::fma(x, y, z);
}

extern "C++" __device__ float fmax(float x, float y); // OpenCL
extern "C++" inline __device__ float fmaxf(float x, float y) {
  return ::fmax(x, y);
}

extern "C++" __device__ float fmin(float x, float y); // OpenCL
extern "C++" inline __device__ float fminf(float x, float y) {
  return ::fmin(x, y);
}

extern "C++" __device__ float fmod(float x, float y); // OpenCL
extern "C++" inline __device__ float fmodf(float x, float y) {
  return ::fmod(x, y);
}

extern "C++" __device__ float frexp(float x, int *nptr); // OpenCL
extern "C++" inline __device__ float frexpf(float x, int *nptr) {
  return ::frexp(x, nptr);
}

extern "C++" __device__ float hypot(float x, float y); // OpenCL
extern "C++" inline __device__ float hypotf(float x, float y) {
  return ::hypot(x, y);
}

extern "C++" __device__ int ilogb(float x); // OpenCL
extern "C++" inline __device__ int ilogbf(float x) { return ::ilogb(x); }

extern "C" __device__  int __ocml_isfinite_f32(float a); // OCML
extern "C++" inline  __device__  int isfinite(float a) { return __ocml_isfinite_f32(a); }

extern "C" __device__  int __ocml_isinf_f32(float a); // OCML
extern "C++" inline __device__  int isinf(float a) { return __ocml_isinf_f32(a); }

extern "C" __device__  int __ocml_isnan_f32(float a); // OCML
extern "C++" inline __device__  int isnan(float a) { return __ocml_isnan_f32(a); }

extern "C" __device__  float __ocml_j0_f32(float x); // OCML
extern "C++" inline __device__ float j0f(float x) { return ::__ocml_j0_f32(x); }

extern "C" __device__  float __ocml_j1_f32(float x); // OCML
extern "C++" inline __device__ float j1f(float x) { return ::__ocml_j1_f32(x); }

extern "C" __device__  float __chip_jn_f32(int n, float x); // Custom
extern "C++" inline __device__ float jnf(int n, float x) {
  return __chip_jn_f32(n, x);
}

extern "C++" __device__ float ldexp(float x, int exp); // OpenCL
extern "C++" inline __device__ float ldexpf(float x, int exp) {
  return ::ldexp(x, exp);
}

extern "C" __device__  float __ocml_lgamma_f32(float x); // OCML
extern "C++" inline __device__ float lgammaf(float x) { return ::__ocml_lgamma_f32(x); };
extern "C++" inline __device__ float lgamma(float x) { return ::lgammaf(x); }

extern "C" __device__  long long int __chip_llrint_f32(float x); // Custom
extern "C++" inline __device__ long long int llrintf(float x) {
  return __chip_llrint_f32(x);
}

extern "C" __device__  long long int __chip_llround_f32(float x); // Custom
extern "C++" inline __device__ long long int llroundf(float x) {
  return __chip_llround_f32(x);
}

extern "C++" __device__ float log10(float x); // OpenCL
extern "C++" __device__ float native_log10(float x); // OpenCL
extern "C++" inline __device__ float log10f(float x) {
#ifdef __FAST_MATH__
  return ::native_log10(x);
#else
  return ::log10(x);
#endif
}

extern "C++" __device__ float log1p(float x); // OpenCL
extern "C++" inline __device__ float log1pf(float x) { return ::log1p(x); }

extern "C++" __device__ float log2(float x); // OpenCL
extern "C++" __device__ float native_log2(float x); // OpenCL
extern "C++" inline __device__ float log2f(float x) {
#ifdef __FAST_MATH__
  return ::native_log2(x);
#else
  return ::log2(x);
#endif
}

extern "C++" __device__ float logb(float x); // OpenCL
extern "C++" inline __device__ float logbf(float x) { return ::logb(x); }

extern "C++" __device__ float log(float x); // OpenCL
extern "C++" __device__ float native_log(float x); // OpenCL
extern "C++" inline __device__ float logf(float x) {
#ifdef __FAST_MATH__
  return ::native_log(x);
#else
  return ::log(x);
#endif
}

extern "C" __device__  long int __chip_lrint_f32(float x); // Custom
extern "C++" inline __device__ long int lrintf(float x) {
  return __chip_lrint_f32(x);
}
extern "C++" inline __device__ long int lrint(float x) {
  return __chip_lrint_f32(x);
}

extern "C" __device__  long int __chip_lround_f32(float x); // Custom
extern "C++" inline __device__ long int lroundf(float x) {
  return __chip_lround_f32(x);
}

// extern "C++" inline __device__ float fmax(const float a, const float b); //
// OpenCL (already declared)
extern "C++" inline __device__ float max(const float a, const float b) {
  return ::fmax(a, b);
}

extern "C++" __device__ float fmin(const float a, const float b); // OpenCL
extern "C++" inline __device__ float min(const float a, const float b) {
  return ::fmin(a, b);
}

extern "C++" __device__ float modf(float x, float *iptr); // OpenCL
extern "C++" inline __device__ float modff(float x, float *iptr) {
  return ::modf(x, iptr);
}

extern "C++" __device__ float nan(uint nancode); // OpenCL
extern "C++" inline __device__ float nanf(const char *tagp) {
  uint nancode = *reinterpret_cast<const uint *>(tagp);
  return ::nan(nancode);
}

// extern "C++" inline __device__ float __builtin_nearbyintf(float x); // LLVM
// (already declared)
extern "C++" inline __device__ float nearbyintf(float x) {
  return __builtin_nearbyintf(x);
}
// extern "C++" inline __device__ float __builtin_nearbyintf(float x); // LLVM
// (already declared)
extern "C++" inline __device__ float nearbyint(float x) {
  return __builtin_nearbyintf(x);
}

extern "C++" __device__ float nextafter(float x, float y); // OpenCL
extern "C++" inline __device__ float nextafterf(float x, float y) {
  return ::nextafter(x, y);
}

extern "C" __device__ float __ocml_len3_f32(float a, float b,
                                                  float c); // OCML
extern "C++" inline __device__ float norm3df(float a, float b, float c) {
  return __ocml_len3_f32(a, b, c);
}

extern "C" __device__ float __ocml_len4_f32(float a, float b, float c,
                                                  float d); // OCML
extern "C++" inline __device__ float norm4df(float a, float b, float c,
                                             float d) {
  return __ocml_len4_f32(a, b, c, d);
}

extern "C" __device__  float __ocml_ncdf_f32(float x); // OCML
extern "C++" inline __device__ float normcdff(float x) { return ::__ocml_ncdf_f32(x); }

extern "C" __device__  float __ocml_ncdfinv_f32(float x); // OCML
extern "C++" inline __device__ float normcdfinvf(float x) {
  return ::__ocml_ncdfinv_f32(x);
}

extern "C" __device__ float __chip_norm_f32(int dim,
                                                  const float *a); // custom
extern "C++" inline __device__ float normf(int dim, const float *a) {
  return ::__chip_norm_f32(dim, a);
}
extern "C++" inline __device__ float norm(int dim, const float *p) {
  return ::__chip_norm_f32(dim, p);
}

extern "C++" __device__ float pow(float x, float y); // OpenCL
extern "C++" inline __device__ float powf(float x, float y) {
  // TODO This function is affected by the --use_fast_math compiler flag
  return ::pow(x, y);
}

extern "C" __device__  float __ocml_rcbrt_f32(float x); // OCML
extern "C++" inline __device__ float rcbrtf(float x) { return ::__ocml_rcbrt_f32(x); }

extern "C++" __device__ float remainder(float x, float y); // OpenCL
extern "C++" inline __device__ float remainderf(float x, float y) {
  return ::remainder(x, y);
}

extern "C++" __device__ float remquo(float x, float y, int *quo); // OpenCL
extern "C++" inline __device__ float remquof(float x, float y, int *quo) {
  return ::remquo(x, y, quo);
}

extern "C" __device__  float __ocml_rhypot_f32(float x, float y); // OCML
extern "C++" inline __device__ float rhypotf(float x, float y) {
  return ::__ocml_rhypot_f32(x, y);
}

extern "C++" __device__ float rint(float x); // OpenCL
extern "C++" inline __device__ float rintf(float x) { return ::rint(x); }

extern "C" __device__ float __ocml_rlen3_f32(float a, float b,
                                              float c); // OCML
extern "C++" inline __device__ float rnorm3df(float a, float b,
                                              float c) {
  return ::__ocml_rlen3_f32(a, b, c);
}

extern "C" __device__ float __ocml_rlen4_f32(float a, float b, float c,
                                              float d); // OCML
extern "C++" inline __device__ float rnorm4df(float a, float b, float c,
                                              float d){
  return ::__ocml_rlen4_f32(a, b, c, d);
}

extern "C" __device__ float __chip_rnorm_f32(int dim,
                                                   const float *a); //// custom
extern "C++" inline __device__ float rnormf(int dim, const float *a) {
  return ::__chip_rnorm_f32(dim, a);
}
extern "C++" inline __device__ float rnorm(int dim, const float *p) {
  return ::rnormf(dim, p);
}

extern "C++" __device__ float round(float x); // OpenCL
extern "C++" inline __device__ float roundf(float x) {
  return static_cast<float>(::round(x));
}

extern "C++" __device__ float rsqrt(float x);        // OpenCL
extern "C++" __device__ float native_rsqrt(float x); // OpenCL
extern "C++" inline __device__ float rsqrtf(float x) {
#ifdef __FAST_MATH__
  return ::native_rsqrt(x);
#else
  return ::rsqrt(x);
#endif
}

extern "C" __device__  float __ocml_scalbn_f32(float x, int n); // OCML
extern "C++" inline __device__ float scalbnf(float x, int n) {
  return ::__ocml_scalbn_f32(x, n);
}

extern "C" __device__  float __ocml_scalb_f32(float x, float n); // OCML
extern "C++" inline __device__ float scalblnf(float x, long int n) {
  return (n < INT_MAX) ? ::__ocml_scalbn_f32(x, (int)n) : ::__ocml_scalb_f32(x, (float)n);
}

extern "C" __device__  int __ocml_signbit_f32(float a); // OCML

extern "C++" __device__ float sincos(float x, float *sptr); // OpenCL
extern "C++" inline __device__ void sincosf(float x, float *sptr, float *cptr) {
  float tmp;
  *sptr = ::sincos(x, &tmp);
  *cptr = tmp;
}

extern "C" __device__ void __chip_sincospi_f32(float x, float *sptr, float *cptr); // Custom
extern "C++" inline __device__ void sincospif(float x, float *sptr,
                                              float *cptr) {
  return __chip_sincospi_f32(x, sptr, cptr);
}

extern "C++" __device__ float sin(float x); // OpenCL
extern "C++" __device__ float native_sin(float x); // OpenCL
extern "C++" inline __device__ float sinf(float x) {
#ifdef __FAST_MATH__
  return ::native_sin(x);
#else
  return ::sin(x);
#endif
}

extern "C++" __device__ float sinh(float x); // OpenCL
extern "C++" inline __device__ float sinhf(float x) { return ::sinh(x); }

extern "C++" __device__ float sinpi(float x); // OpenCL
extern "C++" inline __device__ float sinpif(float x) { return ::sinpi(x); }

extern "C++" __device__ float sqrt(float x); // OpenCL
extern "C++" __device__ float native_sqrt(float x); // OpenCL
extern "C++" inline __device__ float sqrtf(float x) {
#ifdef __FAST_MATH__
  return ::native_sqrt(x);
#else
  return ::sqrt(x);
#endif
}

extern "C++" __device__ float tan(float x); // OpenCL
extern "C++" __device__ float native_tan(float x); // OpenCL
extern "C++" inline __device__ float tanf(float x) {
#ifdef __FAST_MATH__
  return ::native_tan(x);
#else
  return ::tan(x);
#endif
}

extern "C++" __device__ float tanh(float x); // OpenCL
extern "C++" inline __device__ float tanhf(float x) { return ::tanh(x); }

extern "C++" __device__ float tgamma(float x); // OpenCL
extern "C++" inline __device__ float tgammaf(float x) { return ::tgamma(x); }

extern "C++" __device__ float trunc(float x); // OpenCL
extern "C++" inline __device__ float truncf(float x) { return ::trunc(x); }

extern "C" __device__  float __ocml_y0_f32(float x); // OCML
extern "C++" inline __device__ float y0f(float x) { return ::__ocml_y0_f32(x); }

extern "C" __device__  float __ocml_y1_f32(float x); // OCML
extern "C++" inline __device__ float y1f(float x) { return ::__ocml_y1_f32(x); }

extern "C" __device__  float __chip_yn_f32(int n, float x); // custom TODO RENAME
extern "C++" inline __device__ float ynf(int n, float x) {
  return ::__chip_yn_f32(n, x);
}

namespace std {
// Clang does provide device side std:: functions via HIP include
// wrappers but, alas, the wrappers won't compile on chipStar due to
// presence of AMD specific built-ins.
using ::acos;
using ::acosh;
using ::asin;
using ::atan;
using ::ceil;
using ::ceilf;
using ::copysign;
using ::copysignf;
using ::cos;
using ::cosh;
using ::coshf;
using ::erf;
using ::erfc;
using ::erfcf;
using ::erff;
using ::exp;
using ::expf;
using ::expm1;
using ::floor;
using ::floorf;
using ::lgamma;
using ::lgammaf;
using ::log;
using ::log10;
using ::log10f;
using ::log1p;
using ::log2;
using ::log2f;
using ::logf;
using ::lrint;
using ::lrintf;
using ::nearbyint;
using ::nearbyintf;
using ::nextafter;
using ::nextafterf;
using ::sin;
using ::sinh;
using ::sinhf;
using ::sqrt;
using ::sqrtf;
using ::tan;
using ::tanf;
using ::tanh;
using ::tanhf;
using ::trunc;
} // namespace std

#endif // include guard

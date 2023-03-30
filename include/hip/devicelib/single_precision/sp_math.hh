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

#ifndef HIP_INCLUDE_DEVICELIB_SP_MATH_H
#define HIP_INCLUDE_DEVICELIB_SP_MATH_H

#include <hip/devicelib/macros.hh>

/**
 * @brief Declare as extern - we state that these funcitons are implemented and
 * will be found at link time
 *
 */
extern "C++" inline __device__ float acosf(float x);
extern "C++" inline __device__ float acosh(float x);
extern "C++" inline __device__ float asinf(float x);
extern "C++" inline __device__ float asinh(float x);
extern "C++" inline __device__ float atan2(float y, float x);
extern "C++" inline __device__ float atanf(float x);
extern "C++" inline __device__ float atanh(float x);
extern "C++" inline __device__ float cbrt(float x);
extern "C++" inline __device__ float ceilf(float x);
extern "C++" inline __device__ float copysignf(float x, float y);
extern "C++" inline __device__ float cosf(float x);
extern "C++" inline __device__ float coshf(float x);
extern "C++" inline __device__ float cospi(float x);
extern "C" inline __device__ float opencl_cyl_bessel_i0_f(float x);
extern "C" inline __device__ float opencl_cyl_bessel_i1_f(float x);
extern "C++" inline __device__ float erfcf(float x);
extern "C++" inline __device__ float erfcinv(float x);
extern "C++" inline __device__ float erfcx(float x);
extern "C++" inline __device__ float erff(float x);
extern "C++" inline __device__ float erfinv(float x);
extern "C++" inline __device__ float exp10(float x);
extern "C++" inline __device__ float exp2(float x);
extern "C++" inline __device__ float expf(float x);
extern "C++" inline __device__ float expm1f(float x);
extern "C++" inline __device__ float fabs(float x);
extern "C++" inline __device__ float fdim(float x, float y);
extern "C++" inline __device__ float fdividef(float x, float y);
extern "C++" inline __device__ float floorf(float x);
extern "C++" inline __device__ float fma(float x, float y, float z);
extern "C++" inline __device__ float fmax(float x, float y);
extern "C++" inline __device__ float fmin(float x, float y);
extern "C++" inline __device__ float fmod(float x, float y);
extern "C++" inline __device__ float frexp(float x, int *nptr);
extern "C++" inline __device__ float hypot(float x, float y);
extern "C++" inline __device__ int ilogb(float x);
// extern "C++" inline __device__ __RETURN_TYPE isfinite(float a);
// extern "C++" inline __device__ __RETURN_TYPE isinf(float a);
// extern "C++" inline __device__ __RETURN_TYPE isnan(float a);
extern "C++" inline __device__ float j0f(float x);
extern "C++" inline __device__ float j1f(float x);
// extern "C++" inline __device__ float jnf(int n, float x);
// extern "C++" inline __device__ float ldexpf(float x, int exp);
extern "C++" inline __device__ float lgammaf(float x);
extern "C++" inline __device__ long long int llrintf(float x);
extern "C++" inline __device__ long long int llroundf(float x);
extern "C++" inline __device__ float log10f(float x);
extern "C++" inline __device__ float log1pf(float x);
extern "C++" inline __device__ float log2f(float x);
extern "C++" inline __device__ float logbf(float x);
extern "C++" inline __device__ float logf(float x);
extern "C++" inline __device__ long int lrintf(float x);
extern "C++" inline __device__ long int lroundf(float x);
extern "C++" inline __device__ float max(const float a, const float b);
extern "C++" inline __device__ float min(const float a, const float b);
// extern "C++" inline __device__ float modff(float x, float *iptr);
// extern "C++" inline __device__ float nanf(const char *tagp);
extern "C++" inline __device__ float nearbyintf(float x);
extern "C++" inline __device__ float nextafterf(float x, float y);
extern "C" inline __device__ float opencl_norm3d_f(float a, float b, float c); // use OCML?
extern "C" inline __device__ float opencl_norm4d_f(float a, float b, float c, float d); // use OCML
extern "C++" inline __device__ float normcdf(float x);
extern "C++" inline __device__ float normcdfinv(float x);
extern "C++" inline __device__ float norm(int dim, const float *p);
// extern "C++" inline __device__ float pow(float x, float y);
extern "C++" inline __device__ float rcbrt(float x);
extern "C++" inline __device__ float remainderf(float x, float y);
// extern "C++" inline __device__ float remquof(float x, float y, int *quo);
extern "C++" inline __device__ float rhypotf(float x, float y);
extern "C++" inline __device__ float rintf(float x);
extern "C++" inline __device__ float rnorm3df(float a, float b, float c); // OCML - C++ linkage
extern "C++" inline __device__ float rnorm4df(float a, float b, float c,
                                             float d);
extern "C++" inline __device__ float rnormf(int dim, const float *p);
extern "C++" inline __device__ float roundf(float x);
extern "C++" inline __device__ float rsqrtf(float x);
// extern "C++" inline __device__ float scalblnf(float x, long int n);
// extern "C++" inline __device__ float scalbnf(float x, int n);
// extern "C++" inline __device__ __RETURN_TYPE signbit(float a);
extern "C++" inline __device__ void sincos(float x, float *cptr);
extern "C++" inline __device__ void sincospif(float x, float *sptr,
                                              float *cptr);
extern "C++" inline __device__ float sinf(float x);
extern "C++" inline __device__ float sinhf(float x);
extern "C++" inline __device__ float sinpif(float x);
extern "C++" inline __device__ float sqrtf(float x);
extern "C++" inline __device__ float tanf(float x);
extern "C++" inline __device__ float tanhf(float x);
extern "C++" inline __device__ float tgammaf(float x);
extern "C++" inline __device__ float truncf(float x);
extern "C++" inline __device__ float y0f(float x);
extern "C++" inline __device__ float y1f(float x);
// extern "C++" inline __device__ float ynf(int n, float x);

/**
 * @brief Bind these calls to the previously declared extern functions
 *
 */
extern "C++" inline __device__ float acosf(float x) { return ::acos(x); }
extern "C++" inline __device__ float acoshf(float x) { return ::acosh(x); }
extern "C++" inline __device__ float asinf(float x) { return ::asin(x); }
extern "C++" inline __device__ float asinhf(float x) { return ::asinh(x); }
extern "C++" inline __device__ float atan2f(float y, float x) {
  return ::atan2(y, x);
}
extern "C++" inline __device__ float atanf(float x) { return ::atan(x); }
extern "C++" inline __device__ float atanhf(float x) { return ::atanh(x); }
extern "C++" inline __device__ float cbrtf(float x) { return ::cbrt(x); }
extern "C++" inline __device__ float ceilf(float x) { return ::ceil(x); }
extern "C++" inline __device__ float copysignf(float x, float y) {
  return ::copysign(x, y);
}
extern "C++" inline __device__ float cosf(float x) { return ::cos(x); }
extern "C++" inline __device__ float coshf(float x) { return ::cosh(x); }
extern "C++" inline __device__ float cospif(float x) { return ::cospi(x); }
extern "C++" inline __device__ float cyl_bessel_i0f(float x) {
  return ::opencl_cyl_bessel_i0_f(x);
}
extern "C++" inline __device__ float cyl_bessel_i1f(float x) {
  return ::opencl_cyl_bessel_i1_f(x);
}
extern "C++" inline __device__ float erfcf(float x) { return ::erfc(x); }
extern "C++" inline __device__ float erfcinvf(float x) { return ::erfcinv(x); }
extern "C++" inline __device__ float erfcxf(float x) { return ::erfcx(x); }
extern "C++" inline __device__ float erff(float x) { return ::erf(x); }
extern "C++" inline __device__ float erfinvf(float x) { return ::erfinv(x); }
extern "C++" inline __device__ float exp10f(float x) { return ::exp10(x); }
extern "C++" inline __device__ float exp2f(float x) { return ::exp2(x); }
extern "C++" inline __device__ float expf(float x) { return ::exp(x); }
extern "C++" inline __device__ float expm1f(float x) { return ::expm1(x); }
extern "C++" inline __device__ float fabsf(float x) { return ::fabs(x); }
extern "C++" inline __device__ float fdimf(float x, float y) {
  return ::fdim(x, y);
}
extern "C++" inline __device__ float fdividef(float x, float y) {
  return x / y;
}
extern "C++" inline __device__ float floorf(float x) { return ::floor(x); }
extern "C++" inline __device__ float fmaf(float x, float y, float z) {
  return ::fma(x, y, z);
}
extern "C++" inline __device__ float fmaxf(float x, float y) {
  return ::fmax(x, y);
}
extern "C++" inline __device__ float fminf(float x, float y) {
  return ::fmin(x, y);
}
extern "C++" inline __device__ float fmodf(float x, float y) {
  return ::fmod(x, y);
}
extern "C++" inline __device__ float frexpf(float x, int *nptr) {
  return ::frexp(x, nptr);
}
extern "C++" inline __device__ float hypotf(float x, float y) {
  return ::hypot(x, y);
}
extern "C++" inline __device__ int ilogbf(float x) { return ::ilogb(x); }
// extern "C++" inline __device__  __RETURN_TYPE 	isfinite ( float  a )
// extern "C++" inline __device__  __RETURN_TYPE 	isinf ( float  a )
// extern "C++" inline __device__  __RETURN_TYPE 	isnan ( float  a )
extern "C++" inline __device__ float j0f(float x) { return ::j0f(x); }
extern "C++" inline __device__ float j1f(float x) { return ::j1f(x); }
// extern "C++" inline __device__  float jnf ( int  n, float  x )
// extern "C++" inline __device__  float ldexpf ( float  x, int  exp )
extern "C++" inline __device__ float lgammaf(float x) { return (lgamma(x)); };
extern "C++" inline __device__ long long int llrintf(float x) {
  return lrintf(x);
}
extern "C++" inline __device__ long long int llroundf(float x) {
  return lroundf(x);
}
extern "C++" inline __device__ float log10f(float x) { return ::log10(x); }
extern "C++" inline __device__ float log1pf(float x) { return ::log1p(x); }
extern "C++" inline __device__ float log2f(float x) { return ::log2(x); }
extern "C++" inline __device__ float logbf(float x) { return ::logbf(x); }
extern "C++" inline __device__ float logf(float x) { return ::log(x); }
extern "C++" inline __device__ long int lrintf(float x) {
  // return convert_long(rint(x));
}
extern "C++" inline __device__ long int lroundf(float x) {
  // return convert_long(round(x));
}
// extern "C++" inline __device__  float max ( const float  a, const float  b )
// extern "C++" inline __device__  float min ( const float  a, const float  b )
// extern "C++" inline __device__  float modff ( float  x, float* iptr )
// extern "C++" inline __device__  float nanf ( const char* tagp )
extern "C++" inline __device__ float nearbyintf(float x) {
  return __builtin_nearbyintf(x);
}
extern "C++" inline __device__ float nextafterf(float x, float y) {
  return ::nextafter(x, y);
}
extern "C++" inline __device__ float norm3df(float a, float b, float c) {
  return ::opencl_norm3d_f(a, b, c);
}
extern "C++" inline __device__ float norm4df(float a, float b, float c,
                                             float d) {
  return ::opencl_norm4d_f(a, b, c, d);
}
extern "C++" inline __device__ float normcdff(float x) { return ::normcdf(x); }
extern "C++" inline __device__ float normcdfinvf(float x) {
  return ::normcdfinv(x);
}
extern "C++" inline __device__ float normf(int dim, const float *a) {
  float r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return ::sqrtf(r);
}
extern "C++" inline __device__ float powf(float x, float y) {
  return ::pow(x, y);
}
extern "C++" inline __device__ float rcbrtf(float x) { return ::rcbrtf(x); }
extern "C++" inline __device__ float remainderf(float x, float y) {
  return ::remainderf(x, y);
}
//  extern "C++" inline __device__  float remquof ( float  x, float  y, int* quo
//  )
extern "C++" inline __device__ float rhypotf(float x, float y) {
  ::rhypotf(x, y);
}
//  extern
extern "C++" inline __device__ float rintf(float x) { return rint(x); }
// extern "C++" inline __device__ float rnorm3df(float a, float b, float c) {
//   return ::rnorm3d(a, b, c);
// }
// extern "C++" inline __device__ float rnorm4df(float a, float b, float c,
//                                               float d) {
//   return ::rnorm4d(a, b, c, d);
// }
extern "C++" inline __device__ float rnormf(int dim, const float *a) {
  float r = 0;
  while (dim--) {
    r += a[0] * a[0];
    ++a;
  }

  return ::sqrtf(r);
}
extern "C++" inline __device__ float roundf(float x) { return round(x); }
extern "C++" inline __device__ float rsqrtf(float x) { return ::rsqrtf(x); }
// extern "C++" inline __device__  float scalblnf ( float  x, long int  n )
// extern "C++" inline __device__  float scalbnf ( float  x, int  n )
// extern "C++" inline __device__  __RETURN_TYPE 	signbit ( float  a )
extern "C++" inline __device__ void sincosf(float x, float *sptr, float *cptr) {
  // TODO
  // float tmp;
  // *sptr = ::sincos(x, &tmp);
  // *cptr = tmp;
}
extern "C++" inline __device__ void sincospif(float x, float *sptr,
                                              float *cptr) {
  // TODO
  // *sptr = ::sinpif(x);
  // *cptr = ::sincosf(x);
}
extern "C++" inline __device__ float sinf(float x) { return ::sin(x); }
extern "C++" inline __device__ float sinhf(float x) { return ::sinh(x); }
extern "C++" inline __device__ float sinpif(float x) { return ::sinpif(x); }
extern "C++" inline __device__ float sqrtf(float x) { return ::sqrt(x); }
extern "C++" inline __device__ float tanf(float x) { return ::tan(x); }
extern "C++" inline __device__ float tanhf(float x) { return ::tanh(x); }
extern "C++" inline __device__ float tgammaf(float x) { return ::tgammaf(x); }
extern "C++" inline __device__ float truncf(float x) { return ::trunc(x); }
extern "C++" inline __device__ float y0f(float x) { return ::y0f(x); }
extern "C++" inline __device__ float y1f(float x) { return ::y1f(x); }
// extern "C++" inline __device__  float ynf ( int  n, float  x )

/**
 * @brief Bind the overloaded versions i.e. cos(float x) -> cosf(x)
 *
 */
// extern "C++" inline __device__ float acos(float x);
// extern "C++" inline __device__ float acosh(float x);
// extern "C++" inline __device__ float asin(float x);
// extern "C++" inline __device__ float asinh(float x);
// extern "C++" inline __device__ float atan2(float y, float x);
// extern "C++" inline __device__ float atan(float x);
// extern "C++" inline __device__ float atanh(float x);
// extern "C++" inline __device__ float cbrt(float x);
// extern "C++" inline __device__ float ceil(float x);
// extern "C++" inline __device__ float copysign(float x, float y);
// extern "C++" inline __device__ float cos(float x);
// extern "C++" inline __device__ float cosh(float x);
extern "C++" inline __device__ float cospi(float x) { return ::cospi(x); }
extern "C++" inline __device__ float cyl_bessel_i0(float x) {
  return ::cyl_bessel_i0f(x);
}
extern "C++" inline __device__ float cyl_bessel_i1(float x) {
  return ::cyl_bessel_i1f(x);
}
// extern "C++" inline __device__ float erfc(float x);
extern "C++" inline __device__ float erfcinv(float x) { return ::erfcinv(x); }
extern "C++" inline __device__ float erfcx(float x) { return ::erfcx(x); }
// extern "C++" inline __device__ float erf(float x);
extern "C++" inline __device__ float erfinv(float x) { return ::erfinv(x); }
extern "C++" inline __device__ float exp10(float x) { return ::exp10(x); }
extern "C++" inline __device__ float exp2(float x) { return ::exp2(x); }
// extern "C++" inline __device__ float exp(float x);
// extern "C++" inline __device__ float expm1(float x);
extern "C++" inline __device__ float fabs(float x) { return ::fabs(x); }
extern "C++" inline __device__ float fdim(float x, float y) {
  return ::fdim(x, y);
}
// extern "C++" inline __device__ float fdivide(float x, float y);
// extern "C++" inline __device__ float floor(float x);
extern "C++" inline __device__ float fma(float x, float y, float z) {
  return ::fma(x, y, z);
}
extern "C++" inline __device__ float fmax(float x, float y) {
  return ::fmax(x, y);
}
extern "C++" inline __device__ float fmin(float x, float y) {
  return ::fmin(x, y);
}
extern "C++" inline __device__ float fmod(float x, float y) {
  return ::fmod(x, y);
}
// extern "C++" inline __device__ float frexp(float x, int *nptr);
extern "C++" inline __device__ float hypot(float x, float y) {
  return ::hypot(x, y);
}
extern "C++" inline __device__ int ilogb(float x) { return ::ilogb(x); }
// extern "C++" inline __device__ __RETURN_TYPE isfinite(float a);
// extern "C++" inline __device__ __RETURN_TYPE isinf(float a);
// extern "C++" inline __device__ __RETURN_TYPE isnan(float a);
extern "C++" inline __device__ float j0(float x) { return ::j0f(x); }
extern "C++" inline __device__ float j1(float x) { return ::j1f(x); }
// extern "C++" inline __device__ float jn(int n, float x);
// extern "C++" inline __device__ float ldexp(float x, int exp);
// extern "C++" inline __device__ float lgamma(float x);
// extern "C++" inline __device__ long long int llrint(float x);
// extern "C++" inline __device__ long long int llround(float x);
// extern "C++" inline __device__ float log10(float x);
// extern "C++" inline __device__ float log1p(float x);
// extern "C++" inline __device__ float log2(float x);
// extern "C++" inline __device__ float logb(float x);
// extern "C++" inline __device__ float log(float x);
extern "C++" inline __device__ long int lrint(float x) {
  // return convert_long(rint(x));
}
// extern "C++" inline __device__ long int lround(float x);
extern "C++" inline __device__ float max(const float a, const float b) {
  return ::fmax(a, b);
}
extern "C++" inline __device__ float min(const float a, const float b) {
  return ::fmin(a, b);
}
// extern "C++" inline __device__ float modf(float x, float *iptr);
// extern "C++" inline __device__ float nan(const char *tagp);
extern "C++" inline __device__ float nearbyint(float x) {
  return __builtin_nearbyintf(x);
}
// extern "C++" inline __device__ float nextafter(float x, float y);
// extern "C++" inline __device__ float norm3d(float a, float b, float c);
// extern "C++" inline __device__ float norm4d(float a, float b, float c,
//                                              float d);
// extern "C++" inline __device__ float normcdf(float x);
// extern "C++" inline __device__ float normcdfinv(float x);
extern "C++" inline __device__ float norm(int dim, const float *p) {
  return ::normf(dim, p);
}
// extern "C++" inline __device__ float pow(float x, float y);
// extern "C++" inline __device__ float rcbrt(float x);
// extern "C++" inline __device__ float remainder(float x, float y);
// extern "C++" inline __device__ float remquo(float x, float y, int *quo);
// extern "C++" inline __device__ float rhypot(float x, float y);
// extern "C++" inline __device__ float rint(float x);
// extern "C++" inline __device__ float rnorm3d(float a, float b, float c);
// extern "C++" inline __device__ float rnorm4d(float a, float b, float c,
//                                               float d);
extern "C++" inline __device__ float rnorm(int dim, const float *p) {
  return ::rnormf(dim, p);
}
// extern "C++" inline __device__ float round(float x);
// extern "C++" inline __device__ float rsqrt(float x);
// extern "C++" inline __device__ float scalbln(float x, long int n);
// extern "C++" inline __device__ float scalbn(float x, int n);
// extern "C++" inline __device__ __RETURN_TYPE signbit(float a);
// extern "C++" inline __device__ void sincos(float x, float *sptr, float
// *cptr); extern "C++" inline __device__ void sincospi(float x, float *sptr,
//                                               float *cptr);
// extern "C++" inline __device__ float sin(float x);
// extern "C++" inline __device__ float sinh(float x);
// extern "C++" inline __device__ float sinpi(float x);
// extern "C++" inline __device__ float sqrt(float x);
// extern "C++" inline __device__ float tan(float x);
// extern "C++" inline __device__ float tanh(float x);
// extern "C++" inline __device__ float tgamma(float x);
// extern "C++" inline __device__ float trunc(float x);
// extern "C++" inline __device__ float y0(float x);
// extern "C++" inline __device__ float y1(float x);
// extern "C++" inline __device__ float yn(int n, float x);

namespace std {
// Clang does provide device side std:: functions via HIP include
// wrappers but, alas, the wrappers won't compile on CHIP-SPV due to
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

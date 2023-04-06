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

#ifndef HIP_INCLUDE_DEVICELIB_DP_MATH_H
#define HIP_INCLUDE_DEVICELIB_DP_MATH_H

#include <hip/devicelib/macros.hh>

// Declare as extern - we state that these funcitons are implemented and will be
// found at link time
extern "C++" inline __device__ double acos(double x);

extern "C++" inline __device__ double acosh(double x);
extern "C++" inline __device__ double acosh ( double  x );
extern "C++" inline __device__ double asin(double x);

extern "C++" inline __device__ double asinh(double x);
extern "C++" inline __device__ double atan(double x);
extern "C++" inline __device__ double atan2(double y, double x);
extern "C++" inline __device__ double atanh(double x);
extern "C++" inline __device__ double cbrt(double x);
extern "C++" inline __device__ double ceil(double x);
extern "C++" inline __device__ double copysign(double x, double y);
extern "C++" inline __device__ double cos(double x);
extern "C++" inline __device__ double cosh(double x);
extern "C++" inline __device__ double cospi(double x);

extern "C" inline __device__ double opencl_cyl_bessel_i0_d(double x);
extern "C++" inline __device__ double cyl_bessel_i0 ( double  x ) {
  return ::opencl_cyl_bessel_i0_d(x);
}
extern "C" inline __device__ double opencl_cyl_bessel_i1_d(double x);
extern "C++" inline __device__ double cyl_bessel_i1 ( double  x ) {
  return ::opencl_cyl_bessel_i1_d(x);
}

extern "C++" inline __device__ double erf(double x);
extern "C++" inline __device__ double erfc(double x);
extern "C++" inline __device__ double erfcinv(double x);
extern "C++" inline __device__ double erfcx(double x);
extern "C++" inline __device__ double erfinv(double x);
extern "C++" inline __device__ double exp(double x);
extern "C++" inline __device__ double exp10(double x);
extern "C++" inline __device__ double exp2(double x);
extern "C++" inline __device__ double expm1(double x);
extern "C++" inline __device__ double fabs(double x);
extern "C++" inline __device__ double fdim(double x, double y);
extern "C++" inline __device__ double floor(double x);
extern "C++" inline __device__ double fma(double x, double y, double z);
extern "C++" inline __device__ double fmax(double, double);
extern "C++" inline __device__ double fmin(double x, double y);
extern "C++" inline __device__ double fmod(double x, double y);
extern "C++" inline __device__ double frexp(double x, int *nptr);
extern "C++" inline __device__ double hypot(double x, double y);
extern "C++" inline __device__  int ilogb(double x);
extern "C++" inline __device__  int 	isfinite ( double  a );
extern "C++" inline __device__  int 	isinf ( double  a );
extern "C++" inline __device__  int 	isnan ( double  a );
extern "C++" inline __device__ double j0(double x);
extern "C++" inline __device__ double j1(double x);
extern "C++" inline __device__ double jn(int n, double x);
extern "C++" inline __device__ double ldexp(double x, int exp);
extern "C++" inline __device__ double lgamma(double x);
extern "C++" inline __device__ long long int llrint(double x);
extern "C++" inline __device__ long long int llround(double x);
extern "C++" inline __device__ double log(double x);
extern "C++" inline __device__ double log10(double x);
extern "C++" inline __device__ double log1p(double x);
extern "C++" inline __device__ double log2(double x);
extern "C++" inline __device__ double logb(double x);
extern "C++" inline __device__ long int lrint(double x);
extern "C++" inline __device__ long int lround(double x);
extern "C++" inline __device__ double max(const double a, const float b);
extern "C++" inline __device__ double max(const float a, const double b);
extern "C++" inline __device__ double max(const double a, const double b);
extern "C++" inline __device__ double min(const double a, const float b);
extern "C++" inline __device__ double min(const float a, const double b);
extern "C++" inline __device__ double min(const double a, const double b);
extern "C++" inline __device__ double modf(double x, double *iptr);

// extern "C++" inline __device__ double nan(const char *tagp); // OpenCL (already declared)

extern "C++" inline __device__ double nearbyint(double x) {
  return __builtin_nearbyint(x);
}

extern "C++" inline __device__ double nextafter(double x, double y);
extern "C++" inline __device__ double norm(int dim, const double *p); // TODO use OCML

extern "C" inline __device__ double opencl_norm3d_d(double a, double b, double c);
extern "C" inline __device__ double norm3d(double a, double b, double c);

extern "C" inline __device__ double opencl_norm3d_d(double a, double b, double c);
extern "C" inline __device__ double norm4d(double a, double b, double c,
                                             double d);

extern "C++" inline __device__ double normcdf(double x);
extern "C++" inline __device__ double normcdfinv(double x);
extern "C++" inline __device__ double pow(double x, double y);
extern "C++" inline __device__ double rcbrt(double x);
extern "C++" inline __device__ double remainder(double x, double y);
extern "C++" inline __device__ double remquo(double x, double y, int *quo);
extern "C++" inline __device__ double rhypot(double x, double y);
extern "C++" inline __device__ double rint(double x);
extern "C++" inline __device__ double rnorm(int dim, const double *p); // TODO use OCML
extern "C++" inline __device__ double rnorm3d(double a, double b, double c);
extern "C++" inline __device__ double rnorm4d(double a, double b, double c,
                                              double d);
extern "C++" inline __device__ double round(double x);
extern "C++" inline __device__ double rsqrt(double x);
extern "C++" inline __device__ double scalbln(double x, long int n);
extern "C++" inline __device__ double scalbn(double x, int n);
extern "C++" inline __device__ int 	signbit ( double  a );
extern "C++" inline __device__ double sin(double x);
extern "C++" inline __device__ void sincos(double x, double *sptr,
                                           double *cptr);
extern "C++" inline __device__ void sincospi(double x, double *sptr,
                                             double *cptr);
extern "C++" inline __device__ double sinh(double x);
extern "C++" inline __device__ double sinpi(double x);
extern "C++" inline __device__ double sqrt(double x);
extern "C++" inline __device__ double tan(double x);
extern "C++" inline __device__ double tanh(double x);
extern "C++" inline __device__ double tgamma(double x);
extern "C++" inline __device__ double trunc(double x);
extern "C++" inline __device__ double y0(double x);
extern "C++" inline __device__ double y1(double x);
extern "C++" inline __device__ double yn(int n, double x);

namespace std {
// Clang does provide device side std:: functions via HIP include
// wrappers but, alas, the wrappers won't compile on CHIP-SPV due to
// presence of AMD specific built-ins.
using ::acos;
using ::asin;
using ::atan;
using ::ceil;
using ::copysign;
using ::cos;
using ::cosh;
using ::erf;
using ::erfc;
using ::exp;
using ::expm1;
using ::floor;
using ::lgamma;
using ::log;
using ::log10;
using ::log1p;
using ::log2;
using ::nearbyint;
using ::nextafter;
using ::sin;
using ::sinh;
using ::sqrt;
using ::tan;
using ::tanh;
using ::trunc;
} // namespace std

#endif // include guard

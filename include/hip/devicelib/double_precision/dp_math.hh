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

#ifndef HIP_INCLUDE_DEVICELIB_DP_MATH_H
#define HIP_INCLUDE_DEVICELIB_DP_MATH_H

#include <hip/devicelib/macros.hh>

extern "C++" __device__ double acos(double x); // OpenCL

extern "C++" __device__ double acosh(double x); // OpenCL
extern "C++" __device__ double acosh ( double  x ); // OpenCL
extern "C++" __device__ double asin(double x); // OpenCL

extern "C++" __device__ double asinh(double x); // OpenCL
extern "C++" __device__ double atan(double x); // OpenCL
extern "C++" __device__ double atan2(double y, double x); // OpenCL
extern "C++" __device__ double atanh(double x); // OpenCL
extern "C++" __device__ double cbrt(double x); // OpenCL
extern "C++" __device__ double ceil(double x); // OpenCL
extern "C++" __device__ double copysign(double x, double y); // OpenCL

#ifdef __FAST_MATH__
extern "C++" __device__ double native_cos(double x); // OpenCL
extern "C++" inline __device__ double cos(double x) { return ::native_cos(x); }
#else
extern "C++" __device__ double cos(double x); // OpenCL
#endif

extern "C++" __device__ double cosh(double x); // OpenCL
extern "C++" __device__ double cospi(double x); // OpenCL

extern "C" __device__  double __ocml_i0_f64(double x);
extern "C++" inline __device__ double cyl_bessel_i0 ( double  x ) {
  return ::__ocml_i0_f64(x);
}
extern "C" __device__  double __ocml_i1_f64(double x);
extern "C++" inline __device__ double cyl_bessel_i1 ( double  x ) {
  return ::__ocml_i1_f64(x);
}

extern "C++" __device__ double erf(double x);
extern "C++" __device__ double erfc(double x);

extern "C" __device__  double __ocml_erfcinv_f64(double x); // OCML
extern "C++" inline __device__ double erfcinv(double x) {
  return ::__ocml_erfcinv_f64(x);
}

extern "C" __device__  double __ocml_erfcx_f64(double x); // OCML
extern "C++" inline __device__ double erfcx(double x) {
  return ::__ocml_erfcx_f64(x);
}

extern "C" __device__  double __ocml_erfinv_f64(double x);
extern "C++" inline __device__ double erfinv(double x) {
  return ::__ocml_erfinv_f64(x);
}

#ifdef __FAST_MATH__
extern "C++" __device__ double native_exp(double x); // OpenCL
extern "C++" inline __device__ double exp(double x) { return ::native_exp(x); }
#else
extern "C++" __device__ double exp(double x); // OpenCL
#endif

#ifdef __FAST_MATH__
extern "C++" __device__ double native_exp10(double x); // OpenCL
extern "C++" inline __device__ double exp10(double x) {
  return ::native_exp10(x);
}
#else
extern "C++" __device__ double exp10(double x); // OpenCL
#endif

#ifdef __FAST_MATH__
extern "C++" __device__ double native_exp2(double x); // OpenCL
extern "C++" inline __device__ double exp2(double x) {
  return ::native_exp2(x);
}
#else
extern "C++" __device__ double exp2(double x);  // OpenCL
#endif

extern "C++" __device__ double expm1(double x); // OpenCL
extern "C++" __device__ double fabs(double x); // OpenCL
extern "C++" __device__ double fdim(double x, double y); // OpenCL
extern "C++" __device__ double floor(double x); // OpenCL
extern "C++" __device__ double fma(double x, double y, double z); // OpenCL
extern "C++" __device__ double fmax(double, double); // OpenCL
extern "C++" __device__ double fmin(double x, double y); // OpenCL
extern "C++" __device__ double fmod(double x, double y); // OpenCL
extern "C++" __device__ double frexp(double x, int *nptr); // OpenCL
extern "C++" __device__ double hypot(double x, double y); // OpenCL
extern "C++" __device__  int ilogb(double x); // OpenCL

extern "C++" __device__  int 	isfinite ( double  a ); // OpenCL
extern "C++" __device__  int 	isinf ( double  a ); // OpenCL
extern "C++" __device__  int 	isnan ( double  a ); // OpenCL

extern "C" __device__  double __ocml_j0_f64(double x); // OCML
extern "C++" inline __device__ double j0(double x) {
  return ::__ocml_j0_f64(x);
}

extern "C" __device__  double __ocml_j0_f64(double x); // OCML
extern "C++" inline __device__ double j1(double x) {
  return ::__ocml_j0_f64(x);
}

extern "C" __device__  double __chip_jn_f64(int n, double x); // Custom 
extern "C++" inline __device__ double jn(int n, double x) {
  return ::__chip_jn_f64(n, x);
}

extern "C++" __device__ double ldexp(double x, int exp); // OpenCL
extern "C++" __device__ double lgamma(double x); // OpenCL

extern "C" __device__  long long int __chip_llrint_f64(double x); // Custom
extern "C++" inline __device__ long long int llrint(double x) {
  return ::__chip_llrint_f64(x);
}

extern "C" __device__  long long int __chip_llround_f64(double x); // Custom
extern "C++" inline __device__ long long int llround(double x) {
  return ::__chip_llround_f64(x);
}

#ifdef __FAST_MATH__
extern "C++" __device__ double native_log(double x); // OpenCL
extern "C++" inline __device__ double log(double x) { return ::native_log(x); }
#else
extern "C++" __device__ double log(double x);   // OpenCL
#endif

#ifdef __FAST_MATH__
extern "C++" __device__ double native_log10(double x); // OpenCL
extern "C++" inline __device__ double log10(double x) {
  return ::native_log10(x);
}
#else
extern "C++" __device__ double log10(double x); // OpenCL
#endif

extern "C++" __device__ double log1p(double x); // OpenCL

#ifdef __FAST_MATH__
extern "C++" __device__ double native_log2(double x); // OpenCL
extern "C++" inline __device__ double log2(double x) {
  return ::native_log2(x);
}
#else
extern "C++" __device__ double log2(double x);  // OpenCL
#endif

extern "C++" __device__ double logb(double x); // OpenCL

extern "C" __device__  long int __chip_lrint_f64(double x); // Custom
extern "C++" inline __device__ long int lrint(double x) {
  return ::__chip_lrint_f64(x);
}

extern "C" __device__  long int __chip_lround_f64(double x); // Custom
extern "C++" inline __device__ long int lround(double x) {
  return ::__chip_lround_f64(x);
}

extern "C++" __device__ double max(const double a, const float b); // OpenCL
extern "C++" __device__ double max(const float a, const double b); // OpenCL
extern "C++" __device__ double max(const double a, const double b); // OpenCL
extern "C++" __device__ double min(const double a, const float b); // OpenCL
extern "C++" __device__ double min(const float a, const double b); // OpenCL
extern "C++" __device__ double min(const double a, const double b); // OpenCL
extern "C++" __device__ double modf(double x, double *iptr); // OpenCL


extern "C++" __device__ double nan(ulong nancode); // OpenCL
extern "C++" inline __device__ double nan(const char *tagp) {
  ulong nancode = *reinterpret_cast<const ulong *>(tagp);
  return ::nan(nancode);
}

extern "C++" inline __device__ double nearbyint(double x) {
  return __builtin_nearbyint(x);
}

extern "C++" __device__ double nextafter(double x, double y); // OpenCL

extern "C" __device__  double __chip_norm_f64(int dim, const double *p); // Custom
extern "C++" inline __device__ double norm(int dim, const double *p) {
  return ::__chip_norm_f64(dim, p);
}

extern "C" __device__  double __ocml_len3_f64(double a, double b, double c); // OCML
extern "C++" inline __device__ double norm3d(double a, double b, double c) {
  return ::__ocml_len3_f64(a, b, c);
}

extern "C" __device__  double __ocml_len4_f64(double a, double b, double c, double d); // OCML
extern "C++" inline __device__ double norm4d(double a, double b, double c,
                                             double d) {
  return ::__ocml_len4_f64(a, b, c, d);
}

extern "C" __device__  double __ocml_ncdf_f64(double x); // OCML
extern "C++" inline __device__ double normcdf(double x) {
  return ::__ocml_ncdf_f64(x);
}

extern "C" __device__  double __ocml_ncdfinv_f64(double x); // OCML
extern "C++" inline __device__ double normcdfinv(double x) {
  return ::__ocml_ncdfinv_f64(x);
}

extern "C++" __device__ double pow(double x, double y); // OpenCL

extern "C" __device__  double __ocml_rcbrt_f64(double x); // OCML
extern "C++" inline __device__ double rcbrt(double x) {
  return ::__ocml_rcbrt_f64(x);
}

extern "C++" __device__ double remainder(double x, double y); // OpenCL
extern "C++" __device__ double remquo(double x, double y, int *quo); // OpenCL

extern "C" __device__  double __ocml_rhypot_f64(double x, double y);
extern "C++" inline __device__ double rhypot(double x, double y) {
  return ::__ocml_rhypot_f64(x, y);
}

extern "C++" __device__ double rint(double x); // OpenCL

extern "C" __device__  double __chip_rnorm_f64(int dim, const double *p); // Custom
extern "C++" inline __device__ double rnorm(int dim, const double *p) {
  return ::__chip_rnorm_f64(dim, p);
}

extern "C" __device__  double __ocml_rlen3_f64(double a, double b, double c); // OCML
extern "C++" inline __device__ double rnorm3d(double a, double b, double c) {
  return ::__ocml_rlen3_f64(a, b, c);
}

extern "C" __device__ double __ocml_rlen4_f64(double a, double b, double c,
                                              double d); // OCML
extern "C++" inline __device__ double rnorm4d(double a, double b, double c,
                                              double d) {
  return ::__ocml_rlen4_f64(a, b, c, d);
}

extern "C++" __device__ double round(double x); // OpenCL

#ifdef __FAST_MATH__
extern "C++" __device__ double native_rsqrt(double x); // OpenCL
extern "C++" inline __device__ double rsqrt(double x) {
  return ::native_rsqrt(x);
}
#else
extern "C++" __device__ double rsqrt(double x); // OpenCL
#endif

extern "C" __device__  double __ocml_scalb_f64(double x, double n);
extern "C++" inline __device__ double scalbln(double x, long int n) {
  // No implementatin for scalbln(double, long) in OCML so promote 'n'
  // and call OCML's scalb instead.
  return ::__ocml_scalb_f64(x, n);
}

extern "C" __device__  double __ocml_scalbn_f64(double x, int n);
extern "C++" inline __device__ double scalbn(double x, int n)  {
  return ::__ocml_scalbn_f64(x, n);
}

extern "C++" __device__ int 	signbit ( double  a ); // OpenCL

#ifdef __FAST_MATH__
extern "C++" __device__ double native_sin(double x); // OpenCL
extern "C++" inline __device__ double sin(double x) {
  return ::native_sin(x);
}
#else
extern "C++" __device__ double sin(double x); // OpenCL
#endif

extern "C++" __device__ double sincos(double x, double *sptr); // OpenCL
extern "C++" inline __device__ void sincos(double x, double *sptr,
                                           double *cptr) {
  double tmp;
  *sptr = ::sincos(x, &tmp);
  *cptr = tmp;
}

extern "C" __device__ void __chip_sincospi_f64(double x, double *sptr,
                                                    double *cptr); // OCML
extern "C++" inline __device__ void sincospi(double x, double *sptr,
                                             double *cptr) {
 return ::__chip_sincospi_f64(x, sptr, cptr);
}

extern "C++" __device__ double sinh(double x); // OpenCL
extern "C++" __device__ double sinpi(double x); // OpenCL

#ifdef __FAST_MATH__
extern "C++" __device__ double native_sqrt(double x); // OpenCL
extern "C++" inline __device__ double sqrt(double x) {
  return ::native_sqrt(x);
}
#else
extern "C++" __device__ double sqrt(double x); // OpenCL
#endif

#ifdef __FAST_MATH__
extern "C++" __device__ double native_tan(double x); // OpenCL
extern "C++" inline __device__ double tan(double x) { return ::native_tan(x); }
#else
extern "C++" __device__ double tan(double x);  // OpenCL
#endif

extern "C++" __device__ double tanh(double x); // OpenCL
extern "C++" __device__ double tgamma(double x); // OpenCL
extern "C++" __device__ double trunc(double x); // OpenCL

extern "C" __device__  double __ocml_y0_f64(double x); // OCML
extern "C++" inline __device__ double y0(double x) {
  return ::__ocml_y0_f64(x);
}

extern "C" __device__  double __ocml_y1_f64(double x); // OCML
extern "C++" inline __device__ double y1(double x) {
  return ::__ocml_y1_f64(x);
}

extern "C" __device__  double __chip_yn_f64(int n, double x); // custom
extern "C++" inline __device__ double yn(int n, double x) {
  return ::__chip_yn_f64(n, x);
}

namespace std {
// Clang does provide device side std:: functions via HIP include
// wrappers but, alas, the wrappers won't compile on chipStar due to
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
using ::pow;
using ::sin;
using ::sinh;
using ::sqrt;
using ::tan;
using ::tanh;
using ::trunc;
} // namespace std

#endif // include guard

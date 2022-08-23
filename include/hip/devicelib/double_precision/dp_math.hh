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

extern "C++" {

extern __device__ double rint(double x);
extern __device__ double round(double x);
extern __device__ long int convert_long(double x);

__device__ long int lrint(double x) { return convert_long(rint(x)); }
__device__ long int lround(double x) { return convert_long(round(x)); }

__device__ long long int llrint(double x) { return lrint(x); }
__device__ long long int llround(double x) { return lround(x); }

extern __device__ double rnorm3d(double a, double b, double c);
extern __device__ double rnorm4d(double a, double b, double c, double d);

extern __device__ double lgamma ( double  x );
}

// __device__ double acos(double x)
// __device__​ double acosh ( double  x )
// __device__​ double asin ( double  x )
// __device__​ double asinh ( double  x )
// __device__​ double atan ( double  x )
// __device__​ double atan2 ( double  y, double  x )
// __device__​ double atanh ( double  x )
// __device__​ double cbrt ( double  x )
// __device__​ double ceil ( double  x )
// __device__​ double copysign ( double  x, double  y )
// __device__​ double cos ( double  x )
// __device__​ double cosh ( double  x )
// __device__​ double cospi ( double  x )
// __device__​ double cyl_bessel_i0 ( double  x )
// __device__​ double cyl_bessel_i1 ( double  x )
// __device__​ double erf ( double  x )
// __device__​ double erfc ( double  x )
// __device__​ double erfcinv ( double  x )
// __device__​ double erfcx ( double  x )
// __device__​ double erfinv ( double  x )
// __device__​ double exp ( double  x )
// __device__​ double exp10 ( double  x )
// __device__​ double exp2 ( double  x )
// __device__​ double expm1 ( double  x )
// __device__​ double fabs ( double  x )
// __device__​ double fdim ( double  x, double  y )
// __device__​ double floor ( double  x )
// __device__​ double fma ( double  x, double  y, double  z )
// __device__​ double fmax ( double , double )
// __device__​ double fmin ( double  x, double  y )
// __device__​ double fmod ( double  x, double  y )
// __device__​ double frexp ( double  x, int* nptr )
// __device__​ double hypot ( double  x, double  y )
// __device__​ int ilogb ( double  x )
// __device__​ __RETURN_TYPE 	isfinite ( double  a )
// __device__​ __RETURN_TYPE 	isinf ( double  a )
// __device__​ __RETURN_TYPE 	isnan ( double  a )
// __device__​ double j0 ( double  x )
// __device__​ double j1 ( double  x )
// __device__​ double jn ( int  n, double  x )
// __device__​ double ldexp ( double  x, int  exp )
// __device__​ double log ( double  x )
// __device__​ double log10 ( double  x )
// __device__​ double log1p ( double  x )
// __device__​ double log2 ( double  x )
// __device__​ double logb ( double  x )
// __device__​ double max ( const double  a, const float  b )
// __device__​ double max ( const float  a, const double  b )
// __device__​ double max ( const double  a, const double  b )
// __device__​ double min ( const double  a, const float  b )
// __device__​ double min ( const float  a, const double  b )
// __device__​ double min ( const double  a, const double  b )
// __device__​ double modf ( double  x, double* iptr )
// __device__​ double nan ( const char* tagp )
// __device__​ double nearbyint ( double  x )
// __device__​ double nextafter ( double  x, double  y )
// __device__​ double norm ( int  dim, const double* p )
// __device__​ double norm3d ( double  a, double  b, double  c )
// __device__​ double norm4d ( double  a, double  b, double  c, double  d)
// __device__​ double normcdf ( double  x )
// __device__​ double normcdfinv ( double  x )
// __device__​ double pow ( double  x, double  y )
// __device__​ double rcbrt ( double  x )
// __device__​ double remainder ( double  x, double  y )
// __device__​ double remquo ( double  x, double  y, int* quo )
// __device__​ double rhypot ( double  x, double  y )
// __device__​ double rnorm ( int  dim, const double* p )
// __device__​ double rsqrt ( double  x )
// __device__​ double scalbln ( double  x, long int  n )
// __device__​ double scalbn ( double  x, int  n )
// __device__​ __RETURN_TYPE 	signbit ( double  a )
// __device__​ double sin ( double  x )
// __device__​ void sincos ( double  x, double* sptr, double* cptr )
// __device__​ void sincospi ( double  x, double* sptr, double* cptr )
// __device__​ double sinh ( double  x )
// __device__​ double sinpi ( double  x )
// __device__​ double sqrt ( double  x )
// __device__​ double tan ( double  x )
// __device__​ double tanh ( double  x )
// __device__​ double tgamma ( double  x )
// __device__​ double trunc ( double  x )
// __device__​ double y0 ( double  x )
// __device__​ double y1 ( double  x )
// __device__​ double yn ( int  n, double  x )

#endif // include guard

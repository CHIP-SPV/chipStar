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


#ifndef HIP_INCLUDE_DEVICELIB_INT_INTRINSICS_H
#define HIP_INCLUDE_DEVICELIB_INT_INTRINSICS_H

#include <hip/devicelib/macros.hh>

extern "C" __device__  unsigned int 	__chip_brev ( unsigned int  x ); // Custom
extern "C++" inline __device__ unsigned int 	__brev ( unsigned int  x ) { return __chip_brev(x); }

extern "C" __device__  unsigned long long int 	__chip_brevll ( unsigned long long int x); // Custom
extern "C++" inline __device__ unsigned long long int 	__brevll ( unsigned long long int x) { return __chip_brevll(x); }

extern "C" __device__  unsigned int 	__chip_byte_perm ( unsigned int  x, unsigned int y, unsigned int  s ); // Custom
extern "C++" inline __device__ unsigned int 	__byte_perm ( unsigned int  x, unsigned int y, unsigned int  s ) { return __chip_byte_perm(x, y, s); }

extern "C++" __device__ int clz ( int  x ); // OpenCL
extern "C++" inline __device__ int __clz ( int  x ) { return clz(x); }

extern "C++" __device__ int clz ( long long int x ); // OpenCL
extern "C++" inline __device__ int __clzll ( long long int x ) {
  return clz(x);
}

extern "C" __device__  int __chip_ffs ( int  x ); // Custom
extern "C++" inline __device__ int __ffs ( int  x ) { return __chip_ffs(x); }

extern "C" __device__  int __chip_ffsll ( long long int x ); // Custom
extern "C++" inline __device__ int __ffsll ( long long int x ) { return __chip_ffsll(x); }

extern "C" __device__ unsigned int
__chip_funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift); // Custom
extern "C++" inline __device__ unsigned int
__funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift) {
  return __chip_funnelshift_l(lo, hi, shift);
}

extern "C" __device__ unsigned int
__chip_funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift); // Custom
extern "C++" inline __device__ unsigned int
__funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift) {
  return __chip_funnelshift_lc(lo, hi, shift);
}

extern "C" __device__ unsigned int
__chip_funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift); // Custom
extern "C++" inline __device__ unsigned int
__funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift) {
  return __chip_funnelshift_r(lo, hi, shift);
}

extern "C" __device__ unsigned int
__chip_funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift); // Custom
extern "C++" inline __device__ unsigned int
__funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift) {
  return __chip_funnelshift_rc(lo, hi, shift);
}

extern "C++" __device__ int hadd ( int  x, int  y ); // OpenCL
extern "C++" inline __device__ int __hadd ( int  x, int  y ) { return hadd(x, y); }

extern "C++" __device__ int mul24 ( int  x, int  y ); // OpenCL
extern "C++" inline __device__ int __mul24 ( int  x, int  y ) { return mul24(x, y); }

extern "C" __device__  long long int 	__chip_mul64hi ( long long int x, long long int y ); // Custom
extern "C++" inline __device__ long long int 	__mul64hi ( long long int x, long long int y ) { return __chip_mul64hi(x, y); }

extern "C++" __device__ int mul_hi ( int  x, int  y ); // OpenCL
extern "C++" inline __device__ int __mulhi ( int  x, int  y ) { return mul_hi(x, y); }

extern "C++" __device__ int popcount ( unsigned int  x ); // OpenCL
extern "C++" inline __device__ int __popc ( unsigned int  x ) { return popcount(x); }

extern "C++" __device__ int popcount ( unsigned long long int x ); // OpenCL
extern "C++" inline __device__ int __popcll ( unsigned long long int x ) { return popcount(x); }

extern "C++" __device__ int rhadd ( int  x, int  y ); // OpenCL
extern "C++" inline __device__ int __rhadd ( int  x, int  y ) { return rhadd(x, y); }

extern "C" __device__  unsigned int 	__chip_sad ( int  x, int  y, unsigned int  z ); //Custom
extern "C++" inline __device__ unsigned int 	__sad ( int  x, int  y, unsigned int  z ) { return __chip_sad(x, y, z); }

extern "C++" __device__ unsigned int 	hadd ( unsigned int  x, unsigned int  y ); // OpenCL
extern "C++" inline __device__ unsigned int 	__uhadd ( unsigned int  x, unsigned int  y ) { return hadd(x, y); }

extern "C++" __device__ unsigned int 	mul24 ( unsigned int  x, unsigned int  y ); // OpenCL
extern "C++" inline __device__ unsigned int 	__umul24 ( unsigned int  x, unsigned int  y ) { return mul24(x, y); }

extern "C" __device__  unsigned long long int 	__chip_umul64hi ( unsigned long long int x, unsigned long long int y ); // Custom
extern "C++" inline __device__ unsigned long long int 	__umul64hi ( unsigned long long int x, unsigned long long int y ) { return __chip_umul64hi(x, y); }

extern "C++" __device__ unsigned int 	mul_hi ( unsigned int  x, unsigned int  y ); // OpenCL
extern "C++" inline __device__ unsigned int 	__umulhi ( unsigned int  x, unsigned int  y ) { return mul_hi(x, y); }

extern "C++" __device__ unsigned int 	hadd ( unsigned int  x, unsigned int  y ); // OpenCL
extern "C++" inline __device__ unsigned int 	__urhadd ( unsigned int  x, unsigned int  y ) { return rhadd(x, y); }

extern "C" __device__  unsigned int 	__chip_usad ( unsigned int  x, unsigned int  y, unsigned int  z ); // Custom
extern "C++" inline __device__ unsigned int 	__usad ( unsigned int  x, unsigned int  y, unsigned int  z ) { return __chip_usad(x, y, z); }

#endif // include guard

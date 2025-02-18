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

#ifndef HIP_INCLUDE_DEVICELIB_INT_INTRINSICS_H
#define HIP_INCLUDE_DEVICELIB_INT_INTRINSICS_H

#include <hip/devicelib/macros.hh>

// __bitinsert_* intrinsics are not found in the HIP programming
// manual but they are provided by hipamd.
//
// They replace a 'width' bits sized block in 'src0' staring at 'offset'
// with least significant bits extracted from 'src1'.

extern "C" __device__ unsigned int __chip_bitinsert_u32(unsigned int src0,
                                                        unsigned int src1,
                                                        unsigned int offset,
                                                        unsigned int width);
extern "C++" inline __device__ unsigned int
__bitinsert_u32(unsigned int src0, unsigned int src1, unsigned int offset,
                unsigned int width) {
  return __chip_bitinsert_u32(src0, src1, offset, width);
}

extern "C" __device__ unsigned long long int
__chip_bitinsert_u64(unsigned long long int src0, unsigned long long int src1,
                     unsigned long long int offset,
                     unsigned long long int width);
extern "C++" inline __device__ unsigned long long int
__bitinsert_u64(unsigned long long int src0, unsigned long long int src1,
                unsigned long long int offset, unsigned long long int width) {
  return __chip_bitinsert_u64(src0, src1, offset, width);
}

// __bitextract_* intrinsics extract bits from src0 starting at offset with given width
extern "C" __device__ unsigned int __chip_bitextract_u32(unsigned int src0,
                                                        unsigned int offset,
                                                        unsigned int width);
extern "C++" inline __device__ unsigned int
__bitextract_u32(unsigned int src0, unsigned int offset, unsigned int width) {
  return __chip_bitextract_u32(src0, offset, width);
}

extern "C" __device__ unsigned long long int
__chip_bitextract_u64(unsigned long long int src0,
                      unsigned int offset,
                      unsigned int width);
extern "C++" inline __device__ unsigned long long int
__bitextract_u64(unsigned long long int src0, unsigned int offset, unsigned int width) {
  return __chip_bitextract_u64(src0, offset, width);
}

// int was replaced with int
// int64_t was replaced with long long int
extern "C" __device__ int __chip__fns32(unsigned long long int mask,
                                        unsigned int base, int offset);
extern "C++" inline __device__ int __fns32(unsigned long long int mask,
                                           unsigned int base, int offset) {
  return __chip__fns32(mask, base, offset);
}

extern "C" __device__ int __chip__fns64(unsigned long long int mask,
                                        unsigned int base, int offset);
extern "C++" inline __device__ int __fns64(unsigned long long int mask,
                                           unsigned int base, int offset) {
  return __chip__fns64(mask, base, offset);
}

extern "C" __device__ unsigned int __chip_brev(unsigned int x); // Custom
extern "C++" inline __device__ unsigned int __brev(unsigned int x) {
  return __chip_brev(x);
}

extern "C" __device__ unsigned long long int
__chip_brevll(unsigned long long int x); // Custom
extern "C++" inline __device__ unsigned long long int
__brevll(unsigned long long int x) {
  return __chip_brevll(x);
}

extern "C" __device__ unsigned int
__chip_byte_perm(unsigned int x, unsigned int y, unsigned int s); // Custom
extern "C++" inline __device__ unsigned int
__byte_perm(unsigned int x, unsigned int y, unsigned int s) {
  return __chip_byte_perm(x, y, s);
}

extern "C++" inline __device__ int __clz(int x) { return __builtin_clz(x); }

extern "C++" inline __device__ int __clzll(long long int x) {
  return __builtin_clzl(x);
}

extern "C" __device__ int __chip_ffs(int x); // Custom
extern "C++" inline __device__ int __ffs(int x) { return __chip_ffs(x); }

extern "C" __device__ int __chip_ffsll(long long int x); // Custom
extern "C++" inline __device__ int __ffsll(long long int x) {
  return __chip_ffsll(x);
}

extern "C" __device__ unsigned int
__chip_funnelshift_l(unsigned int lo, unsigned int hi,
                     unsigned int shift); // Custom
extern "C++" inline __device__ unsigned int
__funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift) {
  return __chip_funnelshift_l(lo, hi, shift);
}

extern "C" __device__ unsigned int
__chip_funnelshift_lc(unsigned int lo, unsigned int hi,
                      unsigned int shift); // Custom
extern "C++" inline __device__ unsigned int
__funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift) {
  return __chip_funnelshift_lc(lo, hi, shift);
}

extern "C" __device__ unsigned int
__chip_funnelshift_r(unsigned int lo, unsigned int hi,
                     unsigned int shift); // Custom
extern "C++" inline __device__ unsigned int
__funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift) {
  return __chip_funnelshift_r(lo, hi, shift);
}

extern "C" __device__ unsigned int
__chip_funnelshift_rc(unsigned int lo, unsigned int hi,
                      unsigned int shift); // Custom
extern "C++" inline __device__ unsigned int
__funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift) {
  return __chip_funnelshift_rc(lo, hi, shift);
}

extern "C++" __device__ int hadd(int x, int y); // OpenCL
extern "C++" inline __device__ int __hadd(int x, int y) { return hadd(x, y); }

extern "C++" __device__ int mul24(int x, int y); // OpenCL
extern "C++" inline __device__ int __mul24(int x, int y) { return mul24(x, y); }

extern "C" __device__ long long int __chip_mul64hi(long long int x,
                                                   long long int y); // Custom
extern "C++" inline __device__ long long int __mul64hi(long long int x,
                                                       long long int y) {
  return __chip_mul64hi(x, y);
}

extern "C++" __device__ int mul_hi(int x, int y); // OpenCL
extern "C++" inline __device__ int __mulhi(int x, int y) {
  return mul_hi(x, y);
}

extern "C++" inline __device__ int __popc(unsigned int x) {
  return __builtin_popcount(x);
}

extern "C++" inline __device__ int __popcll(unsigned long long int x) {
  return (int)__builtin_popcountl(x);
}

extern "C++" __device__ int rhadd(int x, int y); // OpenCL
extern "C++" inline __device__ int __rhadd(int x, int y) { return rhadd(x, y); }

extern "C" __device__ unsigned int __chip_sad(int x, int y,
                                              unsigned int z); // Custom
extern "C++" inline __device__ unsigned int __sad(int x, int y,
                                                  unsigned int z) {
  return __chip_sad(x, y, z);
}

extern "C++" __device__ unsigned int hadd(unsigned int x,
                                          unsigned int y); // OpenCL
extern "C++" inline __device__ unsigned int __uhadd(unsigned int x,
                                                    unsigned int y) {
  return hadd(x, y);
}

extern "C++" __device__ unsigned int mul24(unsigned int x,
                                           unsigned int y); // OpenCL
extern "C++" inline __device__ unsigned int __umul24(unsigned int x,
                                                     unsigned int y) {
  return mul24(x, y);
}

extern "C" __device__ unsigned long long int
__chip_umul64hi(unsigned long long int x, unsigned long long int y); // Custom
extern "C++" inline __device__ unsigned long long int
__umul64hi(unsigned long long int x, unsigned long long int y) {
  return __chip_umul64hi(x, y);
}

extern "C++" __device__ unsigned int mul_hi(unsigned int x,
                                            unsigned int y); // OpenCL
extern "C++" inline __device__ unsigned int __umulhi(unsigned int x,
                                                     unsigned int y) {
  return mul_hi(x, y);
}

extern "C++" __device__ unsigned int hadd(unsigned int x,
                                          unsigned int y); // OpenCL
extern "C++" inline __device__ unsigned int __urhadd(unsigned int x,
                                                     unsigned int y) {
  return rhadd(x, y);
}

extern "C" __device__ unsigned int __chip_usad(unsigned int x, unsigned int y,
                                               unsigned int z); // Custom
extern "C++" inline __device__ unsigned int
__usad(unsigned int x, unsigned int y, unsigned int z) {
  return __chip_usad(x, y, z);
}

#endif // include guard

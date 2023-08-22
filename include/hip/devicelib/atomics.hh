/*
 * Copyright (c) 2021-23 chipStar developers
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

#ifndef HIP_INCLUDE_DEVICELIB_ATOMICS
#define HIP_INCLUDE_DEVICELIB_ATOMICS

#include <hip/host_defines.h>
#include <hip/devicelib/macros.hh>

// Copied from HIP programming guide:
// https://docs.amd.com/bundle/HIP-Programming-Guide-v5.0/page/Programming_with_HIP.html
// Slightly modified to group operations
extern "C" __device__ int __chip_atomic_add_i(int *address, int val);
extern "C++" inline __device__ int atomicAdd(int *address, int val) {
  return __chip_atomic_add_i(address, val);
}

extern "C" __device__ unsigned int __chip_atomic_add_u(unsigned int *address,
                                                       unsigned int val);
extern "C++" inline __device__ unsigned int atomicAdd(unsigned int *address,
                                                      unsigned int val) {
  return __chip_atomic_add_u(address, val);
}

extern "C" __device__ unsigned long __chip_atomic_add_l(unsigned long *address,
                                                        unsigned long val);

#ifdef CHIP_ENABLE_NON_COMPLIANT_DEVICELIB_CODE
// At least rocPRIM tests call the unsigned long variant although it's not
// listed in the user manual. Annoyingly, size_t is typically defined as
// unsigned long.
// FIXME: We should check that unsigned long is 64bits for the host.
extern "C++" inline __device__ unsigned long atomicAdd(unsigned long *address,
                                                       unsigned long val) {
  return __chip_atomic_add_l(address, val);
}
#endif

// FIXME: We should check that unsigned long long is 64bits for the host.
extern "C" __device__ unsigned long long
__chip_atomic_add_l(unsigned long long *address, unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicAdd(unsigned long long *address, unsigned long long val) {
  return __chip_atomic_add_l(address, val);
}

extern "C" __device__ float __chip_atomic_add_f32(float *address, float val);
extern "C++" inline __device__ float atomicAdd(float *address, float val) {
  return __chip_atomic_add_f32(address, val);
}

extern "C" __device__ double __chip_atomic_add_f64(double *address, double val);
extern "C++" inline __device__ double atomicAdd(double *address, double val) {
  return __chip_atomic_add_f64(address, val);
}

extern "C" __device__ int __chip_atomic_add_system_i(int *address, int val);
extern "C++" inline __device__ int atomicAdd_system(int *address, int val) {
  return __chip_atomic_add_system_i(address, val);
}

extern "C" __device__ unsigned int
__chip_atomic_add_system_u(unsigned int *address, unsigned int val);
extern "C++" inline __device__ unsigned int
atomicAdd_system(unsigned int *address, unsigned int val) {
  return __chip_atomic_add_system_u(address, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_add_system_l(unsigned long long *address, unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicAdd_system(unsigned long long *address, unsigned long long val) {
  return __chip_atomic_add_system_l(address, val);
}

extern "C" __device__ float __chip_atomic_add_system_f32(float *address,
                                                         float val);
extern "C++" inline __device__ float atomicAdd_system(float *address,
                                                      float val) {
  return __chip_atomic_add_system_f32(address, val);
}

extern "C" __device__ double __chip_atomic_add_system_f64(double *address,
                                                          double val);
extern "C++" inline __device__ double atomicAdd_system(double *address,
                                                       double val) {
  return __chip_atomic_add_system_f64(address, val);
}

extern "C" __device__ int __chip_atomic_sub_i(int *address, int val);
extern "C++" inline __device__ int atomicSub(int *address, int val) {
  return __chip_atomic_sub_i(address, val);
}

extern "C" __device__ unsigned int __chip_atomic_sub_u(unsigned int *address,
                                                       unsigned int val);
extern "C++" inline __device__ unsigned int atomicSub(unsigned int *address,
                                                      unsigned int val) {
  return __chip_atomic_sub_u(address, val);
}

extern "C" __device__ int __chip_atomic_sub_system_i(int *address, int val);
extern "C++" inline __device__ int atomicSub_system(int *address, int val) {
  return __chip_atomic_sub_system_i(address, val);
}

extern "C" __device__ unsigned int
__chip_atomic_sub_system_u(unsigned int *address, unsigned int val);
extern "C++" inline __device__ unsigned int
atomicSub_system(unsigned int *address, unsigned int val) {
  return __chip_atomic_sub_system_u(address, val);
}

extern "C" __device__ int __chip_atomic_exch_i(int *address, int val);
extern "C++" inline __device__ int atomicExch(int *address, int val) {
  return __chip_atomic_exch_i(address, val);
}

extern "C" __device__ unsigned int __chip_atomic_exch_u(unsigned int *address,
                                                        unsigned int val);
extern "C++" inline __device__ unsigned int atomicExch(unsigned int *address,
                                                       unsigned int val) {
  return __chip_atomic_exch_u(address, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_exch_ul(unsigned long long int *address,
                      unsigned long long int val);
extern "C++" inline __device__ unsigned long long
atomicExch(unsigned long long int *address, unsigned long long int val) {
  return __chip_atomic_exch_ul(address, val);
}

extern "C" __device__ float __chip_atomic_exch_f32(float *address, float val);
extern "C++" inline __device__ float atomicExch(float *address, float val) {
  return __chip_atomic_exch_f32(address, val);
}

extern "C" __device__ int __chip_atomic_exch_system_i(int *address, int val);
extern "C++" inline __device__ int atomicExch_system(int *address, int val) {
  return __chip_atomic_exch_system_i(address, val);
}

extern "C" __device__ unsigned int
__chip_atomic_exch_system_u(unsigned int *address, unsigned int val);
extern "C++" inline __device__ unsigned int
atomicExch_system(unsigned int *address, unsigned int val) {
  return __chip_atomic_exch_system_u(address, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_exch_system_l(unsigned long long *address,
                            unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicExch_system(unsigned long long *address, unsigned long long val) {
  return __chip_atomic_exch_system_l(address, val);
}

extern "C" __device__ float __chip_atomic_exch_system_f32(float *address,
                                                          float val);
extern "C++" inline __device__ float atomicExch_system(float *address,
                                                       float val) {
  return __chip_atomic_exch_system_f32(address, val);
} // Error on the website

extern "C" __device__ int __chip_atomic_min_i(int *address, int val);
extern "C++" inline __device__ int atomicMin(int *address, int val) {
  return __chip_atomic_min_i(address, val);
}

extern "C" __device__ unsigned int __chip_atomic_min_u(unsigned int *address,
                                                       unsigned int val);
extern "C++" inline __device__ unsigned int atomicMin(unsigned int *address,
                                                      unsigned int val) {
  return __chip_atomic_min_u(address, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_min_l(unsigned long long *address, unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicMin(unsigned long long *address, unsigned long long val) {
  return __chip_atomic_min_l(address, val);
}

extern "C" __device__ int __chip_atomic_min_system_i(int *address, int val);
extern "C++" inline __device__ int atomicMin_system(int *address, int val) {
  return __chip_atomic_min_system_i(address, val);
}

extern "C" __device__ unsigned int
__chip_atomic_min_system_u(unsigned int *address, unsigned int val);
extern "C++" inline __device__ unsigned int
atomicMin_system(unsigned int *address, unsigned int val) {
  return __chip_atomic_min_system_u(address, val);
}

extern "C" __device__ int __chip_atomic_max_i(int *address, int val);
extern "C++" inline __device__ int atomicMax(int *address, int val) {
  return __chip_atomic_max_i(address, val);
}

extern "C" __device__ unsigned int __chip_atomic_max_u(unsigned int *address,
                                                       unsigned int val);
extern "C++" inline __device__ unsigned int atomicMax(unsigned int *address,
                                                      unsigned int val) {
  return __chip_atomic_max_u(address, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_max_l(unsigned long long *address, unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicMax(unsigned long long *address, unsigned long long val) {
  return __chip_atomic_max_l(address, val);
}

extern "C" __device__ int __chip_atomic_max_system_i(int *address, int val);
extern "C++" inline __device__ int atomicMax_system(int *address, int val) {
  return __chip_atomic_max_system_i(address, val);
}

extern "C" __device__ unsigned int
__chip_atomic_max_system_u(unsigned int *address, unsigned int val);
extern "C++" inline __device__ unsigned int
atomicMax_system(unsigned int *address, unsigned int val) {
  return __chip_atomic_max_system_u(address, val);
}

extern "C" __device__ unsigned int __chip_atomic_inc2_u(unsigned int *address,
                                                        unsigned int val);
extern "C++" inline __device__ unsigned atomicInc(unsigned *address) {
  return __chip_atomic_inc2_u(address, 1);
}

extern "C++" inline __device__ unsigned
atomicInc(unsigned *address, unsigned val) { // Undocumented
  return __chip_atomic_inc2_u(address, val);
}

extern "C" __device__ unsigned int __chip_atomic_dec2_u(unsigned int *address,
                                                        unsigned int val);
extern "C++" inline __device__ unsigned int atomicDec(unsigned int *address) {
  return __chip_atomic_dec2_u(address, 1);
}

extern "C++" inline __device__ unsigned int
atomicDec(unsigned *address, unsigned val) { // Undocumented
  return __chip_atomic_dec2_u(address, val);
}

extern "C" __device__ int __chip_atomic_cmpxchg_i(int *address, int compare,
                                                  int val);
extern "C++" inline __device__ int atomicCAS(int *address, int compare,
                                             int val) {
  return __chip_atomic_cmpxchg_i(address, compare, val);
}

extern "C" __device__ unsigned int
__chip_atomic_cmpxchg_u(unsigned int *address, unsigned int compare,
                        unsigned int val);
extern "C++" inline __device__ unsigned int
atomicCAS(unsigned int *address, unsigned int compare, unsigned int val) {
  return __chip_atomic_cmpxchg_u(address, compare, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_cmpxchg_l(unsigned long long *address, unsigned long long compare,
                        unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicCAS(unsigned long long *address, unsigned long long compare,
          unsigned long long val) {
  return __chip_atomic_cmpxchg_l(address, compare, val);
}

extern "C" __device__ int __chip_atomic_cmpxchg_system_i(int *address,
                                                         int compare, int val);
extern "C++" inline __device__ int atomicCAS_system(int *address, int compare,
                                                    int val) {
  return __chip_atomic_cmpxchg_system_i(address, compare, val);
}

extern "C" __device__ unsigned int
__chip_atomic_cmpxchg_system_u(unsigned int *address, unsigned int compare,
                               unsigned int val);
extern "C++" inline __device__ unsigned int
atomicCAS_system(unsigned int *address, unsigned int compare,
                 unsigned int val) {
  return __chip_atomic_cmpxchg_system_u(address, compare, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_cmpxchg_system_l(unsigned long long *address,
                               unsigned long long compare,
                               unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicCAS_system(unsigned long long *address, unsigned long long compare,
                 unsigned long long val) {
  return __chip_atomic_cmpxchg_system_l(address, compare, val);
}

extern "C" __device__ int __chip_atomic_and_i(int *address, int val);
extern "C++" inline __device__ int atomicAnd(int *address, int val) {
  return __chip_atomic_and_i(address, val);
}

extern "C" __device__ unsigned int __chip_atomic_and_u(unsigned int *address,
                                                       unsigned int val);
extern "C++" inline __device__ unsigned int atomicAnd(unsigned int *address,
                                                      unsigned int val) {
  return __chip_atomic_and_u(address, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_and_l(unsigned long long *address, unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicAnd(unsigned long long *address, unsigned long long val) {
  return __chip_atomic_and_l(address, val);
}

extern "C" __device__ int __chip_atomic_and_system_i(int *address, int val);
extern "C++" inline __device__ int atomicAnd_system(int *address, int val) {
  return __chip_atomic_and_system_i(address, val);
}

extern "C" __device__ unsigned int
__chip_atomic_and_system_u(unsigned int *address, unsigned int val);
extern "C++" inline __device__ unsigned int
atomicAnd_system(unsigned int *address, unsigned int val) {
  return __chip_atomic_and_system_u(address, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_and_system_l(unsigned long long *address, unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicAnd_system(unsigned long long *address, unsigned long long val) {
  return __chip_atomic_and_system_l(address, val);
}

extern "C" __device__ int __chip_atomic_or_i(int *address, int val);
extern "C++" inline __device__ int atomicOr(int *address, int val) {
  return __chip_atomic_or_i(address, val);
}

extern "C" __device__ unsigned int __chip_atomic_or_u(unsigned int *address,
                                                      unsigned int val);
extern "C++" inline __device__ unsigned int atomicOr(unsigned int *address,
                                                     unsigned int val) {
  return __chip_atomic_or_u(address, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_or_l(unsigned long long int *address, unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicOr(unsigned long long int *address, unsigned long long val) {
  return __chip_atomic_or_l(address, val);
}

extern "C" __device__ int __chip_atomic_or_system_i(int *address, int val);
extern "C++" inline __device__ int atomicOr_system(int *address, int val) {
  return __chip_atomic_or_system_i(address, val);
}

extern "C" __device__ unsigned int
__chip_atomic_or_system_u(unsigned int *address, unsigned int val);
extern "C++" inline __device__ unsigned int
atomicOr_system(unsigned int *address, unsigned int val) {
  return __chip_atomic_or_system_u(address, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_or_system_l(unsigned long long *address, unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicOr_system(unsigned long long *address, unsigned long long val) {
  return __chip_atomic_or_system_l(address, val);
}

extern "C" __device__ int __chip_atomic_xor_i(int *address, int val);
extern "C++" inline __device__ int atomicXor(int *address, int val) {
  return __chip_atomic_xor_i(address, val);
}

extern "C" __device__ unsigned int __chip_atomic_xor_u(unsigned int *address,
                                                       unsigned int val);
extern "C++" inline __device__ unsigned int atomicXor(unsigned int *address,
                                                      unsigned int val) {
  return __chip_atomic_xor_u(address, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_xor_l(unsigned long long *address, unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicXor(unsigned long long *address, unsigned long long val) {
  return __chip_atomic_xor_l(address, val);
}

extern "C" __device__ int __chip_atomic_xor_system_i(int *address, int val);
extern "C++" inline __device__ int atomicXor_system(int *address, int val) {
  return __chip_atomic_xor_system_i(address, val);
}

extern "C" __device__ unsigned int
__chip_atomic_xor_system_u(unsigned int *address, unsigned int val);
extern "C++" inline __device__ unsigned int
atomicXor_system(unsigned int *address, unsigned int val) {
  return __chip_atomic_xor_system_u(address, val);
}

extern "C" __device__ unsigned long long
__chip_atomic_xor_system_l(unsigned long long *address, unsigned long long val);
extern "C++" inline __device__ unsigned long long
atomicXor_system(unsigned long long *address, unsigned long long val) {
  return __chip_atomic_xor_system_l(address, val);
}

// Undocumented
extern "C++" inline __device__ void atomicAddNoRet(float *address, float val) {
  (void)__chip_atomic_add_f32(address, val);
}

#endif // HIP_INLUDE_DEVICELIB_ATOMICS

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


#ifndef HIP_INCLUDE_DEVICELIB_SYNC_AND_UTIL_H
#define HIP_INCLUDE_DEVICELIB_SYNC_AND_UTIL_H

#include <hip/devicelib/macros.hh>

#if defined(__HIP_DEVICE_COMPILE__)

extern "C" {
NON_OVLD void GEN_NAME(local_barrier)();
NON_OVLD int GEN_NAME(group_all)(int predicate);
NON_OVLD int GEN_NAME(group_any)(int predicate);
NON_OVLD ulong GEN_NAME(group_ballot)(int predicate);
}

EXPORT void __syncthreads() { GEN_NAME(local_barrier)(); }

EXPORT int __syncthreads_and(int predicate) {
  GEN_NAME(local_barrier)();
  return GEN_NAME(group_all)(!!predicate);
}
EXPORT int __syncthreads_or(int predicate) {
  GEN_NAME(local_barrier)();
  return GEN_NAME(group_any)(!!predicate);
}
EXPORT ulong __syncthreads_count(int predicate) {
  GEN_NAME(local_barrier)();
  return GEN_NAME(group_ballot)(!!predicate);
}

extern "C" {
NON_OVLD void GEN_NAME(local_fence)();
}
EXPORT void __threadfence_block() { GEN_NAME(local_fence)(); }

extern "C" {
NON_OVLD void GEN_NAME(system_fence)();
}
EXPORT void __threadfence_system() { GEN_NAME(system_fence)(); }

extern "C" {
NON_OVLD void GEN_NAME(global_fence)();
}
EXPORT void __threadfence() { GEN_NAME(global_fence)(); }

EXPORT clock_t clock() { return 0; }
EXPORT unsigned long long clock64() { return 0; }

extern "C" {
NON_OVLD void *GEN_NAME(memset)(void *ptr, int value, size_t size);
}
EXPORT void *memset(void *ptr, int value, size_t size) {
  return GEN_NAME(memset)(ptr, value, size);
}

extern "C" {
NON_OVLD void *GEN_NAME(memcpy)(void *dest, const void *src, size_t n);
}
EXPORT void *memcpy(void *dest, const void *src, size_t n) {
  return GEN_NAME(memcpy)(dest, src, n);
}

#else

EXPORT void __syncthreads();
EXPORT int __syncthreads_and(int predicate);
EXPORT int __syncthreads_or(int predicate);
EXPORT int __syncthreads_count(int predicate);
EXPORT void __threadfence_block();
EXPORT void __threadfence_system();
EXPORT void __threadfence();
EXPORT clock_t clock();
EXPORT unsigned long long clock64();
EXPORT void *memset(void *ptr, int value, size_t size);
EXPORT void *memcpy(void *dest, const void *src, size_t n);
#endif

#endif // include guard

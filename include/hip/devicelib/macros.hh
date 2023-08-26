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


#ifndef HIP_INCLUDE_DEVICELIB_MACROS_H
#define HIP_INCLUDE_DEVICELIB_MACROS_H

#include <algorithm>
#include <limits>

#define NOOPT __attribute__((optnone))

#if defined(__HIP_DEVICE_COMPILE__)
#define __DEVICE__ __device__
#define EXPORT static inline __device__
#define OVLD __attribute__((overloadable)) __device__
#define NON_OVLD __device__
#define GEN_NAME(N) __chip_##N
#define GEN_NAME2(N, S) __chip_##N##_##S
#else
#define __DEVICE__ extern __device__
#define EXPORT extern __device__
#define NON_OVLD
#define OVLD
#define GEN_NAME(N) N
#define GEN_NAME2(N, S) N
#endif // __HIP_DEVICE_COMPILE__

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

typedef _Float16 api_half;
typedef _Float16 api_half2 __attribute__((ext_vector_type(2)));

typedef unsigned int uint;
typedef unsigned long ulong;

#endif // include guard

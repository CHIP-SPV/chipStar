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


#ifndef HIP_INCLUDE_DEVICELIB_HALF_MATH_H
#define HIP_INCLUDE_DEVICELIB_HALF_MATH_H

#include <hip/devicelib/macros.hh>

extern "C++" {
extern __device__ api_half ceil(api_half x);
extern __device__ api_half cos(api_half x);
extern __device__ api_half exp(api_half x);
extern __device__ api_half floor(api_half x);
extern __device__ api_half log(api_half x);
extern __device__ api_half log10(api_half x);
extern __device__ api_half log2(api_half x);
extern __device__ api_half rint(api_half x);
extern __device__ api_half sin(api_half x);
extern __device__ api_half sqrt(api_half x);
extern __device__ api_half trunc(api_half x);
} // extern "C++"

#endif // include guards

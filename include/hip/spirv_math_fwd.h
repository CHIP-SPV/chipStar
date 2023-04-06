/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

  This file is an almost verbatim copy of
  hipamd /include/hip/amd_detail/hip_fp16_math_fwd.h (revision 348a177).

*/

#pragma once

#if defined(__clang__) && defined(__HIP__)

#include "spirv_hip_vector_types.h"

// DOT FUNCTIONS
__device__ __attribute__((const)) int __ockl_sdot2(
    HIP_vector_base<short, 2>::Native_vec_,
    HIP_vector_base<short, 2>::Native_vec_, int, bool);

__device__ __attribute__((const)) unsigned int __ockl_udot2(
    HIP_vector_base<unsigned short, 2>::Native_vec_,
    HIP_vector_base<unsigned short, 2>::Native_vec_, unsigned int, bool);

__device__ __attribute__((const)) int __ockl_sdot4(
    HIP_vector_base<char, 4>::Native_vec_,
    HIP_vector_base<char, 4>::Native_vec_, int, bool);

__device__ __attribute__((const)) unsigned int __ockl_udot4(
    HIP_vector_base<unsigned char, 4>::Native_vec_,
    HIP_vector_base<unsigned char, 4>::Native_vec_, unsigned int, bool);

__device__ __attribute__((const)) int __ockl_sdot8(int, int, int, bool);

__device__ __attribute__((const)) unsigned int __ockl_udot8(unsigned int,
                                                            unsigned int,
                                                            unsigned int, bool);

#endif // __HIP_CLANG_ONLY__

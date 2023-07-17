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

#ifndef HIP_SPIRV_HIP_HOST_DEFINES_H
#define HIP_SPIRV_HIP_HOST_DEFINES_H

#if defined(__clang__) && defined(__HIP__)

// Undefine the __noinline__ coming from the chipStar's fork of the
// HIP-Common as it is shadowing the keyword in Clang 15+. The
// upstream HIP has removed the __noinline__ definition and also the
// other HIP function attributes and let the implementers define them.
// TODO: We should probably remove the HIP attributes in the HIP-Common fork.
#undef __noinline__

#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#endif  // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__

#if !defined(__has_feature) || !__has_feature(cuda_noinline_keyword)
// In Clang 15+ __noinline__ is a keyword which works within
// __attribute__(()) and standalone. For the earlier Clang versions (and
// compilers which don't recognize the keyword) we have to substitute
// it to empty string so C++ code using __attribute__((__noinline__))
// compiles.
#define __noinline__
#endif

#define __forceinline__ inline __attribute__((always_inline))

#else

// Non-HCC compiler
/**
 * Function and kernel markers
 */
#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__

#define __noinline__
#define __forceinline__ inline

#endif // defined(__clang__) && defined(__HIP__)

#define __launch_bounds__(...)

#endif

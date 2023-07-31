/*
 * Copyright (c) 2023 chipStar developers
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

// A header for applying fixups. To take effect, this header must be first to be
// included before anything else (before any include in user code, before
// all -include options).

#ifndef SPIRV_COMPILER_FIXUPS_H
#define SPIRV_COMPILER_FIXUPS_H

#ifdef __HIP_DEVICE_COMPILE__
// Undefine clang builtin defines (as a workaround) which cause
// errors in the device side compilations.
//
// The troublesome defines "leak" from the host target into the device
// compilation. AFAIK, Clang fuses host and offload target specific
// defines together and passes them both the host side and device side
// compilation. This is known to cause inclusion of code with
// unsupported features (such as __float128 and __bf16 for SPIR-V
// offload target).

// x86_64 target defines these which cause device compilation errors
// with -std=gnu++## and libstdc++ version 12.
#undef __FLOAT128__
#undef __SIZEOF_FLOAT128__

// This define is known to include code with unsupported __bf16 type.
#undef __SSE2__

#endif // __HIP_DEVICE_COMPILE__

// Make sure '__device__ constexpr' definitions, used for implementing
// device side cmath functions, appear before <cmath> include.
// Otherwise, we may encounter 'reference to __host__ function 'xyz' in
// __host__ __device__ function' errors if the <cmath> is included first.
#include <hip/spirv_hip_devicelib.hh>

#endif // SPIRV_COMPILER_FIXUPS_H

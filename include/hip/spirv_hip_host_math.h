/*
 * Copyright (c) 2023 CHIP-SPV developers
 * Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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

// A header for hosting host side math functions.

#ifndef HIP_SPIRV_HIP_HOST_MATH_H
#define HIP_SPIRV_HIP_HOST_MATH_H

#if defined(__clang__) && defined(__HIP__)

#if !defined(__HIPCC_RTC__)
__host__ inline static int min(int arg1, int arg2) {
  return std::min(arg1, arg2);
}

__host__ inline static int max(int arg1, int arg2) {
  return std::max(arg1, arg2);
}
#endif

#endif // defined(__clang__) && defined(__HIP__)
#endif // HIP_SPIRV_HIP_HOST_MATH_H

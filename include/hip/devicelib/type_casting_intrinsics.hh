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


#ifndef HIP_INCLUDE_DEVICELIB_TYPE_CASTING_INTRINSICS_H
#define HIP_INCLUDE_DEVICELIB_TYPE_CASTING_INTRINSICS_H

#include <hip/devicelib/macros.hh>

EXPORT float __double2float_rd(double x);
EXPORT float __double2float_rn(double x);
EXPORT float __double2float_ru(double x);
EXPORT float __double2float_rz(double x);
EXPORT int __double2hiint(double x);
EXPORT int __double2int_rd(double x);
EXPORT int __double2int_rn(double x);
EXPORT int __double2int_ru(double x);
EXPORT int __double2int_rz(double x);
EXPORT long long int __double2ll_rd(double x);
EXPORT long long int __double2ll_rn(double x);
EXPORT long long int __double2ll_ru(double x);
EXPORT long long int __double2ll_rz(double x);
EXPORT int __double2loint(double x);
EXPORT unsigned int __double2uint_rd(double x);
EXPORT unsigned int __double2uint_rn(double x);
EXPORT unsigned int __double2uint_ru(double x);
EXPORT unsigned int __double2uint_rz(double x);
EXPORT unsigned long long int __double2ull_rd(double x);
EXPORT unsigned long long int __double2ull_rn(double x);
EXPORT unsigned long long int __double2ull_ru(double x);
EXPORT unsigned long long int __double2ull_rz(double x);
EXPORT int __float2int_rd(float x);
EXPORT int __float2int_rn(float x);
EXPORT int __float2int_ru(float);
EXPORT int __float2int_rz(float x);
EXPORT long long int __float2ll_rd(float x);
EXPORT long long int __float2ll_rn(float x);
EXPORT long long int __float2ll_ru(float x);
EXPORT long long int __float2ll_rz(float x);
EXPORT unsigned int __float2uint_rd(float x);
EXPORT unsigned int __float2uint_rn(float x);
EXPORT unsigned int __float2uint_ru(float x);
EXPORT unsigned int __float2uint_rz(float x);
EXPORT unsigned long long int __float2ull_rd(float x);
EXPORT unsigned long long int __float2ull_rn(float x);
EXPORT unsigned long long int __float2ull_ru(float x);
EXPORT unsigned long long int __float2ull_rz(float x);
EXPORT int __float_as_int(float x);
EXPORT unsigned int __float_as_uint(float x);
EXPORT double __hiloint2double(int hi, int lo);
EXPORT double __int2double_rn(int x);
EXPORT float __int2float_rd(int x);
EXPORT float __int2float_rn(int x);
EXPORT float __int2float_ru(int x);
EXPORT float __int2float_rz(int x);
EXPORT float __int_as_float(int x);
EXPORT double __ll2double_rd(long long int x);
EXPORT double __ll2double_rn(long long int x);
EXPORT double __ll2double_ru(long long int x);
EXPORT double __ll2double_rz(long long int x);
EXPORT float __ll2float_rd(long long int x);
EXPORT float __ll2float_rn(long long int x);
EXPORT float __ll2float_ru(long long int x);
EXPORT float __ll2float_rz(long long int x);
EXPORT double __uint2double_rn(unsigned int x);
EXPORT float __uint2float_rd(unsigned int x);
EXPORT float __uint2float_rn(unsigned int x);
EXPORT float __uint2float_ru(unsigned int x);
EXPORT float __uint2float_rz(unsigned int x);
EXPORT float __uint_as_float(unsigned int x);
EXPORT double __ull2double_rd(unsigned long long int x);
EXPORT double __ull2double_rn(unsigned long long int x);
EXPORT double __ull2double_ru(unsigned long long int x);
EXPORT double __ull2double_rz(unsigned long long int x);
EXPORT float __ull2float_rd(unsigned long long int x);
EXPORT float __ull2float_rn(unsigned long long int x);
EXPORT float __ull2float_ru(unsigned long long int x);
EXPORT float __ull2float_rz(unsigned long long int x);

extern "C" {
__device__ long long int __double_as_longlong(double x);
__device__ double __longlong_as_double(long long int x);
}

#endif // include guard

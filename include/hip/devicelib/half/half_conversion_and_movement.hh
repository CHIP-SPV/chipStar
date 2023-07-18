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


#ifndef HIP_INCLUDE_DEVICELIB_HALF_CONVERSION_AND_MOVEMENT_H
#define HIP_INCLUDE_DEVICELIB_HALF_CONVERSION_AND_MOVEMENT_H

#include <hip/devicelib/macros.hh>

// __host__​__device__​ __half 	__double2half ( const double  a )
// __host__​__device__​ __half2 	__float22half2_rn ( const float2 a )
// __host__​__device__​ __half 	__float2half ( const float  a )
// __host__​__device__​ __half2 	__float2half2_rn ( const float  a )
// __host__​__device__​ __half 	__float2half_rd ( const float  a )
// __host__​__device__​ __half 	__float2half_rn ( const float  a )
// __host__​__device__​ __half 	__float2half_ru ( const float  a )
// __host__​__device__​ __half 	__float2half_rz ( const float  a )
// __host__​__device__​ __half2 	__floats2half2_rn ( const float  a,
// const float  b )
// __host__​__device__​ float2 	__half22float2 ( const __half2 a )
// __host__​__device__​ float 	__half2float ( const __half a )
// __device__​ __half2 __half2half2 ( const __half a )
// __device__​ int __half2int_rd ( const __half h )
// __device__​ int __half2int_rn ( const __half h )
// __device__​ int __half2int_ru ( const __half h )
// __host__​__device__​ int 	__half2int_rz ( const __half h )
// __device__​ long long int 	__half2ll_rd ( const __half h )
// __device__​ long long int 	__half2ll_rn ( const __half h )
// __device__​ long long int 	__half2ll_ru ( const __half h )
// __host__​__device__​ long long int 	__half2ll_rz ( const __half h )
// __device__​ short int __half2short_rd ( const __half h )
// __device__​ short int __half2short_rn ( const __half h )
// __device__​ short int __half2short_ru ( const __half h )
// __host__​__device__​ short int 	__half2short_rz ( const __half h )
// __device__​ unsigned int 	__half2uint_rd ( const __half h )
// __device__​ unsigned int 	__half2uint_rn ( const __half h )
// __device__​ unsigned int 	__half2uint_ru ( const __half h )
// __host__​__device__​ unsigned int 	__half2uint_rz ( const __half h)
// __device__​ unsigned long long int 	__half2ull_rd ( const __half h )
// __device__​ unsigned long long int 	__half2ull_rn ( const __half h )
// __device__​ unsigned long long int 	__half2ull_ru ( const __half h )
// __host__​__device__​ unsigned long long int 	__half2ull_rz ( const
// __half h )
// __device__​ unsigned short int 	__half2ushort_rd ( const __half h )
// __device__​ unsigned short int 	__half2ushort_rn ( const __half h )
// __device__​ unsigned short int 	__half2ushort_ru ( const __half h )
// __host__​__device__​ unsigned short int 	__half2ushort_rz ( const __half
// h )
// __device__​ short int __half_as_short ( const __half h )
// __device__​ unsigned short int 	__half_as_ushort ( const __half h )
// __device__​ __half2 __halves2half2 ( const __half a, const __half b )
// __host__​__device__​ float 	__high2float ( const __half2 a )
// __device__​ __half __high2half ( const __half2 a )
// __device__​ __half2 __high2half2 ( const __half2 a )
// __device__​ __half2 __highs2half2 ( const __half2 a, const __half2 b )
// __device__​ __half __int2half_rd ( const int  i )
// __host__​__device__​ __half 	__int2half_rn ( const int  i )
// __device__​ __half __int2half_ru ( const int  i )
// __device__​ __half __int2half_rz ( const int  i )
// __device__​ __half __ldca ( const __half* ptr )
// __device__​ __half2 __ldca ( const __half2* ptr )
// __device__​ __half2 __ldcg ( const __half2* ptr )
// __device__​ __half __ldcs ( const __half* ptr )
// __device__​ __half2 __ldcs ( const __half2* ptr )
// __device__​ __half __ldcv ( const __half* ptr )
// __device__​ __half2 __ldcv ( const __half2* ptr )
// __device__​ __half __ldg ( const __half* ptr )
// __device__​ __half2 __ldg ( const __half2* ptr )
// __device__​ __half __ldlu ( const __half* ptr )
// __device__​ __half2 __ldlu ( const __half2* ptr )
// __device__​ __half __ll2half_rd ( const long long int i )
// __host__​__device__​ __half 	__ll2half_rn ( const long long int i )
// __device__​ __half __ll2half_ru ( const long long int i )
// __device__​ __half __ll2half_rz ( const long long int i )
// __host__​__device__​ float 	__low2float ( const __half2 a )
// __device__​ __half __low2half ( const __half2 a )
// __device__​ __half2 __low2half2 ( const __half2 a )
// __device__​ __half2 __lowhigh2highlow ( const __half2 a )
// __device__​ __half2 __lows2half2 ( const __half2 a, const __half2 b )
// __device__​ __half __shfl_down_sync ( const unsigned mask, const __half
// var, const unsigned int  delta, const int  width = warpSize )
// __device__​ __half2 __shfl_down_sync ( const unsigned mask, const __half2
// var, const unsigned int  delta, const int  width = warpSize )
// __device__​ __half __shfl_sync ( const unsigned mask, const __half var,
// const int  delta, const int  width = warpSize )
// __device__​ __half2 __shfl_sync ( const unsigned mask, const __half2 var,
// const int  delta, const int  width = warpSize )
// __device__​ __half __shfl_up_sync ( const unsigned mask, const __half var,
// const unsigned int  delta, const int  width = warpSize )
// __device__​ __half2 __shfl_up_sync ( const unsigned mask, const __half2
// var, const unsigned int  delta, const int  width = warpSize )
// __device__​ __half __shfl_xor_sync ( const unsigned mask, const __half var,
// const int  delta, const int  width = warpSize )
// __device__​ __half2 __shfl_xor_sync ( const unsigned mask, const __half2
// var, const int  delta, const int  width = warpSize )
// __device__​ __half __short2half_rd ( const short int i )
// __host__​__device__​ __half 	__short2half_rn ( const short int i )
// __device__​ __half __short2half_ru ( const short int i )
// __device__​ __half __short2half_rz ( const short int i )
// __device__​ __half __short_as_half ( const short int i )
// __device__​ void __stcg ( const __half* ptr, const __half value )
// __device__​ void __stcg ( const __half2* ptr, const __half2 value )
// __device__​ void __stcs ( const __half* ptr, const __half value )
// __device__​ void __stcs ( const __half2* ptr, const __half2 value )
// __device__​ void __stwb ( const __half* ptr, const __half value )
// __device__​ void __stwb ( const __half2* ptr, const __half2 value )
// __device__​ void __stwt ( const __half* ptr, const __half value )
// __device__​ void __stwt ( const __half2* ptr, const __half2 value )
// __device__​ __half __uint2half_rd ( const unsigned int  i )
// __host__​__device__​ __half 	__uint2half_rn ( const unsigned int  i )
// __device__​ __half __uint2half_ru ( const unsigned int  i )
// __device__​ __half __uint2half_rz ( const unsigned int  i )
// __device__​ __half __ull2half_rd ( const unsigned long long int i )
// __host__​__device__​ __half 	__ull2half_rn ( const unsigned long long
// int i )
// __device__​ __half __ull2half_ru ( const unsigned long long int i )
// __device__​ __half __ull2half_rz ( const unsigned long long int i )
// __device__​ __half __ushort2half_rd ( const unsigned short int i )
// __host__​__device__​ __half 	__ushort2half_rn ( const unsigned short
// int i )
// __device__​ __half __ushort2half_ru ( const unsigned short int i )
// __device__​ __half __ushort2half_rz ( const unsigned short int i )
// __device__​ __half __ushort_as_half ( const unsigned short int i )

#endif // include guards

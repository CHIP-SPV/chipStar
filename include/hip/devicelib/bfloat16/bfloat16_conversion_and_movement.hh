#ifndef HIP_INCLUDE_DEVICELIB_BFLOAT16_CONVERSION_AND_MOVEMENT_H
#define HIP_INCLUDE_DEVICELIB_BFLOAT16_CONVERSION_AND_MOVEMENT_H

#include <hip/devicelib/macros.hh>

// __host__​__device__​ float2 	__bfloat1622float2 ( const
// __nv_bfloat162 a )
// __device__​ __nv_bfloat162 	__bfloat162bfloat162 ( const
// __nv_bfloat16 a )
// __host__​__device__​ float 	__bfloat162float ( const __nv_bfloat16 a
// )
// __device__​ int __bfloat162int_rd ( const __nv_bfloat16 h )
// __device__​ int __bfloat162int_rn ( const __nv_bfloat16 h )
// __device__​ int __bfloat162int_ru ( const __nv_bfloat16 h )
// __host__​__device__​ int 	__bfloat162int_rz ( const __nv_bfloat16
// h
// )
// __device__​ long long int 	__bfloat162ll_rd ( const __nv_bfloat16 h )
// __device__​ long long int 	__bfloat162ll_rn ( const __nv_bfloat16 h )
// __device__​ long long int 	__bfloat162ll_ru ( const __nv_bfloat16 h )
// __host__​__device__​ long long int 	__bfloat162ll_rz ( const
// __nv_bfloat16 h )
// __device__​ short int __bfloat162short_rd ( const __nv_bfloat16 h )
// __device__​ short int __bfloat162short_rn ( const __nv_bfloat16 h )
// __device__​ short int __bfloat162short_ru ( const __nv_bfloat16 h )
// __host__​__device__​ short int 	__bfloat162short_rz ( const
// __nv_bfloat16 h )
// __device__​ unsigned int 	__bfloat162uint_rd ( const __nv_bfloat16 h )
// __device__​ unsigned int 	__bfloat162uint_rn ( const __nv_bfloat16 h )
// __device__​ unsigned int 	__bfloat162uint_ru ( const __nv_bfloat16 h )
// __host__​__device__​ unsigned int 	__bfloat162uint_rz ( const
// __nv_bfloat16 h )
// __device__​ unsigned long long int 	__bfloat162ull_rd ( const
// __nv_bfloat16 h )
// __device__​ unsigned long long int 	__bfloat162ull_rn ( const
// __nv_bfloat16 h )
// __device__​ unsigned long long int 	__bfloat162ull_ru ( const
// __nv_bfloat16 h )
// __host__​__device__​ unsigned long long int 	__bfloat162ull_rz (
// const __nv_bfloat16 h )
// __device__​ unsigned short int 	__bfloat162ushort_rd ( const
// __nv_bfloat16 h )
// __device__​ unsigned short int 	__bfloat162ushort_rn ( const
// __nv_bfloat16 h )
// __device__​ unsigned short int 	__bfloat162ushort_ru ( const
// __nv_bfloat16 h )
// __host__​__device__​ unsigned short int 	__bfloat162ushort_rz ( const
// __nv_bfloat16 h )
// __device__​ short int __bfloat16_as_short ( const __nv_bfloat16 h )
// __device__​ unsigned short int 	__bfloat16_as_ushort ( const
// __nv_bfloat16 h )
// __host__​__device__​ __nv_bfloat16 	__double2bfloat16 ( const double
// a )
// __host__​__device__​ __nv_bfloat162 	__float22bfloat162_rn ( const
// float2 a )
// __host__​__device__​ __nv_bfloat16 	__float2bfloat16 ( const float a
// )
// __host__​__device__​ __nv_bfloat162 	__float2bfloat162_rn ( const
// float  a )
// __host__​__device__​ __nv_bfloat16 	__float2bfloat16_rd ( const
// float a )
// __host__​__device__​ __nv_bfloat16 	__float2bfloat16_rn ( const
// float a )
// __host__​__device__​ __nv_bfloat16 	__float2bfloat16_ru ( const
// float a )
// __host__​__device__​ __nv_bfloat16 	__float2bfloat16_rz ( const
// float a )
// __host__​__device__​ __nv_bfloat162 	__floats2bfloat162_rn ( const
// float  a, const float  b )
// __device__​ __nv_bfloat162 	__halves2bfloat162 ( const __nv_bfloat16
// a, const __nv_bfloat16 b )
// __device__​ __nv_bfloat16 	__high2bfloat16 ( const __nv_bfloat162 a )
// __device__​ __nv_bfloat162 	__high2bfloat162 ( const __nv_bfloat162
// a
// )
// __host__​__device__​ float 	__high2float ( const __nv_bfloat162 a )
// __device__​ __nv_bfloat162 	__highs2bfloat162 ( const __nv_bfloat162
// a, const __nv_bfloat162 b )
// __device__​ __nv_bfloat16 	__int2bfloat16_rd ( const int  i )
// __host__​__device__​ __nv_bfloat16 	__int2bfloat16_rn ( const int  i
// )
// __device__​ __nv_bfloat16 	__int2bfloat16_ru ( const int  i )
// __device__​ __nv_bfloat16 	__int2bfloat16_rz ( const int  i )
// __device__​ __nv_bfloat16 	__ldca ( const __nv_bfloat16* ptr )
// __device__​ __nv_bfloat162 	__ldca ( const __nv_bfloat162* ptr )
// __device__​ __nv_bfloat16 	__ldcg ( const __nv_bfloat16* ptr )
// __device__​ __nv_bfloat162 	__ldcg ( const __nv_bfloat162* ptr )
// __device__​ __nv_bfloat16 	__ldcs ( const __nv_bfloat16* ptr )
// __device__​ __nv_bfloat162 	__ldcs ( const __nv_bfloat162* ptr )
// __device__​ __nv_bfloat16 	__ldcv ( const __nv_bfloat16* ptr )
// __device__​ __nv_bfloat162 	__ldcv ( const __nv_bfloat162* ptr )
// __device__​ __nv_bfloat16 	__ldg ( const __nv_bfloat16* ptr )
// __device__​ __nv_bfloat162 	__ldg ( const __nv_bfloat162* ptr )
// __device__​ __nv_bfloat16 	__ldlu ( const __nv_bfloat16* ptr )
// __device__​ __nv_bfloat162 	__ldlu ( const __nv_bfloat162* ptr )
// __device__​ __nv_bfloat16 	__ll2bfloat16_rd ( const long long int i )
// __host__​__device__​ __nv_bfloat16 	__ll2bfloat16_rn ( const long
// long int i )
// __device__​ __nv_bfloat16 	__ll2bfloat16_rz ( const long long int i )
// __device__​ __nv_bfloat16 	__low2bfloat16 ( const __nv_bfloat162 a )
// __device__​ __nv_bfloat162 	__low2bfloat162 ( const __nv_bfloat162 a
// )
// __host__​__device__​ float 	__low2float ( const __nv_bfloat162 a )
// __device__​ __nv_bfloat162 	__lowhigh2highlow ( const __nv_bfloat162
// a
// )
// __device__​ __nv_bfloat162 	__lows2bfloat162 ( const __nv_bfloat162
// a, const __nv_bfloat162 b )
// __device__​ __nv_bfloat16 	__shfl_down_sync ( const unsigned mask, const
// __nv_bfloat16 var, const unsigned int  delta, const int  width = warpSize )
// __device__​ __nv_bfloat162 	__shfl_down_sync ( const unsigned mask,
// const __nv_bfloat162 var, const unsigned int  delta, const int  width =
// warpSize )
// __device__​ __nv_bfloat16 	__shfl_sync ( const unsigned mask, const
// __nv_bfloat16 var, const int  delta, const int  width = warpSize )
// __device__​ __nv_bfloat162 	__shfl_sync ( const unsigned mask, const
// __nv_bfloat162 var, const int  delta, const int  width = warpSize )
// __device__​ __nv_bfloat16 	__shfl_up_sync ( const unsigned mask, const
// __nv_bfloat16 var, const unsigned int  delta, const int  width = warpSize )
// __device__​ __nv_bfloat162 	__shfl_up_sync ( const unsigned mask,
// const
// __nv_bfloat162 var, const unsigned int  delta, const int  width = warpSize )
// __device__​ __nv_bfloat16 	__shfl_xor_sync ( const unsigned mask, const
// __nv_bfloat16 var, const int  delta, const int  width = warpSize )
// __device__​ __nv_bfloat162 	__shfl_xor_sync ( const unsigned mask,
// const __nv_bfloat162 var, const int  delta, const int  width = warpSize )
// __device__​ __nv_bfloat16 	__short2bfloat16_rd ( const short int i )
// __host__​__device__​ __nv_bfloat16 	__short2bfloat16_rn ( const
// short int i )
// __device__​ __nv_bfloat16 	__short2bfloat16_ru ( const short int i )
// __device__​ __nv_bfloat16 	__short2bfloat16_rz ( const short int i )
// __device__​ __nv_bfloat16 	__short_as_bfloat16 ( const short int i )
// __device__​ void __stcg ( const __nv_bfloat16* ptr, const __nv_bfloat16
// value )
// __device__​ void __stcs ( const __nv_bfloat16* ptr, const __nv_bfloat16
// value )
// __device__​ void __stcs ( const __nv_bfloat162* ptr, const __nv_bfloat162
// value )
// __device__​ void __stwb ( const __nv_bfloat16* ptr, const __nv_bfloat16
// value )
// __device__​ void __stwb ( const __nv_bfloat162* ptr, const __nv_bfloat162
// value )
// __device__​ void __stwt ( const __nv_bfloat16* ptr, const __nv_bfloat16
// value )
// __device__​ void __stwt ( const __nv_bfloat162* ptr, const __nv_bfloat162
// value )
// __device__​ __nv_bfloat16 	__uint2bfloat16_rd ( const unsigned int  i )
// __host__​__device__​ __nv_bfloat16 	__uint2bfloat16_rn ( const
// unsigned int  i )
// __device__​ __nv_bfloat16 	__uint2bfloat16_ru ( const unsigned int  i )
// __device__​ __nv_bfloat16 	__uint2bfloat16_rz ( const unsigned int  i )
// __device__​ __nv_bfloat16 	__ull2bfloat16_rd ( const unsigned long long int
// i )
// __host__​__device__​ __nv_bfloat16 	__ull2bfloat16_rn ( const
// unsigned long long int i )
// __device__​ __nv_bfloat16 	__ull2bfloat16_ru ( const unsigned long long int
// i )
// __device__​ __nv_bfloat16 	__ull2bfloat16_rz ( const unsigned long long int
// i )
// __device__​ __nv_bfloat16 	__ushort2bfloat16_rd ( const unsigned short int
// i )
// __host__​__device__​ __nv_bfloat16 	__ushort2bfloat16_rn ( const
// unsigned short int i )
// __device__​ __nv_bfloat16 	__ushort2bfloat16_ru ( const unsigned short int
// i )
// __device__​ __nv_bfloat16 	__ushort2bfloat16_rz ( const unsigned short int
// i )
// __device__​ __nv_bfloat16 	__ushort_as_bfloat16 ( const unsigned short int
// i )

#endif // include guard
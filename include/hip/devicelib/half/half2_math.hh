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


#ifndef HIP_INCLUDE_DEVICELIB_HALF2_MATH_H
#define HIP_INCLUDE_DEVICELIB_HALF2_MATH_H

#include <hip/devicelib/macros.hh>

extern "C++" {

extern __device__ api_half2 rint(api_half2 x);

}

static inline __device__ api_half2 rint_2h(api_half2 x) { return rint(x); }


//__device__ __half2 h2ceil ( const __half2 h )
//__device__ __half2 h2cos ( const __half2 a )
//__device__ __half2 h2exp ( const __half2 a )
//__device__ __half2 h2exp10 ( const __half2 a )
//__device__ __half2 h2exp2 ( const __half2 a )
//__device__ __half2 h2floor ( const __half2 h )
//__device__ __half2 h2log ( const __half2 a )
//__device__ __half2 h2log10 ( const __half2 a )
//__device__ __half2 h2log2 ( const __half2 a )
//__device__ __half2 h2rcp ( const __half2 a )
//__device__ __half2 h2rint ( const __half2 h )
//__device__ __half2 h2rsqrt ( const __half2 a )
//__device__ __half2 h2sin ( const __half2 a )
//__device__ __half2 h2sqrt ( const __half2 a )
//__device__ __half2 h2trunc ( const __half2 h )

#endif // include guards

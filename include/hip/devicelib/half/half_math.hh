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


#ifndef HIP_INCLUDE_DEVICELIB_HALF_MATH_H
#define HIP_INCLUDE_DEVICELIB_HALF_MATH_H

#include <hip/devicelib/macros.hh>


extern "C++" {

extern __device__ api_half rint(api_half x);

}

static inline __device__ api_half rint_h(api_half x) { return rint(x); }


//__device__ __half hceil ( const __half h )
//__device__ __half hcos ( const __half a )
//__device__ __half hexp ( const __half a )
//__device__ __half hexp10 ( const __half a )
//__device__ __half hexp2 ( const __half a )
//__device__ __half hfloor ( const __half h )
//__device__ __half hlog ( const __half a )
//__device__ __half hlog10 ( const __half a )
//__device__ __half hlog2 ( const __half a )
//__device__ __half hrcp ( const __half a )
//__device__ __half hrint ( const __half h )
//__device__ __half hrsqrt ( const __half a )
//__device__ __half hsin ( const __half a )
//__device__ __half hsqrt ( const __half a )
//__device__ __half htrunc ( const __half h )

#endif // include guards

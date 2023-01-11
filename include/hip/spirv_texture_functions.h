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
*/

#ifndef SPIRV_HIP_TEXTURE_FUNCTIONS_H
#define SPIRV_HIP_TEXTURE_FUNCTIONS_H

#ifdef __cplusplus
#include "spirv_hip_vector_types.h"
#include "spirv_hip_texture_types.h"

#define __TEXTURE_FUNCTIONS_DECL__ static inline __device__

typedef float NativeFloat2 __attribute__((ext_vector_type(2)));
typedef float NativeFloat4 __attribute__((ext_vector_type(4)));
typedef int NativeInt4 __attribute__((ext_vector_type(4)));
typedef unsigned int NativeUint4 __attribute__((ext_vector_type(4)));

extern "C" __device__ NativeInt4 _chip_tex1dfetchi(hipTextureObject_t TexObj,
                                                   int Pos);

extern "C" __device__ NativeUint4 _chip_tex1dfetchu(hipTextureObject_t TexObj,
                                                    int Pos);

extern "C" __device__ NativeFloat4 _chip_tex1dfetchf(hipTextureObject_t TexObj,
                                                     int Pos);

extern "C" __device__ NativeInt4 _chip_tex1di(hipTextureObject_t TexObj,
                                              float Pos);

extern "C" __device__ NativeUint4 _chip_tex1du(hipTextureObject_t TexObj,
                                               float Pos);

extern "C" __device__ NativeFloat4 _chip_tex1df(hipTextureObject_t TexObj,
                                                float Pos);

extern "C" __device__ NativeFloat4 _chip_tex2df(hipTextureObject_t TexObj,
                                                NativeFloat2 Pos);

extern "C" __device__ NativeInt4 _chip_tex2di(hipTextureObject_t TexObj,
                                              NativeFloat2 Pos);

extern "C" __device__ NativeUint4 _chip_tex2du(hipTextureObject_t TexObj,
                                               NativeFloat2 Pos);

#define DEF_TEX1D_SCL(_NAME, _RES_TY, _POS_TY, _IFN)                           \
  __TEXTURE_FUNCTIONS_DECL__ void _NAME(                                       \
      _RES_TY *RetVal, hipTextureObject_t TexObj, _POS_TY x) {                 \
    *RetVal = _IFN(TexObj, x).x;                                               \
  }                                                                            \
  __asm__("")

#define DEF_TEX1D_VEC1(_NAME, _RES_TY, _POS_TY, _IFN)                          \
  __TEXTURE_FUNCTIONS_DECL__ void _NAME(                                       \
      _RES_TY *RetVal, hipTextureObject_t TexObj, _POS_TY X) {                 \
    auto Res = _IFN(TexObj, X);                                                \
    *RetVal = make_##_RES_TY(Res.x);                                           \
  }                                                                            \
  __asm__("")

#define DEF_TEX1D_VEC2(_NAME, _RES_TY, _POS_TY, _IFN)                          \
  __TEXTURE_FUNCTIONS_DECL__ void _NAME(                                       \
      _RES_TY *RetVal, hipTextureObject_t TexObj, _POS_TY X) {                 \
    auto Res = _IFN(TexObj, X);                                                \
    *RetVal = make_##_RES_TY(Res.x, Res.y);                                    \
  }                                                                            \
  __asm__("")

#define DEF_TEX1D_VEC4(_NAME, _RES_TY, _POS_TY, _IFN)                          \
  __TEXTURE_FUNCTIONS_DECL__ void _NAME(                                       \
      _RES_TY *RetVal, hipTextureObject_t TexObj, _POS_TY X) {                 \
    auto Res = _IFN(TexObj, X);                                                \
    *RetVal = make_##_RES_TY(Res.x, Res.y, Res.z, Res.w);                      \
  }                                                                            \
  __asm__("")

#define DEF_TEX2D_SCL(_NAME, _RES_TY, _IFN)                                    \
  __TEXTURE_FUNCTIONS_DECL__ void _NAME(                                       \
      _RES_TY *RetVal, hipTextureObject_t TexObj, float X, float Y) {          \
    NativeFloat2 Pos;                                                          \
    Pos.x = X;                                                                 \
    Pos.y = Y;                                                                 \
    *RetVal = _IFN(TexObj, Pos).x;                                             \
  }                                                                            \
  __asm__("")

#define DEF_TEX2D_VEC1(_NAME, _RES_TY, _IFN)                                   \
  __TEXTURE_FUNCTIONS_DECL__ void _NAME(                                       \
      _RES_TY *RetVal, hipTextureObject_t TexObj, float X, float Y) {          \
    NativeFloat2 Pos;                                                          \
    Pos.x = X;                                                                 \
    Pos.y = Y;                                                                 \
    auto Res = _IFN(TexObj, Pos);                                              \
    *RetVal = make_##_RES_TY(Res.x);                                           \
  }                                                                            \
  __asm__("")

#define DEF_TEX2D_VEC2(_NAME, _RES_TY, _IFN)                                   \
  __TEXTURE_FUNCTIONS_DECL__ void _NAME(                                       \
      _RES_TY *RetVal, hipTextureObject_t TexObj, float X, float Y) {          \
    NativeFloat2 Pos;                                                          \
    Pos.x = X;                                                                 \
    Pos.y = Y;                                                                 \
    auto Res = _IFN(TexObj, Pos);                                              \
    *RetVal = make_##_RES_TY(Res.x, Res.y);                                    \
  }                                                                            \
  __asm__("")

#define DEF_TEX2D_VEC4(_NAME, _RES_TY, _IFN)                                   \
  __TEXTURE_FUNCTIONS_DECL__ void _NAME(                                       \
      _RES_TY *RetVal, hipTextureObject_t TexObj, float X, float Y) {          \
    NativeFloat2 Pos;                                                          \
    Pos.x = X;                                                                 \
    Pos.y = Y;                                                                 \
    auto Res = _IFN(TexObj, Pos);                                              \
    *RetVal = make_##_RES_TY(Res.x, Res.y, Res.z, Res.y);                      \
  }                                                                            \
  __asm__("")

// tex1DFetch //

DEF_TEX1D_SCL(tex1Dfetch, char, int, _chip_tex1dfetchi);
DEF_TEX1D_SCL(tex1Dfetch, unsigned char, int, _chip_tex1dfetchu);
DEF_TEX1D_SCL(tex1Dfetch, short, int, _chip_tex1dfetchi);
DEF_TEX1D_SCL(tex1Dfetch, unsigned short, int, _chip_tex1dfetchu);
DEF_TEX1D_SCL(tex1Dfetch, int, int, _chip_tex1dfetchi);
DEF_TEX1D_SCL(tex1Dfetch, unsigned int, int, _chip_tex1dfetchu);
DEF_TEX1D_SCL(tex1Dfetch, float, int, _chip_tex1dfetchf);

DEF_TEX1D_VEC1(tex1Dfetch, char1, int, _chip_tex1dfetchi);
DEF_TEX1D_VEC1(tex1Dfetch, uchar1, int, _chip_tex1dfetchu);
DEF_TEX1D_VEC1(tex1Dfetch, short1, int, _chip_tex1dfetchi);
DEF_TEX1D_VEC1(tex1Dfetch, ushort1, int, _chip_tex1dfetchu);
DEF_TEX1D_VEC1(tex1Dfetch, int1, int, _chip_tex1dfetchi);
DEF_TEX1D_VEC1(tex1Dfetch, uint1, int, _chip_tex1dfetchu);
DEF_TEX1D_VEC1(tex1Dfetch, float1, int, _chip_tex1dfetchf);

DEF_TEX1D_VEC2(tex1Dfetch, char2, int, _chip_tex1dfetchi);
DEF_TEX1D_VEC2(tex1Dfetch, uchar2, int, _chip_tex1dfetchu);
DEF_TEX1D_VEC2(tex1Dfetch, short2, int, _chip_tex1dfetchi);
DEF_TEX1D_VEC2(tex1Dfetch, ushort2, int, _chip_tex1dfetchu);
DEF_TEX1D_VEC2(tex1Dfetch, int2, int, _chip_tex1dfetchi);
DEF_TEX1D_VEC2(tex1Dfetch, uint2, int, _chip_tex1dfetchu);
DEF_TEX1D_VEC2(tex1Dfetch, float2, int, _chip_tex1dfetchf);

DEF_TEX1D_VEC4(tex1Dfetch, char4, int, _chip_tex1dfetchi);
DEF_TEX1D_VEC4(tex1Dfetch, uchar4, int, _chip_tex1dfetchu);
DEF_TEX1D_VEC4(tex1Dfetch, short4, int, _chip_tex1dfetchi);
DEF_TEX1D_VEC4(tex1Dfetch, ushort4, int, _chip_tex1dfetchu);
DEF_TEX1D_VEC4(tex1Dfetch, int4, int, _chip_tex1dfetchi);
DEF_TEX1D_VEC4(tex1Dfetch, uint4, int, _chip_tex1dfetchu);
DEF_TEX1D_VEC4(tex1Dfetch, float4, int, _chip_tex1dfetchf);

// tex1D //

DEF_TEX1D_SCL(tex1D, char, float, _chip_tex1di);
DEF_TEX1D_SCL(tex1D, unsigned char, float, _chip_tex1du);
DEF_TEX1D_SCL(tex1D, short, float, _chip_tex1di);
DEF_TEX1D_SCL(tex1D, unsigned short, float, _chip_tex1du);
DEF_TEX1D_SCL(tex1D, int, float, _chip_tex1di);
DEF_TEX1D_SCL(tex1D, unsigned int, float, _chip_tex1du);
DEF_TEX1D_SCL(tex1D, float, float, _chip_tex1df);

DEF_TEX1D_VEC1(tex1D, char1, float, _chip_tex1di);
DEF_TEX1D_VEC1(tex1D, uchar1, float, _chip_tex1du);
DEF_TEX1D_VEC1(tex1D, short1, float, _chip_tex1di);
DEF_TEX1D_VEC1(tex1D, ushort1, float, _chip_tex1du);
DEF_TEX1D_VEC1(tex1D, int1, float, _chip_tex1di);
DEF_TEX1D_VEC1(tex1D, uint1, float, _chip_tex1du);
DEF_TEX1D_VEC1(tex1D, float1, float, _chip_tex1df);

DEF_TEX1D_VEC2(tex1D, char2, float, _chip_tex1di);
DEF_TEX1D_VEC2(tex1D, uchar2, float, _chip_tex1du);
DEF_TEX1D_VEC2(tex1D, short2, float, _chip_tex1di);
DEF_TEX1D_VEC2(tex1D, ushort2, float, _chip_tex1du);
DEF_TEX1D_VEC2(tex1D, int2, float, _chip_tex1di);
DEF_TEX1D_VEC2(tex1D, uint2, float, _chip_tex1du);
DEF_TEX1D_VEC2(tex1D, float2, float, _chip_tex1df);

DEF_TEX1D_VEC4(tex1D, char4, float, _chip_tex1di);
DEF_TEX1D_VEC4(tex1D, uchar4, float, _chip_tex1du);
DEF_TEX1D_VEC4(tex1D, short4, float, _chip_tex1di);
DEF_TEX1D_VEC4(tex1D, ushort4, float, _chip_tex1du);
DEF_TEX1D_VEC4(tex1D, int4, float, _chip_tex1di);
DEF_TEX1D_VEC4(tex1D, uint4, float, _chip_tex1du);
DEF_TEX1D_VEC4(tex1D, float4, float, _chip_tex1df);

// tex2D //

DEF_TEX2D_SCL(tex2D, char, _chip_tex2di);
DEF_TEX2D_SCL(tex2D, unsigned char, _chip_tex2du);
DEF_TEX2D_SCL(tex2D, short, _chip_tex2di);
DEF_TEX2D_SCL(tex2D, unsigned short, _chip_tex2du);
DEF_TEX2D_SCL(tex2D, int, _chip_tex2di);
DEF_TEX2D_SCL(tex2D, unsigned int, _chip_tex2du);
DEF_TEX2D_SCL(tex2D, float, _chip_tex2df);

DEF_TEX2D_VEC1(tex2D, char1, _chip_tex2di);
DEF_TEX2D_VEC1(tex2D, uchar1, _chip_tex2du);
DEF_TEX2D_VEC1(tex2D, short1, _chip_tex2di);
DEF_TEX2D_VEC1(tex2D, ushort1, _chip_tex2du);
DEF_TEX2D_VEC1(tex2D, int1, _chip_tex2di);
DEF_TEX2D_VEC1(tex2D, uint1, _chip_tex2du);
DEF_TEX2D_VEC1(tex2D, float1, _chip_tex2df);

DEF_TEX2D_VEC2(tex2D, char2, _chip_tex2di);
DEF_TEX2D_VEC2(tex2D, uchar2, _chip_tex2du);
DEF_TEX2D_VEC2(tex2D, short2, _chip_tex2di);
DEF_TEX2D_VEC2(tex2D, ushort2, _chip_tex2du);
DEF_TEX2D_VEC2(tex2D, int2, _chip_tex2di);
DEF_TEX2D_VEC2(tex2D, uint2, _chip_tex2du);
DEF_TEX2D_VEC2(tex2D, float2, _chip_tex2df);

DEF_TEX2D_VEC4(tex2D, char4, _chip_tex2di);
DEF_TEX2D_VEC4(tex2D, uchar4, _chip_tex2du);
DEF_TEX2D_VEC4(tex2D, short4, _chip_tex2di);
DEF_TEX2D_VEC4(tex2D, ushort4, _chip_tex2du);
DEF_TEX2D_VEC4(tex2D, int4, _chip_tex2di);
DEF_TEX2D_VEC4(tex2D, uint4, _chip_tex2du);
DEF_TEX2D_VEC4(tex2D, float4, _chip_tex2df);

// nvidia texture object API
template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex1Dfetch(hipTextureObject_t TexObj, int X) {
  T Ret;
  tex1Dfetch(&Ret, TexObj, X);
  return Ret;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex1D(hipTextureObject_t TexObj, float X) {
  T Ret;
  tex1D(&Ret, TexObj, X);
  return Ret;
}

template <class T>
__TEXTURE_FUNCTIONS_DECL__ T tex2D(hipTextureObject_t TexObj, float X, float Y) {
  T Ret;
  tex2D(&Ret, TexObj, X, Y);
  return Ret;
}

// nvidia texture reference API

#define DEF_TEXREF(DataType, FloatType) \
__TEXTURE_FUNCTIONS_DECL__ DataType tex1D(texture<DataType, hipTextureType2D, hipReadModeElementType> texRef, float x) { \
  DataType RetVal; \
  tex1D(&RetVal, texRef.textureObject, x); \
  return RetVal; \
} \
__TEXTURE_FUNCTIONS_DECL__ FloatType tex1D(texture<DataType, hipTextureType2D, hipReadModeNormalizedFloat> texRef, float x) { \
  FloatType RetVal; \
  tex1D(&RetVal, texRef.textureObject, x); \
  return RetVal; \
} \
__TEXTURE_FUNCTIONS_DECL__ DataType tex2D(texture<DataType, hipTextureType2D, hipReadModeElementType> texRef, float x, float y) { \
  DataType RetVal; \
  tex2D(&RetVal, texRef.textureObject, x, y); \
  return RetVal; \
} \
__TEXTURE_FUNCTIONS_DECL__ FloatType tex2D(texture<DataType, hipTextureType2D, hipReadModeNormalizedFloat> texRef, float x, float y) { \
  FloatType RetVal; \
  tex2D(&RetVal, texRef.textureObject, x, y); \
  return RetVal; \
} \
__asm__("")
/*
__TEXTURE_FUNCTIONS_DECL__ DataType tex3D(texture<DataType, hipTextureType2D, hipReadModeElementType> texRef, float x, float y, float z) { \
  DataType RetVal; \
  tex3D(&RetVal, texRef.textureObject, x, y, z); \
  return RetVal; \
} \
__TEXTURE_FUNCTIONS_DECL__ FloatType tex3D(texture<DataType, hipTextureType2D, hipReadModeNormalizedFloat> texRef, float x, float y, float z) { \
  FloatType RetVal; \
  tex3D(&RetVal, texRef.textureObject, x, y, z); \
  return RetVal; \
} \
*/

DEF_TEXREF(char, float);
DEF_TEXREF(unsigned char, float);
DEF_TEXREF(short, float);
DEF_TEXREF(unsigned short, float);
DEF_TEXREF(int, float);
DEF_TEXREF(unsigned int, float);
DEF_TEXREF(float, float);

DEF_TEXREF(char1, float1);
DEF_TEXREF(uchar1, float1);
DEF_TEXREF(short1, float1);
DEF_TEXREF(ushort1, float1);
DEF_TEXREF(int1, float1);
DEF_TEXREF(uint1, float1);
DEF_TEXREF(float1, float1);

DEF_TEXREF(char2, float2);
DEF_TEXREF(uchar2, float2);
DEF_TEXREF(short2, float2);
DEF_TEXREF(ushort2, float2);
DEF_TEXREF(int2, float2);
DEF_TEXREF(uint2, float2);
DEF_TEXREF(float2, float2);

DEF_TEXREF(char4, float4);
DEF_TEXREF(uchar4, float4);
DEF_TEXREF(short4, float4);
DEF_TEXREF(ushort4, float4);
DEF_TEXREF(int4, float4);
DEF_TEXREF(uint4, float4);
DEF_TEXREF(float4, float4);

#endif // cplusplus
#endif // include guard

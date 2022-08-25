/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 * (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory
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

// Texture object type declaration. Must be tandem with
// <Common-HIP>/include/hip/texture_types.h.
struct __hip_texture;
typedef struct __hip_texture *hipTextureObject_t;

// vvv DECLARATIONS INTENTIONALLY WITHOUT DEFINITION vvv

// Texture function lowering pass (HipTextureLowering.cpp) use these
// to recognize the texture function calls in kernels. The pass
// replaces the calls to them with "_chip_*_impl" variant.
int4 _chip_tex1dfetchi(hipTextureObject_t TextureObject, int Pos);
uint4 _chip_tex1dfetchu(hipTextureObject_t TextureObject, int Pos);
float4 _chip_tex1dfetchf(hipTextureObject_t TextureObject, int Pos);
int4 _chip_tex1di(hipTextureObject_t TextureObject, int Pos);
uint4 _chip_tex1du(hipTextureObject_t TextureObject, int Pos);
float4 _chip_tex1df(hipTextureObject_t TextureObject, int Pos);
float4 _chip_tex2df(hipTextureObject_t TextureObject, float2 Pos);
int4 _chip_tex2di(hipTextureObject_t TextureObject, float2 Pos);
uint4 _chip_tex2du(hipTextureObject_t TextureObject, float2 Pos);

// ^^^ DECLARATIONS INTENTIONALLY WITHOUT DEFINITION ^^^

static int4 __attribute__((used))
_chip_tex1dfetchi_impl(image1d_t I, sampler_t S, int Pos) {
  return read_imagei(I, S, Pos);
}

static uint4 __attribute__((used))
_chip_tex1dfetchu_impl(image1d_t I, sampler_t S, int Pos) {
  return read_imageui(I, S, Pos);
}

static float4 __attribute__((used))
_chip_tex1dfetchf_impl(image1d_t I, sampler_t S, int Pos) {
  return read_imagef(I, S, Pos);
}

static int4 __attribute__((used))
_chip_tex1di_impl(image1d_t I, sampler_t S, float Pos) {
  return read_imagei(I, S, Pos);
}

static uint4 __attribute__((used))
_chip_tex1du_impl(image1d_t I, sampler_t S, float Pos) {
  return read_imageui(I, S, Pos);
}

static float4 __attribute__((used))
_chip_tex1df_impl(image1d_t I, sampler_t S, float Pos) {
  return read_imagef(I, S, Pos);
}

static float4 __attribute__((used))
_chip_tex2df_impl(image2d_t I, sampler_t S, float2 Pos) {
  return read_imagef(I, S, Pos);
}

static int4 __attribute__((used))
_chip_tex2di_impl(image2d_t I, sampler_t S, float2 Pos) {
  return read_imagei(I, S, Pos);
}

static uint4 __attribute__((used))
_chip_tex2du_impl(image2d_t I, sampler_t S, float2 Pos) {
  return read_imageui(I, S, Pos);
}

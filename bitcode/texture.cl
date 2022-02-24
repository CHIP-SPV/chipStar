// Declarations and implementations for HIP texture functions.
//
// (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory

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

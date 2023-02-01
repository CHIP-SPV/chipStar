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

/*
 * This is counterpart to hipcl_mathlib.hh
 * ATM it can't be used right after compilation because of a problem with mangling.
 *
 * HIP with default AS set to 4 mangles functions with pointer args to:
 *   float @_Z13opencl_sincosfPf(float, float addrspace(4)*)
 * while OpenCL code compiled for SPIR mangles to either
 *   float @_Z6sincosfPU3AS4f(float, float addrspace(4)*)
 * or
 *   float @_Z6sincosfPf(float, float *)
*/

#define NON_OVLD
#define OVLD __attribute__((overloadable))
//#define AI __attribute__((always_inline))
#define EXPORT NON_OVLD

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define DEFAULT_AS __generic

#define CL_NAME(N) opencl_##N
#define CL_NAME2(N, S) opencl_##N##_##S

#define CL_NAME_MANGLED_ATOM(NAME, S) CL_NAME2(atomic_##NAME, S)

#define DEFOCML_OPENCL1F(NAME) \
float OVLD NAME(float f); \
double OVLD NAME(double f); \
half OVLD NAME(half f); \
half2 OVLD NAME(half2 f); \
EXPORT float CL_NAME2(NAME, f)(float x) { return NAME(x); } \
EXPORT double CL_NAME2(NAME, d)(double x) { return NAME(x); } \
EXPORT half CL_NAME2(NAME, h)(half x) { return NAME(x); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x) { return NAME(x); }


#define DEFOCML_OPENCL2F(NAME) \
float OVLD NAME(float x, float y); \
double OVLD NAME(double x, double y); \
half OVLD NAME(half x, half y); \
half2 OVLD NAME(half2 x, half2 y); \
EXPORT float CL_NAME2(NAME, f)(float x, float y) { return NAME(x, y); } \
EXPORT double CL_NAME2(NAME, d)(double x, double y) { return NAME(x, y); } \
EXPORT half CL_NAME2(NAME, h)(half x, half y) { return NAME(x, y); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x, half2 y) { return NAME(x, y); }

/*****************************************************************/

#define DEF_OPENCL1F(NAME) \
EXPORT float CL_NAME2(NAME, f)(float x) { return NAME(x); } \
EXPORT double CL_NAME2(NAME, d)(double x) { return NAME(x); } \
EXPORT half CL_NAME2(NAME, h)(half x) { return NAME(x); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x) { return NAME(x); }

#define DEF_OPENCL2F(NAME) \
EXPORT float CL_NAME2(NAME, f)(float x, float y) { return NAME(x, y); } \
EXPORT double CL_NAME2(NAME, d)(double x, double y) { return NAME(x, y); } \
EXPORT half CL_NAME2(NAME, h)(half x, half y) { return NAME(x, y); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x, half2 y) { return NAME(x, y); }

#define DEF_OPENCL3F(NAME) \
EXPORT float CL_NAME2(NAME, f)(float x, float y, float z) { return NAME(x, y, z); } \
EXPORT double CL_NAME2(NAME, d)(double x, double y, double z) { return NAME(x, y, z); } \
EXPORT half CL_NAME2(NAME, h)(half x, half y, half z) { return NAME(x, y, z); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x, half2 y, half2 z) { return NAME(x, y, z); }

#define DEF_OPENCL4F(NAME) \
EXPORT float CL_NAME2(NAME, f)(float x, float y, float z, float w) { return NAME(x, y, z, w); } \
EXPORT double CL_NAME2(NAME, d)(double x, double y, double z, double w) { return NAME(x, y, z, w); } \
EXPORT half CL_NAME2(NAME, h)(half x, half y, half z, half w) { return NAME(x, y, z, w); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x, half2 y, half2 z, half2 w) { return NAME(x, y, z, w); }

#define DEF_OPENCL1B(NAME) \
EXPORT int CL_NAME2(NAME, f)(float x) { return NAME(x); } \
EXPORT long CL_NAME2(NAME, d)(double x) { return NAME(x); } \
EXPORT half CL_NAME2(NAME, h)(half x) { return as_half(convert_short(NAME(x))); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x) { return as_half2(NAME(x)); }


#define DEF_OPENCL1INT(NAME) \
EXPORT int CL_NAME2(NAME, f)(float x) { return NAME(x); } \
EXPORT int CL_NAME2(NAME, d)(double x) { return NAME(x); } \
EXPORT int CL_NAME2(NAME, h)(half x) { return NAME(x); }
//EXPORT int CL_NAME(NAME, h2)(half2 x) { return NAME(x); }


#ifdef CHIP_NONPORTABLE_MATH_INTRISINCS

// The OpenCL native_* functions have no accuracy quarantees whatsoever
// whereas the __* math intrinsics of CUDA do. Thus, we cannot use the
// native functions by default if we want to retain portable correctness
// in the produced SPIR-Vs.

#define DEF_OPENCL1F_NATIVE(NAME) \
float OVLD native_##NAME(float x); \
double OVLD native_##NAME(double x); \
EXPORT float CL_NAME2(NAME##_native, f)(float x) { return native_##NAME(x); } \
EXPORT double CL_NAME2(NAME##_native, d)(double x) { return native_##NAME(x); }

#define DEF_OPENCL2F_NATIVE(NAME) \
float OVLD native_##NAME(float x, float y); \
double OVLD native_##NAME(double x, double y); \
EXPORT float CL_NAME2(NAME##_native, f)(float x, float y) { return native_##NAME(x, y); } \
EXPORT double CL_NAME2(NAME##_native, d)(double x, double y) { return native_##NAME(x, y); }

#else

#define DEF_OPENCL1F_NATIVE(NAME) \
float OVLD native_##NAME(float x); \
double OVLD native_##NAME(double x); \
EXPORT float CL_NAME2(NAME##_native, f)(float x) { return NAME(x); } \
EXPORT double CL_NAME2(NAME##_native, d)(double x) { return NAME(x); }

#define DEF_OPENCL2F_NATIVE(NAME) \
float OVLD native_##NAME(float x, float y); \
double OVLD native_##NAME(double x, double y); \
EXPORT float CL_NAME2(NAME##_native, f)(float x, float y) { return NAME(x, y); } \
EXPORT double CL_NAME2(NAME##_native, d)(double x, double y) { return NAME(x, y); }

#endif

// +7
DEF_OPENCL1F(acos)
DEF_OPENCL1F(asin)
DEF_OPENCL1F(acosh)
DEF_OPENCL1F(asinh)
DEF_OPENCL1F(atan)
DEF_OPENCL2F(atan2)
DEF_OPENCL1F(atanh)
DEF_OPENCL1F(cbrt)
DEF_OPENCL1F(ceil)

DEF_OPENCL2F(copysign)

DEF_OPENCL1F(cos)
DEF_OPENCL1F(cosh)
DEF_OPENCL1F(cospi)

// OCML
float OVLD i0(float f);
double OVLD i0(double f);
EXPORT float CL_NAME2(cyl_bessel_i0, f)(float x) { return i0(x); }
EXPORT double CL_NAME2(cyl_bessel_i0, d)(double x) { return i0(x); }
float OVLD i1(float f);
double OVLD i1(double f);
EXPORT float CL_NAME2(cyl_bessel_i1, f)(float x) { return i1(x); }
EXPORT double CL_NAME2(cyl_bessel_i1, d)(double x) { return i1(x); }


DEF_OPENCL1F(erfc)
DEF_OPENCL1F(erf)

// OCML
DEFOCML_OPENCL1F(erfcinv)
DEFOCML_OPENCL1F(erfcx)
DEFOCML_OPENCL1F(erfinv)

DEF_OPENCL1F(exp10)
DEF_OPENCL1F(exp2)
DEF_OPENCL1F(exp)
DEF_OPENCL1F(expm1)
DEF_OPENCL1F(fabs)
DEF_OPENCL2F(fdim)
DEF_OPENCL1F(floor)

DEF_OPENCL3F(fma)

DEF_OPENCL2F(fmax)
DEF_OPENCL2F(fmin)
DEF_OPENCL2F(fmod)

EXPORT float CL_NAME2(frexp, f)(float x, DEFAULT_AS int *i) {
  int tmp;
  float ret = frexp(x, &tmp);
  *i = tmp;
  return ret;
}
EXPORT double CL_NAME2(frexp, d)(double x, DEFAULT_AS int *i) {
  int tmp;
  double ret = frexp(x, &tmp);
  *i = tmp;
  return ret;
}

DEF_OPENCL2F(hypot)
DEF_OPENCL1INT(ilogb)

DEF_OPENCL1B(isfinite)
DEF_OPENCL1B(isinf)
DEF_OPENCL1B(isnan)

EXPORT float CL_NAME2(ldexp, f)(float x, int k) { return ldexp(x, k); }
EXPORT double CL_NAME2(ldexp, d)(double x, int k) { return ldexp(x, k); }


DEF_OPENCL1F(log10)
DEF_OPENCL1F(log1p)
DEF_OPENCL1F(log2)
DEF_OPENCL1F(logb)
DEF_OPENCL1F(log)

EXPORT float CL_NAME2(modf, f)(float x, DEFAULT_AS float *i) {
  float tmp;
  float ret = modf(x, &tmp);
  *i = tmp;
  return ret;
}
EXPORT double CL_NAME2(modf, d)(double x, DEFAULT_AS double *i) {
  double tmp;
  double ret = modf(x, &tmp);
  *i = tmp;
  return ret;
}

// OCML
DEFOCML_OPENCL1F(nearbyint)
DEFOCML_OPENCL2F(nextafter)

float OVLD length(float4 f);
double OVLD length(double4 f);
EXPORT float CL_NAME2(norm4d, f)(float x, float y, float z, float w) { float4 temp = (float4)(x, y, z, w); return length(temp); }
EXPORT double CL_NAME2(norm4d, d)(double x, double y, double z, double w) { double4 temp = (double4)(x, y, z, w); return length(temp); }
EXPORT float CL_NAME2(norm3d, f)(float x, float y, float z) { float4 temp = (float4)(x, y, z, 0.0f); return length(temp); }
EXPORT double CL_NAME2(norm3d, d)(double x, double y, double z) { double4 temp = (double4)(x, y, z, 0.0); return length(temp); }


// OCML ncdf / ncdfinv
DEFOCML_OPENCL1F(normcdf)
DEFOCML_OPENCL1F(normcdfinv)

DEF_OPENCL2F(pow)
DEF_OPENCL2F(remainder)
// OCML
DEFOCML_OPENCL1F(rcbrt)

// remquo
EXPORT float CL_NAME2(remquo, f)(float x, float y, DEFAULT_AS int *quo) {
  int tmp;
  float rem = remquo(x, y, &tmp);
  *quo = tmp;
  return rem;
}
EXPORT double CL_NAME2(remquo, d)(double x, double y, DEFAULT_AS int *quo) {
  int tmp;
  double rem = remquo(x, y, &tmp);
  *quo = tmp;
  return rem;
}

// OCML
DEFOCML_OPENCL2F(rhypot)

DEF_OPENCL1F(rsqrt)

// OCML
float OVLD scalbn(float f, int k);
double OVLD scalbn(double f, int k);
EXPORT float CL_NAME2(scalbn, f)(float x, int k) { return scalbn(x, k); }
EXPORT double CL_NAME2(scalbn, d)(double x, int k) { return scalbn(x, k); }
// OCML
DEFOCML_OPENCL2F(scalb)

DEF_OPENCL1B(signbit)

DEF_OPENCL1F(sin)
DEF_OPENCL1F(sinh)
DEF_OPENCL1F(sinpi)
DEF_OPENCL1F(sqrt)
DEF_OPENCL1F(tan)
DEF_OPENCL1F(tanh)
DEF_OPENCL1F(tgamma)
DEF_OPENCL1F(trunc)



// sincos
EXPORT float CL_NAME2(sincos, f)(float x, DEFAULT_AS float *cos) {
  float tmp;
  float sin = sincos(x, &tmp);
  *cos = tmp;
  return sin;
}

EXPORT double CL_NAME2(sincos, d)(double x, DEFAULT_AS double *cos) {
  double tmp;
  double sin = sincos(x, &tmp);
  *cos = tmp;
  return sin;
}

// OCML
DEFOCML_OPENCL1F(y0)
DEFOCML_OPENCL1F(y1)

DEFOCML_OPENCL1F(j0)
DEFOCML_OPENCL1F(j1)

/* native */

DEF_OPENCL1F_NATIVE(cos)
DEF_OPENCL1F_NATIVE(sin)
DEF_OPENCL1F_NATIVE(tan)

DEF_OPENCL1F_NATIVE(exp10)
DEF_OPENCL1F_NATIVE(exp2)
DEF_OPENCL1F_NATIVE(exp)

DEF_OPENCL1F_NATIVE(log10)
DEF_OPENCL1F_NATIVE(log2)
DEF_OPENCL1F_NATIVE(log)

DEF_OPENCL1F_NATIVE(rsqrt)
DEF_OPENCL1F_NATIVE(sqrt)

DEF_OPENCL2F_NATIVE(powr)

#ifdef CHIP_NONPORTABLE_MATH_INTRISINCS

DEF_OPENCL1F_NATIVE(recip)
DEF_OPENCL2F_NATIVE(divide)

#else

EXPORT float OVLD __fdividef(float x, float y) { return x / y; }

#endif


/* other */

EXPORT void CL_NAME(local_barrier)() { barrier(CLK_LOCAL_MEM_FENCE); }

EXPORT void CL_NAME(local_fence)() { mem_fence(CLK_LOCAL_MEM_FENCE); }

EXPORT void CL_NAME(global_fence)() { mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); }

EXPORT void CL_NAME(system_fence)() { mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); }
/* memory routines */

// sets size bytes of the memory pointed to by ptr to value
// interpret ptr as a unsigned char so that it writes as bytes
EXPORT void* CL_NAME(memset)(DEFAULT_AS void* ptr, int value, size_t size) {
  volatile unsigned char* temporary = ptr;

  for(int i=0;i<size;i++)
    temporary[i] = value;
  
    return ptr;
}

EXPORT void* CL_NAME(memcpy)(DEFAULT_AS void *dest, DEFAULT_AS const void * src, size_t n) {
  volatile unsigned char* temporary_dest = dest;
  volatile const unsigned char* temporary_src = src;

  for(int i=0;i<n;i++)
    temporary_dest[i] = temporary_src[i];

  return dest;
}

/**********************************************************************/

EXPORT uint CL_NAME2(popcount, ui)(uint var) {
  return popcount(var);
}

EXPORT ulong CL_NAME2(popcount, ul)(ulong var) {
  return popcount(var);
}


EXPORT int CL_NAME2(clz, i)(int var) {
  return clz(var);
}

EXPORT long CL_NAME2(clz, li)(long var) {
  return clz(var);
}

EXPORT int CL_NAME2(ctz, i)(int var) {
  return ctz(var);
}

EXPORT long CL_NAME2(ctz, li)(long var) {
  return ctz(var);
}


EXPORT int CL_NAME2(hadd, i)(int x, int y) {
  return hadd(x, y);
}

EXPORT int CL_NAME2(rhadd, i)(int x, int y) {
  return hadd(x, y);
}

EXPORT uint CL_NAME2(uhadd, ui)(uint x, uint y) {
  return hadd(x, y);
}

EXPORT uint CL_NAME2(urhadd, ui)(uint x, uint y) {
  return hadd(x, y);
}


EXPORT int CL_NAME2(mul24, i)(int x, int y) {
  return mul24(x, y);
}

EXPORT int CL_NAME2(mulhi, i)(int x, int y) {
  return mul_hi(x, y);
}

EXPORT long CL_NAME2(mul64hi, li)(long x, long y) {
  return mul_hi(x, y);
}


EXPORT uint CL_NAME2(umul24, ui)(uint x, uint y) {
  return mul24(x, y);
}

EXPORT uint CL_NAME2(umulhi, ui)(uint x, uint y) {
  return mul_hi(x, y);
}

EXPORT ulong CL_NAME2(umul64hi, uli)(ulong x, ulong y) {
  return mul_hi(x, y);
}




/**********************************************************************/

#define DEF_OPENCL_ATOMIC2(NAME)                                               \
  int CL_NAME_MANGLED_ATOM(NAME, i)(DEFAULT_AS int *address, int i) {          \
    volatile global int *gi = to_global(address);                              \
    if (gi)                                                                    \
      return atomic_##NAME(gi, i);                                             \
    else {                                                                     \
      volatile local int *li = to_local(address);                              \
      if (li)                                                                  \
        return atomic_##NAME(li, i);                                           \
      else                                                                     \
        return 0;                                                              \
    }                                                                          \
  };                                                                           \
  uint CL_NAME_MANGLED_ATOM(NAME, u)(                                          \
      DEFAULT_AS uint *address, uint ui) {                                     \
    volatile global uint *gi = to_global(address);                             \
    if (gi)                                                                    \
      return atomic_##NAME(gi, ui);                                            \
    else {                                                                     \
      volatile local uint *li = to_local(address);                             \
      if (li)                                                                  \
        return atomic_##NAME(li, ui);                                          \
      else                                                                     \
        return 0;                                                              \
    }                                                                          \
  };                                                                           \
  ulong CL_NAME_MANGLED_ATOM(NAME, l)(                                         \
      DEFAULT_AS ulong *address,                                               \
      ulong ull) {                                                             \
    volatile global ulong *gi =                                                \
        to_global((DEFAULT_AS ulong *)address);                                \
    if (gi)                                                                    \
      return atom_##NAME(gi, ull);                                             \
    else {                                                                     \
      volatile local ulong *li =                                               \
          to_local((DEFAULT_AS ulong *)address);                               \
      if (li)                                                                  \
        return atom_##NAME(li, ull);                                           \
      else                                                                     \
        return 0;                                                              \
    }                                                                          \
  };

DEF_OPENCL_ATOMIC2(add)
DEF_OPENCL_ATOMIC2(sub)
DEF_OPENCL_ATOMIC2(xchg)
DEF_OPENCL_ATOMIC2(min)
DEF_OPENCL_ATOMIC2(max)
DEF_OPENCL_ATOMIC2(and)
DEF_OPENCL_ATOMIC2(or)
DEF_OPENCL_ATOMIC2(xor)

#define DEF_OPENCL_ATOMIC1(NAME)                                               \
  int CL_NAME_MANGLED_ATOM(NAME, i)(DEFAULT_AS int *address) {                 \
    volatile global int *gi = to_global(address);                              \
    if (gi)                                                                    \
      return atomic_##NAME(gi);                                                \
    volatile local int *li = to_local(address);                                \
    if (li)                                                                    \
      return atomic_##NAME(li);                                                \
    return 0;                                                                  \
  };                                                                           \
  uint CL_NAME_MANGLED_ATOM(NAME, u)(                                          \
      DEFAULT_AS uint *address) {                                              \
    volatile global uint *gi = to_global(address);                             \
    if (gi)                                                                    \
      return atomic_##NAME(gi);                                                \
    volatile local uint *li = to_local(address);                               \
    if (li)                                                                    \
      return atomic_##NAME(li);                                                \
    return 0;                                                                  \
  };                                                                           \
  ulong CL_NAME_MANGLED_ATOM(NAME, l)(                                         \
      DEFAULT_AS ulong *address) {                                             \
    volatile global ulong *gi =                                                \
        to_global((DEFAULT_AS ulong *)address);                                \
    if (gi)                                                                    \
      return atom_##NAME(gi);                                                  \
    volatile local ulong *li = to_local((DEFAULT_AS ulong *)address);          \
    if (li)                                                                    \
      return atom_##NAME(li);                                                  \
    return 0;                                                                  \
  };

DEF_OPENCL_ATOMIC1(inc)
DEF_OPENCL_ATOMIC1(dec)

#define DEF_OPENCL_ATOMIC3(NAME)                                               \
  int CL_NAME_MANGLED_ATOM(NAME, i)(                                           \
      DEFAULT_AS int *address, int cmp, int val) {                             \
    volatile global int *gi = to_global(address);                              \
    if (gi)                                                                    \
      return atomic_##NAME(gi, cmp, val);                                      \
    volatile local int *li = to_local(address);                                \
    if (li)                                                                    \
      return atomic_##NAME(li, cmp, val);                                      \
    return 0;                                                                  \
  };                                                                           \
  uint CL_NAME_MANGLED_ATOM(NAME, u)(                                          \
      DEFAULT_AS uint *address, uint cmp,                                      \
      uint val) {                                                              \
    volatile global uint *gi = to_global(address);                             \
    if (gi)                                                                    \
      return atomic_##NAME(gi, cmp, val);                                      \
    volatile local uint *li = to_local(address);                               \
    if (li)                                                                    \
      return atomic_##NAME(li, cmp, val);                                      \
    return 0;                                                                  \
  };                                                                           \
  ulong CL_NAME_MANGLED_ATOM(NAME, l)(                                         \
      DEFAULT_AS ulong *address, ulong cmp,                                    \
      ulong val) {                                                             \
    volatile global ulong *gi =                                                \
        to_global((DEFAULT_AS ulong *)address);                                \
    if (gi)                                                                    \
      return atom_##NAME(gi, cmp, val);                                        \
    volatile local ulong *li = to_local((DEFAULT_AS ulong *)address);          \
    if (li)                                                                    \
      return atom_##NAME(li, cmp, val);                                        \
    return 0;                                                                  \
  };

DEF_OPENCL_ATOMIC3(cmpxchg)

/* This code adapted from AMD's HIP sources */

static OVLD float atomic_add_f(volatile local float *address, float val) {
  volatile local uint *uaddr = (volatile local uint *)address;
  uint old = *uaddr;
  uint r;

  do {
    r = old;
    old = atomic_cmpxchg(uaddr, r, as_uint(val + as_float(r)));
  } while (r != old);

  return as_float(r);
}

static OVLD double atom_add_d(volatile local double *address, double val) {
  volatile local ulong *uaddr = (volatile local ulong *)address;
  ulong old = *uaddr;
  ulong r;

  do {
    r = old;
    old = atom_cmpxchg(uaddr, r, as_ulong(val + as_double(r)));
  } while (r != old);

  return as_double(r);
}

static OVLD float atomic_exch_f(volatile local float *address, float val) {
  return as_float(atomic_xchg((volatile local uint *)(address), as_uint(val)));
}

static OVLD float atomic_add_f(volatile global float *address, float val) {
  volatile global uint *uaddr = (volatile global uint *)address;
  uint old = *uaddr;
  uint r;

  do {
    r = old;
    old = atomic_cmpxchg(uaddr, r, as_uint(val + as_float(r)));
  } while (r != old);

  return as_float(r);
}

static OVLD double atom_add_d(volatile global double *address, double val) {
  volatile global ulong *uaddr = (volatile global ulong *)address;
  ulong old = *uaddr;
  ulong r;

  do {
    r = old;
    old = atom_cmpxchg(uaddr, r, as_ulong(val + as_double(r)));
  } while (r != old);

  return as_double(r);
}

static OVLD float atomic_exch_f(volatile global float *address, float val) {
  return as_float(atomic_xchg((volatile global uint *)(address), as_uint(val)));
}

static OVLD uint atomic_inc2_u(volatile local uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, ((r >= val) ? 0 : (r+1)));
  } while (r != old);

  return r;
}

static OVLD uint atomic_dec2_u(volatile local uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, (((r == 0) || (r > val)) ? val : (r-1)));
  } while (r != old);

  return r;
}

static OVLD uint atomic_inc2_u(volatile global uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, ((r >= val) ? 0 : (r+1)));
  } while (r != old);

  return r;
}

static OVLD uint atomic_dec2_u(volatile global uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, (((r == 0) || (r > val)) ? val : (r-1)));
  } while (r != old);

  return r;
}

EXPORT float CL_NAME_MANGLED_ATOM(add, f)(DEFAULT_AS float *address,
                                 float val) {
  volatile global float *gi = to_global(address);
  if (gi)
    return atomic_add_f(gi, val);
  volatile local float *li = to_local(address);
  if (li)
    return atomic_add_f(li, val);
  return 0;
}

EXPORT double CL_NAME_MANGLED_ATOM(add, d)(DEFAULT_AS double *address,
                                  double val) {
  volatile global double *gi = to_global((DEFAULT_AS double *)address);
  if (gi)
    return atom_add_d(gi, val);
  volatile local double *li = to_local((DEFAULT_AS double *)address);
  if (li)
    return atom_add_d(li, val);
  return 0;
}

EXPORT float CL_NAME_MANGLED_ATOM(exch, f)(DEFAULT_AS float *address,
                                 float val) {
  volatile global float *gi = to_global(address);
  if (gi)
    return atomic_exch_f(gi, val);
  volatile local float *li = to_local(address);
  if (li)
    return atomic_exch_f(li, val);
  return 0;
}

EXPORT uint CL_NAME_MANGLED_ATOM(inc2, u)(DEFAULT_AS uint *address,
                                 uint val) {
  volatile global uint *gi = to_global((DEFAULT_AS uint *)address);
  if (gi)
    return atomic_inc2_u(gi, val);
  volatile local uint *li = to_local((DEFAULT_AS uint *)address);
  if (li)
    return atomic_inc2_u(li, val);
  return 0;
}

EXPORT uint CL_NAME_MANGLED_ATOM(dec2, u)(DEFAULT_AS uint *address,
                                 uint val) {
  volatile global uint *gi = to_global((DEFAULT_AS uint *)address);
  if (gi)
    return atomic_dec2_u(gi, val);
  volatile local uint *li = to_local((DEFAULT_AS uint *)address);
  if (li)
    return atomic_dec2_u(li, val);
  return 0;
}
/**********************************************************************/

// Use the Intel versions for now by default, since the Intel OpenCL CPU
// driver still implements only them, not the KHR versions.
#define sub_group_shuffle intel_sub_group_shuffle
#define sub_group_shuffle_xor intel_sub_group_shuffle_xor

int OVLD sub_group_shuffle(int var, uint srcLane);
float OVLD sub_group_shuffle(float var, uint srcLane);
int OVLD sub_group_shuffle_xor(int var, uint value);
float OVLD sub_group_shuffle_xor(float var, uint value);

// Compute the full warp lane id given a subwarp of size wSize and
// a "logical" lane id within it.
//
// Assumes that each subwarp behaves as a separate entity
// with a starting logical lane ID of 0.
__attribute__((always_inline))
static int warpLaneId(int subWarpLaneId, int wSize) {
  if (wSize == DEFAULT_WARP_SIZE)
    return subWarpLaneId;
  unsigned laneId = get_sub_group_local_id();
  unsigned logicalSubWarp = laneId / wSize;
  return logicalSubWarp * wSize + subWarpLaneId;
}

EXPORT OVLD int __shfl(int var, int srcLane, int wSize) {
  int laneId = get_sub_group_local_id();
  return sub_group_shuffle(var, warpLaneId(srcLane, wSize));
}
EXPORT OVLD float __shfl(float var, int srcLane, int wSize) {
  return sub_group_shuffle(var, warpLaneId(srcLane, wSize));
}

EXPORT OVLD int __shfl_xor(int var, int value, int warpSizeOverride) {
  return sub_group_shuffle_xor(var, value);
}
EXPORT OVLD float __shfl_xor(float var, int value, int warpSizeOverride) {
  return sub_group_shuffle_xor(var, value);
}


#define __SHFL_UP(T)							\
  EXPORT OVLD T __shfl_up(T var, uint delta, int wSize) {		\
    int laneId = get_sub_group_local_id();				\
    int logicalSubWarp = laneId / wSize;				\
    int logicalSubWarpLaneId = laneId % wSize;				\
    int subWarpSrcId = logicalSubWarpLaneId - delta;			\
    if (subWarpSrcId < 0)						\
      subWarpSrcId = logicalSubWarpLaneId;				\
    return sub_group_shuffle(var, logicalSubWarp * wSize + subWarpSrcId); \
}

__SHFL_UP(int);
__SHFL_UP(float);

#define __SHFL_DOWN(T)							\
EXPORT OVLD T __shfl_down(T var, uint delta, int wSize) {		\
  int laneId = get_sub_group_local_id();				\
  int logicalSubWarp = laneId / wSize;					\
  int logicalSubWarpLaneId = laneId % wSize;				\
  int subWarpSrcId = logicalSubWarpLaneId + delta;			\
  if (subWarpSrcId >= wSize)						\
    subWarpSrcId = logicalSubWarpLaneId;				\
  return sub_group_shuffle(var, logicalSubWarp * wSize + subWarpSrcId); \
}

__SHFL_DOWN(int);
__SHFL_DOWN(float);

uint4 sub_group_ballot( int predicate );
EXPORT OVLD ulong __ballot(int predicate) {
#if DEFAULT_WARP_SIZE <= 32
  return sub_group_ballot(predicate).x;
#else
  return sub_group_ballot(predicate).x |
    (sub_group_ballot(predicate).y << 32);
#endif
}

EXPORT OVLD int __all(int predicate) {
  return __ballot(predicate) == ~0;
}

EXPORT OVLD int __any(int predicate) {
  return __ballot(predicate) != 0;
}

EXPORT OVLD unsigned __lane_id() {
  return get_sub_group_local_id();
}

typedef struct {
  intptr_t  image;
  intptr_t  sampler;
} *hipTextureObject_t;

EXPORT float CL_NAME2(tex2D, f)(hipTextureObject_t textureObject,
				float x, float y) {
  return read_imagef(
    __builtin_astype(textureObject->image, read_only image2d_t),
    __builtin_astype(textureObject->sampler, sampler_t),
    (float2)(x, y)).x;
}

// In HIP long long is 64-bit integer. In OpenCL it's 128-bit integer.
EXPORT long __double_as_longlong(double x) { return as_long(x); }
EXPORT double __longlong_as_double(long int x) { return as_double(x); }

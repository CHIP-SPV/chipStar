/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// OCML prototypes
//#include "ocml.h"

// Tables
#include "tables.h"

// Builtins
//#include "builtins.h"

// Mangling
#define MATH_MANGLE(N) N
#define MATH_PRIVATE(N) __priv##N
#define MATH_PRIVATE_OVLD(N) __attribute__((overloadable)) __priv##N

// mine
#define MATH_CLZI(x) (x == 0 ? 32 : clz(x))
#define MATH_MAD(x, y, z) mad(x, y, z)
#define FINITE_ONLY_OPT() 0
#define BUILTIN_FMA_F32(x, y, z) fma(x, y, z)
#define MATH_SQRT(x) sqrt(x)
#define MATH_RCP(x) native_recip(x)
#define AS_FLOAT(x) as_float(x)
#define AS_INT(x) as_int(x)
#define AS_INT2(x) as_int2(x)
#define AS_UINT(x) as_uint(x)
#define BUILTIN_ABS_F32(x) fabs(x)
#define BUILTIN_COPYSIGN_F32(x, y) copysign(x, y)
#define HAVE_FAST_FMA32() 1
#define BUILTIN_RSQRT_F32(x) native_rsqrt(x)
#define MATH_FAST_RCP(x) native_recip(x)
#define MATH_FAST_DIV(x, y) ((x) / (y))
#define MATH_FAST_SQRT(x) native_sqrt(x)
#define MATH_DIV(x, y) ((x) / (y))

#define BUILTIN_CLAMP_F32(x, y, z) clamp(x, y, z)
#define BUILTIN_MAX_U32(x, y) max(x, y)
#define BUILTIN_MIN_U32(x, y) min(x, y)
#define BUILTIN_ISINF_F32(x) isinf(x)
#define BUILTIN_ISNAN_F32(x) isnan(x)

#define BUILTIN_LOG2_F32(x) native_log2(x)
#define BUILTIN_EXP2_F32(x) native_exp2(x)

#define BUILTIN_RINT_F32(x) rint(x)

static inline float BUILTIN_FRACTION_F32(float x) {
  float temp;
  float retval = fract(x, &temp);
  return retval;
}

static inline int frexp_exp(float x) {
  int e;
  float mant = frexp(x, &e);
  return e;
}

#define BUILTIN_FREXP_EXP_F32(x) frexp_exp(x)
#define BUILTIN_FLDEXP_F32(x, k) ldexp(x, (int)k)

static inline uint BUILTIN_BITALIGN_B32(uint x, uint y, uint shift) {
  return (x << (32-shift)) | (y >> shift);
}

// Optimization Controls
//#include "opts.h"

// Attributes
#define PUREATTR __attribute__((pure)) __attribute__((overloadable))
#define CONSTATTR __attribute__((const)) __attribute__((overloadable))

// Math controls
//#include "privF.h"

// Floating point patterns
#define SIGNBIT_SP32      (int)0x80000000
#define EXSIGNBIT_SP32    0x7fffffff
#define EXPBITS_SP32      0x7f800000
#define MANTBITS_SP32     0x007fffff
#define ONEEXPBITS_SP32   0x3f800000
#define TWOEXPBITS_SP32   0x40000000
#define HALFEXPBITS_SP32  0x3f000000
#define IMPBIT_SP32       0x00800000
#define QNANBITPATT_SP32  0x7fc00000
#define PINFBITPATT_SP32  0x7f800000
#define NINFBITPATT_SP32  (int)0xff800000
#define EXPBIAS_SP32      127
#define EXPSHIFTBITS_SP32 23
#define BIASEDEMIN_SP32   1
#define EMIN_SP32         -126
#define BIASEDEMAX_SP32   254
#define EMAX_SP32         127
#define MANTLENGTH_SP32   24
#define BASEDIGITS_SP32   7

#define CLASS_PINF 2
#define CLASS_NINF 4
#define CLASS_QNAN 8
#define CLASS_SNAN 16
#define CLASS_PSUB 32
#define CLASS_NSUB 64
#define CLASS_PZER 128
#define CLASS_NZER 256


static inline int CONSTATTR BUILTIN_CLASS_F32(float x, int klass)
{
  if ((klass & CLASS_PINF) && (as_int(x) == PINFBITPATT_SP32))
    return -1;
  if ((klass & CLASS_NINF) && (as_int(x) == NINFBITPATT_SP32))
    return -1;

  if ((klass & (CLASS_QNAN|CLASS_SNAN)) && (as_int(x) & QNANBITPATT_SP32))
    return -1;

  if ((klass & (CLASS_NZER|CLASS_PZER)) && ((as_int(x) & (~SIGNBIT_SP32)) == 0) )
    return -1;

  if (
        (klass & (CLASS_NSUB|CLASS_PSUB)) &&
        (
            ((as_int(x) & EXPBITS_SP32) == 0) && ((as_int(x) & MANTBITS_SP32) != 0)
        )
     )
    return -1;

  return 0;
}

// types
struct redret;
struct scret;

// declarations
PUREATTR float j1(float x);
PUREATTR float j0(float x);

CONSTATTR float erfinv(float x);
CONSTATTR float erfcinv(float x);

CONSTATTR float MATH_PRIVATE_OVLD(ba0)(float t);
CONSTATTR float MATH_PRIVATE_OVLD(ba1)(float t);

CONSTATTR float MATH_PRIVATE_OVLD(bp0)(float t);
CONSTATTR float MATH_PRIVATE_OVLD(bp1)(float t);

float MATH_PRIVATE_OVLD(cosb)(float x, int n, float p);
float MATH_PRIVATE_OVLD(sinb)(float x, int n, float p);

CONSTATTR struct scret MATH_PRIVATE_OVLD(sincosred)(float x);
CONSTATTR struct scret MATH_PRIVATE_OVLD(sincosred2)(float x, float y);

CONSTATTR struct redret MATH_PRIVATE_OVLD(trigredsmall)(float x);
CONSTATTR struct redret MATH_PRIVATE_OVLD(trigredlarge)(float x);
CONSTATTR struct redret MATH_PRIVATE_OVLD(trigred)(float x);

/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// OCML prototypes
//#include "ocml.h"

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Tables
#include "tables.h"

// Builtins
//#include "builtins.h"

// Mangling
#define MATH_MANGLE(N) N
#define MATH_PRIVATE(N) __priv##N
#define MATH_PRIVATE_OVLD(N) __attribute__((overloadable)) __priv##N

// mine
#define MATH_CLZI(x) (x == 0 ? 64 : clz(x))
#define MATH_MAD(x, y, z) fma(x, y, z)
#define FINITE_ONLY_OPT() 0
#define BUILTIN_FMA_F64(x, y, z) fma(x, y, z)
#define MATH_SQRT(x) sqrt(x)
#define MATH_RCP(x) native_recip(x)
#define AS_DOUBLE(x) as_double(x)
#define AS_INT2(x) as_int2(x)
#define AS_LONG(x) as_long(x)
#define BUILTIN_ABS_F64(x) fabs(x)
#define BUILTIN_COPYSIGN_F64(x, y) copysign(x, y)
#define MATH_FAST_SQRT(x) native_sqrt(x)
#define MATH_DIV(x, y) ((x) / (y))
#define BUILTIN_ISNAN_F64(x) isnan(x)
#define BUILTIN_MAX_F64(x, y) fmax(x, y)
#define BUILTIN_MIN_F64(x, y) fmin(x, y)

#define BUILTIN_RSQRT_F64(x) native_rsqrt(x)
#define BUILTIN_ISINF_F64(x) isinf(x)

#define BUILTIN_LOG2_F32(x) native_log2(x)
#define BUILTIN_EXP2_F32(x) native_exp2(x)

#define BUILTIN_RINT_F32(x) rint(x)
#define BUILTIN_RINT_F64(x) rint(x)

static inline double BUILTIN_FRACTION_F64(double x) {
  double temp;
  double retval = fract(x, &temp);
  return retval;
}


static inline int frexp_exp(double x) {
  int e;
  double mant = frexp(x, &e);
  return e;
}

#define BUILTIN_FREXP_EXP_F64(x) frexp_exp(x)
#define BUILTIN_FLDEXP_F64(x, k) ldexp(x, (int)k)

double BUILTIN_TRIG_PREOP_F64(double input, int shift);

// Optimization Controls
//#include "opts.h"

// Attributes
#define PUREATTR __attribute__((pure)) __attribute__((overloadable))
#define CONSTATTR __attribute__((const)) __attribute__((overloadable))

// Math controls
//#include "privD.h"

// Bit patterns
#define SIGNBIT_DP64      0x8000000000000000L
#define EXSIGNBIT_DP64    0x7fffffffffffffffL
#define EXPBITS_DP64      0x7ff0000000000000L
#define MANTBITS_DP64     0x000fffffffffffffL
#define ONEEXPBITS_DP64   0x3ff0000000000000L
#define TWOEXPBITS_DP64   0x4000000000000000L
#define HALFEXPBITS_DP64  0x3fe0000000000000L
#define IMPBIT_DP64       0x0010000000000000L
#define QNANBITPATT_DP64  0x7ff8000000000000L
#define INDEFBITPATT_DP64 0xfff8000000000000L
#define PINFBITPATT_DP64  0x7ff0000000000000L
#define NINFBITPATT_DP64  0xfff0000000000000L
#define EXPBIAS_DP64      1023
#define EXPSHIFTBITS_DP64 52
#define BIASEDEMIN_DP64   1
#define EMIN_DP64         -1022
#define BIASEDEMAX_DP64   2046
#define EMAX_DP64         1023
#define LAMBDA_DP64       1.0e300
#define MANTLENGTH_DP64   53
#define BASEDIGITS_DP64   15

#define CLASS_PINF 2
#define CLASS_NINF 4
#define CLASS_QNAN 8
#define CLASS_SNAN 16
#define CLASS_PSUB 32
#define CLASS_NSUB 64
#define CLASS_PZER 128
#define CLASS_NZER 256


static inline long CONSTATTR BUILTIN_CLASS_F64(double x, int klass)
{
  if ((klass & CLASS_PINF) && (as_long(x) == PINFBITPATT_DP64))
    return -1;
  if ((klass & CLASS_NINF) && (as_long(x) == NINFBITPATT_DP64))
    return -1;

  if ((klass & (CLASS_QNAN|CLASS_SNAN)) && (as_long(x) & QNANBITPATT_DP64))
    return -1;

  if ((klass & (CLASS_NZER|CLASS_PZER)) && ((as_long(x) & (~SIGNBIT_DP64)) == 0) )
    return -1;

  if (
        (klass & (CLASS_NSUB|CLASS_PSUB)) &&
        (
            ((as_long(x) & EXPBITS_DP64) == 0) && ((as_long(x) & MANTBITS_DP64) != 0)
        )
     )
    return -1;

  return 0;
}

// types
struct redret;
struct scret;

// declarations

extern PUREATTR double j1(double x);
extern PUREATTR double j0(double x);
extern CONSTATTR double erfinv(double x);
extern CONSTATTR double erfcinv(double x);

extern CONSTATTR double MATH_PRIVATE_OVLD(ba0)(double t);
extern CONSTATTR double MATH_PRIVATE_OVLD(ba1)(double t);

extern CONSTATTR double MATH_PRIVATE_OVLD(bp0)(double t);
extern CONSTATTR double MATH_PRIVATE_OVLD(bp1)(double t);

double MATH_PRIVATE_OVLD(cosb)(double x, int n, double p);
double MATH_PRIVATE_OVLD(sinb)(double x, int n, double p);

extern CONSTATTR struct scret MATH_PRIVATE_OVLD(sincosred)(double x);
extern CONSTATTR struct scret MATH_PRIVATE_OVLD(sincosred2)(double x, double y);

extern CONSTATTR struct redret MATH_PRIVATE_OVLD(trigredsmall)(double x);
extern CONSTATTR struct redret MATH_PRIVATE_OVLD(trigredlarge)(double x);
extern CONSTATTR struct redret MATH_PRIVATE_OVLD(trigred)(double x);

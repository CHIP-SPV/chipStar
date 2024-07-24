#include <math.h>
#include <string.h> // for std::memcpy

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

// Floating point patterns
#define SIGNBIT_HP16      0x8000
#define EXSIGNBIT_HP16    0x7fff
#define EXPBITS_HP16      0x7c00
#define MANTBITS_HP16     0x03ff
#define ONEEXPBITS_HP16   0x3c00
#define TWOEXPBITS_HP16   0x4000
#define HALFEXPBITS_HP16  0x3800
#define IMPBIT_HP16       0x0400
#define QNANBITPATT_HP16  0x7e00
#define PINFBITPATT_HP16  0x7c00
#define NINFBITPATT_HP16  0xfc00
#define EXPBIAS_HP16      15
#define EXPSHIFTBITS_HP16 10
#define BIASEDEMIN_HP16   1
#define EMIN_HP16         -14
#define BIASEDEMAX_HP16   30
#define EMAX_HP16         15
#define MANTLENGTH_HP16   11
#define BASEDIGITS_HP16   5



typedef int bool;
#define true 1
#define false 0
typedef float half;
typedef struct int2 {
    union {
        struct { int x, y; };
        struct { int lo, hi; };
    };
} int2;

typedef struct float2 {
    union {
        struct { float x, y; };
        struct { float lo, hi; };
    };
} float2;

typedef struct double2 {
    union {
        struct { double x, y; };
        struct { double lo, hi; };
    };
} double2;

typedef struct half2 {
    union {
        struct { half x, y; };
        struct { half lo, hi; };
    };
} half2;

typedef struct uint2 {
    union {
        struct { unsigned int x, y; };
        struct { unsigned int lo, hi; };
    };
} uint2;

typedef struct ushort2 {
    union {
        struct { unsigned short x, y; };
        struct { unsigned short lo, hi; };
    };
} ushort2;

typedef struct short2 {
    union {
        struct { short x, y; };
        struct { short lo, hi; };
    };
} short2;

typedef struct long2 {
    union {
        struct { long x, y; };
        struct { long lo, hi; };
    };
} long2;

typedef struct ulong2 {
    union {
        struct { unsigned long x, y; };
        struct { unsigned long lo, hi; };
    };
} ulong2;

#define __private 
#define __constant 
#include <stdint.h>

typedef uint32_t uint;
typedef uint16_t ushort;
typedef uint64_t ulong;
#include "ocml.h"

#define bit_cast(To, From, src) \
    ({ \
        To dst; \
        From tmp = src; \
        memcpy(&dst, &tmp, sizeof(To)); \
        dst; \
    })
#define AS_SHORT(X) bit_cast(short, typeof(X), X)
#define AS_SHORT2(X) bit_cast(short2, typeof(X), X)
#define AS_USHORT(X) bit_cast(unsigned short, typeof(X), X)
#define AS_USHORT2(X) bit_cast(ushort2, typeof(X), X)
#define AS_INT(X) bit_cast(int, typeof(X), X)
#define AS_INT2(X) bit_cast(int2, typeof(X), X)
#define AS_UINT(X) bit_cast(unsigned int, typeof(X), X)
#define AS_UINT2(X) bit_cast(uint2, typeof(X), X)
#define AS_LONG(X) bit_cast(long, typeof(X), X)
#define AS_ULONG(X) bit_cast(unsigned long, typeof(X), X)
#define AS_DOUBLE(X) bit_cast(double, typeof(X), X)
#define AS_FLOAT(X) bit_cast(float, typeof(X), X)
#define AS_HALF(X) bit_cast(half, typeof(X), X)
#define AS_HALF2(X) bit_cast(half2, typeof(X), X)
// // Class mask bits
#define CLASS_SNAN 0x001
#define CLASS_QNAN 0x002
#define CLASS_NINF 0x004
#define CLASS_NNOR 0x008
#define CLASS_NSUB 0x010
#define CLASS_NZER 0x020
#define CLASS_PZER 0x040
#define CLASS_PSUB 0x080
#define CLASS_PNOR 0x100
#define CLASS_PINF 0x200

#define BUILTIN_ABS_F32(x) __builtin_fabs(x)
#define BUILTIN_FMA_F32(a, b, c) __builtin_fma(a, b, c)
#define BUILTIN_FMA_F64(a, b, c) __builtin_fma(a, b, c)
#define BUILTIN_COPYSIGN_F32(a, b) __builtin_copysignf(a, b)
#define BUILTIN_ISNAN_F32(x) __builtin_isnanf(x)
#define BUILTIN_ISINF_F32(x) __builtin_isinf(x)
#define BUILTIN_ISINF_F64(x) __builtin_isinf(x)
static inline double BUILTIN_FREXP_MANT_F64(double x) {
    int exp;
    return frexp(x, &exp);
}
#define BUILTIN_EXP_F32(x) __builtin_expf(x)
#define BUILTIN_LOG_F32(x) __builtin_logf(x)
#define BUILTIN_SQRT_F32(x) __builtin_sqrtf(x)

#define BUILTIN_RSQRT_F32(x) (1.0f / __builtin_sqrtf(x))
#define BUILTIN_RSQRT_F64(x) __builtin_fabs(1.0 / __builtin_sqrt(x))

#define BUILTIN_DIV_F32(a, b) (a) / (b)
#define BUILTIN_DIV_F64(a, b) (a) / (b)

#define UGEN(x) x

#define BUILTIN_MAD_F32(a, b, c) __builtin_fmaf(a, b, c)
#define BUILTIN_MAD_2F32(a, b, c) vec2(__builtin_fmaf(a.x, b.x, c.x), __builtin_fmaf(a.y, b.y, c.y))
#define BUILTIN_MAD_F64(a, b, c) __builtin_fma(a, b, c)
#define BUILTIN_MAD_F16(a, b, c) __builtin_fmaf(a, b, c)
#define BUILTIN_MAD_2F16(a, b, c) vec2(__builtin_fmaf(a.x, b.x, c.x), __builtin_fmaf(a.y, b.y, c.y))

#define BUILTIN_RCP_F32(x) (1.0f / (x))
#define BUILTIN_RCP_F64(x) (1.0 / (x))

#define BUILTIN_ABS_F64(x) __builtin_fabs(x)
#define BUILTIN_COPYSIGN_F64(a, b) __builtin_copysignf(a, b)

#define BUILTIN_ISFINITE_F32(x) __builtin_isfinite(x)

// __builtin_modff and __builtin_modf return the fractional part of x, and store the integer part in the address of the second argument
// so we need to subtract the integer part from the fractional part to get the fractional part
static inline float BUILTIN_FRACTION_F32(float x) { float rem; return x - __builtin_modff(x, &rem); }
static inline double BUILTIN_FRACTION_F64(double x) { double rem; return x - __builtin_modf(x, &rem); }
#define BUILTIN_RINT_F32(x) __builtin_rintf(x)
#define BUILTIN_RINT_F64(x) __builtin_rint(x)
#define BUILTIN_ISFINITE_F64(x) __builtin_isfinite(x)



// Attributes
#define ALIGNEDATTR(X) __attribute__((aligned(X)))
#define INLINEATTR __attribute__((always_inline))
#define PUREATTR __attribute__((pure))
#define CONSTATTR __attribute__((const))

// #include "opts.h"
static inline bool HAVE_FAST_FMA32() { return true; }
static inline bool FINITE_ONLY_OPT() { return false; }
static inline bool UNSAFE_MATH_OPT() { return false; }
static inline bool DAZ_OPT() { return false; }
static inline bool CORRECTLY_ROUNDED_SQRT32() { return false; }


#define BUILTIN_CLASS_F32(x, class) __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, x) == (class)
#define BUILTIN_CLASS_F64(x, class) __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, x) == (class)
#define BUILTIN_CLASS_F16(x, class) __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, x) == (class)

#define BUILTIN_FREXP_EXP_F64(x) ({ \
    int exp; \
    (void)frexp(x, &exp); \
    exp; \
})

#define BUILTIN_FLDEXP_F64(x, exp) ldexp(x, exp)

#define BUILTIN_LOG2_F32(x) __builtin_log2f(x)

#define BUILTIN_EXP2_F32(x) __builtin_exp2f(x)
#define BUILTIN_CANONICALIZE_F32(x) (x)

#define BUILTIN_FLDEXP_F32(x, exp) ldexpf(x, exp)



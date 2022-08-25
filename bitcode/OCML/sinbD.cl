/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

#define FSUM2(A, B, H, L) \
    do { \
        double __s = A + B; \
        double __t = B - (__s - A); \
        H = __s; \
        L = __t; \
    } while (0)

#define FDIF2(A, B, H, L) \
    do { \
        double __d = A - B; \
        double __e = (A - __d) - B; \
        H = __d; \
        L = __e; \
    } while (0)

double
MATH_PRIVATE_OVLD(sinb)(double x, int n, double p)
{
    struct redret r = MATH_PRIVATE(trigred)(x);
    bool b = r.hi < p;
    r.i = (r.i - b - n) & 3;

    // This is a properly signed extra precise pi/4
    double ph = AS_DOUBLE((uint2)(0x54442d18, 0xbfe921fb ^ (b ? 0x80000000 : 0)));
    double pl = AS_DOUBLE((uint2)(0x33145c07, 0xbc81a626 ^ (b ? 0x80000000 : 0)));

    double sh, sl;

    FDIF2(ph, p, ph, sl);
    pl += sl;
    FSUM2(ph, pl, ph, pl);

    FSUM2(ph, r.hi, sh, sl);
    sl += pl + r.lo;
    FSUM2(sh, sl, sh, sl);

    struct scret sc = MATH_PRIVATE(sincosred2)(sh, sl);

    int2 s = AS_INT2((r.i & 1) == 0 ? sc.s : sc.c);
    s.hi ^= r.i > 1 ? 0x80000000 : 0;

    return AS_DOUBLE(s);
}


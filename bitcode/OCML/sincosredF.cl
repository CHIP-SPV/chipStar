/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

CONSTATTR struct scret
MATH_PRIVATE_OVLD(sincosred)(float x)
{
    float t = x * x;

    float s = MATH_MAD(x, t*MATH_MAD(t, MATH_MAD(t, -0x1.983304p-13f, 0x1.110388p-7f), -0x1.55553ap-3f), x);
    float c = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                  0x1.aea668p-16f, -0x1.6c9e76p-10f), 0x1.5557eep-5f), -0x1.000008p-1f), 1.0f);

    struct scret ret;
    ret.c = c;
    ret.s = s;
    return ret;
}


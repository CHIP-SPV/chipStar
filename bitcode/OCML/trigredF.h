/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define SMALL_BOUND 0x1.0p+17f

#if defined EXTRA_PRECISION
struct redret {
    float hi;
    float lo;
    int i;
};
#else
struct redret {
    float hi;
    int i;
};
#endif

struct scret {
    float s;
    float c;
};

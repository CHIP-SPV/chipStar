/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 *
 * Copyright (c) 2017 Michal Babej / Tampere University of Technology
 *
 * Copyright (c) 2022 Michal Babej / Parmance
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

static const unsigned char PIBITS_TBL[] = {
    224, 241, 27, 193, 12, 88, 33, 116, 53, 126, 196, 126, 237, 175,
    169, 75, 74, 41, 222, 231, 28, 244, 236, 197, 151, 175, 31,
    235, 158, 212, 181, 168, 127, 121, 154, 253, 24, 61, 221, 38,
    44, 159, 60, 251, 217, 180, 125, 180, 41, 104, 45, 70, 188,
    188, 63, 96, 22, 120, 255, 95, 226, 127, 236, 160, 228, 247,
    46, 126, 17, 114, 210, 231, 76, 13, 230, 88, 71, 230, 4, 249,
    125, 209, 154, 192, 113, 166, 19, 18, 237, 186, 212, 215, 8,
    162, 251, 156, 166, 196, 114, 172, 119, 248, 115, 72, 70, 39,
    168, 187, 36, 25, 128, 75, 55, 9, 233, 184, 145, 220, 134, 21,
    239, 122, 175, 142, 69, 249, 7, 65, 14, 241, 100, 86, 138, 109,
    3, 119, 211, 212, 71, 95, 157, 240, 167, 84, 16, 57, 185, 13,
    230, 139, 2, 0, 0, 0, 0, 0, 0, 0
};


// implements BUILTIN_TRIG_PREOP_F64(x, 0);

/* from RDNA instruction set manual:

V_TRIG_PREOP_F64

Look Up 2/PI (S0.d) with segment select S1.u[4:0]. This operation
returns an aligned, double precision segment of 2/PI needed to do
range reduction on S0.d (double-precision value). Multiple
segments can be specified through S1.u[4:0]. Rounding is round-
to-zero. Large inputs (exp > 1968) are scaled to avoid loss of
precision through denormalization.

shift = S1.u * 53;

if exponent(S0.d) > 1077 then
  shift += exponent(S0.d) - 1077;
endif

result = (double) ((2/PI[1200:0] << shift) &
0x1fffff_ffffffff);

scale = (-53 - shift);

if exponent(S0.d) >= 1968 then
  scale += 128;
endif

D.d = ldexp(result, scale).

*/


//  utmp2 = *(unsigned char *)(ptr + sizeof(ulong));

double BUILTIN_TRIG_PREOP_F64(double input, int shift) {
  shift = shift * 53;
  int expon = 0;
  double mant = frexp(input, &expon);
  if (expon > 1077) {
    shift += expon - 1077;
  }
  const unsigned char *ptr = PIBITS_TBL + shift / 8;
  ulong tmp = *(ulong *)(ptr);
  ulong tmp2 = *(ulong *)(ptr+1);
  int rem = shift % 8;
  if (rem) {
    int mask = (1 << rem) - 1;
    
  }
  double result = as_double(tmp & 0x1fffffffffffffUL);
  int scale = (-53 - shift);
  if (expon >= 1968) {
    scale += 128;
  }
  return ldexp(result, scale);
}

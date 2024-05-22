/*
 * Copyright (c) 2023 chipStar developers
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

// Implementations for emulated 64-bit floating point atomic add operations.

#include "cl_utils.h"

#ifndef __opencl_c_generic_address_space
#error __opencl_c_generic_address_space needed!
#endif

#define OVERLOADED __attribute__((overloadable))

/* This code is adapted from AMD's HIP sources */
static OVERLOADED double __chip_atomic_add_f64(volatile local double *address,
                                               double val) {
  volatile local ulong *uaddr = (volatile local ulong *)address;
  ulong old = *uaddr;
  ulong r;

  do {
    r = old;
    old = atom_cmpxchg(uaddr, r, as_ulong(val + as_double(r)));
  } while (r != old);

  return as_double(r);
}

static OVERLOADED double __chip_atomic_add_f64(volatile global double *address,
                                               double val) {
  volatile global ulong *uaddr = (volatile global ulong *)address;
  ulong old = *uaddr;
  ulong r;

  do {
    r = old;
    old = atom_cmpxchg(uaddr, r, as_ulong(val + as_double(r)));
  } while (r != old);

  return as_double(r);
}

double __chip_atomic_add_f64(__chip_obfuscated_ptr_t address, double val) {
  volatile global double *gi = to_global(UNCOVER_OBFUSCATED_PTR(address));
  if (gi)
    return __chip_atomic_add_f64(gi, val);
  volatile local double *li = to_local(UNCOVER_OBFUSCATED_PTR(address));
  if (li)
    return __chip_atomic_add_f64(li, val);
  return 0;
}

double __chip_atomic_add_system_f64(__chip_obfuscated_ptr_t address,
                                    double val) {
  return __chip_atomic_add_f64(address, val);
}

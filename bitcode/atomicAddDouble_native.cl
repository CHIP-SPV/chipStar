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

// Implementations for 64-bit floating point atomic operations using
// OpenCL built-in extension.

#include "cl_utils.h"

#ifndef __opencl_c_generic_address_space
#error __opencl_c_generic_address_space needed!
#endif

#if !defined(__opencl_c_ext_fp64_global_atomic_add) ||                         \
    !defined(__opencl_c_ext_fp64_local_atomic_add)
#error cl_ext_float_atomics needed!
#endif

#define OVERLOADED __attribute__((overloadable))

/* https://registry.khronos.org/OpenCL/extensions/ext/cl_ext_float_atomics.html
 */
#define DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME, OP, ORDER, SCOPE)                  \
  double __chip_atomic_##NAME##_f64(__chip_obfuscated_ptr_t address,           \
                                    double i) {                                \
    return atomic_##OP##_explicit(                                             \
        (volatile __generic double *)UNCOVER_OBFUSCATED_PTR(address), i,       \
        memory_order_##ORDER, memory_scope_##SCOPE);                           \
  }

#define DEF_CHIP_ATOMIC2F(NAME, OP)                                            \
  DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME, OP, relaxed, device)                     \
  DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME##_system, OP, relaxed, all_svm_devices)   \
  DEF_CHIP_ATOMIC2F_ORDER_SCOPE(NAME##_block, OP, relaxed, work_group)

DEF_CHIP_ATOMIC2F(add, fetch_add);

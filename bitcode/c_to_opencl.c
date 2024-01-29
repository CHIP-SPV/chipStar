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

// See c_to_opencl.def for details.

#define DEF_UNARY_FN_MAP(FROM_FN_, TO_FN_, TYPE_)                              \
  extern TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_);                                 \
  TYPE_ FROM_FN_(TYPE_ x) { return __chip_c2ocl_##FROM_FN_(x); }

#define DEF_BINARY_FN_MAP(FROM_FN_, TO_FN_, TYPE_)                             \
  extern TYPE_ __chip_c2ocl_##FROM_FN_(TYPE_, TYPE_);                          \
  TYPE_ FROM_FN_(TYPE_ x, TYPE_ y) { return __chip_c2ocl_##FROM_FN_(x, y); }

#include "c_to_opencl.def"

#undef UNARY_FN
#undef BINARY_FN

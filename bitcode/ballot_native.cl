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
// __ballot implementation with sub_group_ballot().

// SPIR-V requirements: 1.3+ unless environment supports
//                      cl_khr_subgroup_ballot or ZE_extension_subgroups.
// OpenCL requirements: cl_khr_subgroup_ballot.
// Level0 requirements: ZE_extension_subgroups.

#define OVERLOADED __attribute__((overloadable))

OVERLOADED ulong __chip_ballot(int predicate) {
#if DEFAULT_WARP_SIZE <= 32
  return sub_group_ballot(predicate).x;
#else
  return sub_group_ballot(predicate).x | (sub_group_ballot(predicate).y << 32);
#endif
}

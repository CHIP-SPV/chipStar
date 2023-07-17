/*
 * Copyright (c) 2021-22 chipStar developers
 * Copyright (c) 2022 Pekka Jääskeläinen / Parmance for Argonne National Laboratory
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


#ifndef CHIP_PRINTF_TESTS_COMMON_H
#define CHIP_PRINTF_TESTS_COMMON_H

#define CHECK(cmd, golden)                                              \
  do {                                                                  \
    auto produced = (cmd);                                              \
    if (produced != golden) {                                           \
      std::cerr << "Check failed: " << produced << " is not "           \
                << golden << " at line " << __LINE__ << std::endl;      \
      failures++;                                                       \
    }                                                                   \
  } while (0)                                                           \

#define CHECK_GT(cmd, golden)                                           \
  do {                                                                  \
    auto produced = (cmd);                                              \
    if (produced <= golden) {                                           \
      std::cerr << "Check failed: " << produced << " is not greater "   \
                << "than " << golden << " at line "                     \
                << __LINE__ << std::endl;                               \
      failures++;                                                       \
    }                                                                   \
  } while (0)                                                           \

#endif

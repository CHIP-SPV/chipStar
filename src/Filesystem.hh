/*
 * Copyright (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory
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

// Provides a namespace alias (fs) over std::filesystem-like module.

#ifndef SRC_FILESYSTEM_HH
#define SRC_FILESYSTEM_HH

#include "CHIPSPVConfig.hh"

#if defined(HAS_FILESYSTEM) && HAS_FILESYSTEM == 1

#include <filesystem>
namespace fs = std::filesystem;

#elif defined(HAS_EXPERIMENTAL_FILESYSTEM) && HAS_EXPERIMENTAL_FILESYSTEM == 1

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

#else
#error filesystem is not available!
#endif

#endif

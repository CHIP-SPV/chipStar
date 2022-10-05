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


#ifndef SRC_UTILS_HH
#define SRC_UTILS_HH

#include "common.hh"
#include "Filesystem.hh"
#include "hip/hip_fatbin.h"

#include <optional>

/// Clamps 'Val' to [0, INT_MAX] range.
static inline int clampToInt(size_t Val) {
  return std::min<size_t>(Val, std::numeric_limits<int>::max());
}

// Round a value up to next power of two - e.g. 13 -> 16, 8 -> 8.
static inline size_t roundUpToPowerOfTwo(size_t Val) {
  size_t Pow = 1;
  while (Pow < Val)
    Pow *= 2;
  return Pow;
}

/// Round a value up e.g. roundUp(9, 8) -> 16.
static inline size_t roundUp(size_t Val, size_t Rounding) {
  return ((Val + Rounding - 1) / Rounding) * Rounding;
}

/// Return a random 'N' length string.
std::string getRandomString(size_t N);

/// Return a unique directory for temporary use.
std::optional<fs::path> createTemporaryDirectory();

/// Write 'Data' into 'Path' file. If the file exists, its content is
/// overwriten. Return false on errors.
bool writeToFile(const fs::path Path, const std::string &Data);

/// Reads contents of file from 'Path' into a std::string.
std::optional<std::string> readFromFile(const fs::path Path);

/// Locate hipcc tool. Return an absolute path to it if found.
std::optional<fs::path> getHIPCCPath();

/// Returns a span (string_view) over SPIR-V module in the given clang
/// offload bundle.  Returns empty span if an error was encountered
/// and 'ErrorMsg' is set to describe the encountered error.
std::string_view extractSPIRVModule(const void *ClangOffloadBundle,
                                    std::string &ErrorMsg);

/// Convert "extra" kernel argument passing style to pointer array
/// style (an array of pointers to the arguments).
std::vector<void *> convertExtraArgsToPointerArray(void *ExtraArgBuf,
                                                   const OCLFuncInfo &FuncInfo);

#endif

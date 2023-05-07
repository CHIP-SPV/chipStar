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
#include <cstring>
#include <string_view>

std::string readEnvVar(std::string EnvVar, bool Lower = true);
void dumpSpirv(std::string_view Spirv);

/// Reinterpret the pointed region, starting from BaseAddr +
/// ByteOffset, as a value of the given type.
template <class T>
static T copyAs(const void *BaseAddr, size_t ByteOffset = 0) {
  T Res;
  std::memcpy(&Res, (const char *)BaseAddr + ByteOffset, sizeof(T));
  return Res;
}

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
                                                   const SPVFuncInfo &FuncInfo);

std::string_view trim(std::string_view Str);
bool startsWith(std::string_view Str, std::string_view WithStr);

/// A class for forming a iterator range which can be used in
/// 'for (auto E : C) ...' expressions.
template <typename IteratorT> class IteratorRange {
  // The implemention is copied from LLVM.
  IteratorT Begin_;
  IteratorT End_;

public:
  IteratorRange(IteratorT Begin, IteratorT End) : Begin_(Begin), End_(End) {}

  IteratorT begin() const { return Begin_; }
  IteratorT end() const { return End_; }
  bool empty() const { return Begin_ == End_; }
};

/// An iterator adaptor for map-like containers for iterating its keys only.
template <typename MapT>
class ConstMapKeyIterator : public MapT::const_iterator {
public:
  ConstMapKeyIterator(typename MapT::const_iterator It)
      : MapT::const_iterator(std::move(It)) {}

  const typename MapT::key_type &operator*() const {
    return MapT::const_iterator::operator*().first;
  }
};

// A less comparator for comparing mixed raw and smart pointers.
//
// From https://stackoverflow.com/questions/18939882. Formatted for
// CHIP-SPV.
template <class T> struct PointerCmp {
  typedef std::true_type is_transparent;
  struct Helper {
    T *Ptr;
    Helper() : Ptr(nullptr) {}
    Helper(Helper const &) = default;
    Helper(T *p) : Ptr(p) {}
    template <class U> Helper(std::shared_ptr<U> const &Sp) : Ptr(Sp.get()) {}
    template <class U, class... Ts>
    Helper(std::unique_ptr<U, Ts...> const &Up) : Ptr(Up.get()) {}
    // && optional: enforces rvalue use only
    bool operator<(Helper o) const { return std::less<T *>()(Ptr, o.Ptr); }
  };

  bool operator()(Helper const &&lhs, Helper const &&rhs) const {
    return lhs < rhs;
  }
};

#endif

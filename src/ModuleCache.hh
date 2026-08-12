/*
 * Copyright (c) 2026 chipStar developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/// Backend-agnostic module cache: key construction, compiler identity via
/// the loader delta, and validated atomic store/load of compiled device
/// binaries.
///
/// This file must not include any backend (OpenCL / Level Zero) header: it
/// is built unconditionally, and the key builder and file format are unit
/// tested without a GPU.

#ifndef SRC_MODULECACHE_HH
#define SRC_MODULECACHE_HH

#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace chipstar {
namespace cache {

/// Identifies a keyed field. These values are baked into every on-disk key:
/// never renumber, only append.
enum class KeyField : uint8_t {
  KeyFormatVersion = 0, ///< emitted automatically by the KeyBuilder ctor
  BackendTag = 1,       ///< "opencl" / "level0"
  Il = 2,               ///< SPIR-V bytes handed to the driver
  BuildOptions = 3,     ///< the string given to clBuildProgram/clCompileProgram
                        ///< or a Level Zero per-module build-flags string
  LinkFlags = 4,        ///< the string given to clLinkProgram
  BranchFlag = 5,       ///< e.g. needs-rtdevlib: which compile path ran
  RtDevLibName = 6,     ///< one rtdevlib module's name, in link order
  RtDevLibBytes = 7,    ///< that module's bitcode bytes
  DeviceName = 8,
  DriverVersion = 9, ///< driver version string; floor for the degraded
                     ///< case where the loader delta is unavailable
  DeviceId = 10,
  VendorId = 11,
  LoaderDelta = 12, ///< digest of libraries the runtime loaded at init
  Environment = 13, ///< collectCompilerEnvironmentVariables() result
};

/// Builds a cache key from length-prefixed tagged fields.
///
/// Each add() appends: u8 tag, u64 little-endian length, then the raw bytes.
/// Length prefixes make the serialization injective by construction: no
/// concatenation of fields can be re-split as different fields, which
/// delimiter schemes cannot guarantee when the data is arbitrary SPIR-V or
/// user-controlled flag strings. The tag byte additionally distinguishes
/// "field absent" from "field present but empty".
///
/// Field order is preserved, not sorted: repeated RtDevLibName/RtDevLibBytes
/// pairs are emitted in link order, which is a real input to the driver.
class KeyBuilder {
public:
  KeyBuilder();

  KeyBuilder &add(KeyField F, std::string_view Bytes);
  /// String literals would otherwise convert to bool (a standard
  /// conversion, which beats string_view's user-defined one) and silently
  /// collapse every such field to a single bit.
  KeyBuilder &add(KeyField F, const char *Str) {
    return add(F, std::string_view(Str));
  }
  KeyBuilder &add(KeyField F, const void *Data, size_t Size);
  KeyBuilder &add(KeyField F, uint64_t Value); ///< 8 raw LE bytes
  KeyBuilder &add(KeyField F, bool Value);

  /// 16 lowercase hex digits of fnv1a64 over the framed buffer. Used
  /// directly as the cache file name.
  std::string finish() const;

  /// The framed buffer itself; for tests and diagnostics, not for hashing
  /// by callers.
  std::string_view framed() const { return Buf_; }

private:
  std::string Buf_;
};

/// One shared library's identity: resolved path plus stat() facts.
struct LibStamp {
  std::string RealPath;
  uint64_t Size = 0;
  uint64_t MTimeNs = 0;
};

/// The token the loader-delta digest degrades to when nothing was observed
/// (non-Linux, or the application initialized the compute API before
/// libCHIP loaded so the runtime's libraries predate our snapshot).
inline constexpr const char *LoaderDeltaUnavailable =
    "loader-delta-unavailable";

/// Pure digest over an explicit stamp list; unit-testable. Empty input
/// yields exactly LoaderDeltaUnavailable.
std::string loaderDeltaDigestFrom(const std::vector<LibStamp> &Stamps);

/// Paths of all shared objects currently mapped into this process
/// (/proc/self/maps). Empty set on platforms without procfs.
std::set<std::string> snapshotMappedObjects();

/// Compute and memoize the loader-delta digest from the libraries mapped
/// since 'Before'. Called once, from chipstar::Backend::initialize(),
/// bracketing initializeImpl() — the window in which the runtime loads its
/// device compiler (NEO dlopens libigc during device enumeration).
void recordLoaderDelta(const std::set<std::string> &Before);

/// The memoized digest; LoaderDeltaUnavailable if recordLoaderDelta never
/// ran or observed nothing (a warning is logged in that case, since the key
/// then cannot see a compiler-only upgrade).
const std::string &loaderDeltaDigest();

/// A loaded cache entry. Owns its bytes; movable but not copyable, so a raw
/// pointer taken from data() cannot outlive the object via a silent copy.
/// Level Zero hands data() to zeModuleCreate, which requires the Entry to
/// stay in scope across that call.
class Entry {
public:
  Entry() = default;
  Entry(Entry &&) = default;
  Entry &operator=(Entry &&) = default;
  Entry(const Entry &) = delete;
  Entry &operator=(const Entry &) = delete;

  explicit operator bool() const { return !Data_.empty(); }
  const std::vector<uint8_t> &data() const { return Data_; }

private:
  std::vector<uint8_t> Data_;
  friend Entry load(const std::string &, std::string_view, const std::string &);
};

/// Look up Key under CacheDir/BackendTag. Never throws, never aborts. Any
/// I/O error, bad magic, wrong version, truncation or digest mismatch is a
/// miss, reported with one info-level "module-cache: MISS ... reason=..."
/// marker line.
Entry load(const std::string &CacheDir, std::string_view BackendTag,
           const std::string &Key);

/// Store a compiled binary under CacheDir/BackendTag/Key. Atomic
/// (pid+counter temp file, full-write check, fsync, rename). Never throws,
/// never aborts: on any failure it logs at debug level, removes its temp
/// file, and returns false.
bool store(const std::string &CacheDir, std::string_view BackendTag,
           const std::string &Key, const void *Data, size_t Size);

/// The canonical outcome markers the ctest driver greps for. load() emits
/// Miss itself; the backend emits Hit or Rejected once it knows whether the
/// cached bytes actually produced a working program.
enum class Outcome { Hit, Miss, Rejected };
void logOutcome(std::string_view BackendTag, const std::string &Key, Outcome O,
                std::string_view Detail);

} // namespace cache
} // namespace chipstar

#endif // SRC_MODULECACHE_HH

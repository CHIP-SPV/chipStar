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

#include "ModuleCache.hh"

#include "Utils.hh"
#include "logging.hh"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>

// POSIX file I/O for the store path: available on Linux and macOS alike.
// Only the loader-delta snapshot (/proc/self/maps) is Linux-specific.
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace chipstar {
namespace cache {

// ---------------------------------------------------------------------------
// KeyBuilder

/// On-key format version. Bump when the framing itself changes.
static constexpr uint64_t KeyFormatVersionValue = 1;

static void appendU64LE(std::string &Buf, uint64_t V) {
  for (int I = 0; I < 8; ++I)
    Buf.push_back(static_cast<char>((V >> (8 * I)) & 0xFF));
}

KeyBuilder::KeyBuilder() {
  add(KeyField::KeyFormatVersion, KeyFormatVersionValue);
}

KeyBuilder &KeyBuilder::add(KeyField F, std::string_view Bytes) {
  Buf_.push_back(static_cast<char>(F));
  appendU64LE(Buf_, Bytes.size());
  Buf_.append(Bytes.data(), Bytes.size());
  return *this;
}

KeyBuilder &KeyBuilder::add(KeyField F, const void *Data, size_t Size) {
  return add(F, std::string_view(static_cast<const char *>(Data), Size));
}

KeyBuilder &KeyBuilder::add(KeyField F, uint64_t Value) {
  Buf_.push_back(static_cast<char>(F));
  appendU64LE(Buf_, 8);
  appendU64LE(Buf_, Value);
  return *this;
}

KeyBuilder &KeyBuilder::add(KeyField F, bool Value) {
  Buf_.push_back(static_cast<char>(F));
  appendU64LE(Buf_, 1);
  Buf_.push_back(Value ? '\1' : '\0');
  return *this;
}

std::string KeyBuilder::finish() const {
  uint64_t H = fnv1a64(Buf_);
  char Hex[17];
  std::snprintf(Hex, sizeof(Hex), "%016llx",
                static_cast<unsigned long long>(H));
  return std::string(Hex, 16);
}

// ---------------------------------------------------------------------------
// Loader delta

std::set<std::string> snapshotMappedObjects() {
  std::set<std::string> Objects;
#ifdef __linux__
  std::ifstream Maps("/proc/self/maps");
  for (std::string Line; std::getline(Maps, Line);) {
    // Line: address perms offset dev inode  /path. The path is the first
    // (and only) field containing '/'.
    auto Pos = Line.find('/');
    if (Pos == std::string::npos)
      continue;
    std::string Path = Line.substr(Pos);
    if (Path.find(".so") != std::string::npos)
      Objects.insert(std::move(Path)); // one lib maps several times; set dedups
  }
#endif
  return Objects;
}

std::string loaderDeltaDigestFrom(const std::vector<LibStamp> &Stamps) {
  if (Stamps.empty())
    return LoaderDeltaUnavailable;
  std::string Buf;
  for (const auto &S : Stamps) {
    Buf += S.RealPath;
    Buf += '|';
    Buf += std::to_string(S.Size);
    Buf += '|';
    Buf += std::to_string(S.MTimeNs);
    Buf += '\n';
  }
  uint64_t H = fnv1a64(Buf);
  char Hex[17];
  std::snprintf(Hex, sizeof(Hex), "%016llx",
                static_cast<unsigned long long>(H));
  return std::string(Hex, 16);
}

static std::string &deltaDigestStorage() {
  static std::string Digest;
  return Digest;
}

void recordLoaderDelta(const std::set<std::string> &Before) {
  std::vector<LibStamp> Stamps;
#ifdef __linux__
  for (const auto &Path : snapshotMappedObjects()) {
    if (Before.count(Path))
      continue;
    LibStamp Stamp;
    char Resolved[PATH_MAX];
    Stamp.RealPath = ::realpath(Path.c_str(), Resolved) ? Resolved : Path;
    struct stat St;
    if (::stat(Stamp.RealPath.c_str(), &St) == 0) {
      Stamp.Size = static_cast<uint64_t>(St.st_size);
      Stamp.MTimeNs = static_cast<uint64_t>(St.st_mtim.tv_sec) * 1000000000ull +
                      static_cast<uint64_t>(St.st_mtim.tv_nsec);
    }
    Stamps.push_back(std::move(Stamp));
  }
  // snapshotMappedObjects returns a sorted set, and realpath preserves no
  // useful order, so re-sort by resolved path for load-order independence.
  std::sort(Stamps.begin(), Stamps.end(),
            [](const LibStamp &A, const LibStamp &B) {
              return A.RealPath < B.RealPath;
            });
#endif
  deltaDigestStorage() = loaderDeltaDigestFrom(Stamps);
  if (Stamps.empty()) {
#ifdef __linux__
    // On Linux an empty delta is a surprise worth flagging: it usually means
    // the application initialized the compute API before libCHIP loaded.
    logWarn("module-cache: no libraries were loaded during backend init; "
            "the cache key cannot see a device-compiler-only upgrade "
            "(was the compute runtime initialized before libCHIP?)");
#else
    // Without /proc/self/maps there is nothing to observe; this is the
    // normal state, not a misconfiguration.
    logDebug("module-cache: loader delta unavailable on this platform");
#endif
  } else
    logDebug("module-cache: loader delta covers {} libraries, digest {}",
             Stamps.size(), deltaDigestStorage());
}

const std::string &loaderDeltaDigest() {
  static const std::string Unavailable = LoaderDeltaUnavailable;
  return deltaDigestStorage().empty() ? Unavailable : deltaDigestStorage();
}

// ---------------------------------------------------------------------------
// Outcome markers

void logOutcome(std::string_view BackendTag, const std::string &Key, Outcome O,
                std::string_view Detail) {
  switch (O) {
  case Outcome::Hit:
    logInfo("module-cache: HIT backend={} key={} {}", BackendTag, Key, Detail);
    break;
  case Outcome::Miss:
    logInfo("module-cache: MISS backend={} key={} reason={}", BackendTag, Key,
            Detail);
    break;
  case Outcome::Rejected:
    // Always means corruption or a format/driver mismatch: make it loud.
    logWarn("module-cache: REJECTED backend={} key={} stage={}", BackendTag,
            Key, Detail);
    break;
  }
}

// ---------------------------------------------------------------------------
// On-disk format CHM2 (single binary per entry; the device is in the key)
//
//   offset size field
//     0     4   magic          'C','H','M','2'
//     4     2   format_version u16 LE (= 2)
//     6     2   reserved       u16 LE (= 0)
//     8     8   payload_len    u64 LE
//    16     8   digest         u64 LE, fnv1a64 over the payload
//    24    ...  payload
//
// payload_len is checked against the actual file size before anything is
// allocated (a magic alone cannot catch truncation: a file cut at a page
// boundary still starts with CHM2). The digest catches a full-length file
// with corrupt content.

static constexpr char Magic[4] = {'C', 'H', 'M', '2'};
static constexpr uint16_t FormatVersion = 2;
static constexpr size_t HeaderSize = 24;

static void putU16LE(uint8_t *P, uint16_t V) {
  P[0] = V & 0xFF;
  P[1] = (V >> 8) & 0xFF;
}
static void putU64LE(uint8_t *P, uint64_t V) {
  for (int I = 0; I < 8; ++I)
    P[I] = (V >> (8 * I)) & 0xFF;
}
static uint16_t getU16LE(const uint8_t *P) {
  return static_cast<uint16_t>(P[0]) | (static_cast<uint16_t>(P[1]) << 8);
}
static uint64_t getU64LE(const uint8_t *P) {
  uint64_t V = 0;
  for (int I = 0; I < 8; ++I)
    V |= static_cast<uint64_t>(P[I]) << (8 * I);
  return V;
}

static uint64_t digestBytes(const void *Data, size_t Size) {
  // fnv1a64 in Utils takes std::string; avoid the copy for large binaries.
  uint64_t Hash = UINT64_C(14695981039346656037);
  const unsigned char *P = static_cast<const unsigned char *>(Data);
  for (size_t I = 0; I < Size; ++I) {
    Hash ^= P[I];
    Hash *= UINT64_C(1099511628211);
  }
  return Hash;
}

static std::string entryPath(const std::string &CacheDir,
                             std::string_view BackendTag,
                             const std::string &Key) {
  std::string P = CacheDir;
  P += '/';
  P.append(BackendTag.data(), BackendTag.size());
  P += '/';
  P += Key;
  return P;
}

Entry load(const std::string &CacheDir, std::string_view BackendTag,
           const std::string &Key) {
  Entry Result;
  std::string Path = entryPath(CacheDir, BackendTag, Key);

  std::ifstream In(Path, std::ios::in | std::ios::binary);
  if (!In) {
    logOutcome(BackendTag, Key, Outcome::Miss, "absent");
    return Result;
  }

  In.seekg(0, std::ios::end);
  auto EndPos = In.tellg();
  if (EndPos < 0) {
    logOutcome(BackendTag, Key, Outcome::Miss, "open-failed");
    return Result;
  }
  size_t FileSize = static_cast<size_t>(EndPos);
  In.seekg(0, std::ios::beg);

  if (FileSize < HeaderSize) {
    logOutcome(BackendTag, Key, Outcome::Miss, "size-mismatch");
    return Result;
  }

  uint8_t Header[HeaderSize];
  if (!In.read(reinterpret_cast<char *>(Header), HeaderSize)) {
    logOutcome(BackendTag, Key, Outcome::Miss, "short-read");
    return Result;
  }

  if (std::memcmp(Header, Magic, sizeof(Magic)) != 0) {
    logOutcome(BackendTag, Key, Outcome::Miss, "bad-magic");
    return Result;
  }
  if (getU16LE(Header + 4) != FormatVersion) {
    logOutcome(BackendTag, Key, Outcome::Miss, "bad-version");
    return Result;
  }
  uint64_t PayloadLen = getU64LE(Header + 8);
  uint64_t Digest = getU64LE(Header + 16);

  // Non-overflowing: compare against what actually remains in the file, so a
  // corrupt PayloadLen can neither wrap the arithmetic nor reach the
  // allocator.
  if (FileSize - HeaderSize != PayloadLen) {
    logOutcome(BackendTag, Key, Outcome::Miss, "size-mismatch");
    return Result;
  }

  std::vector<uint8_t> Payload(PayloadLen);
  if (PayloadLen > 0 &&
      !In.read(reinterpret_cast<char *>(Payload.data()), PayloadLen)) {
    logOutcome(BackendTag, Key, Outcome::Miss, "short-read");
    return Result;
  }

  if (digestBytes(Payload.data(), Payload.size()) != Digest) {
    logOutcome(BackendTag, Key, Outcome::Miss, "digest-mismatch");
    return Result;
  }

  Result.Data_ = std::move(Payload);
  return Result;
}

bool store(const std::string &CacheDir, std::string_view BackendTag,
           const std::string &Key, const void *Data, size_t Size) {
  std::string Dir = CacheDir;
  Dir += '/';
  Dir.append(BackendTag.data(), BackendTag.size());

  std::error_code EC;
  fs::create_directories(Dir, EC); // non-throwing overload
  if (EC) {
    logDebug("module-cache: cannot create {}: {}", Dir, EC.message());
    return false;
  }

  std::string Final = entryPath(CacheDir, BackendTag, Key);

  // pid alone does not distinguish two threads of one process storing the
  // same key concurrently; add a process-wide counter.
  static std::atomic<unsigned> TempCounter{0};
  std::string Temp = Final + ".tmp." + std::to_string(::getpid()) + "." +
                     std::to_string(TempCounter.fetch_add(1));

  int Fd = ::open(Temp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (Fd == -1) {
    logDebug("module-cache: open({}) failed: {}", Temp, std::strerror(errno));
    return false;
  }

  uint8_t Header[HeaderSize];
  std::memcpy(Header, Magic, sizeof(Magic));
  putU16LE(Header + 4, FormatVersion);
  putU16LE(Header + 6, 0);
  putU64LE(Header + 8, Size);
  putU64LE(Header + 16, digestBytes(Data, Size));

  auto WriteAll = [&](const void *Buf, size_t Len) {
    const char *P = static_cast<const char *>(Buf);
    while (Len > 0) {
      ssize_t N = ::write(Fd, P, Len);
      if (N < 0) {
        if (errno == EINTR)
          continue; // retry; a short write is completed by the loop
        return false;
      }
      P += N;
      Len -= static_cast<size_t>(N);
    }
    return true;
  };

  bool Ok =
      WriteAll(Header, HeaderSize) && WriteAll(Data, Size) && ::fsync(Fd) == 0;
  Ok = (::close(Fd) == 0) && Ok; // NFS can surface write errors at close
  if (Ok)
    Ok = ::rename(Temp.c_str(), Final.c_str()) == 0;
  if (!Ok) {
    logDebug("module-cache: store of {} failed: {}", Final,
             std::strerror(errno));
    ::unlink(Temp.c_str());
    return false;
  }
  logDebug("module-cache: stored {} ({} bytes)", Final, Size);
  return true;
}

} // namespace cache
} // namespace chipstar

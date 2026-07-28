/*
 * Copyright (c) 2026 chipStar developers
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

// The content of a header included via an -I search path must participate in the
// HIPRTC compilation cache key: with the source text and compile options held
// constant, a compile is served from cache when (and only when) the -I header
// content matches a previously compiled one.
//
// This is a compile-only test: it never launches a kernel, so it needs no device
// and can run wherever the build runs. It compiles the same source four times
// with the same options, changing only the content of an -I header, and reads
// HIPRTC's own cache decision (hit vs. store) from the compile's stderr output:
//
//   compile 1  header = A  -> miss   (cold: A's content is new)
//   compile 2  header = A  -> hit    (unchanged: A's entry is reused)
//   compile 3  header = B  -> miss   (only the header changed: different key)
//   compile 4  header = A  -> hit    (A's entry is still distinct from B's)
//
// Compile 2's hit proves caching is genuinely exercised (the test cannot pass
// vacuously if caching never happens). Compile 3's miss proves the header
// content is in the key. Compile 4's hit proves the key is a function of the
// header's CONTENT: had the key keyed on, say, the header's timestamp, reverting
// the content would still be a miss (a fresh timestamp), never a hit.
//
// Requirements (provided by the harness, not by this program):
//   - CHIP_MODULE_CACHE_DIR points at a cache directory that is EMPTY at start,
//     so compile 1 is a cold miss. CMakeLists.txt wipes a dedicated directory via
//     a ctest fixture before the test runs.
//   - CHIP_LOGLEVEL=info, so HIPRTC's hit/miss reporting is emitted.
// libCHIP reads both at static-init time, before main(); CMakeLists.txt sets them
// for ctest.

#include "TestCommon.hh"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

// Fixed across every compile. The only thing that varies between compiles is the
// value defined in the -I header, so the raw source text and options are
// constant while the preprocessed translation unit (and thus the key) differs.
static constexpr auto Source = R"---(
#include "hiprtc_cache_inc.h"
extern "C" __global__ void addConst(int *Out, const int *In) {
  *Out = *In + HIPRTC_CACHE_TEST_ADDEND;
}
)---";

static const char *IncludeHeaderName = "hiprtc_cache_inc.h";

struct CacheDecision {
  bool Hit;     // HIPRTC served this compile from cache
  bool Touched; // HIPRTC reported any cache activity (hit or store)
};

// Overwrite the -I header so it defines HIPRTC_CACHE_TEST_ADDEND to Value. The
// numeric value only needs to differ to make the header content differ.
static void writeHeader(const fs::path &HeaderPath, int Value) {
  std::ofstream F(HeaderPath, std::ios::binary | std::ios::trunc);
  TEST_ASSERT(F.is_open());
  F << "#define HIPRTC_CACHE_TEST_ADDEND " << Value << "\n";
  F.close();
  TEST_ASSERT(F.good());
}

// Compile the fixed Source with -I<IncDir> and report HIPRTC's cache decision,
// read from the diagnostics it writes to stderr during the compile. spdlog's
// sink is stderr (fd 2); we redirect it to a temp file for the compile, then
// restore it and echo the captured text so it still appears in the test log.
static CacheDecision compileAndGetCacheDecision(const fs::path &IncDir) {
  hiprtcProgram Prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&Prog, Source, "cache_inc.hip", 0, nullptr,
                                   nullptr));

  std::string IncludeOpt = "-I" + IncDir.string();
  std::vector<const char *> Options = {IncludeOpt.c_str()};

  std::fflush(stderr);
  int SavedStderr = dup(STDERR_FILENO);
  TEST_ASSERT(SavedStderr >= 0);
  std::FILE *Cap = std::tmpfile();
  TEST_ASSERT(Cap != nullptr);
  TEST_ASSERT(dup2(fileno(Cap), STDERR_FILENO) >= 0);

  hiprtcResult R = hiprtcCompileProgram(Prog, Options.size(), Options.data());

  std::fflush(stderr);
  TEST_ASSERT(dup2(SavedStderr, STDERR_FILENO) >= 0);
  close(SavedStderr);

  std::string Captured;
  std::fseek(Cap, 0, SEEK_END);
  long N = std::ftell(Cap);
  if (N > 0) {
    std::fseek(Cap, 0, SEEK_SET);
    Captured.resize(static_cast<size_t>(N));
    size_t Got = std::fread(&Captured[0], 1, static_cast<size_t>(N), Cap);
    Captured.resize(Got);
  }
  std::fclose(Cap);
  std::cerr << Captured;

  if (R != HIPRTC_SUCCESS) {
    size_t LogSize = 0;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(Prog, &LogSize));
    if (LogSize) {
      std::string Log(LogSize, '\0');
      hiprtcGetProgramLog(Prog, Log.data());
      std::cerr << Log << "\n";
    }
    HIPRTC_CHECK(R);
  }
  HIPRTC_CHECK(hiprtcDestroyProgram(&Prog));

  bool Hit = Captured.find("Cache hit") != std::string::npos;
  bool Stored = Captured.find("Saved SPIRV to cache") != std::string::npos;
  bool Loaded = Captured.find("Loaded SPIRV from cache") != std::string::npos;
  return {Hit, (Hit || Stored || Loaded)};
}

int main() {
  // The cache must be enabled for this test to mean anything; the harness points
  // CHIP_MODULE_CACHE_DIR at a directory that is empty at start (see the file
  // header). We do not manage that directory here.
  const char *CacheDir = std::getenv("CHIP_MODULE_CACHE_DIR");
  if (!CacheDir || !*CacheDir) {
    std::cerr << "CHIP_MODULE_CACHE_DIR is not set; this test needs an (empty) "
                 "cache directory. The CMake build provides one for ctest.\n";
    return 1;
  }

  // Private -I directory holding the header we mutate between compiles.
  std::error_code Ec;
  fs::path IncDir = fs::temp_directory_path() / "chipstar_hiprtc_cache_inc_test";
  fs::remove_all(IncDir, Ec);
  TEST_ASSERT(fs::create_directories(IncDir, Ec));
  fs::path HeaderPath = IncDir / IncludeHeaderName;

  // Compile 1: header content A. Cold cache -> miss (store).
  writeHeader(HeaderPath, 111);
  CacheDecision D1 = compileAndGetCacheDecision(IncDir);
  std::cerr << "compile #1 (header A): hit=" << D1.Hit << "\n";
  // Cache must actually be exercised, otherwise the remaining checks are vacuous;
  // and a fresh cache means this cold compile is a miss (also catches a stale,
  // non-empty cache directory).
  TEST_ASSERT(D1.Touched);
  TEST_ASSERT(!D1.Hit);

  // Compile 2: identical source, options, and header content -> hit. This is what
  // proves caching is genuinely happening and the key is stable for equal input.
  CacheDecision D2 = compileAndGetCacheDecision(IncDir);
  std::cerr << "compile #2 (header A, unchanged): hit=" << D2.Hit << "\n";
  TEST_ASSERT(D2.Hit);

  // Compile 3: only the -I header content changes (B). The key reflects header
  // content, so this is a miss.
  writeHeader(HeaderPath, 222);
  CacheDecision D3 = compileAndGetCacheDecision(IncDir);
  std::cerr << "compile #3 (header B): hit=" << D3.Hit << "\n";
  TEST_ASSERT(!D3.Hit);

  // Compile 4: revert to header content A. Its entry is distinct from B's, so
  // this is a hit -- proving the key tracks header content, not (e.g.) mtime.
  writeHeader(HeaderPath, 111);
  CacheDecision D4 = compileAndGetCacheDecision(IncDir);
  std::cerr << "compile #4 (header A, reverted): hit=" << D4.Hit << "\n";
  TEST_ASSERT(D4.Hit);

  fs::remove_all(IncDir, Ec);
  std::cerr << "Test passed: -I #include content participates in the HIPRTC "
               "cache key.\n";
  return 0;
}

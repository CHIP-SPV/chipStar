/*
 * Copyright (c) 2026 chipStar developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit passing to whom the Software is
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

// The program log returned by hiprtcGetProgramLog() must not depend on whether
// the compilation was served from the HIPRTC cache, and a diagnostic must not be
// repeated because of internal re-processing of the options.
//
// Both properties were broken:
//
//   * hiprtcCompileProgram() used to process the user options only on the
//     cache-miss path (inside compile()). processOptions() is what appends
//     "warning: ignored option '...'" to the log, so on a cache hit the log came
//     back EMPTY. The visible symptom was TestHiprtcOptions passing against a
//     cold ~/.cache/chipStar and failing on every run after it.
//
//   * When the cache key started being computed from preprocessed source, the
//     preprocess pass processed the options a second time, so an
//     include-bearing source logged the same warning twice.
//
// This test compiles the same program twice against one cache directory that is
// empty at start, so compile 1 is a cold miss and compile 2 is a hit, and
// requires exactly one warning from each. It is run for a self-contained source
// and for an #include-bearing one, because only the latter takes the preprocess
// path where the double-count occurred.
//
// This is a compile-only test: it never launches a kernel, so it needs no device.
//
// Requirements (provided by the harness, not by this program):
//   - CHIP_MODULE_CACHE_DIR points at a cache directory that is EMPTY at start.
//   - CHIP_LOGLEVEL=info, so HIPRTC's hit/miss reporting is emitted.
// libCHIP reads both at static-init time, before main(); the ctest registration
// in CMakeLists.txt routes this test through run_test_with_fresh_cache.cmake,
// which provides both.

#include "TestCommon.hh"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

// Self-contained: no #include, so hiprtc keys on the raw source and never runs
// the preprocess pass.
static constexpr auto PlainSource = R"---(
extern "C" __global__ void add1(int *Out, const int *In) { *Out = *In + 1; }
)---";

// Includes a header, so hiprtc preprocesses the source to build the cache key.
// This is the path that used to process the options a second time.
static constexpr auto IncludingSource = R"---(
#include <hip/hip_runtime.h>
extern "C" __global__ void add2(int *Out, const int *In) { *Out = *In + 2; }
)---";

// An option hiprtc does not recognize. processOptions() is required to drop it
// and record one "warning: ignored option" entry in the program log.
static const char *IgnoredOption = "--nonexistent-flag";

static size_t countOccurrences(const std::string &Haystack,
                               const std::string &Needle) {
  size_t Count = 0;
  for (size_t Pos = Haystack.find(Needle); Pos != std::string::npos;
       Pos = Haystack.find(Needle, Pos + 1))
    ++Count;
  return Count;
}

struct CompileResult {
  size_t Warnings; // "warning: ignored option" entries in the program log
  bool Hit;        // HIPRTC served this compile from cache
};

// Compile Source with the unrecognized option and report both the number of
// ignored-option warnings in the program log and whether the compile was a
// cache hit. The hit/miss decision is only reported on hiprtc's stderr, so
// redirect fd 2 (spdlog's sink) for the duration of the compile, then restore it
// and echo what was captured so it still shows up in the test log.
static CompileResult compileAndInspect(const char *Source,
                                       const char *SourceName) {
  hiprtcProgram Prog;
  HIPRTC_CHECK(
      hiprtcCreateProgram(&Prog, Source, SourceName, 0, nullptr, nullptr));

  std::vector<const char *> Options = {IgnoredOption};

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

  // An unrecognized option must be ignored, not fatal.
  HIPRTC_CHECK(R);

  size_t LogSize = 0;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(Prog, &LogSize));
  std::string Log;
  if (LogSize) {
    Log.resize(LogSize);
    HIPRTC_CHECK(hiprtcGetProgramLog(Prog, Log.data()));
  }
  HIPRTC_CHECK(hiprtcDestroyProgram(&Prog));

  return {countOccurrences(Log, "warning: ignored option"),
          Captured.find("Cache hit") != std::string::npos};
}

// Compile Source twice against the (initially empty) cache and require exactly
// one ignored-option warning each time: once on the cold miss, and again on the
// hit that follows.
static void checkLogIsCacheIndependent(const char *Source,
                                       const char *SourceName,
                                       const char *Description) {
  std::cerr << "--- " << Description << " ---\n";

  CompileResult Cold = compileAndInspect(Source, SourceName);
  std::cerr << "compile #1 (cold): hit=" << Cold.Hit
            << " warnings=" << Cold.Warnings << "\n";
  // A fresh cache directory means this must be a miss. If it were a hit the
  // second check below would be testing the same path twice.
  TEST_ASSERT(!Cold.Hit);
  TEST_ASSERT(Cold.Warnings == 1);

  CompileResult Warm = compileAndInspect(Source, SourceName);
  std::cerr << "compile #2 (warm): hit=" << Warm.Hit
            << " warnings=" << Warm.Warnings << "\n";
  // Proves the cache is genuinely exercised, so the log check that follows is
  // actually covering the cache-hit path.
  TEST_ASSERT(Warm.Hit);
  TEST_ASSERT(Warm.Warnings == 1);
}

int main() {
  const char *CacheDir = std::getenv("CHIP_MODULE_CACHE_DIR");
  if (!CacheDir || !*CacheDir) {
    std::cerr << "CHIP_MODULE_CACHE_DIR is not set; this test needs an (empty) "
                 "cache directory. The CMake build provides one for ctest.\n";
    return 1;
  }

  checkLogIsCacheIndependent(PlainSource, "plain.hip",
                             "self-contained source (no preprocess pass)");
  checkLogIsCacheIndependent(IncludingSource, "including.hip",
                             "#include-bearing source (preprocess pass runs)");

  std::cerr << "Test passed: the program log is identical on cache hit and "
               "miss, and diagnostics are not duplicated.\n";
  return 0;
}

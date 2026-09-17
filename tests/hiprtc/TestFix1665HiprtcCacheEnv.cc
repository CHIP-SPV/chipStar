// Reproduces CHIP-SPV/chipStar#1665: hipcc environment overrides must not be
// served SPIR-V cached under other overrides. Compile-only, needs no device.

#include "TestCommon.hh"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static constexpr auto Source = R"---(
extern "C" __global__ void add1(int *Out, const int *In) { *Out = *In + 1; }
)---";

static size_t compileAndCountEntries() {
  hiprtcProgram Prog;
  HIPRTC_CHECK(
      hiprtcCreateProgram(&Prog, Source, "p.hip", 0, nullptr, nullptr));
  HIPRTC_CHECK(hiprtcCompileProgram(Prog, 0, nullptr));
  HIPRTC_CHECK(hiprtcDestroyProgram(&Prog));
  fs::path Dir = fs::path(std::getenv("CHIP_MODULE_CACHE_DIR")) / "hiprtc";
  return std::distance(fs::directory_iterator(Dir), fs::directory_iterator());
}

int main() {
  unsetenv("HIPCC_COMPILE_FLAGS_APPEND");
  unsetenv("HIP_COMPILER_BIN");
  TEST_ASSERT(compileAndCountEntries() == 1);

  setenv("HIPCC_COMPILE_FLAGS_APPEND", "-DTEST_FIX_1665", 1);
  TEST_ASSERT(compileAndCountEntries() == 2);

  auto Wrapper = fs::path(std::getenv("CHIP_MODULE_CACHE_DIR")) / "cxx.sh";
  std::ofstream(Wrapper) << "#!/bin/sh\nexec " CLANGXX " \"$@\"\n";
  fs::permissions(Wrapper, fs::perms::owner_all);
  setenv("HIP_COMPILER_BIN", Wrapper.c_str(), 1);
  TEST_ASSERT(compileAndCountEntries() == 3);

  // Stand in for editing the wrapper in place.
  fs::last_write_time(Wrapper,
                      fs::last_write_time(Wrapper) + std::chrono::seconds(1));
  TEST_ASSERT(compileAndCountEntries() == 4);

  // A bare name resolves through PATH, so it must not be cached at all.
  setenv("PATH", (Wrapper.parent_path().string() + ":" + getenv("PATH")).c_str(),
         1);
  setenv("HIP_COMPILER_BIN", Wrapper.filename().c_str(), 1);
  TEST_ASSERT(compileAndCountEntries() == 4);

  std::cerr << "Test passed\n";
  return 0;
}

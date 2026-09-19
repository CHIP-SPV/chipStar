// Reproduces CHIP-SPV/chipStar#1335: editing a force-included header must not
// serve SPIR-V cached for an include-free source. Compile-only, needs no device.

#include "TestCommon.hh"

#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static constexpr auto Source = R"---(
extern "C" __global__ void set(int *Out) { *Out = TEST_FIX_1335_VALUE; }
)---";

static size_t compileAndCountEntries() {
  hiprtcProgram Prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&Prog, Source, "p.hip", 0, nullptr, nullptr));
  HIPRTC_CHECK(hiprtcCompileProgram(Prog, 0, nullptr));
  HIPRTC_CHECK(hiprtcDestroyProgram(&Prog));
  fs::path Dir = fs::path(std::getenv("CHIP_MODULE_CACHE_DIR")) / "hiprtc";
  return std::distance(fs::directory_iterator(Dir), fs::directory_iterator());
}

int main() {
  auto Header = fs::path(std::getenv("CHIP_MODULE_CACHE_DIR")) / "forced.h";
  std::string Flags = "-include " + Header.string();
  setenv("HIPCC_COMPILE_FLAGS_APPEND", Flags.c_str(), 1);

  std::ofstream(Header) << "#define TEST_FIX_1335_VALUE 1\n";
  TEST_ASSERT(compileAndCountEntries() == 1);
  TEST_ASSERT(compileAndCountEntries() == 1);

  std::ofstream(Header) << "#define TEST_FIX_1335_VALUE 2\n";
  TEST_ASSERT(compileAndCountEntries() == 2);

  std::cerr << "Test passed\n";
  return 0;
}

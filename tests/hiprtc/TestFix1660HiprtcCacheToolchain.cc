// Reproduces CHIP-SPV/chipStar#1660: a rebuilt pass plugin must not be served
// SPIR-V cached by the previous build. Compile-only, needs no device.

#include "TestCommon.hh"

#include <chrono>
#include <cstdlib>
#include <filesystem>

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
  TEST_ASSERT(compileAndCountEntries() == 1);
  TEST_ASSERT(compileAndCountEntries() == 1);

  // Stand in for a rebuild: give the plugin a new mtime, then restore it.
  auto MTime = fs::last_write_time(PASS_PLUGIN);
  fs::last_write_time(PASS_PLUGIN, MTime + std::chrono::seconds(1));
  size_t Entries = compileAndCountEntries();
  fs::last_write_time(PASS_PLUGIN, MTime);

  TEST_ASSERT(Entries == 2);
  std::cerr << "Test passed\n";
  return 0;
}

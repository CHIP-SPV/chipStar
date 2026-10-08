// Reproduces CHIP-SPV/chipStar#1660: the hipRTC cache key must follow the
// compile command libCHIP builds, not the raw option strings. Compile-only.

#include "TestCommon.hh"

#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

static constexpr auto Source = R"---(
extern "C" __global__ void add1(int *Out, const int *In) { *Out = *In + 1; }
)---";

static size_t compileAndCountEntries(const char *Option) {
  hiprtcProgram Prog;
  HIPRTC_CHECK(
      hiprtcCreateProgram(&Prog, Source, "p.hip", 0, nullptr, nullptr));
  HIPRTC_CHECK(hiprtcCompileProgram(Prog, Option ? 1 : 0, &Option));
  HIPRTC_CHECK(hiprtcDestroyProgram(&Prog));
  fs::path Dir = fs::path(std::getenv("CHIP_MODULE_CACHE_DIR")) / "hiprtc";
  return std::distance(fs::directory_iterator(Dir), fs::directory_iterator());
}

int main() {
  TEST_ASSERT(compileAndCountEntries(nullptr) == 1);
  // An ignored option does not reach the compiler, so it must share the entry.
  TEST_ASSERT(compileAndCountEntries("--nonexistent-flag") == 1);
  // An accepted option does reach it, so it must not.
  TEST_ASSERT(compileAndCountEntries("-O1") == 2);
  std::cerr << "Test passed\n";
  return 0;
}

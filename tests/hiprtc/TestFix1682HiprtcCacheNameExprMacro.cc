// Reproduces CHIP-SPV/chipStar#1682: a header macro used only in a name
// expression must reach the cache key. Compile-only, needs no device.

#include "TestCommon.hh"

#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static constexpr auto Source = R"---(
#include "nameexpr.h"
template <int N> __global__ void k() {}
__global__ void f() {}
#define f(...) 0
)---";

static std::string compileAndLower(const std::string &IncludeOpt) {
  const char *Expr = "(k<TEST_FIX_1682_VALUE>)";
  const char *Opts[] = {IncludeOpt.c_str()};
  hiprtcProgram Prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&Prog, Source, "p.hip", 0, nullptr, nullptr));
  HIPRTC_CHECK(hiprtcAddNameExpression(Prog, Expr));
  HIPRTC_CHECK(hiprtcAddNameExpression(Prog, "&f"));
  HIPRTC_CHECK(hiprtcCompileProgram(Prog, 1, Opts));
  const char *Lowered = nullptr;
  HIPRTC_CHECK(hiprtcGetLoweredName(Prog, Expr, &Lowered));
  std::string Result = Lowered;
  HIPRTC_CHECK(hiprtcDestroyProgram(&Prog));
  return Result;
}

int main() {
  fs::path Dir = std::getenv("CHIP_MODULE_CACHE_DIR");
  std::string IncludeOpt = "-I" + Dir.string();

  std::ofstream(Dir / "nameexpr.h") << "#define TEST_FIX_1682_VALUE 1\n";
  TEST_ASSERT(compileAndLower(IncludeOpt) == "_Z1kILi1EEvv");
  TEST_ASSERT(compileAndLower(IncludeOpt) == "_Z1kILi1EEvv");

  std::ofstream(Dir / "nameexpr.h") << "#define TEST_FIX_1682_VALUE 2\n";
  TEST_ASSERT(compileAndLower(IncludeOpt) == "_Z1kILi2EEvv");

  std::cerr << "Test passed\n";
  return 0;
}

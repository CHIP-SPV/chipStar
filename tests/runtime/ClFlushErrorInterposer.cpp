// LD_PRELOAD interposer for TestFix1604ClFlushErrorMap.
//
// Returns CL_OUT_OF_RESOURCES from one clFlush call, selected by ordinal in
// CLFLUSH_FAIL_AT, and chains to the real implementation for every other call,
// so preloading this library is inert unless that variable is set.
//
// CL_OUT_OF_RESOURCES is one of the three errors the OpenCL 3.0 runtime layer
// lists for clFlush, and it is the one chipStar's conversion table omitted:
// CHIPERR_CHECK_LOG_AND_THROW_TABLE aborts the process on a code the table
// does not map, so before the fix a driver returning it killed the
// application instead of raising a HIP error.
//
// The chaining prototype takes a plain pointer rather than cl_command_queue so
// the interposer builds whether or not the OpenCL headers are present, which
// matches ExitDuringModuleBuildInterposer.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

namespace {
constexpr int ClOutOfResources = -5; // CL_OUT_OF_RESOURCES

int failAtOrdinal() {
  static int Ordinal = [] {
    const char *Env = std::getenv("CLFLUSH_FAIL_AT");
    return Env ? std::atoi(Env) : -1;
  }();
  return Ordinal;
}
} // namespace

extern "C" int clFlush(void *CommandQueue) {
  using ClFlushFn = int (*)(void *);
  static auto RealClFlush =
      reinterpret_cast<ClFlushFn>(dlsym(RTLD_NEXT, "clFlush"));

  static int Calls = 0;
  int Ordinal = __atomic_add_fetch(&Calls, 1, __ATOMIC_SEQ_CST);

  if (Ordinal == failAtOrdinal()) {
    std::fprintf(stderr,
                 "clflush-interposer: failing call %d with "
                 "CL_OUT_OF_RESOURCES\n",
                 Ordinal);
    std::fflush(stderr);
    return ClOutOfResources;
  }

  if (!RealClFlush) {
    std::fprintf(stderr, "clflush-interposer: no real clFlush to chain to\n");
    std::fflush(stderr);
    return ClOutOfResources;
  }
  return RealClFlush(CommandQueue);
}

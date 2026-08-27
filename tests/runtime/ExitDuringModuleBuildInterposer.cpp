// LD_PRELOAD interposer for TestFix1393ExitDuringModuleBuild.
//
// Terminates the process from inside a backend module-build call, but only
// while chipStar's Device::getOrCreateModule is on the stack, which is the
// window in which Device::DeviceVarMtx is held. That models
// SPIRV-LLVM-Translator's SPIRVErrorLog::checkError(), which defaults to
// SPIRVDbgErrorHandlingKinds::Exit and so calls std::exit() out of
// clBuildProgram on a module it rejects.
//
// Every entry point chains to the real implementation when the marker frame is
// absent, so preloading this library is inert outside the one call it targets.
// The chaining prototypes take plain integers and pointers instead of the
// OpenCL and Level Zero types: each argument is in the same ABI class as the
// real one, and using them keeps the interposer buildable whether or not
// either backend's headers are present.

#include <dlfcn.h>
#include <execinfo.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Kept in sync with EXPECTED_EXIT_CODE in
// tests/run_exit_during_module_build_test.cmake.
constexpr int InjectedExitCode = 93;

bool lockHolderOnStack() {
  void *Frames[64];
  int NumFrames = backtrace(Frames, 64);
  char **Symbols = backtrace_symbols(Frames, NumFrames);
  if (!Symbols)
    return false;

  bool Found = false;
  for (int I = 0; I < NumFrames; ++I)
    if (std::strstr(Symbols[I], "getOrCreateModule"))
      Found = true;

  std::free(Symbols);
  return Found;
}

void exitIfHoldingLock(const char *Api) {
  if (!lockHolderOnStack()) {
    std::fprintf(stderr, "interposer: %s reached without getOrCreateModule\n",
                 Api);
    std::fflush(stderr);
    return;
  }
  std::fprintf(stderr, "interposer: exiting from inside %s\n", Api);
  std::fflush(stderr);
  std::exit(InjectedExitCode);
}

template <typename FnTy> FnTy realSymbol(const char *Name) {
  void *Symbol = dlsym(RTLD_NEXT, Name);
  if (!Symbol) {
    // Reached only if something calls an interposed entry point that the real
    // runtime does not provide. Say so instead of jumping through a null.
    std::fprintf(stderr, "interposer: no real %s to chain to\n", Name);
    std::fflush(stderr);
    std::abort();
  }
  return reinterpret_cast<FnTy>(Symbol);
}

} // namespace

extern "C" {

int clBuildProgram(void *Program, unsigned NumDevices, const void *DeviceList,
                   const char *Options, void *Notify, void *UserData) {
  exitIfHoldingLock("clBuildProgram");
  using FnTy = int (*)(void *, unsigned, const void *, const char *, void *,
                       void *);
  return realSymbol<FnTy>("clBuildProgram")(Program, NumDevices, DeviceList,
                                            Options, Notify, UserData);
}

int clCompileProgram(void *Program, unsigned NumDevices, const void *DeviceList,
                     const char *Options, unsigned NumInputHeaders,
                     const void *InputHeaders, const char **HeaderIncludeNames,
                     void *Notify, void *UserData) {
  exitIfHoldingLock("clCompileProgram");
  using FnTy = int (*)(void *, unsigned, const void *, const char *, unsigned,
                       const void *, const char **, void *, void *);
  return realSymbol<FnTy>("clCompileProgram")(
      Program, NumDevices, DeviceList, Options, NumInputHeaders, InputHeaders,
      HeaderIncludeNames, Notify, UserData);
}

int zeModuleCreate(void *Context, void *Device, const void *Desc, void *Module,
                   void *BuildLog) {
  exitIfHoldingLock("zeModuleCreate");
  using FnTy = int (*)(void *, void *, const void *, void *, void *);
  return realSymbol<FnTy>("zeModuleCreate")(Context, Device, Desc, Module,
                                            BuildLog);
}

} // extern "C"

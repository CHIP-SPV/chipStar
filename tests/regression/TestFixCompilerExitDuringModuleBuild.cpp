// Reproduces: chipStar deadlocks at exit when the backend compiler terminates
// the process from inside a module build.
//
// Some OpenCL implementations call exit() instead of returning an error when
// they reject a SPIR-V module.  PoCL does this through the SPIR-V reader when
// the module needs a capability the target does not have, for example
// SPV_INTEL_function_pointers:
//
//   InvalidModule: Invalid SPIR-V module: input SPIR-V module uses extension
//   'SPV_INTEL_function_pointers' which were disabled by --spirv-ext option
//
// exit() runs the atexit handlers, one of which is chipStar's
// __hip_module_dtor -> __hipUnregisterFatBinary -> CHIPUninitialize ->
// Device::deallocateDeviceVariables.  That takes Device::DeviceVarMtx, which
// the very same thread is already holding in Device::getOrCreateModule() for
// the duration of the backend compile.  std::mutex is not recursive, so the
// process wedges forever instead of failing.
//
// This test interposes the backend module build entry points so the failure is
// reproduced without needing a driver that rejects the module.  A definition in
// the executable takes precedence over the shared library one, so libCHIP's
// call lands here.  The test is bounded: the parent kills the child and reports
// FAIL rather than letting the harness time out.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE // for RTLD_NEXT
#endif

#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

// Exit code the interposed build call terminates the child with.
static const int ExitCodeFromCompiler = 42;
// Set in the child so the interposers only fire there.
static const char *ArmEnvVar = "CHIPSTAR_TESTFIX_EXIT_IN_MODULE_BUILD";
// Upper bound on how long the child may take, in tenths of a second.
static const int ChildTimeoutTenths = 450;

static bool armed() { return getenv(ArmEnvVar) != nullptr; }

static void exitLikeARejectingCompiler(const char *Fn) {
  fprintf(stderr, "interposed %s: terminating the process the way a compiler "
                  "rejecting the module would\n",
          Fn);
  exit(ExitCodeFromCompiler);
}

// clBuildProgram and zeModuleCreate take only pointer-sized and integer
// arguments, so these opaque prototypes are ABI compatible with the real ones.
extern "C" int clBuildProgram(void *Program, unsigned NumDevices,
                              const void *DeviceList, const char *Options,
                              void *Notify, void *UserData) {
  if (armed())
    exitLikeARejectingCompiler("clBuildProgram");
  using FnTy = int (*)(void *, unsigned, const void *, const char *, void *,
                       void *);
  FnTy Real = (FnTy)dlsym(RTLD_NEXT, "clBuildProgram");
  if (!Real)
    return -6; // CL_OUT_OF_HOST_MEMORY
  return Real(Program, NumDevices, DeviceList, Options, Notify, UserData);
}

extern "C" int zeModuleCreate(void *Context, void *Device, const void *Desc,
                              void *Module, void *BuildLog) {
  if (armed())
    exitLikeARejectingCompiler("zeModuleCreate");
  using FnTy = int (*)(void *, void *, const void *, void *, void *);
  FnTy Real = (FnTy)dlsym(RTLD_NEXT, "zeModuleCreate");
  if (!Real)
    return 0x78000001; // ZE_RESULT_ERROR_UNINITIALIZED
  return Real(Context, Device, Desc, Module, BuildLog);
}

__global__ void touch(int *Out) { *Out = 1; }

// Runs in the child: the first launch compiles the module, which lands in the
// interposed build call above and terminates the process from underneath the
// held DeviceVarMtx.
static int runChild() {
  int *Ptr = nullptr;
  if (hipMalloc(&Ptr, sizeof(int)) != hipSuccess)
    return 0; // No usable device; the parent reports this as a skip.
  touch<<<1, 1>>>(Ptr);
  (void)hipDeviceSynchronize();
  (void)hipFree(Ptr);
  return 0; // The interposers never fired.
}

int main(int Argc, char **Argv) {
  (void)Argc;
  if (armed())
    return runChild();

  pid_t Pid = fork();
  if (Pid < 0) {
    printf("HIP_SKIP_THIS_TEST: fork failed\n");
    return 0;
  }
  if (Pid == 0) {
    setenv(ArmEnvVar, "1", 1);
    execv(Argv[0], Argv);
    _exit(127);
  }

  int Status = 0;
  for (int Tenths = 0; Tenths < ChildTimeoutTenths; ++Tenths) {
    if (waitpid(Pid, &Status, WNOHANG) == Pid) {
      if (WIFEXITED(Status) && WEXITSTATUS(Status) == ExitCodeFromCompiler) {
        printf("PASS: runtime shut down cleanly after the backend compiler "
               "terminated the process during a module build\n");
        return 0;
      }
      printf("HIP_SKIP_THIS_TEST: module build was not interposed (status "
             "0x%x)\n",
             Status);
      return 0;
    }
    usleep(100000);
  }

  kill(Pid, SIGKILL);
  (void)waitpid(Pid, &Status, 0);
  printf("FAIL: runtime deadlocked in its exit handler after the backend "
         "compiler terminated the process during a module build\n");
  return 1;
}

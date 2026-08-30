// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1539: a device
// compiler that accepts a module with an unresolved extern __device__ function
// (Mali) builds the program with an empty log but leaves out the kernel that
// calls the function. The runtime then treated the module as compiled and
// failed later, in the kernel binding of Device::getOrCreateModule, with
// hipErrorLaunchFailure ("Failed to find kernel via kernel name"), a code
// meant for a faulting kernel, and retried the whole lookup on every launch.
//
// Where the device compiler rejects the module instead (Intel OpenCL fails
// clBuildProgram, Level Zero fails zeModuleCreate) the launch already fails
// with a module load error, so the exact code is not asserted. What is
// asserted on every backend: both launches fail, the second the same way as
// the first, neither with hipErrorLaunchFailure, the runtime log names the
// unresolved symbol, and the kernel lookup failure never appears.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

extern __device__ void undefinedFn(int);

__global__ void kernel(int X) { undefinedFn(X); }

static bool logHas(const std::string &Path, const char *Needle) {
  std::ifstream In(Path);
  std::string Line;
  while (std::getline(In, Line))
    if (Line.find(Needle) != std::string::npos)
      return true;
  return false;
}

int main(int argc, char **argv) {
  // The unresolved symbol is reported at warn level, so re-exec once with
  // the log level pinned in case the environment lowered it.
  if (!getenv("TESTFIX1539_CHILD")) {
    setenv("TESTFIX1539_CHILD", "1", 1);
    setenv("CHIP_LOGLEVEL", "warn", 1);
    execv(argv[0], argv);
    perror("execv");
    return 1;
  }

  // Divert the runtime log into a file for the duration of the launches.
  char LogPath[] = "/tmp/TestFix1539.XXXXXX";
  int LogFd = mkstemp(LogPath);
  if (LogFd < 0) {
    perror("mkstemp");
    return 1;
  }
  fflush(stderr);
  int SavedStderr = dup(STDERR_FILENO);
  dup2(LogFd, STDERR_FILENO);

  (void)hipFree(nullptr);

  using Clock = std::chrono::steady_clock;
  kernel<<<1, 1>>>(1);
  hipError_t Err1 = hipGetLastError();
  auto T1 = Clock::now();
  kernel<<<1, 1>>>(2);
  hipError_t Err2 = hipGetLastError();
  auto T2 = Clock::now();

  fflush(stderr);
  dup2(SavedStderr, STDERR_FILENO);
  close(LogFd);

  double SecondMs = std::chrono::duration<double, std::milli>(T2 - T1).count();
  bool NamesSymbol = logHas(LogPath, "_Z11undefinedFni");
  bool LookupFailed = logHas(LogPath, "Failed to find kernel via kernel name");
  unlink(LogPath);

  printf("first launch: %s\n", hipGetErrorName(Err1));
  printf("second launch: %s in %.1f ms\n", hipGetErrorName(Err2), SecondMs);

  bool Ok = true;
  if (Err1 == hipSuccess) {
    printf("FAILED: first launch succeeded, expected a module load error\n");
    Ok = false;
  }
  if (Err1 == hipErrorLaunchFailure) {
    printf("FAILED: first launch reported hipErrorLaunchFailure for a module "
           "that never loaded\n");
    Ok = false;
  }
  if (Err2 != Err1) {
    printf("FAILED: second launch returned %s, expected the first launch's "
           "%s\n",
           hipGetErrorName(Err2), hipGetErrorName(Err1));
    Ok = false;
  }
  if (SecondMs > 5000.0) {
    printf("FAILED: second launch took %.1f ms\n", SecondMs);
    Ok = false;
  }
  if (!NamesSymbol) {
    printf("FAILED: runtime log does not name the unresolved symbol "
           "_Z11undefinedFni\n");
    Ok = false;
  }
  if (LookupFailed) {
    printf("FAILED: runtime log reports a kernel lookup failure, the module "
           "should have failed to load instead\n");
    Ok = false;
  }
  if (Ok)
    printf("PASSED\n");
  return Ok ? 0 : 1;
}

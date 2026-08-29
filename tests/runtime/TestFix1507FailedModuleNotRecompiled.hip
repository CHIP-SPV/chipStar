// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1507: a source
// module whose device program fails to build (here: an unresolved extern
// __device__ function) is compiled again from scratch by every later API call
// that touches one of its symbols, instead of being remembered as failed. The
// failure was also reported with a placeholder error code.
//
// The test drives two symbol/launch APIs at the broken module and checks that
// both return hipErrorSharedObjectInitFailed, that the second returns promptly,
// and that the runtime log holds exactly one compile attempt. Every attempt
// starts with a module cache lookup, which logs one "module-cache: MISS" line
// at info level in both backends, so that line is the attempt counter.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

extern __device__ void undefinedFn(int);
__device__ int DevVar = 0;

__global__ void kernel(int X) { undefinedFn(X); }

static int countLines(const std::string &Path, const char *Needle) {
  std::ifstream In(Path);
  std::string Line;
  int Count = 0;
  while (std::getline(In, Line))
    if (Line.find(Needle) != std::string::npos)
      Count++;
  return Count;
}

int main(int argc, char **argv) {
  // The attempt counter is an info-level line that only appears when the
  // module cache is enabled, so re-exec once with the log level set and the
  // cache pointed at a private, empty directory.
  if (!getenv("TESTFIX1507_CHILD")) {
    char CacheDir[] = "/tmp/TestFix1507-cache.XXXXXX";
    if (!mkdtemp(CacheDir)) {
      perror("mkdtemp");
      return 1;
    }
    setenv("TESTFIX1507_CHILD", "1", 1);
    setenv("CHIP_LOGLEVEL", "info", 1);
    setenv("CHIP_MODULE_CACHE_DIR", CacheDir, 1);
    execv(argv[0], argv);
    perror("execv");
    return 1;
  }

  // Divert the runtime log into a file for the duration of the API calls.
  char LogPath[] = "/tmp/TestFix1507.XXXXXX";
  int LogFd = mkstemp(LogPath);
  if (LogFd < 0) {
    perror("mkstemp");
    return 1;
  }
  fflush(stderr);
  int SavedStderr = dup(STDERR_FILENO);
  dup2(LogFd, STDERR_FILENO);

  // Initialize the runtime up front so the timings below cover the module
  // compilation only.
  (void)hipFree(nullptr);

  using Clock = std::chrono::steady_clock;
  auto T0 = Clock::now();
  kernel<<<1, 1>>>(1);
  hipError_t Err1 = hipGetLastError();
  auto T1 = Clock::now();
  int Value = 0;
  hipError_t Err2 =
      hipMemcpyToSymbol(HIP_SYMBOL(DevVar), &Value, sizeof(Value));
  auto T2 = Clock::now();

  fflush(stderr);
  dup2(SavedStderr, STDERR_FILENO);
  close(LogFd);

  double FirstMs = std::chrono::duration<double, std::milli>(T1 - T0).count();
  double SecondMs = std::chrono::duration<double, std::milli>(T2 - T1).count();
  int Attempts = countLines(LogPath, "module-cache: MISS");
  unlink(LogPath);
  rmdir(getenv("CHIP_MODULE_CACHE_DIR"));

  printf("first call: %s in %.1f ms\n", hipGetErrorName(Err1), FirstMs);
  printf("second call: %s in %.1f ms\n", hipGetErrorName(Err2), SecondMs);
  printf("compile attempts logged: %d\n", Attempts);

  bool Ok = true;
  if (Err1 != hipErrorSharedObjectInitFailed) {
    printf("FAILED: first call returned %s, expected "
           "hipErrorSharedObjectInitFailed\n",
           hipGetErrorName(Err1));
    Ok = false;
  }
  if (Err2 != hipErrorSharedObjectInitFailed) {
    printf("FAILED: second call returned %s, expected "
           "hipErrorSharedObjectInitFailed\n",
           hipGetErrorName(Err2));
    Ok = false;
  }
  if (SecondMs > 5000.0) {
    printf("FAILED: second call took %.1f ms\n", SecondMs);
    Ok = false;
  }
  if (Attempts != 1) {
    printf("FAILED: %d compile attempts logged, expected 1\n", Attempts);
    Ok = false;
  }
  if (Ok)
    printf("PASSED\n");
  return Ok ? 0 : 1;
}

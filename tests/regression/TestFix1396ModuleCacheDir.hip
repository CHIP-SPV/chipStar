// Regression test for: CHIP_MODULE_CACHE_DIR must be used verbatim.
//
// Bug (#1396): src/CHIPDriver.hh read the variable through readEnvVar with
// Lower = true, which lowercases the whole string. The value is a filesystem
// path, so a path containing any uppercase letter ended up pointing somewhere
// else. If the lowercased path was not creatable the process aborted:
//
//   terminate called after throwing an instance of
//   'std::filesystem::__cxx11::filesystem_error'
//     what():  filesystem error: cannot create directories: Permission denied
//
// and if it was creatable the cache silently went to the wrong place, so
// nothing was ever reused and every process recompiled the whole device
// module from scratch.
//
// Fix: read it with Lower = false, like the other path-valued variables.
//
// The test re-executes itself rather than calling setenv and carrying on,
// because chipStar reads its configuration when the driver initializes, which
// happens before main. The parent sets the variable, the child compiles a
// kernel, and the parent inspects the directories once the child has exited
// and flushed its cache.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

static const char *ChildMarker = "TESTFIX1396_CHILD";

__global__ void touchIt(int *Out) { Out[0] = 7; }

static int runChildWorkload() {
  int *D = nullptr;
  if (hipMalloc(&D, sizeof(int)) != hipSuccess)
    return 2; // no GPU
  hipMemset(D, 0, sizeof(int));
  hipLaunchKernelGGL(touchIt, dim3(1), dim3(1), 0, 0, D);
  if (hipDeviceSynchronize() != hipSuccess)
    return 3;
  hipFree(D);
  return 0;
}

int main(int argc, char **argv) {
  if (getenv(ChildMarker))
    return runChildWorkload();

  const fs::path Wanted =
      fs::temp_directory_path() / "TestFix1396ChipStarCacheDir";
  const fs::path Lowered =
      fs::temp_directory_path() / "testfix1396chipstarcachedir";

  std::error_code Ec;
  fs::remove_all(Wanted, Ec);
  fs::remove_all(Lowered, Ec);

  setenv("CHIP_MODULE_CACHE_DIR", Wanted.string().c_str(), 1);
  setenv(ChildMarker, "1", 1);

  pid_t Pid = fork();
  if (Pid < 0) {
    printf("FAIL: fork failed\n");
    return 1;
  }
  if (Pid == 0) {
    execv(argv[0], argv);
    _exit(127);
  }

  int Status = 0;
  if (waitpid(Pid, &Status, 0) < 0 || !WIFEXITED(Status)) {
    printf("FAIL: child did not exit normally\n");
    return 1;
  }
  const int Rc = WEXITSTATUS(Status);
  if (Rc == 2) {
    printf("SKIP: no GPU\n");
    return 0;
  }
  if (Rc != 0) {
    printf("FAIL: child returned %d\n", Rc);
    return 1;
  }

  const bool WantedExists = fs::exists(Wanted);
  const bool LoweredExists = fs::exists(Lowered);
  printf("requested '%s' exists=%d, lowercased '%s' exists=%d\n",
         Wanted.string().c_str(), (int)WantedExists, Lowered.string().c_str(),
         (int)LoweredExists);

  fs::remove_all(Wanted, Ec);
  fs::remove_all(Lowered, Ec);

  if (!WantedExists || LoweredExists) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

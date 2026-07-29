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

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

static const char *ChildMarker = "TESTFIX1396_CHILD";

static const char *WantedName = "TestFix1396ChipStarCacheDir";
static const char *LoweredName = "testfix1396chipstarcachedir";

static std::string toLower(std::string S) {
  for (char &C : S)
    C = (char)std::tolower((unsigned char)C);
  return S;
}

// Return the names the cache directory was actually created under, as the
// parent directory spells them. Existence checks cannot tell the wanted path
// from its lowercased twin on a case-insensitive filesystem (APFS on macOS
// answers yes to both), but those filesystems are case-preserving, so the
// directory entry still carries the case it was created with.
static std::vector<std::string> onDiskNames(const fs::path &Parent) {
  std::vector<std::string> Names;
  std::error_code Ec;
  for (fs::directory_iterator It(Parent, Ec), End; It != End; It.increment(Ec)) {
    if (Ec)
      break;
    std::string Name = It->path().filename().string();
    if (toLower(Name) == LoweredName)
      Names.push_back(Name);
  }
  return Names;
}

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

  const fs::path Wanted = fs::temp_directory_path() / WantedName;
  const fs::path Lowered = fs::temp_directory_path() / LoweredName;

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

  const std::vector<std::string> Found = onDiskNames(fs::temp_directory_path());
  printf("requested '%s', created as:", WantedName);
  for (const std::string &Name : Found)
    printf(" '%s'", Name.c_str());
  if (Found.empty())
    printf(" <nothing>");
  printf("\n");

  fs::remove_all(Wanted, Ec);
  fs::remove_all(Lowered, Ec);

  if (Found.size() != 1 || Found[0] != WantedName) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

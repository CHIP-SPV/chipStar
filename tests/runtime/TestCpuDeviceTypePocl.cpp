// CHIP_DEVICE_TYPE=cpu must select a PoCL CPU device when no other OpenCL CPU
// platform exists. Only meaningful on PoCL lanes (CHIP_DEVICE_TYPE=pocl).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc == 1) {
    const char *Type = getenv("CHIP_DEVICE_TYPE");
    if (!Type || strcmp(Type, "pocl") != 0) {
      printf("HIP_SKIP_THIS_TEST: not a PoCL lane\n");
      return CHIP_SKIP_TEST;
    }
    // The env var is read at library load, so re-exec with it changed.
    setenv("CHIP_DEVICE_TYPE", "cpu", 1);
    char *Args[] = {argv[0], const_cast<char *>("reexec"), nullptr};
    execv(argv[0], Args);
    perror("execv");
    return 1;
  }
  int Count = 0;
  hipError_t Err = hipGetDeviceCount(&Count);
  if (Err != hipSuccess || Count < 1) {
    printf("FAIL: hipGetDeviceCount returned %d, count=%d\n", Err, Count);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

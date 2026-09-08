// Deliberately failing child for TestFix1592ExtractorExitStatus.bash: exits 1,
// or dies of SIGABRT given "abort". A fatbinary so the wrapper has real SPIR-V
// to inspect; the kernel never launches. Listed in RUNTIME_TESTS_MANUAL, so the
// .bash is the test, not this.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

__global__ void k(int *p) { *p = 1; }

// Argument-free: exit 1. With "abort": die of SIGABRT, so the .bash can check
// the 128+signal mapping too. Both are outcomes the wrapper claims to report.
int main(int Argc, char **Argv) {
  if (Argc > 1 && strcmp(Argv[1], "abort") == 0) {
    printf("FAIL: deliberate, this program must die of SIGABRT\n");
    fflush(stdout);
    abort();
  }
  printf("FAIL: deliberate, this program must exit 1\n");
  return 1;
}

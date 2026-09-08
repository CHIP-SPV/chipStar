// Reproducer for chipStar issue #1592: a deliberately failing HIP program.
// TestFix1592ExtractorExitStatus.bash runs it directly and through
// `spirv-extractor --check-for-doubles`, which is how every test is wrapped
// when CHIP_SKIP_TESTS_WITH_DOUBLES=ON, and requires the wrapper to report the
// same non-zero exit status. Before the fix the wrapper returned system()'s
// raw wait status (256 for exit 1), which main() truncates to 0, so ctest saw
// a passing test.
//
// It is compiled as a fatbinary so the wrapper has real SPIR-V to inspect; the
// kernel is never launched. Not registered as a ctest itself (see
// RUNTIME_TESTS_MANUAL); the .bash is the test.
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

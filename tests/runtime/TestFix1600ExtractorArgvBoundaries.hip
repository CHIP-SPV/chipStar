// Reproducer for chipStar issue #1600: prints and checks its own argv.
// TestFix1600ExtractorArgvBoundaries.bash runs it through
// `spirv-extractor --check-for-doubles`, which is how every test is wrapped
// when CHIP_SKIP_TESTS_WITH_DOUBLES=ON, passing one argument that contains
// spaces. The wrapper built the child's command line by joining arguments
// with spaces and handing the string to system(), so the shell re-split that
// argument and the child saw three.
//
// Catch2 is the real victim: catch_discover_tests passes each test case name
// as one argument and many contain spaces, e.g.
// "Unit_Device_sincos_Accuracy_Positive - float".
//
// Compiled as a fatbinary so the wrapper has real SPIR-V to inspect; the
// kernel is never launched. Not registered as a ctest itself (see
// RUNTIME_TESTS_MANUAL); the .bash is the test.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

__global__ void k(int *p) { *p = 1; }

// The argument the .bash passes. Spaces are the point.
static const char *Expected = "Unit_Device_sincos_Accuracy_Positive - float";

int main(int argc, char **argv) {
  printf("argc=%d\n", argc);
  for (int I = 1; I < argc; ++I)
    printf("  argv[%d]=<%s>\n", I, argv[I]);

  if (argc != 2) {
    printf("FAILED: expected exactly 1 argument, got %d\n", argc - 1);
    return 1;
  }
  if (strcmp(argv[1], Expected) != 0) {
    printf("FAILED: argument altered\n  expected <%s>\n  got      <%s>\n",
           Expected, argv[1]);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}

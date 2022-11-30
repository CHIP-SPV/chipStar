#include <hip/hip_runtime.h>

// Vector implementations use Clang recognized attribute - should
// still compile in other compilers.
int4 test_make_vector(int x, int y, int z, int w) {
  return make_int4(x, y, z, w);
}

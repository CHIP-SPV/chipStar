// Reproduces #2315: hipMalloc3D rejected zero extents instead of returning a null pointer.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

int main() {
  hipExtent Extents[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}};
  int Failed = 0;
  for (hipExtent E : Extents) {
    hipPitchedPtr P;
    P.ptr = (void *)&P;
    P.pitch = 42;
    hipError_t Err = hipMalloc3D(&P, E);
    printf("extent {%zu,%zu,%zu}: %s ptr=%p pitch=%zu\n", E.width, E.height,
           E.depth, hipGetErrorName(Err), P.ptr, P.pitch);
    if (Err != hipSuccess || P.ptr != nullptr || P.pitch != 0) {
      printf("FAIL\n");
      Failed = 1;
    }
  }
  // Nonzero extents whose pitch or size wraps must fail, not look empty.
  hipExtent Wraps[] = {{SIZE_MAX, 1, 1}, {2, size_t{1} << 63, 1}};
  for (hipExtent E : Wraps) {
    hipPitchedPtr P;
    P.ptr = (void *)&P;
    P.pitch = 42;
    hipError_t Err = hipMalloc3D(&P, E);
    printf("extent {%zu,%zu,%zu}: %s ptr=%p pitch=%zu\n", E.width, E.height,
           E.depth, hipGetErrorName(Err), P.ptr, P.pitch);
    if (Err == hipSuccess) {
      printf("FAIL\n");
      Failed = 1;
    }
  }
  if (!Failed)
    printf("PASS\n");
  return Failed;
}

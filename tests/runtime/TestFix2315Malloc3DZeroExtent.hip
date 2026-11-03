// Reproduces #2315: hipMalloc3D rejected zero extents instead of returning a null pointer.
#include <hip/hip_runtime.h>
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
  if (!Failed)
    printf("PASS\n");
  return Failed;
}

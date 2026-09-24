// printf lowered to each shape of call HipPrintf builds.
#include <hip/hip_runtime.h>
__global__ void k(const char *s) {
  printf("a %s b %d\n", s, 1);
  printf("%d\n");
}
int main() { return 0; }

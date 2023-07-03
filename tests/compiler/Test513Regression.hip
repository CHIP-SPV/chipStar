#include <hip/hip_runtime.h>
__global__ void k(uchar2 x, uchar2 y) {
  auto t = x ^ y;
  printf("t=(%d, %d)\n", t.x, t.y);
}

int main() {
  k<<<1, 1>>>(make_uchar2(1, 2), make_uchar2(3, 3)); // Should print t=(2, 1)
  hipDeviceSynchronize();
  return 0;
}
// Device code: double2 * long double must compile (CHIP-SPV/chipStar#1890).
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void k(double2 *o) { o[0] = o[0] * -0.5l; }

int main() {
  double2 h = make_double2(2.0, 4.0), *d;
  (void)hipMalloc(&d, sizeof(h));
  (void)hipMemcpy(d, &h, sizeof(h), hipMemcpyHostToDevice);
  k<<<1, 1>>>(d);
  (void)hipMemcpy(&h, d, sizeof(h), hipMemcpyDeviceToHost);
  (void)hipFree(d);
  bool ok = h.x == -1.0 && h.y == -2.0;
  printf("%s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}

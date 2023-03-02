#include <hip/hip_runtime.h>

 // Declaration is repeated in b.hip. Added for uncovering possible issues.
static __device__ int s = 999;
extern __device__ int f(int);
__global__ void k(int *Out, int In) { *Out = f(In); }
int main() {
  int *OutD, OutH = 0;
  (void)hipMalloc(&OutD, sizeof(int));
  k<<<1, 1>>>(OutD, 20);
  (void)hipMemcpy(&OutH, OutD, sizeof(int), hipMemcpyDeviceToHost);
  return !(OutH == 123);
}

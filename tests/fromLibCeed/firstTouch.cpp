#include <hip/hip_runtime.h>

struct Data {
  int *A_d;
} typedef Data;

__global__ void setOne(Data datain) {
  datain.A_d[0] = 1;
}

int main() {
  Data data;
  int A_h[1] = {0};
  hipMalloc(&data.A_d, sizeof(int));
  hipLaunchKernelGGL(setOne, dim3(1), dim3(1), 0, 0, data);
  hipDeviceSynchronize();
  hipMemcpy(A_h, data.A_d, sizeof(int), hipMemcpyDeviceToHost);
  if (A_h[0] == 1) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }
  hipFree(data.A_d);
  return 0;
}
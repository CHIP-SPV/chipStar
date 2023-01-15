#include <hip/hip_runtime.h>
#include <iostream>
static constexpr auto LEN{1};
static constexpr auto SIZE{LEN * sizeof(int)};

__global__ void addOne(int *A) { A[0] = 1; }

void testHipHostMalloc() {
  hipError_t Status;
  int *A_h;
  Status = hipHostMalloc(reinterpret_cast<void **>(&A_h), SIZE,
                         hipHostMallocDefault);
  assert(Status == hipSuccess);
  A_h[0] = 0;

  hipLaunchKernelGGL(addOne, 1, 1, 0, 0, static_cast<int *>(A_h));
  hipDeviceSynchronize();
  if (A_h[0] != 1) {
    std::cout << "FAILED\n";
  } else {
    std::cout << "PASSED\n";
  }

  hipHostFree(A_h);
}

void testHipDeviceMalloc() {
  hipError_t Status;
  int *A_h, *A_d;
  
  A_h = new int[LEN];
  Status = hipMalloc(reinterpret_cast<void **>(&A_d), SIZE);
  // Status = hipHostMalloc(reinterpret_cast<void **>(&A_d), SIZE,
  //                       hipHostMallocDefault);
  assert(Status == hipSuccess);
  A_h[0] = 0;
  hipMemcpy(A_d, A_h, SIZE, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(addOne, 1, 1, 0, 0, static_cast<int *>(A_d));
  hipMemcpy(A_h, A_d, SIZE, hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  if (A_h[0] != 1) {
    std::cout << "FAILED\n";
  } else {
    std::cout << "PASSED\n";
  }

  hipFree(A_h);
}

int main() {
  testHipHostMalloc();
  testHipDeviceMalloc();
  return 0;
}

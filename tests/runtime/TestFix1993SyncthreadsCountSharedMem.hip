// __syncthreads_count must return per-block counts and must not reserve more
// shared memory than __syncthreads_or (IGC gives work_group_reduce_add 4 KB).

#include <hip/hip_runtime.h>

#include <cstdio>

#define HIPCHECK(Expr)                                                         \
  do {                                                                         \
    hipError_t Err = (Expr);                                                   \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %s\n", #Expr, hipGetErrorString(Err));         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void countKernel(int *Bad) {
  int N = blockDim.x * blockDim.y * blockDim.z;
  int Tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int Want = min((int)blockIdx.x, N);
  int Below = __syncthreads_count(Tid < (int)blockIdx.x ? -7 : 0);
  int AtOrAbove = __syncthreads_count(Tid >= (int)blockIdx.x);
  if (Below != Want || AtOrAbove != N - Want)
    atomicAdd(Bad, 1);
}

__global__ void orKernel(int *Out) {
  Out[threadIdx.x] = __syncthreads_or(threadIdx.x & 1);
}

int main() {
  hipFuncAttributes CountAttr, OrAttr;
  HIPCHECK(hipFuncGetAttributes(&CountAttr, (const void *)countKernel));
  HIPCHECK(hipFuncGetAttributes(&OrAttr, (const void *)orKernel));
  printf("static shared memory: __syncthreads_count %zu B, "
         "__syncthreads_or %zu B\n",
         CountAttr.sharedSizeBytes, OrAttr.sharedSizeBytes);
  bool Ok = CountAttr.sharedSizeBytes <= OrAttr.sharedSizeBytes + 256;
  if (!Ok)
    printf("FAIL: __syncthreads_count reserves too much shared memory\n");

  int MaxThreads = CountAttr.maxThreadsPerBlock;
  dim3 Shapes[] = {dim3(1), dim3(7), dim3(33), dim3(48), dim3(100),
                   dim3(5, 3, 2), dim3(MaxThreads)};
  for (dim3 Block : Shapes) {
    int N = Block.x * Block.y * Block.z;
    if (N > MaxThreads)
      continue;
    int *BadD, Bad = 0;
    HIPCHECK(hipMalloc(&BadD, sizeof(int)));
    HIPCHECK(hipMemcpy(BadD, &Bad, sizeof(int), hipMemcpyHostToDevice));
    countKernel<<<N + 2, Block>>>(BadD);
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipMemcpy(&Bad, BadD, sizeof(int), hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(BadD));
    if (Bad) {
      printf("FAIL: block size %d: %d threads got a wrong count\n", N, Bad);
      Ok = false;
    }
  }

  if (!Ok)
    return 1;
  printf("PASSED\n");
  return 0;
}

// Reproduces https://github.com/CHIP-SPV/chipStar/issues/1855:
// hipGraphAddKernelNode segfaulted for a kernel with a spilled by-value
// argument because the node's exec item had no queue.
#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(X)                                                               \
  if ((X) != hipSuccess) {                                                     \
    printf("FAIL: %s\n", #X);                                                  \
    return 1;                                                                  \
  }

constexpr int N = 512;
struct Big {
  int V[N]; // 2 KiB, large enough to be spilled.
};

__global__ void copyBig(Big B, int *Out) {
  int I = threadIdx.x;
  Out[I] = B.V[I];
}

int main() {
  Big B;
  for (int I = 0; I < N; I++)
    B.V[I] = I * 3 + 1;
  int *Out;
  CHECK(hipMalloc(&Out, N * sizeof(int)));
  CHECK(hipMemset(Out, 0, N * sizeof(int)));

  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0));
  void *Args[] = {&B, &Out};
  hipKernelNodeParams P = {};
  P.func = reinterpret_cast<void *>(copyBig);
  P.gridDim = dim3(1);
  P.blockDim = dim3(N);
  P.kernelParams = Args;
  hipGraphNode_t Node;
  CHECK(hipGraphAddKernelNode(&Node, Graph, nullptr, 0, &P));
  // The node must have captured B by value.
  for (int I = 0; I < N; I++)
    B.V[I] = -1;

  hipGraphExec_t Exec;
  CHECK(hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0));
  hipStream_t S;
  CHECK(hipStreamCreateWithFlags(&S, hipStreamNonBlocking));
  CHECK(hipGraphLaunch(Exec, S));
  CHECK(hipStreamSynchronize(S));

  int H[N];
  CHECK(hipMemcpy(H, Out, sizeof(H), hipMemcpyDeviceToHost));
  for (int I = 0; I < N; I++)
    if (H[I] != I * 3 + 1) {
      printf("FAIL: Out[%d] = %d, expected %d\n", I, H[I], I * 3 + 1);
      return 1;
    }
  CHECK(hipGraphExecDestroy(Exec));
  CHECK(hipGraphDestroy(Graph));
  CHECK(hipStreamDestroy(S));
  CHECK(hipFree(Out));
  printf("PASS\n");
  return 0;
}

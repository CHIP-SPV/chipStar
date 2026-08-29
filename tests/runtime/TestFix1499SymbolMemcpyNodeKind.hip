// Regression test for issue #1499: the copy kind handed to
// hipGraphMemcpyNodeSetParamsToSymbol / hipGraphMemcpyNodeSetParamsFromSymbol
// must replace the kind the node was created with.
//
// Each node is created with a device-to-device copy and then re-pointed at a
// host buffer with the matching host kind. If the node keeps the stale
// device-to-device kind the runtime validates the host pointer as a device
// allocation and hipGraphLaunch fails with hipErrorInvalidValue.

#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %s (line %d)\n", #Call, hipGetErrorName(Err),  \
             __LINE__);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

constexpr int N = 16;
constexpr size_t Bytes = N * sizeof(int);

__device__ int Sym[N];

int main() {
  int *DevBuf;
  CHECK(hipMalloc(&DevBuf, Bytes));
  CHECK(hipMemset(DevBuf, 0, Bytes));

  // To-symbol node: created as device-to-device from DevBuf, updated to copy
  // HostSrc into the symbol with hipMemcpyHostToDevice.
  int HostSrc[N];
  for (int I = 0; I < N; I++)
    HostSrc[I] = I + 1;

  hipGraph_t Graph;
  hipGraphNode_t ToNode;
  CHECK(hipGraphCreate(&Graph, 0));
  CHECK(hipGraphAddMemcpyNodeToSymbol(&ToNode, Graph, nullptr, 0,
                                      HIP_SYMBOL(Sym), DevBuf, Bytes, 0,
                                      hipMemcpyDeviceToDevice));
  CHECK(hipGraphMemcpyNodeSetParamsToSymbol(ToNode, HIP_SYMBOL(Sym), HostSrc,
                                            Bytes, 0, hipMemcpyHostToDevice));

  hipGraphExec_t Exec;
  CHECK(hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(Exec, 0));
  CHECK(hipStreamSynchronize(0));
  CHECK(hipGraphExecDestroy(Exec));
  CHECK(hipGraphDestroy(Graph));

  int HostOut[N] = {0};
  CHECK(hipMemcpyFromSymbol(HostOut, HIP_SYMBOL(Sym), Bytes, 0,
                            hipMemcpyDeviceToHost));
  for (int I = 0; I < N; I++)
    if (HostOut[I] != I + 1) {
      printf("FAIL: to-symbol node: Sym[%d] = %d, expected %d\n", I,
             HostOut[I], I + 1);
      return 1;
    }

  // From-symbol node: created as device-to-device into DevBuf, updated to copy
  // the symbol into HostDst with hipMemcpyDeviceToHost.
  int HostDst[N] = {0};
  hipGraphNode_t FromNode;
  CHECK(hipGraphCreate(&Graph, 0));
  CHECK(hipGraphAddMemcpyNodeFromSymbol(&FromNode, Graph, nullptr, 0, DevBuf,
                                        HIP_SYMBOL(Sym), Bytes, 0,
                                        hipMemcpyDeviceToDevice));
  CHECK(hipGraphMemcpyNodeSetParamsFromSymbol(FromNode, HostDst,
                                              HIP_SYMBOL(Sym), Bytes, 0,
                                              hipMemcpyDeviceToHost));

  CHECK(hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(Exec, 0));
  CHECK(hipStreamSynchronize(0));
  CHECK(hipGraphExecDestroy(Exec));
  CHECK(hipGraphDestroy(Graph));

  for (int I = 0; I < N; I++)
    if (HostDst[I] != I + 1) {
      printf("FAIL: from-symbol node: HostDst[%d] = %d, expected %d\n", I,
             HostDst[I], I + 1);
      return 1;
    }

  CHECK(hipFree(DevBuf));
  printf("PASS\n");
  return 0;
}

/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 *
*/

#include "hip/hip_runtime.h"

__global__ void doNoting() {

}

int main(int argc, char **argv) {
  hipGraph_t Graph;
  hipGraphCreate(&Graph, 0);

  hipGraphNode_t M1;
  hipGraphAddMemcpyNode1D(&M1, Graph, nullptr, 0, nullptr, nullptr, 0, hipMemcpyHostToDevice);

  hipGraphNode_t M2;
  hipGraphAddMemcpyNode1D(&M2, Graph, nullptr, 0, nullptr, nullptr, 0, hipMemcpyHostToDevice);

  hipGraphNode_t K1;
  hipGraphNode_t K2;
  hipGraphNode_t K3;

  hipGraphNode_t M1M2Deps[2] = {M1, M2};
  hipKernelNodeParams kernelParams;
  kernelParams.blockDim = dim3(1, 1, 1);
  kernelParams.gridDim = dim3(1, 1, 1);
  kernelParams.func = (void*)&doNoting;
  hipGraphAddKernelNode(&K1, Graph, M1M2Deps, 2, &kernelParams);
  hipGraphAddKernelNode(&K2, Graph, M1M2Deps, 2, &kernelParams);
  hipGraphAddKernelNode(&K3, Graph, M1M2Deps, 2, &kernelParams);

  hipGraphNode_t kernelDeps[3] = {K1, K2, K3};
  hipGraphNode_t Mout;
  hipGraphAddMemcpyNode1D(&Mout, Graph, kernelDeps, 3, nullptr, nullptr, 0, hipMemcpyDeviceToHost);

  hipGraphExec_t GraphExec;
  hipGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0);

  hipGraphLaunch(GraphExec, 0);
}

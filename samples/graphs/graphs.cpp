/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * @file CHIPBindings.hh
 * @author Paulius Velesko (pvelesko@pglc.io)
 * @brief Sample for testing hipGraphs
 * @version 0.1
 * @date 2022-11-17
 *
 * @copyright Copyright (c) 2022
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

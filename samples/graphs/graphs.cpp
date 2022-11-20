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
#include <iostream>
#include "util.hh"

__global__ void doNoting() {}

void case1() {
  hipError_t err;
  hipGraph_t Graph;
  err = hipGraphCreate(&Graph, 0);
  ERR_CHECK(err);

  hipGraphNode_t M1;
  err = hipGraphAddMemcpyNode1D(&M1, Graph, nullptr, 0, nullptr, nullptr, 0,
                                hipMemcpyHostToDevice);
  ERR_CHECK(err);

  hipGraphNode_t M2;
  err = hipGraphAddMemcpyNode1D(&M2, Graph, nullptr, 0, nullptr, nullptr, 0,
                                hipMemcpyHostToDevice);
  ERR_CHECK(err);

  hipGraphNode_t K1;
  hipGraphNode_t K2;
  hipGraphNode_t K3;

  hipKernelNodeParams kernelParams;
  kernelParams.blockDim = dim3(1, 1, 1);
  kernelParams.gridDim = dim3(1, 1, 1);
  kernelParams.func = (void *)&doNoting;
  err = hipGraphAddKernelNode(&K1, Graph, (hipGraphNode_t[]){M1, M2}, 2,
                              &kernelParams);
  ERR_CHECK(err);
  err = hipGraphAddKernelNode(&K2, Graph, (hipGraphNode_t[]){M1, M2}, 2,
                              &kernelParams);
  ERR_CHECK(err);
  err = hipGraphAddKernelNode(&K3, Graph, (hipGraphNode_t[]){M1, M2}, 2,
                              &kernelParams);
  ERR_CHECK(err);

  hipGraphNode_t Mout;
  err = hipGraphAddMemcpyNode1D(&Mout, Graph, (hipGraphNode_t[]){K1, K2, K3}, 3,
                                nullptr, nullptr, 0, hipMemcpyDeviceToHost);
  ERR_CHECK(err);

  hipGraphExec_t GraphExec;
  err = hipGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0);
  ERR_CHECK(err);

  err = hipGraphLaunch(GraphExec, 0);
  ERR_CHECK(err);

  std::cout << "PASSED\n";
}

void case2() {
  hipError_t err;
  hipGraph_t Graph;
  err = hipGraphCreate(&Graph, 0);
  ERR_CHECK(err);

  hipGraphNode_t M1, M2, M3, M4, M5, M6, M7, M8, M9;
  err = hipGraphAddEmptyNode(&M1, Graph, nullptr, 0);
  ERR_CHECK(err);
  err = hipGraphAddEmptyNode(&M2, Graph, nullptr, 0);
  ERR_CHECK(err);
  err = hipGraphAddEmptyNode(&M3, Graph, &M1, 1);
  ERR_CHECK(err);
  err = hipGraphAddEmptyNode(&M4, Graph, &M2, 1);
  ERR_CHECK(err);
  err = hipGraphAddEmptyNode(&M5, Graph, (hipGraphNode_t[]){M1, M3}, 2);
  ERR_CHECK(err);
  err = hipGraphAddEmptyNode(&M6, Graph, (hipGraphNode_t[]){M3, M4}, 2);
  ERR_CHECK(err);
  err = hipGraphAddEmptyNode(&M7, Graph, (hipGraphNode_t[]){M3, M5, M6}, 3);
  ERR_CHECK(err);
  err = hipGraphAddEmptyNode(&M8, Graph, (hipGraphNode_t[]){M5, M7}, 2);
  ERR_CHECK(err);
  err = hipGraphAddEmptyNode(&M9, Graph, (hipGraphNode_t[]){M5, M8}, 2);
  ERR_CHECK(err);

  hipGraphExec_t GraphExec;
  err = hipGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0);
  ERR_CHECK(err);

  err = hipGraphLaunch(GraphExec, 0);
  ERR_CHECK(err);
  std::cout << "PASSED\n";
}

int main(int argc, char **argv) {
  case1();
  case2();
}

/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 * Copyright (c) 2021-22 Paulius Velesko <paulius.velesko@intel.com>
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

#include "../../../include/ze_api.h"
#include "CL/sycl/backend/level_zero.hpp"
#include "oneapi/mkl.hpp"
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <stdio.h>

#include "hip_sycl_interop.h"

using namespace std;

int onemkl_gemm(sycl::queue& my_queue, float* A, float* B, float* C, int m,
                int n, int k, int ldA, int ldB, int ldC, float alpha,
                float beta) {
  auto my_exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const& e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
                  << e.what() << std::endl;
      } catch (std::exception const& e) {
        std::cout << "Caught asynchronous STL exception:\n"
                  << e.what() << std::endl;
      }
    }
  };

  // create sycl buffers of matrix data for offloading between device and host
  sycl::buffer<float, 1> A_buffer(A, m * n);
  sycl::buffer<float, 1> B_buffer(B, m * n);
  sycl::buffer<float, 1> C_buffer(C, m * n);
  // add oneapi::mkl::blas::gemm to execution queue and catch any synchronous
  // exceptions
  try {
    oneapi::mkl::blas::column_major::gemm(
        my_queue, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, m, n, k, alpha, A_buffer, ldA,
        B_buffer, ldB, beta, C_buffer, ldC);
  } catch (sycl::exception const& e) {
    std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
              << e.what() << std::endl;
  } catch (std::exception const& e) {
    std::cout << "\t\tCaught synchronous STL exception during GEMM:\n"
              << e.what() << std::endl;
  }

  // ensure any asynchronous exceptions caught are handled before proceeding
  my_queue.wait_and_throw();

  return 0;
}

// Run GEMM test via oneMKL
int oneMKLGemmTest(uintptr_t* nativeHandlers, float* A, float* B, float* C,
                   int M, int N, int K, int ldA, int ldB, int ldC, float alpha,
                   float beta) {
  // Extract the native information
  ze_driver_handle_t hDriver = (ze_driver_handle_t)nativeHandlers[0];
  ze_device_handle_t hDevice = (ze_device_handle_t)nativeHandlers[1];
  ze_context_handle_t hContext = (ze_context_handle_t)nativeHandlers[2];
  ze_command_queue_handle_t hQueue =
      (ze_command_queue_handle_t)nativeHandlers[3];

  auto keep_ownership = static_cast<sycl::ext::oneapi::level_zero::ownership>(1);
  sycl::platform sycl_platform =
      sycl::level_zero::make<sycl::platform>(hDriver);

  // make devices from converted platform and L0 device
  sycl::device sycl_device =
      sycl::level_zero::make<sycl::device>(sycl_platform, hDevice);
  std::vector<sycl::device> devices;
  devices.push_back(sycl_device);
  sycl::context sycl_context =
      sycl::level_zero::make<sycl::context>(devices, hContext, keep_ownership);
  sycl::queue queue = sycl::level_zero::make<sycl::queue>(sycl_context, hQueue);

  // Test the oneMKL
  return onemkl_gemm(queue, A, B, C, M, N, K, ldA, ldB, ldC, alpha, beta);
}

/*
 * Copyright (c) 2021-22 chipStar developers
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

#include <oneapi/mkl.hpp>

// Check if UR API is available
#if INTEL_MKL_VERSION >= 20250000
  #include <sycl/backend.hpp>
  // Check if ur_native_handle_t is defined
  #ifdef UR_API_H_INCLUDED
    #define HAS_UR_API 1
  #else
    #define HAS_UR_API 0
  #endif
#elif INTEL_MKL_VERSION >= 20230000
  #include <sycl/ext/oneapi/backend/level_zero.hpp>
  #define HAS_UR_API 0
#else
  #include <CL/sycl/backend/level_zero.hpp>
  #define HAS_UR_API 0
#endif

#include "ze_api.h"

#include <stdlib.h>
#include <vector>
#include <string.h>
#include <stdio.h>

#include "hip_sycl_interop.h"

using namespace std;

int onemkl_gemm(sycl::queue &my_queue, float *A, float *B, float *C, int m,
                int n, int k, int ldA, int ldB, int ldC, float alpha,
                float beta) {
  auto my_exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "Caught asynchronous SYCL exception:\n"
                  << e.what() << std::endl;
      } catch (std::exception const &e) {
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
  } catch (sycl::exception const &e) {
    std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
              << e.what() << std::endl;
  } catch (std::exception const &e) {
    std::cout << "\t\tCaught synchronous STL exception during GEMM:\n"
              << e.what() << std::endl;
  }

  // ensure any asynchronous exceptions caught are handled before proceeding
  my_queue.wait_and_throw();

  return 0;
}

// Run GEMM test via oneMKL
int oneMKLGemmTest(uintptr_t *nativeHandlers, float *A,
                   float *B, float *C, int M, int N, int K, int ldA, int ldB,
                   int ldC, float alpha, float beta) {
  std::string hipBackendName((char *)nativeHandlers[0]);
  sycl::queue sycl_queue;
  if (!hipBackendName.compare("opencl")) {
        // handle openCl case here
        cl_platform_id hPlatformId = (cl_platform_id)nativeHandlers[1];
        cl_device_id hDeviceId = (cl_device_id)nativeHandlers[2];
        cl_context hContext = (cl_context)nativeHandlers[3];
        cl_command_queue hQueue = (cl_command_queue)nativeHandlers[4];
        
#if HAS_UR_API
        // MKL 2025 uses UR API
        sycl::platform sycl_platform =
            sycl::detail::make_platform((ur_native_handle_t)hPlatformId, sycl::backend::opencl);
        sycl::device sycl_device =
            sycl::detail::make_device((ur_native_handle_t)hDeviceId, sycl::backend::opencl);
        sycl::context sycl_context =
            sycl::detail::make_context((ur_native_handle_t)hContext, {}, sycl::backend::opencl, false);
        sycl_queue =
            sycl::detail::make_queue((ur_native_handle_t)hQueue, false, sycl_context, &sycl_device, false, {}, {}, sycl::backend::opencl);
#else
        // MKL 2024 and earlier use PI API
        sycl::platform sycl_platform =
            sycl::opencl::make_platform((pi_native_handle)hPlatformId);
        sycl::device sycl_device =
            sycl::opencl::make_device((pi_native_handle)hDeviceId);
        sycl::context sycl_context =
            sycl::opencl::make_context((pi_native_handle)hContext);
        sycl_queue =
            sycl::opencl::make_queue(sycl_context, (pi_native_handle)hQueue);
#endif
  } else if (!hipBackendName.compare("level0")) {
    // handle L0 here
    // Extract the native information
    ze_driver_handle_t hDriver = (ze_driver_handle_t)nativeHandlers[1];
    ze_device_handle_t hDevice = (ze_device_handle_t)nativeHandlers[2];
    ze_context_handle_t hContext = (ze_context_handle_t)nativeHandlers[3];
    ze_command_queue_handle_t hQueue =
        (ze_command_queue_handle_t)nativeHandlers[4];
    ze_command_list_handle_t hCommandList =
        (ze_command_list_handle_t)nativeHandlers[5];

    bool isImmCmdList = hCommandList ? true : false;

#if HAS_UR_API
    // MKL 2025 uses UR API
    sycl::platform sycl_platform =
        sycl::detail::make_platform((ur_native_handle_t)hDriver, sycl::backend::ext_oneapi_level_zero);
    sycl::device sycl_device = 
        sycl::detail::make_device((ur_native_handle_t)hDevice, sycl::backend::ext_oneapi_level_zero);

    std::vector<sycl::device> sycl_devices(1);
    sycl_devices[0] = sycl_device;
    // Use the specific device from CHIP-SPV, not all system devices
    sycl::context sycl_context = 
        sycl::detail::make_context((ur_native_handle_t)hContext, {}, sycl::backend::ext_oneapi_level_zero, false,
	 sycl_platform.get_devices());

    if (isImmCmdList) {
        sycl_queue = sycl::detail::make_queue((ur_native_handle_t)hCommandList, true, sycl_context, &sycl_device, true, 
                                             {sycl::property::queue::in_order()}, {}, sycl::backend::ext_oneapi_level_zero);
    } else {
        sycl_queue = sycl::detail::make_queue((ur_native_handle_t)hQueue, false, sycl_context, &sycl_device, true, 
                                             {sycl::property::queue::in_order()}, {}, sycl::backend::ext_oneapi_level_zero);
    }
#elif __INTEL_LLVM_COMPILER >= 20240000
    auto keep_ownership =
        static_cast<sycl::ext::oneapi::level_zero::ownership>(1);
    sycl::platform sycl_platform =
        sycl::ext::oneapi::level_zero::make_platform((pi_native_handle)hDriver);
    sycl::device sycl_device = sycl::ext::oneapi::level_zero::make_device(
        sycl_platform, (pi_native_handle)hDevice);

    std::vector<sycl::device> sycl_devices(1);
    sycl_devices[0] = sycl_device;
    sycl::context sycl_context = sycl::ext::oneapi::level_zero::make_context(
        sycl_devices, (pi_native_handle)hContext, 1);

    if (isImmCmdList) {
      sycl_queue = sycl::ext::oneapi::level_zero::make_queue(
          sycl_context, sycl_device, (pi_native_handle)hCommandList, true, 1,
          sycl::property::queue::in_order());
    } else {
      sycl_queue = sycl::ext::oneapi::level_zero::make_queue(
          sycl_context, sycl_device, (pi_native_handle)hQueue, false, 1,
          sycl::property::queue::in_order());
    }
#else
    auto keep_ownership =
        static_cast<sycl::ext::oneapi::level_zero::ownership>(1);
    sycl::platform sycl_platform =
        sycl::ext::oneapi::level_zero::make_platform((pi_native_handle)hDriver);
    sycl::device sycl_device = sycl::ext::oneapi::level_zero::make_device(
        sycl_platform, (pi_native_handle)hDevice);

    std::vector<sycl::device> sycl_devices(1);
    sycl_devices[0] = sycl_device;
    sycl::context sycl_context = sycl::ext::oneapi::level_zero::make_context(
        sycl_devices, (pi_native_handle)hContext, 1);

    sycl_queue = sycl::ext::oneapi::level_zero::make_queue(
        sycl_context, sycl_device, (pi_native_handle)hQueue, 1);
#endif
  } else {
    std::cout << "Unsupported backend: " << hipBackendName << std::endl;
    std::abort();
  }

  // Test the oneMKL
  return onemkl_gemm(sycl_queue, A, B, C, M, N, K, ldA, ldB, ldC, alpha, beta);
}

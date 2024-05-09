/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasSwapModel = ArgumentModel<e_a_type, e_N, e_incx, e_incy>;

inline void testname_swap(const Arguments& arg, std::string& name)
{
    hipblasSwapModel{}.test_name(arg, name);
}

template <typename T>
void testing_swap_bad_arg(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSwapFn = FORTRAN ? hipblasSwap<T, true> : hipblasSwap<T, false>;
    auto hipblasSwapFn_64
        = arg.api == FORTRAN_64 ? hipblasSwap_64<T, true> : hipblasSwap_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t N    = 100;
    int64_t incx = 1;
    int64_t incy = 1;

    device_vector<T> dx(N * incx);
    device_vector<T> dy(N * incy);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED, hipblasSwapFn, (nullptr, N, dx, incx, dy, incy));

    if(arg.bad_arg_all)
    {
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE, hipblasSwapFn, (handle, N, nullptr, incx, dy, incy));
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE, hipblasSwapFn, (handle, N, dx, incx, nullptr, incy));
    }
}

template <typename T>
void testing_swap(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSwapFn = FORTRAN ? hipblasSwap<T, true> : hipblasSwap<T, false>;
    auto hipblasSwapFn_64
        = arg.api == FORTRAN_64 ? hipblasSwap_64<T, true> : hipblasSwap_64<T, false>;

    int64_t N          = arg.N;
    int64_t incx       = arg.incx;
    int64_t incy       = arg.incy;
    int     unit_check = arg.unit_check;
    int     norm_check = arg.norm_check;
    int     timing     = arg.timing;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        DAPI_CHECK(hipblasSwapFn, (handle, N, nullptr, incx, nullptr, incy));
        return;
    }

    int64_t abs_incx = incx >= 0 ? incx : -incx;
    int64_t abs_incy = incy >= 0 ? incy : -incy;
    size_t  sizeX    = size_t(N) * abs_incx;
    size_t  sizeY    = size_t(N) * abs_incy;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hy(sizeY);
    host_vector<T> hx_cpu(sizeX);
    host_vector<T> hy_cpu(sizeY);

    // allocate memory on device
    device_vector<T> dx(sizeX);
    device_vector<T> dy(sizeY);
    int              device_pointer = 1;

    double gpu_time_used = 0.0, cpu_time_used = 0.0;
    double hipblas_error = 0.0;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, N, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy, arg, N, abs_incy, 0, 1, hipblas_client_alpha_sets_nan, false);
    hx_cpu = hx;
    hy_cpu = hy;

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        DAPI_CHECK(hipblasSwapFn, (handle, N, dx, incx, dy, incy));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * sizeX, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * sizeY, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        ref_swap<T>(N, hx.data(), incx, hy.data(), incy);

        if(unit_check)
        {
            unit_check_general<T>(1, N, abs_incx, hx_cpu.data(), hx.data());
            unit_check_general<T>(1, N, abs_incy, hy_cpu.data(), hy.data());
        }
        if(norm_check)
        {
            hipblas_error
                = std::max(norm_check_general<T>('F', 1, N, abs_incx, hx_cpu.data(), hx.data()),
                           norm_check_general<T>('F', 1, N, abs_incy, hy_cpu.data(), hy.data()));
        }

    } // end of if unit/norm check

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_CHECK(hipblasSwapFn, (handle, N, dx, incx, dy, incy));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasSwapModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       swap_gflop_count<T>(N),
                                       swap_gbyte_count<T>(N),
                                       hipblas_error);
    }
}

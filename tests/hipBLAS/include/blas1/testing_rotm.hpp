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

using hipblasRotmModel = ArgumentModel<e_a_type, e_N, e_incx, e_incy>;

inline void testname_rotm(const Arguments& arg, std::string& name)
{
    hipblasRotmModel{}.test_name(arg, name);
}

template <typename T>
void testing_rotm_bad_arg(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotmFn = FORTRAN ? hipblasRotm<T, true> : hipblasRotm<T, false>;
    auto hipblasRotmFn_64
        = arg.api == FORTRAN_64 ? hipblasRotm_64<T, true> : hipblasRotm_64<T, false>;

    int64_t N    = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T       h_param[5];

    hipblasLocalHandle handle(arg);

    device_vector<T> dx(N * incx);
    device_vector<T> dy(N * incy);
    device_vector<T> dparam(5);
    T*               param = dparam;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
        {
            param = h_param;

            // if pointer_mode_host and param[0] == -2, can quick return
            param[0] = -2;
            DAPI_CHECK(hipblasRotmFn, (handle, N, nullptr, incx, nullptr, incy, param));
            param[0] = 0;
        }

        DAPI_EXPECT(
            HIPBLAS_STATUS_NOT_INITIALIZED, hipblasRotmFn, (nullptr, N, dx, incx, dy, incy, param));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasRotmFn,
                        (handle, N, nullptr, incx, dy, incy, param));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasRotmFn,
                        (handle, N, dx, incx, nullptr, incy, param));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasRotmFn,
                        (handle, N, dx, incx, dy, incy, nullptr));
        }
    }
}

template <typename T>
void testing_rotm(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotmFn = FORTRAN ? hipblasRotm<T, true> : hipblasRotm<T, false>;
    auto hipblasRotmFn_64
        = arg.api == FORTRAN_64 ? hipblasRotm_64<T, true> : hipblasRotm_64<T, false>;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0)
    {
        DAPI_CHECK(hipblasRotmFn, (handle, N, nullptr, incx, nullptr, incy, nullptr));
        return;
    }

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    int64_t abs_incx = incx >= 0 ? incx : -incx;
    int64_t abs_incy = incy >= 0 ? incy : -incy;
    size_t  size_x   = N * size_t(abs_incx);
    size_t  size_y   = N * size_t(abs_incy);
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;

    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<T> dparam(5);

    // Initial Data on CPU
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);
    host_vector<T> hdata(4);
    host_vector<T> hparam(5);

    hipblas_init_vector(hx, arg, N, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy, arg, N, abs_incy, 0, 1, hipblas_client_alpha_sets_nan, false);
    hipblas_init_vector(hdata, arg, 4, 1, 0, 1, hipblas_client_alpha_sets_nan, false);

    // CPU BLAS reference data
    ref_rotmg<T>(&hdata[0], &hdata[1], &hdata[2], &hdata[3], hparam);
    const int FLAG_COUNT        = 4;
    const T   FLAGS[FLAG_COUNT] = {-1, 0, 1, -2};
    for(int i = 0; i < FLAG_COUNT; ++i)
    {
        if(arg.unit_check || arg.norm_check)
        {
            hparam[0]         = FLAGS[i];
            host_vector<T> cx = hx;
            host_vector<T> cy = hy;
            ref_rotm<T>(N, cx, incx, cy, incy, hparam);

            // Test host
            {
                CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
                CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
                DAPI_CHECK(hipblasRotmFn, (handle, N, dx, incx, dy, incy, hparam));
                host_vector<T> rx(size_x);
                host_vector<T> ry(size_y);
                CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
                if(arg.unit_check)
                {
                    near_check_general(1, N, abs_incx, cx.data(), rx.data(), rel_error);
                    near_check_general(1, N, abs_incy, cy.data(), ry.data(), rel_error);
                }
                if(arg.norm_check)
                {
                    hipblas_error_host = norm_check_general<T>('F', 1, N, abs_incx, cx, rx);
                    hipblas_error_host += norm_check_general<T>('F', 1, N, abs_incy, cy, ry);
                }
            }

            // Test device
            {
                CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
                CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dparam, hparam, sizeof(T) * 5, hipMemcpyHostToDevice));
                DAPI_CHECK(hipblasRotmFn, (handle, N, dx, incx, dy, incy, dparam));
                host_vector<T> rx(size_x);
                host_vector<T> ry(size_y);
                CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
                if(arg.unit_check)
                {
                    near_check_general(1, N, abs_incx, cx.data(), rx.data(), rel_error);
                    near_check_general(1, N, abs_incy, cy.data(), ry.data(), rel_error);
                }
                if(arg.norm_check)
                {
                    hipblas_error_device = norm_check_general<T>('F', 1, N, abs_incx, cx, rx);
                    hipblas_error_device += norm_check_general<T>('F', 1, N, abs_incy, cy, ry);
                }
            }
        }
    }

    if(arg.timing)
    {
        hparam[0] = 0;
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dparam, hparam, sizeof(T) * 5, hipMemcpyHostToDevice));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_CHECK(hipblasRotmFn, (handle, N, dx, incx, dy, incy, dparam));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotmModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       rotm_gflop_count<T>(N, hparam[0]),
                                       rotm_gbyte_count<T>(N, hparam[0]),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}

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

using hipblasRotModel = ArgumentModel<e_a_type, e_c_type, e_compute_type, e_N, e_incx, e_incy>;

inline void testname_rot(const Arguments& arg, std::string& name)
{
    hipblasRotModel{}.test_name(arg, name);
}

template <typename T, typename U = T, typename V = T>
void testing_rot_bad_arg(const Arguments& arg)
{
    bool FORTRAN      = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotFn = FORTRAN ? hipblasRot<T, U, V, true> : hipblasRot<T, U, V, false>;
    auto hipblasRotFn_64
        = arg.api == FORTRAN_64 ? hipblasRot_64<T, U, V, true> : hipblasRot_64<T, U, V, false>;

    int64_t N    = 100;
    int64_t incx = 1;
    int64_t incy = 1;

    hipblasLocalHandle handle(arg);

    device_vector<T> dx(N * incx);
    device_vector<T> dy(N * incy);
    device_vector<U> dc(1);
    device_vector<V> ds(1);

    EXPECT_HIPBLAS_STATUS(hipblasRotFn(nullptr, N, dx, incx, dy, incy, dc, ds),
                          HIPBLAS_STATUS_NOT_INITIALIZED);

    if(arg.bad_arg_all)
    {
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasRotFn,
                    (handle, N, nullptr, incx, dy, incy, dc, ds));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasRotFn,
                    (handle, N, dx, incx, nullptr, incy, dc, ds));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasRotFn,
                    (handle, N, dx, incx, dy, incy, nullptr, ds));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasRotFn,
                    (handle, N, dx, incx, dy, incy, dc, nullptr));
    }
}

template <typename T, typename U = T, typename V = T>
void testing_rot(const Arguments& arg)
{
    bool FORTRAN      = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotFn = FORTRAN ? hipblasRot<T, U, V, true> : hipblasRot<T, U, V, false>;
    auto hipblasRotFn_64
        = arg.api == FORTRAN_64 ? hipblasRot_64<T, U, V, true> : hipblasRot_64<T, U, V, false>;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0)
    {
        DAPI_CHECK(hipblasRotFn, (handle, N, nullptr, incx, nullptr, incy, nullptr, nullptr));
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
    device_vector<U> dc(1);
    device_vector<V> ds(1);

    // Initial Data on CPU
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);
    host_vector<U> hc(1);
    host_vector<V> hs(1);

    // Random alpha (0 - 10)
    host_vector<int> alpha(1);

    hipblas_init_vector(hx, arg, N, abs_incx, 0, 1, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hy, arg, N, abs_incy, 0, 1, hipblas_client_never_set_nan, false);
    hipblas_init_vector(alpha, arg, 1, 1, 0, 1, hipblas_client_never_set_nan, false);

    // cos and sin of alpha (in rads)
    hc[0] = cos(alpha[0]);
    hs[0] = sin(alpha[0]);

    // CPU BLAS reference data
    host_vector<T> cx = hx;
    host_vector<T> cy = hy;

    ref_rot<T, U, V>(N, cx.data(), incx, cy.data(), incy, *hc, *hs);

    if(arg.unit_check || arg.norm_check)
    {
        // Test host
        {
            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
            DAPI_CHECK(hipblasRotFn, (handle, N, dx, incx, dy, incy, hc, hs));

            host_vector<T> rx(size_x);
            host_vector<T> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                near_check_general(1, N, abs_incx, cx.data(), rx.data(), double(rel_error));
                near_check_general(1, N, abs_incy, cy.data(), ry.data(), double(rel_error));
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
            CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));
            DAPI_CHECK(hipblasRotFn, (handle, N, dx, incx, dy, incy, dc, ds));
            host_vector<T> rx(size_x);
            host_vector<T> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                near_check_general(1, N, abs_incx, cx.data(), rx.data(), double(rel_error));
                near_check_general(1, N, abs_incy, cy.data(), ry.data(), double(rel_error));
            }
            if(arg.norm_check)
            {
                hipblas_error_device = norm_check_general<T>('F', 1, N, abs_incx, cx, rx);
                hipblas_error_device += norm_check_general<T>('F', 1, N, abs_incy, cy, ry);
            }
        }
    }

    if(arg.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_CHECK(hipblasRotFn, (handle, N, dx, incx, dy, incy, dc, ds));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotModel{}.log_args<T>(std::cout,
                                      arg,
                                      gpu_time_used,
                                      rot_gflop_count<T, T, U, V>(N),
                                      rot_gbyte_count<T>(N),
                                      hipblas_error_host,
                                      hipblas_error_device);
    }
}

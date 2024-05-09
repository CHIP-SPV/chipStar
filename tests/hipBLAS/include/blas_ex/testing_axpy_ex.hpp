/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

using hipblasAxpyExModel
    = ArgumentModel<e_a_type, e_b_type, e_c_type, e_compute_type, e_N, e_alpha, e_incx, e_incy>;

inline void testname_axpy_ex(const Arguments& arg, std::string& name)
{
    hipblasAxpyExModel{}.test_name(arg, name);
}

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_ex_bad_arg(const Arguments& arg)
{
    bool FORTRAN         = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasAxpyExFn = FORTRAN ? hipblasAxpyExFortran : hipblasAxpyEx;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasDatatype_t alphaType     = arg.a_type;
        hipblasDatatype_t xType         = arg.b_type;
        hipblasDatatype_t yType         = arg.c_type;
        hipblasDatatype_t executionType = arg.compute_type;

        int64_t N    = 100;
        int64_t incx = 1;
        int64_t incy = 1;

        device_vector<Ta> d_alpha(1), d_zero(1);
        device_vector<Tx> dx(N * incx);
        device_vector<Ty> dy(N * incy);

        const Ta  h_alpha(1), h_zero(0);
        const Ta* alpha = &h_alpha;
        const Ta* zero  = &h_zero;

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(h_alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, &h_zero, sizeof(h_zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        EXPECT_HIPBLAS_STATUS(
            hipblasAxpyExFn(
                nullptr, N, alpha, alphaType, dx, xType, incx, dy, yType, incy, executionType),
            HIPBLAS_STATUS_NOT_INITIALIZED);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasAxpyExFn(
                    handle, N, nullptr, alphaType, dx, xType, incx, dy, yType, incy, executionType),
                HIPBLAS_STATUS_INVALID_VALUE);

            // Can only check for nullptr for dx/dy with host mode because
            // device mode may not check as it could be quick-return success
            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                EXPECT_HIPBLAS_STATUS(hipblasAxpyExFn(handle,
                                                      N,
                                                      alpha,
                                                      alphaType,
                                                      nullptr,
                                                      xType,
                                                      incx,
                                                      dy,
                                                      yType,
                                                      incy,
                                                      executionType),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasAxpyExFn(handle,
                                                      N,
                                                      alpha,
                                                      alphaType,
                                                      dx,
                                                      xType,
                                                      incx,
                                                      nullptr,
                                                      yType,
                                                      incy,
                                                      executionType),
                                      HIPBLAS_STATUS_INVALID_VALUE);
            }
        }

        CHECK_HIPBLAS_ERROR(hipblasAxpyExFn(handle,
                                            0,
                                            nullptr,
                                            alphaType,
                                            nullptr,
                                            xType,
                                            incx,
                                            nullptr,
                                            yType,
                                            incy,
                                            executionType));
        CHECK_HIPBLAS_ERROR(hipblasAxpyExFn(
            handle, N, zero, alphaType, nullptr, xType, incx, nullptr, yType, incy, executionType));
    }
}

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_ex(const Arguments& arg)
{
    bool FORTRAN         = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasAxpyExFn = FORTRAN ? hipblasAxpyExFortran : hipblasAxpyEx;

    int N    = arg.N;
    int incx = arg.incx;
    int incy = arg.incy;

    hipblasLocalHandle handle(arg);

    hipblasDatatype_t alphaType     = arg.a_type;
    hipblasDatatype_t xType         = arg.b_type;
    hipblasDatatype_t yType         = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasAxpyExFn(handle,
                                            N,
                                            nullptr,
                                            alphaType,
                                            nullptr,
                                            xType,
                                            incx,
                                            nullptr,
                                            yType,
                                            incy,
                                            executionType));
        return;
    }

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    size_t sizeX = size_t(N) * abs_incx;
    size_t sizeY = size_t(N) * abs_incy;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    Ta h_alpha = arg.get_alpha<Ta>();

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(sizeX);
    host_vector<Ty> hy_host(sizeY);
    host_vector<Ty> hy_device(sizeY);
    host_vector<Ty> hy_cpu(sizeY);

    device_vector<Tx> dx(sizeX);
    device_vector<Ty> dy(sizeY);
    device_vector<Ta> d_alpha(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, N, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy_host, arg, N, abs_incy, 0, 1, hipblas_client_alpha_sets_nan, false);
    // hx[0] = (Tx)5.0f;
    // hy_host[0] = (Ty)22.0f;

    hy_device = hy_host;
    hy_cpu    = hy_host;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy_host, sizeof(Ty) * sizeY, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasAxpyExFn(
        handle, N, &h_alpha, alphaType, dx, xType, incx, dy, yType, incy, executionType));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hy_host, dy, sizeof(Ty) * sizeY, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy_device, sizeof(Ty) * sizeY, hipMemcpyHostToDevice));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasAxpyExFn(
        handle, N, d_alpha, alphaType, dx, xType, incx, dy, yType, incy, executionType));
    CHECK_HIP_ERROR(hipMemcpy(hy_device, dy, sizeof(Ty) * sizeY, hipMemcpyDeviceToHost));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        ref_axpy(N, h_alpha, hx.data(), incx, hy_cpu.data(), incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<Ty>(1, N, abs_incy, hy_cpu, hy_host);
            unit_check_general<Ty>(1, N, abs_incy, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host   = norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_host);
            hipblas_error_device = norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_device);
        }

    } // end of if unit check

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasAxpyExFn(
                handle, N, d_alpha, alphaType, dx, xType, incx, dy, yType, incy, executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasAxpyExModel{}.log_args<Ta>(std::cout,
                                          arg,
                                          gpu_time_used,
                                          axpy_gflop_count<Ta>(N),
                                          axpy_gbyte_count<Ta>(N),
                                          hipblas_error_host,
                                          hipblas_error_device);
    }
}

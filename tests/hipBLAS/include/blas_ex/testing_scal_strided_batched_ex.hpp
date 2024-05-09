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

using hipblasScalStridedBatchedExModel = ArgumentModel<e_a_type,
                                                       e_b_type,
                                                       e_compute_type,
                                                       e_N,
                                                       e_alpha,
                                                       e_incx,
                                                       e_stride_scale,
                                                       e_batch_count>;

inline void testname_scal_strided_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasScalStridedBatchedExModel{}.test_name(arg, name);
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_strided_batched_ex_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasScalStridedBatchedExFn
        = FORTRAN ? hipblasScalStridedBatchedExFortran : hipblasScalStridedBatchedEx;

    hipblasDatatype_t alphaType     = arg.a_type;
    hipblasDatatype_t xType         = arg.b_type;
    hipblasDatatype_t executionType = arg.compute_type;

    int64_t N           = 100;
    int64_t incx        = 1;
    int64_t batch_count = 2;

    hipblasStride stridex = N * incx;

    Ta alpha = (Ta)0.6;

    hipblasLocalHandle handle(arg);

    device_vector<Tx> dx(stridex * batch_count);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // Notably scal differs from axpy such that x can /never/ be a nullptr, regardless of alpha.

        // None of these test cases will write to result so using device pointer is fine for both modes
        EXPECT_HIPBLAS_STATUS(hipblasScalStridedBatchedExFn(nullptr,
                                                            N,
                                                            &alpha,
                                                            alphaType,
                                                            dx,
                                                            xType,
                                                            incx,
                                                            stridex,
                                                            batch_count,
                                                            executionType),
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(hipblasScalStridedBatchedExFn(handle,
                                                                N,
                                                                nullptr,
                                                                alphaType,
                                                                dx,
                                                                xType,
                                                                incx,
                                                                stridex,
                                                                batch_count,
                                                                executionType),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasScalStridedBatchedExFn(handle,
                                                                N,
                                                                &alpha,
                                                                alphaType,
                                                                nullptr,
                                                                xType,
                                                                incx,
                                                                stridex,
                                                                batch_count,
                                                                executionType),
                                  HIPBLAS_STATUS_INVALID_VALUE);
        }
    }
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_strided_batched_ex(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasScalStridedBatchedExFn
        = FORTRAN ? hipblasScalStridedBatchedExFortran : hipblasScalStridedBatchedEx;

    int    N            = arg.N;
    int    incx         = arg.incx;
    double stride_scale = arg.stride_scale;
    int    batch_count  = arg.batch_count;

    int unit_check = arg.unit_check;
    int timing     = arg.timing;
    int norm_check = arg.norm_check;

    hipblasStride stridex = size_t(N) * incx * stride_scale;
    size_t        sizeX   = stridex * batch_count;

    Ta h_alpha = arg.get_alpha<Ta>();

    hipblasLocalHandle handle(arg);

    hipblasDatatype_t alphaType     = arg.a_type;
    hipblasDatatype_t xType         = arg.b_type;
    hipblasDatatype_t executionType = arg.compute_type;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasScalStridedBatchedExFn(handle,
                                                          N,
                                                          nullptr,
                                                          alphaType,
                                                          nullptr,
                                                          xType,
                                                          incx,
                                                          stride_scale,
                                                          batch_count,
                                                          executionType));
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx_host(sizeX);
    host_vector<Tx> hx_device(sizeX);
    host_vector<Tx> hx_cpu(sizeX);

    host_vector<Tex> hx_host_ex(sizeX);
    host_vector<Tex> hx_device_ex(sizeX);
    host_vector<Tex> hx_cpu_ex(sizeX);

    device_vector<Tx> dx(sizeX);
    device_vector<Ta> d_alpha(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(
        hx_host, arg, N, incx, stridex, batch_count, hipblas_client_alpha_sets_nan, true);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hx_device = hx_cpu = hx_host;

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx_host, sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasScalStridedBatchedExFn(
            handle, N, &h_alpha, alphaType, dx, xType, incx, stridex, batch_count, executionType));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx_host, dx, sizeof(Tx) * sizeX, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dx, hx_device, sizeof(Tx) * sizeX, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasScalStridedBatchedExFn(
            handle, N, d_alpha, alphaType, dx, xType, incx, stridex, batch_count, executionType));

        CHECK_HIP_ERROR(hipMemcpy(hx_device, dx, sizeof(Tx) * sizeX, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            ref_scal<Tx, Ta>(N, h_alpha, hx_cpu + b * stridex, incx);
        }

        for(size_t b = 0; b < batch_count; b++)
        {
            for(size_t i = 0; i < N; i++)
            {
                hx_host_ex[i * incx + b * stridex]   = (Tex)hx_host[i * incx + b * stridex];
                hx_device_ex[i * incx + b * stridex] = (Tex)hx_device[i * incx + b * stridex];
                hx_cpu_ex[i * incx + b * stridex]    = (Tex)hx_cpu[i * incx + b * stridex];
            }
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(unit_check)
        {
            unit_check_general<Tex>(1, N, batch_count, incx, stridex, hx_cpu_ex, hx_host_ex);
            unit_check_general<Tex>(1, N, batch_count, incx, stridex, hx_cpu_ex, hx_device_ex);
        }

        if(norm_check)
        {
            hipblas_error_host = norm_check_general<Tex>(
                'F', 1, N, incx, stridex, hx_cpu_ex, hx_host_ex, batch_count);
            hipblas_error_device = norm_check_general<Tex>(
                'F', 1, N, incx, stridex, hx_cpu_ex, hx_device_ex, batch_count);
        }

    } // end of if unit check

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasScalStridedBatchedExFn(handle,
                                                              N,
                                                              d_alpha,
                                                              alphaType,
                                                              dx,
                                                              xType,
                                                              incx,
                                                              stridex,
                                                              batch_count,
                                                              executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasScalStridedBatchedExModel{}.log_args<Tx>(std::cout,
                                                        arg,
                                                        gpu_time_used,
                                                        scal_gflop_count<Tx, Ta>(N),
                                                        scal_gbyte_count<Tx>(N),
                                                        hipblas_error_host,
                                                        hipblas_error_device);
    }
}

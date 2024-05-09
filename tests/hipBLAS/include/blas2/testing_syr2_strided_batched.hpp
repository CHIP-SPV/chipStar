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

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasSyr2StridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_uplo,
                                                     e_N,
                                                     e_alpha,
                                                     e_incx,
                                                     e_incy,
                                                     e_lda,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_syr2_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasSyr2StridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_syr2_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSyr2StridedBatchedFn
        = FORTRAN ? hipblasSyr2StridedBatched<T, true> : hipblasSyr2StridedBatched<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasFillMode_t uplo        = HIPBLAS_FILL_MODE_UPPER;
        int64_t           N           = 100;
        int64_t           lda         = 100;
        int64_t           incx        = 1;
        int64_t           incy        = 1;
        int64_t           batch_count = 2;
        hipblasStride     stridex     = N * incx;
        hipblasStride     stridey     = N * incy;
        hipblasStride     strideA     = N * lda;

        device_vector<T> d_alpha(1), d_zero(1);

        const T  h_alpha(1), h_zero(0);
        const T* alpha = &h_alpha;
        const T* zero  = &h_zero;

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        device_vector<T> dA(strideA * batch_count);
        device_vector<T> dx(stridex * batch_count);
        device_vector<T> dy(stridey * batch_count);

        EXPECT_HIPBLAS_STATUS(hipblasSyr2StridedBatchedFn(nullptr,
                                                          uplo,
                                                          N,
                                                          alpha,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          dy,
                                                          incy,
                                                          stridey,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          batch_count),
                              HIPBLAS_STATUS_NOT_INITIALIZED);
        EXPECT_HIPBLAS_STATUS(hipblasSyr2StridedBatchedFn(handle,
                                                          HIPBLAS_FILL_MODE_FULL,
                                                          N,
                                                          alpha,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          dy,
                                                          incy,
                                                          stridey,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasSyr2StridedBatchedFn(handle,
                                                          (hipblasFillMode_t)HIPBLAS_OP_N,
                                                          N,
                                                          alpha,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          dy,
                                                          incy,
                                                          stridey,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);

        EXPECT_HIPBLAS_STATUS(hipblasSyr2StridedBatchedFn(handle,
                                                          uplo,
                                                          N,
                                                          nullptr,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          dy,
                                                          incy,
                                                          stridey,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);

        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
        {
            // For device mode in rocBLAS we don't have checks for dA, dx as we may be able to quick return
            EXPECT_HIPBLAS_STATUS(hipblasSyr2StridedBatchedFn(handle,
                                                              uplo,
                                                              N,
                                                              alpha,
                                                              nullptr,
                                                              incx,
                                                              stridex,
                                                              dy,
                                                              incy,
                                                              stridey,
                                                              dA,
                                                              lda,
                                                              strideA,
                                                              batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasSyr2StridedBatchedFn(handle,
                                                              uplo,
                                                              N,
                                                              alpha,
                                                              dx,
                                                              incx,
                                                              stridex,
                                                              nullptr,
                                                              incy,
                                                              stridey,
                                                              dA,
                                                              lda,
                                                              strideA,
                                                              batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasSyr2StridedBatchedFn(handle,
                                                              uplo,
                                                              N,
                                                              alpha,
                                                              dx,
                                                              incx,
                                                              stridex,
                                                              dy,
                                                              incy,
                                                              stridey,
                                                              nullptr,
                                                              lda,
                                                              strideA,
                                                              batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);
        }

        // With N == 0, can have all nullptrs
        CHECK_HIPBLAS_ERROR(hipblasSyr2StridedBatchedFn(handle,
                                                        uplo,
                                                        0,
                                                        nullptr,
                                                        nullptr,
                                                        incx,
                                                        stridex,
                                                        nullptr,
                                                        incy,
                                                        stridey,
                                                        nullptr,
                                                        lda,
                                                        strideA,
                                                        batch_count));
        CHECK_HIPBLAS_ERROR(hipblasSyr2StridedBatchedFn(handle,
                                                        uplo,
                                                        N,
                                                        nullptr,
                                                        nullptr,
                                                        incx,
                                                        stridex,
                                                        nullptr,
                                                        incy,
                                                        stridey,
                                                        nullptr,
                                                        lda,
                                                        strideA,
                                                        0));

        // With alpha == 0, can have all nullptrs
        CHECK_HIPBLAS_ERROR(hipblasSyr2StridedBatchedFn(handle,
                                                        uplo,
                                                        N,
                                                        zero,
                                                        nullptr,
                                                        incx,
                                                        stridex,
                                                        nullptr,
                                                        incy,
                                                        stridey,
                                                        nullptr,
                                                        lda,
                                                        strideA,
                                                        batch_count));
    }
}

template <typename T>
void testing_syr2_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSyr2StridedBatchedFn
        = FORTRAN ? hipblasSyr2StridedBatched<T, true> : hipblasSyr2StridedBatched<T, false>;

    hipblasFillMode_t uplo         = char2hipblas_fill(arg.uplo);
    int               N            = arg.N;
    int               incx         = arg.incx;
    int               incy         = arg.incy;
    int               lda          = arg.lda;
    double            stride_scale = arg.stride_scale;
    int               batch_count  = arg.batch_count;

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    hipblasStride strideA = lda * N * stride_scale;
    hipblasStride stridex = abs_incx * N * stride_scale;
    hipblasStride stridey = abs_incy * N * stride_scale;
    size_t        A_size  = strideA * batch_count;
    size_t        x_size  = stridex * batch_count;
    size_t        y_size  = stridey * batch_count;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || !incy || lda < N || lda < 1 || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        hipblasStatus_t actual = hipblasSyr2StridedBatchedFn(handle,
                                                             uplo,
                                                             N,
                                                             nullptr,
                                                             nullptr,
                                                             incx,
                                                             stridex,
                                                             nullptr,
                                                             incy,
                                                             stridey,
                                                             nullptr,
                                                             lda,
                                                             strideA,
                                                             batch_count);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA_cpu(A_size);
    host_vector<T> hA_host(A_size);
    host_vector<T> hA_device(A_size);
    host_vector<T> hx(x_size);
    host_vector<T> hy(y_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(x_size);
    device_vector<T> dy(y_size);
    device_vector<T> d_alpha(1);

    T h_alpha = arg.get_alpha<T>();

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, N, N, lda, strideA, batch_count, hipblas_client_never_set_nan, true);
    hipblas_init_vector(
        hx, arg, N, abs_incx, stridex, batch_count, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, arg, N, abs_incy, stridey, batch_count, hipblas_client_alpha_sets_nan);
    hA_cpu = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * x_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * y_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasSyr2StridedBatchedFn(handle,
                                                        uplo,
                                                        N,
                                                        &h_alpha,
                                                        dx,
                                                        incx,
                                                        stridex,
                                                        dy,
                                                        incy,
                                                        stridey,
                                                        dA,
                                                        lda,
                                                        strideA,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hA_host.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasSyr2StridedBatchedFn(handle,
                                                        uplo,
                                                        N,
                                                        d_alpha,
                                                        dx,
                                                        incx,
                                                        stridex,
                                                        dy,
                                                        incy,
                                                        stridey,
                                                        dA,
                                                        lda,
                                                        strideA,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hA_device.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            ref_syr2<T>(uplo,
                        N,
                        h_alpha,
                        hx.data() + b * stridex,
                        incx,
                        hy.data() + b * stridey,
                        incy,
                        hA_cpu.data() + b * strideA,
                        lda);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, lda, strideA, hA_cpu, hA_host);
            unit_check_general<T>(N, N, batch_count, lda, strideA, hA_cpu, hA_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>(
                'F', N, N, lda, strideA, hA_cpu.data(), hA_host.data(), batch_count);
            hipblas_error_device = norm_check_general<T>(
                'F', N, N, lda, strideA, hA_cpu.data(), hA_device.data(), batch_count);
        }
    }

    if(arg.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasSyr2StridedBatchedFn(handle,
                                                            uplo,
                                                            N,
                                                            d_alpha,
                                                            dx,
                                                            incx,
                                                            stridex,
                                                            dy,
                                                            incy,
                                                            stridey,
                                                            dA,
                                                            lda,
                                                            strideA,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasSyr2StridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     syr2_gflop_count<T>(N),
                                                     syr2_gbyte_count<T>(N),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}

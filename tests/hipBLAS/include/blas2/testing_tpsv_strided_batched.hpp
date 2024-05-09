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

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasTpsvStridedBatchedModel
    = ArgumentModel<e_a_type, e_uplo, e_transA, e_diag, e_N, e_incx, e_stride_scale, e_batch_count>;

inline void testname_tpsv_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasTpsvStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_tpsv_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTpsvStridedBatchedFn
        = FORTRAN ? hipblasTpsvStridedBatched<T, true> : hipblasTpsvStridedBatched<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasOperation_t transA      = HIPBLAS_OP_N;
        hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_UPPER;
        hipblasDiagType_t  diag        = HIPBLAS_DIAG_NON_UNIT;
        int64_t            N           = 100;
        int64_t            incx        = 1;
        int64_t            batch_count = 2;
        int64_t            A_size      = N * (N + 1) / 2;
        hipblasStride      strideA     = A_size;
        hipblasStride      stridex     = N * incx;

        device_vector<T> dA(strideA * batch_count);
        device_vector<T> dx(stridex * batch_count);

        EXPECT_HIPBLAS_STATUS(
            hipblasTpsvStridedBatchedFn(
                nullptr, uplo, transA, diag, N, dA, strideA, dx, incx, stridex, batch_count),
            HIPBLAS_STATUS_NOT_INITIALIZED);
        EXPECT_HIPBLAS_STATUS(hipblasTpsvStridedBatchedFn(handle,
                                                          HIPBLAS_FILL_MODE_FULL,
                                                          transA,
                                                          diag,
                                                          N,
                                                          dA,
                                                          strideA,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasTpsvStridedBatchedFn(handle,
                                                          (hipblasFillMode_t)HIPBLAS_OP_N,
                                                          transA,
                                                          diag,
                                                          N,
                                                          dA,
                                                          strideA,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(
            hipblasTpsvStridedBatchedFn(handle,
                                        uplo,
                                        (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                        diag,
                                        N,
                                        dA,
                                        strideA,
                                        dx,
                                        incx,
                                        stridex,
                                        batch_count),
            HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasTpsvStridedBatchedFn(handle,
                                                          uplo,
                                                          transA,
                                                          (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                                                          N,
                                                          dA,
                                                          strideA,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);

        EXPECT_HIPBLAS_STATUS(
            hipblasTpsvStridedBatchedFn(
                handle, uplo, transA, diag, N, nullptr, strideA, dx, incx, stridex, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(
            hipblasTpsvStridedBatchedFn(
                handle, uplo, transA, diag, N, dA, strideA, nullptr, incx, stridex, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);

        // With N == 0, can have all nullptrs
        CHECK_HIPBLAS_ERROR(hipblasTpsvStridedBatchedFn(
            handle, uplo, transA, diag, 0, nullptr, strideA, nullptr, incx, stridex, batch_count));
        CHECK_HIPBLAS_ERROR(hipblasTpsvStridedBatchedFn(
            handle, uplo, transA, diag, N, nullptr, strideA, nullptr, incx, stridex, 0));
    }
}

template <typename T>
void testing_tpsv_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTpsvStridedBatchedFn
        = FORTRAN ? hipblasTpsvStridedBatched<T, true> : hipblasTpsvStridedBatched<T, false>;

    hipblasFillMode_t  uplo         = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(arg.diag);
    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    int                N            = arg.N;
    int                incx         = arg.incx;
    double             stride_scale = arg.stride_scale;
    int                batch_count  = arg.batch_count;

    int           dim_AP   = N * (N + 1) / 2;
    int           abs_incx = incx < 0 ? -incx : incx;
    hipblasStride strideA  = N * N; // only for test setup
    hipblasStride strideAP = dim_AP * stride_scale;
    hipblasStride stridex  = abs_incx * N * stride_scale;
    size_t        size_A   = strideA * batch_count;
    size_t        size_AP  = strideAP * batch_count;
    size_t        size_x   = stridex * batch_count;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        hipblasStatus_t actual = hipblasTpsvStridedBatchedFn(
            handle, uplo, transA, diag, N, nullptr, strideAP, nullptr, incx, stridex, batch_count);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hAP(size_AP);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);
    host_vector<T> hx_or_b_2(size_x);
    host_vector<T> cpu_x_or_b(size_x);

    device_vector<T> dAP(size_AP);
    device_vector<T> dx_or_b(size_x);

    double gpu_time_used, hipblas_error, cumulative_hipblas_error = 0;

    // Initial Data on CPU
    hipblas_init_matrix_type(hipblas_diagonally_dominant_triangular_matrix,
                             (T*)hA,
                             arg,
                             N,
                             N,
                             N,
                             strideA,
                             batch_count,
                             hipblas_client_never_set_nan,
                             true);
    hipblas_init_vector(
        hx, arg, N, abs_incx, stridex, batch_count, hipblas_client_never_set_nan, false, true);
    hb = hx;

    for(int b = 0; b < batch_count; b++)
    {
        T* hAb  = hA.data() + b * strideA;
        T* hAPb = hAP.data() + b * strideAP;
        T* hbb  = hb.data() + b * stridex;

        if(diag == HIPBLAS_DIAG_UNIT)
        {
            make_unit_diagonal(uplo, hAb, N, N);
        }

        // Calculate hb = hA*hx;
        ref_trmv<T>(uplo, transA, diag, N, hAb, N, hbb, incx);

        regular_to_packed(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hAb, (T*)hAPb, N);
    }

    cpu_x_or_b = hb; // cpuXorB <- B
    hx_or_b_1  = hb;
    hx_or_b_2  = hb;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dAP, hAP.data(), sizeof(T) * size_AP, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dx_or_b, hx_or_b_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasTpsvStridedBatchedFn(
            handle, uplo, transA, diag, N, dAP, strideAP, dx_or_b, incx, stridex, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hx_or_b_1.data(), dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        // Calculating error
        // For norm_check/bench, currently taking the cumulative sum of errors over all batches
        for(int b = 0; b < batch_count; b++)
        {
            hipblas_error = hipblas_abs(vector_norm_1<T>(
                N, abs_incx, hx.data() + b * stridex, hx_or_b_1.data() + b * stridex));
            if(arg.unit_check)
            {
                double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * N;
                unit_check_error(hipblas_error, tolerance);
            }

            cumulative_hipblas_error += hipblas_error;
        }
    }
    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTpsvStridedBatchedFn(
                handle, uplo, transA, diag, N, dAP, strideAP, dx_or_b, incx, stridex, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasTpsvStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     tpsv_gflop_count<T>(N),
                                                     tpsv_gbyte_count<T>(N),
                                                     cumulative_hipblas_error);
    }
}

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

using hipblasTrmvBatchedModel
    = ArgumentModel<e_a_type, e_uplo, e_transA, e_diag, e_N, e_lda, e_incx, e_batch_count>;

inline void testname_trmv_batched(const Arguments& arg, std::string& name)
{
    hipblasTrmvBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_trmv_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrmvBatchedFn
        = FORTRAN ? hipblasTrmvBatched<T, true> : hipblasTrmvBatched<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasOperation_t transA      = HIPBLAS_OP_N;
        hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_UPPER;
        hipblasDiagType_t  diag        = HIPBLAS_DIAG_NON_UNIT;
        int64_t            N           = 100;
        int64_t            lda         = 100;
        int64_t            incx        = 1;
        int64_t            batch_count = 2;

        device_batch_vector<T> dA(N * lda, 1, batch_count);
        device_batch_vector<T> dx(N, incx, batch_count);

        EXPECT_HIPBLAS_STATUS(hipblasTrmvBatchedFn(nullptr,
                                                   uplo,
                                                   transA,
                                                   diag,
                                                   N,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_NOT_INITIALIZED);
        EXPECT_HIPBLAS_STATUS(hipblasTrmvBatchedFn(handle,
                                                   HIPBLAS_FILL_MODE_FULL,
                                                   transA,
                                                   diag,
                                                   N,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasTrmvBatchedFn(handle,
                                                   (hipblasFillMode_t)HIPBLAS_OP_N,
                                                   transA,
                                                   diag,
                                                   N,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasTrmvBatchedFn(handle,
                                                   uplo,
                                                   (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                                   diag,
                                                   N,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasTrmvBatchedFn(handle,
                                                   uplo,
                                                   transA,
                                                   (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                                                   N,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);

        EXPECT_HIPBLAS_STATUS(
            hipblasTrmvBatchedFn(
                handle, uplo, transA, diag, N, nullptr, lda, dx.ptr_on_device(), incx, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(
            hipblasTrmvBatchedFn(
                handle, uplo, transA, diag, N, dA.ptr_on_device(), lda, nullptr, incx, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);

        // With N == 0, can have all nullptrs
        CHECK_HIPBLAS_ERROR(hipblasTrmvBatchedFn(
            handle, uplo, transA, diag, 0, nullptr, lda, nullptr, incx, batch_count));
        CHECK_HIPBLAS_ERROR(
            hipblasTrmvBatchedFn(handle, uplo, transA, diag, N, nullptr, lda, nullptr, incx, 0));
    }
}

template <typename T>
void testing_trmv_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrmvBatchedFn
        = FORTRAN ? hipblasTrmvBatched<T, true> : hipblasTrmvBatched<T, false>;

    hipblasFillMode_t  uplo        = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA      = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(arg.diag);
    int                N           = arg.N;
    int                lda         = arg.lda;
    int                incx        = arg.incx;
    int                batch_count = arg.batch_count;

    int    abs_incx = incx >= 0 ? incx : -incx;
    size_t A_size   = size_t(lda) * N;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        hipblasStatus_t actual = hipblasTrmvBatchedFn(
            handle, uplo, transA, diag, N, nullptr, lda, nullptr, incx, batch_count);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return;
    }

    double gpu_time_used, hipblas_error;

    // arrays of pointers-to-host on host
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hres(N, incx, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dx(N, incx, batch_count);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());

    hipblas_init_vector(hA, arg, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hx, arg, hipblas_client_never_set_nan, false, true);
    hres.copy_from(hx);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasTrmvBatchedFn(handle,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 N,
                                                 dA.ptr_on_device(),
                                                 lda,
                                                 dx.ptr_on_device(),
                                                 incx,
                                                 batch_count));

        CHECK_HIP_ERROR(hres.transfer_from(dx));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            ref_trmv<T>(uplo, transA, diag, N, hA[b], lda, hx[b], incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incx, hx, hres);
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, N, abs_incx, hx, hres, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasTrmvBatchedFn(handle,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     N,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     dx.ptr_on_device(),
                                                     incx,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrmvBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              trmv_gflop_count<T>(N),
                                              trmv_gbyte_count<T>(N),
                                              hipblas_error);
    }
}

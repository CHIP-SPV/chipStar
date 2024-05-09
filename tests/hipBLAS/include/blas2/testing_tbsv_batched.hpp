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

using hipblasTbsvBatchedModel
    = ArgumentModel<e_a_type, e_uplo, e_transA, e_diag, e_N, e_K, e_lda, e_incx, e_batch_count>;

inline void testname_tbsv_batched(const Arguments& arg, std::string& name)
{
    hipblasTbsvBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_tbsv_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTbsvBatchedFn
        = FORTRAN ? hipblasTbsvBatched<T, true> : hipblasTbsvBatched<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_UPPER;
        hipblasOperation_t transA      = HIPBLAS_OP_N;
        hipblasDiagType_t  diag        = HIPBLAS_DIAG_NON_UNIT;
        int64_t            N           = 100;
        int64_t            K           = 5;
        int64_t            lda         = 100;
        int64_t            incx        = 1;
        int64_t            batch_count = 2;

        device_batch_vector<T> dA(N * lda, 1, batch_count);
        device_batch_vector<T> dx(N, incx, batch_count);

        EXPECT_HIPBLAS_STATUS(hipblasTbsvBatchedFn(nullptr,
                                                   uplo,
                                                   transA,
                                                   diag,
                                                   N,
                                                   K,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_NOT_INITIALIZED);
        EXPECT_HIPBLAS_STATUS(hipblasTbsvBatchedFn(handle,
                                                   HIPBLAS_FILL_MODE_FULL,
                                                   transA,
                                                   diag,
                                                   N,
                                                   K,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasTbsvBatchedFn(handle,
                                                   (hipblasFillMode_t)HIPBLAS_OP_N,
                                                   transA,
                                                   diag,
                                                   N,
                                                   K,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasTbsvBatchedFn(handle,
                                                   uplo,
                                                   (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                                   diag,
                                                   N,
                                                   K,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasTbsvBatchedFn(handle,
                                                   uplo,
                                                   transA,
                                                   (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                                                   N,
                                                   K,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);

        EXPECT_HIPBLAS_STATUS(hipblasTbsvBatchedFn(handle,
                                                   uplo,
                                                   transA,
                                                   diag,
                                                   N,
                                                   K,
                                                   nullptr,
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasTbsvBatchedFn(handle,
                                                   uplo,
                                                   transA,
                                                   diag,
                                                   N,
                                                   K,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   nullptr,
                                                   incx,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);

        // With N == 0, can have all nullptrs
        CHECK_HIPBLAS_ERROR(hipblasTbsvBatchedFn(
            handle, uplo, transA, diag, 0, K, nullptr, lda, nullptr, incx, batch_count));
        CHECK_HIPBLAS_ERROR(
            hipblasTbsvBatchedFn(handle, uplo, transA, diag, N, K, nullptr, lda, nullptr, incx, 0));
    }
}

template <typename T>
void testing_tbsv_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTbsvBatchedFn
        = FORTRAN ? hipblasTbsvBatched<T, true> : hipblasTbsvBatched<T, false>;

    hipblasFillMode_t  uplo        = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(arg.diag);
    hipblasOperation_t transA      = char2hipblas_operation(arg.transA);
    int                N           = arg.N;
    int                K           = arg.K;
    int                incx        = arg.incx;
    int                lda         = arg.lda;
    int                batch_count = arg.batch_count;

    int    abs_incx = incx < 0 ? -incx : incx;
    size_t size_A   = size_t(N) * N;
    size_t size_AB  = size_t(lda) * N;
    size_t size_x   = abs_incx * size_t(N);

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || K < 0 || lda < K + 1 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        hipblasStatus_t actual = hipblasTbsvBatchedFn(
            handle, uplo, transA, diag, N, K, nullptr, lda, nullptr, incx, batch_count);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(size_A, 1, batch_count);
    host_batch_vector<T> hAB(size_AB, 1, batch_count);
    host_batch_vector<T> hb(N, incx, batch_count);
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hx_or_b(N, incx, batch_count);

    device_batch_vector<T> dAB(size_AB, 1, batch_count);
    device_batch_vector<T> dx_or_b(N, incx, batch_count);

    CHECK_HIP_ERROR(dAB.memcheck());
    CHECK_HIP_ERROR(dx_or_b.memcheck());

    double gpu_time_used, hipblas_error, cumulative_hipblas_error = 0;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true, true);
    hb.copy_from(hx);

    for(int b = 0; b < batch_count; b++)
    {
        hipblas_init_matrix_type(hipblas_diagonally_dominant_triangular_matrix,
                                 (T*)hA[b],
                                 arg,
                                 N,
                                 N,
                                 N,
                                 0,
                                 1,
                                 hipblas_client_never_set_nan,
                                 false);

        banded_matrix_setup(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA[b], N, N, K);

        if(diag == HIPBLAS_DIAG_UNIT)
        {
            make_unit_diagonal(uplo, (T*)hA[b], N, N);
        }

        regular_to_banded(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA[b], N, (T*)hAB[b], lda, N, K);

        // Calculate hb = hA*hx;
        ref_tbmv<T>(uplo, transA, diag, N, K, hAB[b], lda, hb[b], incx);
    }

    hx_or_b.copy_from(hb);

    CHECK_HIP_ERROR(dAB.transfer_from(hAB));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasTbsvBatchedFn(handle,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 N,
                                                 K,
                                                 dAB.ptr_on_device(),
                                                 lda,
                                                 dx_or_b.ptr_on_device(),
                                                 incx,
                                                 batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx_or_b.transfer_from(dx_or_b));

        // Calculating error
        // For norm_check/bench, currently taking the cumulative sum of errors over all batches
        for(int b = 0; b < batch_count; b++)
        {
            hipblas_error = hipblas_abs(vector_norm_1<T>(N, abs_incx, hx[b], hx_or_b[b]));
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

            CHECK_HIPBLAS_ERROR(hipblasTbsvBatchedFn(handle,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     N,
                                                     K,
                                                     dAB.ptr_on_device(),
                                                     lda,
                                                     dx_or_b.ptr_on_device(),
                                                     incx,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasTbsvBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              tbsv_gflop_count<T>(N, K),
                                              tbsv_gbyte_count<T>(N, K),
                                              cumulative_hipblas_error);
    }
}

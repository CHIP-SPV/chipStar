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

using hipblasTrtriBatchedModel = ArgumentModel<e_a_type, e_uplo, e_diag, e_N, e_lda, e_batch_count>;

inline void testname_trtri_batched(const Arguments& arg, std::string& name)
{
    hipblasTrtriBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_trtri_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrtriBatchedFn
        = FORTRAN ? hipblasTrtriBatched<T, true> : hipblasTrtriBatched<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t           N           = 100;
    int64_t           lda         = 102;
    int64_t           batch_count = 2;
    hipblasFillMode_t uplo        = HIPBLAS_FILL_MODE_LOWER;
    hipblasDiagType_t diag        = HIPBLAS_DIAG_NON_UNIT;

    device_batch_vector<T> dA(N * lda, 1, batch_count);
    device_batch_vector<T> dinvA(N * lda, 1, batch_count);

    EXPECT_HIPBLAS_STATUS(hipblasTrtriBatchedFn(nullptr,
                                                uplo,
                                                diag,
                                                N,
                                                dA.ptr_on_device(),
                                                lda,
                                                dinvA.ptr_on_device(),
                                                lda,
                                                batch_count),
                          HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(hipblasTrtriBatchedFn(handle,
                                                HIPBLAS_FILL_MODE_FULL,
                                                diag,
                                                N,
                                                dA.ptr_on_device(),
                                                lda,
                                                dinvA.ptr_on_device(),
                                                lda,
                                                batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasTrtriBatchedFn(handle,
                                                (hipblasFillMode_t)HIPBLAS_OP_N,
                                                diag,
                                                N,
                                                dA.ptr_on_device(),
                                                lda,
                                                dinvA.ptr_on_device(),
                                                lda,
                                                batch_count),
                          HIPBLAS_STATUS_INVALID_ENUM);
    EXPECT_HIPBLAS_STATUS(hipblasTrtriBatchedFn(handle,
                                                uplo,
                                                (hipblasDiagType_t)HIPBLAS_OP_N,
                                                N,
                                                dA.ptr_on_device(),
                                                lda,
                                                dinvA.ptr_on_device(),
                                                lda,
                                                batch_count),
                          HIPBLAS_STATUS_INVALID_ENUM);

    if(arg.bad_arg_all)
    {
        EXPECT_HIPBLAS_STATUS(
            hipblasTrtriBatchedFn(
                handle, uplo, diag, N, nullptr, lda, dinvA.ptr_on_device(), lda, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(
            hipblasTrtriBatchedFn(
                handle, uplo, diag, N, dA.ptr_on_device(), lda, nullptr, lda, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
    }

    // If N == 0, can have nullptrs
    CHECK_HIPBLAS_ERROR(
        hipblasTrtriBatchedFn(handle, uplo, diag, 0, nullptr, lda, nullptr, lda, batch_count));
    CHECK_HIPBLAS_ERROR(
        hipblasTrtriBatchedFn(handle, uplo, diag, N, nullptr, lda, nullptr, lda, 0));
}

template <typename T>
void testing_trtri_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrtriBatchedFn
        = FORTRAN ? hipblasTrtriBatched<T, true> : hipblasTrtriBatched<T, false>;

    const double rel_error = get_epsilon<T>() * 1000;

    hipblasFillMode_t uplo        = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t diag        = char2hipblas_diagonal(arg.diag);
    int               N           = arg.N;
    int               lda         = arg.lda;
    int               batch_count = arg.batch_count;

    int    ldinvA = lda;
    size_t A_size = size_t(lda) * N;

    // check here to prevent undefined memory allocation error
    if(N < 0 || lda < 0 || lda < N || batch_count < 0)
    {
        return;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hB(A_size, 1, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dinvA(A_size, 1, batch_count);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dinvA.memcheck());

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(arg);

    hipblas_init(hA, true);

    for(int b = 0; b < batch_count; b++)
    {
        // proprocess the matrix to avoid ill-conditioned matrix
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                hA[b][i + j * lda] *= 0.01;

                if(j % 2)
                    hA[b][i + j * lda] *= -1;
                if(uplo == HIPBLAS_FILL_MODE_LOWER && j > i)
                    hA[b][i + j * lda] = 0.0f;
                else if(uplo == HIPBLAS_FILL_MODE_UPPER && j < i)
                    hA[b][i + j * lda] = 0.0f;
                if(i == j)
                {
                    if(diag == HIPBLAS_DIAG_UNIT)
                        hA[b][i + j * lda] = 1.0;
                    else
                        hA[b][i + j * lda] *= 100.0;
                }
            }
        }
    }

    hB.copy_from(hA);
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dinvA.transfer_from(hA));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasTrtriBatchedFn(handle,
                                                  uplo,
                                                  diag,
                                                  N,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dinvA.ptr_on_device(),
                                                  ldinvA,
                                                  batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hA.transfer_from(dinvA));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            ref_trtri<T>(arg.uplo, arg.diag, N, hB[b], lda);
        }

        if(arg.unit_check)
        {
            for(int b = 0; b < batch_count; b++)
                near_check_general<T>(N, N, lda, hB[b], hA[b], rel_error);
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', N, N, lda, hB, hA, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasTrtriBatchedFn(handle,
                                                      uplo,
                                                      diag,
                                                      N,
                                                      dA.ptr_on_device(),
                                                      lda,
                                                      dinvA.ptr_on_device(),
                                                      ldinvA,
                                                      batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrtriBatchedModel{}.log_args<T>(std::cout,
                                               arg,
                                               gpu_time_used,
                                               trtri_gflop_count<T>(N),
                                               trtri_gbyte_count<T>(N),
                                               hipblas_error);
    }
}

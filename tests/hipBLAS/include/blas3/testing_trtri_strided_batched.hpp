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

using hipblasTrtriStridedBatchedModel
    = ArgumentModel<e_a_type, e_uplo, e_diag, e_N, e_lda, e_stride_scale, e_batch_count>;

inline void testname_trtri_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasTrtriStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_trtri_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrtriStridedBatchedFn
        = FORTRAN ? hipblasTrtriStridedBatched<T, true> : hipblasTrtriStridedBatched<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t           N           = 100;
    int64_t           lda         = 102;
    int64_t           batch_count = 2;
    hipblasFillMode_t uplo        = HIPBLAS_FILL_MODE_LOWER;
    hipblasDiagType_t diag        = HIPBLAS_DIAG_NON_UNIT;

    hipblasStride    strideA    = N * lda;
    hipblasStride    strideinvA = N * lda;
    device_vector<T> dA(strideA * batch_count);
    device_vector<T> dinvA(strideinvA * batch_count);

    EXPECT_HIPBLAS_STATUS(
        hipblasTrtriStridedBatchedFn(
            nullptr, uplo, diag, N, dA, lda, strideA, dinvA, lda, strideinvA, batch_count),
        HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(hipblasTrtriStridedBatchedFn(handle,
                                                       HIPBLAS_FILL_MODE_FULL,
                                                       diag,
                                                       N,
                                                       dA,
                                                       lda,
                                                       strideA,
                                                       dinvA,
                                                       lda,
                                                       strideinvA,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasTrtriStridedBatchedFn(handle,
                                                       (hipblasFillMode_t)HIPBLAS_OP_N,
                                                       diag,
                                                       N,
                                                       dA,
                                                       lda,
                                                       strideA,
                                                       dinvA,
                                                       lda,
                                                       strideinvA,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_ENUM);
    EXPECT_HIPBLAS_STATUS(hipblasTrtriStridedBatchedFn(handle,
                                                       uplo,
                                                       (hipblasDiagType_t)HIPBLAS_OP_N,
                                                       N,
                                                       dA,
                                                       lda,
                                                       strideA,
                                                       dinvA,
                                                       lda,
                                                       strideinvA,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_ENUM);

    if(arg.bad_arg_all)
    {
        EXPECT_HIPBLAS_STATUS(
            hipblasTrtriStridedBatchedFn(
                handle, uplo, diag, N, nullptr, lda, strideA, dinvA, lda, strideinvA, batch_count),
            HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasTrtriStridedBatchedFn(handle,
                                                           uplo,
                                                           diag,
                                                           N,
                                                           nullptr,
                                                           lda,
                                                           strideA,
                                                           nullptr,
                                                           lda,
                                                           strideinvA,
                                                           batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
    }

    // If N == 0, can have nullptrs
    CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatchedFn(
        handle, uplo, diag, 0, nullptr, lda, strideA, nullptr, lda, strideinvA, batch_count));
    CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatchedFn(
        handle, uplo, diag, N, nullptr, lda, strideA, nullptr, lda, strideinvA, 0));
}

template <typename T>
void testing_trtri_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrtriStridedBatchedFn
        = FORTRAN ? hipblasTrtriStridedBatched<T, true> : hipblasTrtriStridedBatched<T, false>;

    const double rel_error = get_epsilon<T>() * 1000;

    hipblasFillMode_t uplo         = char2hipblas_fill(arg.uplo);
    hipblasDiagType_t diag         = char2hipblas_diagonal(arg.diag);
    int               N            = arg.N;
    int               lda          = arg.lda;
    double            stride_scale = arg.stride_scale;
    int               batch_count  = arg.batch_count;

    int           ldinvA  = lda;
    hipblasStride strideA = size_t(lda) * N * stride_scale;
    size_t        A_size  = strideA * batch_count;

    // check here to prevent undefined memory allocation error
    if(N < 0 || lda < 0 || lda < N || batch_count < 0)
    {
        return;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(A_size);

    device_vector<T> dA(A_size);
    device_vector<T> dinvA(A_size);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(arg);

    srand(1);
    hipblas_init_symmetric<T>(hA, N, lda, strideA, batch_count);
    for(int b = 0; b < batch_count; b++)
    {
        T* hAb = hA.data() + b * strideA;

        // proprocess the matrix to avoid ill-conditioned matrix
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                hAb[i + j * lda] *= 0.01;

                if(j % 2)
                    hAb[i + j * lda] *= -1;
                if(uplo == HIPBLAS_FILL_MODE_LOWER && j > i)
                    hAb[i + j * lda] = 0.0f;
                else if(uplo == HIPBLAS_FILL_MODE_UPPER && j < i)
                    hAb[i + j * lda] = 0.0f;
                if(i == j)
                {
                    if(diag == HIPBLAS_DIAG_UNIT)
                        hAb[i + j * lda] = 1.0;
                    else
                        hAb[i + j * lda] *= 100.0;
                }
            }
        }
    }

    hB = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dinvA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatchedFn(
            handle, uplo, diag, N, dA, lda, strideA, dinvA, ldinvA, strideA, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hA, dinvA, sizeof(T) * A_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            ref_trtri<T>(arg.uplo, arg.diag, N, hB.data() + b * strideA, lda);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            near_check_general<T>(N, N, batch_count, lda, strideA, hB, hA, rel_error);
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', N, N, lda, strideA, hB, hA, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatchedFn(
                handle, uplo, diag, N, dA, lda, strideA, dinvA, ldinvA, strideA, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrtriStridedBatchedModel{}.log_args<T>(std::cout,
                                                      arg,
                                                      gpu_time_used,
                                                      trtri_gflop_count<T>(N),
                                                      trtri_gbyte_count<T>(N),
                                                      hipblas_error);
    }
}

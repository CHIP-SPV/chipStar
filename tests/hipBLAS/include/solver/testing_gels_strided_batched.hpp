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

#include "gtest/gtest.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using hipblasGelsStridedBatchedModel
    = ArgumentModel<e_a_type, e_transA, e_M, e_N, e_lda, e_ldb, e_stride_scale, e_batch_count>;

inline void testname_gels_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasGelsStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_gels_strided_batched_bad_arg(const Arguments& arg)
{
    auto hipblasGelsStridedBatchedFn = arg.api == hipblas_client_api::FORTRAN
                                           ? hipblasGelsStridedBatched<T, true>
                                           : hipblasGelsStridedBatched<T, false>;

    hipblasLocalHandle       handle(arg);
    const int                M          = 100;
    const int                N          = 101;
    const int                nrhs       = 10;
    const int                lda        = 102;
    const int                ldb        = 103;
    const int                batchCount = 2;
    const hipblasOperation_t opN        = HIPBLAS_OP_N;
    const hipblasOperation_t opBad      = is_complex<T> ? HIPBLAS_OP_T : HIPBLAS_OP_C;

    const hipblasStride strideA = size_t(lda) * N;
    const hipblasStride strideB = size_t(ldb) * nrhs;
    const size_t        A_size  = strideA * batchCount;
    const size_t        B_size  = strideB * batchCount;

    device_vector<T>   dA(A_size);
    device_vector<T>   dB(B_size);
    device_vector<int> dInfo(batchCount);
    int                info = 0;
    int                expectedInfo;

    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opN,
                                                      M,
                                                      N,
                                                      nrhs,
                                                      dA,
                                                      lda,
                                                      strideA,
                                                      dB,
                                                      ldb,
                                                      strideB,
                                                      nullptr,
                                                      dInfo,
                                                      batchCount),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opBad,
                                                      M,
                                                      N,
                                                      nrhs,
                                                      dA,
                                                      lda,
                                                      strideA,
                                                      dB,
                                                      ldb,
                                                      strideB,
                                                      &info,
                                                      dInfo,
                                                      batchCount),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -1;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsStridedBatchedFn(
            handle, opN, -1, N, nrhs, dA, lda, strideA, dB, ldb, strideB, &info, dInfo, batchCount),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -2;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsStridedBatchedFn(
            handle, opN, M, -1, nrhs, dA, lda, strideA, dB, ldb, strideB, &info, dInfo, batchCount),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -3;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsStridedBatchedFn(
            handle, opN, M, N, -1, dA, lda, strideA, dB, ldb, strideB, &info, dInfo, batchCount),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -4;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opN,
                                                      M,
                                                      N,
                                                      nrhs,
                                                      nullptr,
                                                      lda,
                                                      strideA,
                                                      dB,
                                                      ldb,
                                                      strideB,
                                                      &info,
                                                      dInfo,
                                                      batchCount),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -5;
    unit_check_general(1, 1, 1, &expectedInfo, &info);
    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opN,
                                                      M,
                                                      N,
                                                      nrhs,
                                                      dA,
                                                      M - 1,
                                                      strideA,
                                                      dB,
                                                      ldb,
                                                      strideB,
                                                      &info,
                                                      dInfo,
                                                      batchCount),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -6;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opN,
                                                      M,
                                                      N,
                                                      nrhs,
                                                      dA,
                                                      lda,
                                                      strideA,
                                                      nullptr,
                                                      ldb,
                                                      strideB,
                                                      &info,
                                                      dInfo,
                                                      batchCount),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -8;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // Explicit values to check for ldb < M and ldb < N
    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opN,
                                                      100,
                                                      200,
                                                      nrhs,
                                                      dA,
                                                      lda,
                                                      strideA,
                                                      dB,
                                                      199,
                                                      strideB,
                                                      &info,
                                                      dInfo,
                                                      batchCount),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -9;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opN,
                                                      200,
                                                      100,
                                                      nrhs,
                                                      dA,
                                                      201,
                                                      strideA,
                                                      dB,
                                                      199,
                                                      strideB,
                                                      &info,
                                                      dInfo,
                                                      batchCount),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -9;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opN,
                                                      M,
                                                      N,
                                                      nrhs,
                                                      dA,
                                                      lda,
                                                      strideA,
                                                      dB,
                                                      ldb,
                                                      strideB,
                                                      &info,
                                                      nullptr,
                                                      batchCount),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -12;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsStridedBatchedFn(
            handle, opN, M, N, nrhs, dA, lda, strideA, dB, ldb, strideB, &info, dInfo, -1),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -13;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // If M == 0 || N == 0, A can be nullptr
    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opN,
                                                      0,
                                                      N,
                                                      nrhs,
                                                      nullptr,
                                                      lda,
                                                      strideA,
                                                      dB,
                                                      ldb,
                                                      strideB,
                                                      &info,
                                                      dInfo,
                                                      batchCount),
                          HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opN,
                                                      M,
                                                      0,
                                                      nrhs,
                                                      nullptr,
                                                      lda,
                                                      strideA,
                                                      dB,
                                                      ldb,
                                                      strideB,
                                                      &info,
                                                      dInfo,
                                                      batchCount),
                          HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // if nrhs == 0, B can be nullptr
    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opN,
                                                      M,
                                                      N,
                                                      0,
                                                      dA,
                                                      lda,
                                                      strideA,
                                                      nullptr,
                                                      ldb,
                                                      strideB,
                                                      &info,
                                                      dInfo,
                                                      batchCount),
                          HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // if M == 0 && N == 0, B can be nullptr
    EXPECT_HIPBLAS_STATUS(hipblasGelsStridedBatchedFn(handle,
                                                      opN,
                                                      0,
                                                      0,
                                                      nrhs,
                                                      dA,
                                                      lda,
                                                      strideA,
                                                      nullptr,
                                                      ldb,
                                                      strideB,
                                                      &info,
                                                      dInfo,
                                                      batchCount),
                          HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // if batchCount == 0, dInfo can be nullptr
    EXPECT_HIPBLAS_STATUS(
        hipblasGelsStridedBatchedFn(
            handle, opN, M, N, nrhs, dA, lda, strideA, dB, ldb, strideB, &info, nullptr, 0),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);
}

template <typename T>
void testing_gels_strided_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGelsStridedBatchedFn
        = FORTRAN ? hipblasGelsStridedBatched<T, true> : hipblasGelsStridedBatched<T, false>;

    char   transc      = arg.transA;
    int    N           = arg.N;
    int    M           = arg.M;
    int    nrhs        = arg.K;
    int    lda         = arg.lda;
    int    ldb         = arg.ldb;
    double strideScale = arg.stride_scale;
    int    batchCount  = arg.batch_count;

    if(is_complex<T> && transc == 'T')
        transc = 'C';
    else if(!is_complex<T> && transc == 'C')
        transc = 'T';

    // this makes logging incorrect as overriding arg
    hipblasOperation_t trans = char2hipblas_operation(transc);

    hipblasStride strideA = size_t(lda) * N * strideScale;
    hipblasStride strideB = size_t(ldb) * nrhs * strideScale;
    size_t        A_size  = strideA * batchCount;
    size_t        B_size  = strideB * batchCount;

    // Check to prevent memory allocation error
    if(M < 0 || N < 0 || nrhs < 0 || lda < M || ldb < M || ldb < N || batchCount < 0)
    {
        return;
    }
    if(batchCount == 0)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T>   hA(A_size);
    host_vector<T>   hB(B_size);
    host_vector<T>   hB_res(B_size);
    host_vector<int> info_res(batchCount);
    host_vector<int> info(batchCount);
    int              info_input(-1);

    device_vector<T>   dA(A_size);
    device_vector<T>   dB(B_size);
    device_vector<int> dInfo(batchCount);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(arg);

    // Initial hA, hB, hX on CPU
    srand(1);
    hipblas_init<T>(hA, true);
    hipblas_init<T>(hB);
    for(int b = 0; b < batchCount; b++)
    {
        T* hAb = hA.data() + b * strideA;

        // scale A to avoid singularities
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    hAb[i + j * lda] += 400;
                else
                    hAb[i + j * lda] -= 4;
            }
        }
    }

    hB_res = hB;

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, B_size * sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasGelsStridedBatchedFn(handle,
                                                        trans,
                                                        M,
                                                        N,
                                                        nrhs,
                                                        dA,
                                                        lda,
                                                        strideA,
                                                        dB,
                                                        ldb,
                                                        strideB,
                                                        &info_input,
                                                        dInfo,
                                                        batchCount));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hB_res, dB, B_size * sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(info_res.data(), dInfo, sizeof(int) * batchCount, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */
        int            sizeW = std::max(1, std::min(M, N) + std::max(std::min(M, N), nrhs));
        host_vector<T> hW(sizeW);

        for(int b = 0; b < batchCount; b++)
        {
            info[b] = ref_gels(transc,
                               M,
                               N,
                               nrhs,
                               hA.data() + b * strideA,
                               lda,
                               hB.data() + b * strideB,
                               ldb,
                               hW.data(),
                               sizeW);
        }

        hipblas_error = norm_check_general<T>(
            'F', std::max(M, N), nrhs, ldb, strideB, hB.data(), hB_res.data(), batchCount);

        if(info_input != 0)
            hipblas_error += 1.0;
        for(int b = 0; b < batchCount; b++)
        {
            if(info[b] != info_res[b])
                hipblas_error += 1.0;
        }

        if(arg.unit_check)
        {
            double eps       = std::numeric_limits<U>::epsilon();
            double tolerance = N * eps * 100;
            int    zero      = 0;

            unit_check_error(hipblas_error, tolerance);
            unit_check_general(1, 1, 1, &zero, &info_input);
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

            CHECK_HIPBLAS_ERROR(hipblasGelsStridedBatchedFn(handle,
                                                            trans,
                                                            M,
                                                            N,
                                                            nrhs,
                                                            dA,
                                                            lda,
                                                            strideA,
                                                            dB,
                                                            ldb,
                                                            strideB,
                                                            &info_input,
                                                            dInfo,
                                                            batchCount));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGelsStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     ArgumentLogging::NA_value,
                                                     ArgumentLogging::NA_value,
                                                     hipblas_error);
    }
}

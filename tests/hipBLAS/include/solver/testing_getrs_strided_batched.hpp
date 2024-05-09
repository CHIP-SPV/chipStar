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

using hipblasGetrsStridedBatchedModel
    = ArgumentModel<e_a_type, e_N, e_lda, e_ldb, e_stride_scale, e_batch_count>;

inline void testname_getrs_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasGetrsStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void setup_getrs_strided_batched_testing(host_vector<T>&     hA,
                                         host_vector<T>&     hB,
                                         host_vector<T>&     hX,
                                         host_vector<int>&   hIpiv,
                                         device_vector<T>&   dA,
                                         device_vector<T>&   dB,
                                         device_vector<int>& dIpiv,
                                         int                 N,
                                         int                 lda,
                                         int                 ldb,
                                         hipblasStride       strideA,
                                         hipblasStride       strideB,
                                         hipblasStride       strideP,
                                         int                 batch_count)
{
    size_t A_size    = strideA * batch_count;
    size_t B_size    = strideB * batch_count;
    size_t Ipiv_size = strideP * batch_count;

    // Initial hA, hB, hX on CPU
    srand(1);
    hipblasOperation_t op = HIPBLAS_OP_N;
    for(int b = 0; b < batch_count; b++)
    {
        T*   hAb    = hA.data() + b * strideA;
        T*   hXb    = hX.data() + b * strideB;
        T*   hBb    = hB.data() + b * strideB;
        int* hIpivb = hIpiv.data() + b * strideP;

        hipblas_init<T>(hAb, N, N, lda);
        hipblas_init<T>(hXb, N, 1, ldb);

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

        // Calculate hB = hA*hX;
        ref_gemm<T>(op, op, N, 1, N, (T)1, hAb, lda, hXb, ldb, (T)0, hBb, ldb);

        // LU factorize hA on the CPU
        int info = ref_getrf<T>(N, N, hAb, lda, hIpivb);
        if(info != 0)
        {
            std::cerr << "LU decomposition failed" << std::endl;
            int expectedInfo = 0;
            unit_check_general(1, 1, 1, &expectedInfo, &info);
        }
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, B_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv, Ipiv_size * sizeof(int), hipMemcpyHostToDevice));
}

template <typename T>
void testing_getrs_strided_batched_bad_arg(const Arguments& arg)
{
    auto hipblasGetrsStridedBatchedFn = arg.api == hipblas_client_api::FORTRAN
                                            ? hipblasGetrsStridedBatched<T, true>
                                            : hipblasGetrsStridedBatched<T, false>;

    hipblasLocalHandle handle(arg);
    const int          N           = 100;
    const int          nrhs        = 1;
    const int          lda         = 101;
    const int          ldb         = 102;
    const int          batch_count = 2;
    hipblasStride      strideA     = size_t(lda) * N;
    hipblasStride      strideB     = size_t(ldb) * 1;
    hipblasStride      strideP     = size_t(N);
    size_t             A_size      = strideA * batch_count;
    size_t             B_size      = strideB * batch_count;
    size_t             Ipiv_size   = strideP * batch_count;

    const hipblasOperation_t op = HIPBLAS_OP_N;

    host_vector<T>   hA(A_size);
    host_vector<T>   hB(B_size);
    host_vector<T>   hX(B_size);
    host_vector<int> hIpiv(Ipiv_size);

    device_vector<T>   dA(A_size);
    device_vector<T>   dB(B_size);
    device_vector<int> dIpiv(Ipiv_size);
    int                info = 0;
    int                expectedInfo;

    // Need initialization code because even with bad params we call roc/cu-solver
    // so want to give reasonable data

    setup_getrs_strided_batched_testing(
        hA, hB, hX, hIpiv, dA, dB, dIpiv, N, lda, ldb, strideA, strideB, strideP, batch_count);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsStridedBatchedFn(handle,
                                                       op,
                                                       N,
                                                       nrhs,
                                                       dA,
                                                       lda,
                                                       strideA,
                                                       dIpiv,
                                                       strideP,
                                                       dB,
                                                       ldb,
                                                       strideB,
                                                       nullptr,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsStridedBatchedFn(handle,
                                                       op,
                                                       -1,
                                                       nrhs,
                                                       dA,
                                                       lda,
                                                       strideA,
                                                       dIpiv,
                                                       strideP,
                                                       dB,
                                                       ldb,
                                                       strideB,
                                                       &info,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -2;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsStridedBatchedFn(handle,
                                                       op,
                                                       N,
                                                       -1,
                                                       dA,
                                                       lda,
                                                       strideA,
                                                       dIpiv,
                                                       strideP,
                                                       dB,
                                                       ldb,
                                                       strideB,
                                                       &info,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -3;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsStridedBatchedFn(handle,
                                                       op,
                                                       N,
                                                       nrhs,
                                                       nullptr,
                                                       lda,
                                                       strideA,
                                                       dIpiv,
                                                       strideP,
                                                       dB,
                                                       ldb,
                                                       strideB,
                                                       &info,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -4;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsStridedBatchedFn(handle,
                                                       op,
                                                       N,
                                                       nrhs,
                                                       dA,
                                                       N - 1,
                                                       strideA,
                                                       dIpiv,
                                                       strideP,
                                                       dB,
                                                       ldb,
                                                       strideB,
                                                       &info,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -5;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsStridedBatchedFn(handle,
                                                       op,
                                                       N,
                                                       nrhs,
                                                       dA,
                                                       lda,
                                                       strideA,
                                                       nullptr,
                                                       strideP,
                                                       dB,
                                                       ldb,
                                                       strideB,
                                                       &info,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -7;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsStridedBatchedFn(handle,
                                                       op,
                                                       N,
                                                       nrhs,
                                                       dA,
                                                       lda,
                                                       strideA,
                                                       dIpiv,
                                                       strideP,
                                                       nullptr,
                                                       ldb,
                                                       strideB,
                                                       &info,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -9;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGetrsStridedBatchedFn(handle,
                                                       op,
                                                       N,
                                                       nrhs,
                                                       dA,
                                                       lda,
                                                       strideA,
                                                       dIpiv,
                                                       strideP,
                                                       dB,
                                                       N - 1,
                                                       strideB,
                                                       &info,
                                                       batch_count),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -10;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGetrsStridedBatchedFn(
            handle, op, N, nrhs, dA, lda, strideA, dIpiv, strideP, dB, ldb, strideB, &info, -1),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -13;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // If N == 0, A, B, and ipiv can be nullptr
    EXPECT_HIPBLAS_STATUS(hipblasGetrsStridedBatchedFn(handle,
                                                       op,
                                                       0,
                                                       nrhs,
                                                       nullptr,
                                                       lda,
                                                       strideA,
                                                       nullptr,
                                                       strideP,
                                                       nullptr,
                                                       ldb,
                                                       strideB,
                                                       &info,
                                                       batch_count),
                          HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // if nrhs == 0, B can be nullptr
    EXPECT_HIPBLAS_STATUS(hipblasGetrsStridedBatchedFn(handle,
                                                       op,
                                                       N,
                                                       0,
                                                       dA,
                                                       lda,
                                                       strideA,
                                                       dIpiv,
                                                       strideP,
                                                       nullptr,
                                                       ldb,
                                                       strideB,
                                                       &info,
                                                       batch_count),
                          HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // can't make any assumptions about ptrs when batch_count < 0, this is handled by rocSOLVER
}

template <typename T>
void testing_getrs_strided_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGetrsStridedBatchedFn
        = FORTRAN ? hipblasGetrsStridedBatched<T, true> : hipblasGetrsStridedBatched<T, false>;

    int    N            = arg.N;
    int    lda          = arg.lda;
    int    ldb          = arg.ldb;
    double stride_scale = arg.stride_scale;
    int    batch_count  = arg.batch_count;

    hipblasStride strideA   = size_t(lda) * N * stride_scale;
    hipblasStride strideB   = size_t(ldb) * 1 * stride_scale;
    hipblasStride strideP   = size_t(N) * stride_scale;
    size_t        A_size    = strideA * batch_count;
    size_t        B_size    = strideB * batch_count;
    size_t        Ipiv_size = strideP * batch_count;

    // Check to prevent memory allocation error
    if(N < 0 || lda < N || ldb < N || batch_count < 0)
    {
        return;
    }
    if(batch_count == 0)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T>   hA(A_size);
    host_vector<T>   hX(B_size);
    host_vector<T>   hB(B_size);
    host_vector<T>   hB1(B_size);
    host_vector<int> hIpiv(Ipiv_size);
    host_vector<int> hIpiv1(Ipiv_size);
    int              info;

    device_vector<T>   dA(A_size);
    device_vector<T>   dB(B_size);
    device_vector<int> dIpiv(Ipiv_size);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(arg);
    hipblasOperation_t op = HIPBLAS_OP_N;

    setup_getrs_strided_batched_testing(
        hA, hB, hX, hIpiv, dA, dB, dIpiv, N, lda, ldb, strideA, strideB, strideP, batch_count);

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasGetrsStridedBatchedFn(handle,
                                                         op,
                                                         N,
                                                         1,
                                                         dA,
                                                         lda,
                                                         strideA,
                                                         dIpiv,
                                                         strideP,
                                                         dB,
                                                         ldb,
                                                         strideB,
                                                         &info,
                                                         batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hB1.data(), dB, B_size * sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            ref_getrs('N',
                      N,
                      1,
                      hA.data() + b * strideA,
                      lda,
                      hIpiv.data() + b * strideP,
                      hB.data() + b * strideB,
                      ldb);
        }

        hipblas_error = norm_check_general<T>('F', N, 1, ldb, strideB, hB, hB1, batch_count);

        if(arg.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = N * eps * 100;
            int    zero      = 0;

            unit_check_error(hipblas_error, tolerance);
            unit_check_general(1, 1, 1, &zero, &info);
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

            CHECK_HIPBLAS_ERROR(hipblasGetrsStridedBatchedFn(handle,
                                                             op,
                                                             N,
                                                             1,
                                                             dA,
                                                             lda,
                                                             strideA,
                                                             dIpiv,
                                                             strideP,
                                                             dB,
                                                             ldb,
                                                             strideB,
                                                             &info,
                                                             batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGetrsStridedBatchedModel{}.log_args<T>(std::cout,
                                                      arg,
                                                      gpu_time_used,
                                                      getrs_gflop_count<T>(N, 1),
                                                      ArgumentLogging::NA_value,
                                                      hipblas_error);
    }
}

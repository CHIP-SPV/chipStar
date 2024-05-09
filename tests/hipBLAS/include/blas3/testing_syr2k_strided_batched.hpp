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

using hipblasSyr2kStridedBatchedModel = ArgumentModel<e_a_type,
                                                      e_uplo,
                                                      e_transA,
                                                      e_N,
                                                      e_K,
                                                      e_alpha,
                                                      e_lda,
                                                      e_ldb,
                                                      e_beta,
                                                      e_ldc,
                                                      e_stride_scale,
                                                      e_batch_count>;

inline void testname_syr2k_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasSyr2kStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_syr2k_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSyrk2StridedBatchedFn
        = FORTRAN ? hipblasSyr2kStridedBatched<T, true> : hipblasSyr2kStridedBatched<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t            N           = 101;
    int64_t            K           = 100;
    int64_t            lda         = 102;
    int64_t            ldb         = 103;
    int64_t            ldc         = 104;
    int64_t            batch_count = 2;
    hipblasOperation_t transA      = HIPBLAS_OP_N;
    hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_LOWER;

    int64_t cols = transA == HIPBLAS_OP_N ? K : N;

    hipblasStride strideA = cols * lda;
    hipblasStride strideB = cols * ldb;
    hipblasStride strideC = N * ldc;

    device_vector<T> dA(strideA * batch_count);
    device_vector<T> dB(strideB * batch_count);
    device_vector<T> dC(strideC * batch_count);

    device_vector<T> d_alpha(1), d_zero(1), d_beta(1), d_one(1);
    const T          h_alpha(1), h_zero(0), h_beta(2), h_one(1);

    const T* alpha = &h_alpha;
    const T* beta  = &h_beta;
    const T* one   = &h_one;
    const T* zero  = &h_zero;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_beta, beta, sizeof(*beta), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_one, one, sizeof(*one), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            beta  = d_beta;
            one   = d_one;
            zero  = d_zero;
        }

        EXPECT_HIPBLAS_STATUS(hipblasSyrk2StridedBatchedFn(nullptr,
                                                           uplo,
                                                           transA,
                                                           N,
                                                           K,
                                                           alpha,
                                                           dA,
                                                           lda,
                                                           strideA,
                                                           dB,
                                                           ldb,
                                                           strideB,
                                                           beta,
                                                           dC,
                                                           ldc,
                                                           strideC,
                                                           batch_count),
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(hipblasSyrk2StridedBatchedFn(handle,
                                                           HIPBLAS_FILL_MODE_FULL,
                                                           transA,
                                                           N,
                                                           K,
                                                           alpha,
                                                           dA,
                                                           lda,
                                                           strideA,
                                                           dB,
                                                           ldb,
                                                           strideB,
                                                           beta,
                                                           dC,
                                                           ldc,
                                                           strideC,
                                                           batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasSyrk2StridedBatchedFn(handle,
                                                           (hipblasFillMode_t)HIPBLAS_OP_N,
                                                           transA,
                                                           N,
                                                           K,
                                                           alpha,
                                                           dA,
                                                           lda,
                                                           strideA,
                                                           dB,
                                                           ldb,
                                                           strideB,
                                                           beta,
                                                           dC,
                                                           ldc,
                                                           strideC,
                                                           batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(
            hipblasSyrk2StridedBatchedFn(handle,
                                         uplo,
                                         (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                         N,
                                         K,
                                         alpha,
                                         dA,
                                         lda,
                                         strideA,
                                         dB,
                                         ldb,
                                         strideB,
                                         beta,
                                         dC,
                                         ldc,
                                         strideC,
                                         batch_count),
            HIPBLAS_STATUS_INVALID_ENUM);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(hipblasSyrk2StridedBatchedFn(handle,
                                                               uplo,
                                                               transA,
                                                               N,
                                                               K,
                                                               nullptr,
                                                               dA,
                                                               lda,
                                                               strideA,
                                                               dB,
                                                               ldb,
                                                               strideB,
                                                               beta,
                                                               dC,
                                                               ldc,
                                                               strideC,
                                                               batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasSyrk2StridedBatchedFn(handle,
                                                               uplo,
                                                               transA,
                                                               N,
                                                               K,
                                                               alpha,
                                                               dA,
                                                               lda,
                                                               strideA,
                                                               dB,
                                                               ldb,
                                                               strideB,
                                                               nullptr,
                                                               dC,
                                                               ldc,
                                                               strideC,
                                                               batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                EXPECT_HIPBLAS_STATUS(hipblasSyrk2StridedBatchedFn(handle,
                                                                   uplo,
                                                                   transA,
                                                                   N,
                                                                   K,
                                                                   alpha,
                                                                   nullptr,
                                                                   lda,
                                                                   strideA,
                                                                   dB,
                                                                   ldb,
                                                                   strideB,
                                                                   beta,
                                                                   dC,
                                                                   ldc,
                                                                   strideC,
                                                                   batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasSyrk2StridedBatchedFn(handle,
                                                                   uplo,
                                                                   transA,
                                                                   N,
                                                                   K,
                                                                   alpha,
                                                                   dA,
                                                                   lda,
                                                                   strideA,
                                                                   nullptr,
                                                                   ldb,
                                                                   strideB,
                                                                   beta,
                                                                   dC,
                                                                   ldc,
                                                                   strideC,
                                                                   batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasSyrk2StridedBatchedFn(handle,
                                                                   uplo,
                                                                   transA,
                                                                   N,
                                                                   K,
                                                                   alpha,
                                                                   dA,
                                                                   lda,
                                                                   strideA,
                                                                   dB,
                                                                   ldb,
                                                                   strideB,
                                                                   beta,
                                                                   nullptr,
                                                                   ldc,
                                                                   strideC,
                                                                   batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
            }

            // If k == 0 && beta == 1, A, B, C may be nullptr
            CHECK_HIPBLAS_ERROR(hipblasSyrk2StridedBatchedFn(handle,
                                                             uplo,
                                                             transA,
                                                             N,
                                                             0,
                                                             alpha,
                                                             nullptr,
                                                             lda,
                                                             strideA,
                                                             nullptr,
                                                             ldb,
                                                             strideB,
                                                             one,
                                                             nullptr,
                                                             ldc,
                                                             strideC,
                                                             batch_count));

            // If alpha == 0 && beta == 1, A, B, C may be nullptr
            CHECK_HIPBLAS_ERROR(hipblasSyrk2StridedBatchedFn(handle,
                                                             uplo,
                                                             transA,
                                                             N,
                                                             K,
                                                             zero,
                                                             nullptr,
                                                             lda,
                                                             strideA,
                                                             nullptr,
                                                             ldb,
                                                             strideB,
                                                             one,
                                                             nullptr,
                                                             ldc,
                                                             strideC,
                                                             batch_count));
        }

        // If N == 0 || batch_count, can have nullptrs
        CHECK_HIPBLAS_ERROR(hipblasSyrk2StridedBatchedFn(handle,
                                                         uplo,
                                                         transA,
                                                         0,
                                                         K,
                                                         nullptr,
                                                         nullptr,
                                                         lda,
                                                         strideA,
                                                         nullptr,
                                                         ldb,
                                                         strideB,
                                                         nullptr,
                                                         nullptr,
                                                         ldc,
                                                         strideC,
                                                         batch_count));
        CHECK_HIPBLAS_ERROR(hipblasSyrk2StridedBatchedFn(handle,
                                                         uplo,
                                                         transA,
                                                         0,
                                                         K,
                                                         nullptr,
                                                         nullptr,
                                                         lda,
                                                         strideA,
                                                         nullptr,
                                                         ldb,
                                                         strideB,
                                                         nullptr,
                                                         nullptr,
                                                         ldc,
                                                         strideC,
                                                         0));
    }
}

template <typename T>
void testing_syr2k_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSyrk2StridedBatchedFn
        = FORTRAN ? hipblasSyr2kStridedBatched<T, true> : hipblasSyr2kStridedBatched<T, false>;

    hipblasFillMode_t  uplo         = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    int                N            = arg.N;
    int                K            = arg.K;
    int                lda          = arg.lda;
    int                ldb          = arg.ldb;
    int                ldc          = arg.ldc;
    double             stride_scale = arg.stride_scale;
    int                batch_count  = arg.batch_count;

    int           K1       = (transA == HIPBLAS_OP_N ? K : N);
    hipblasStride stride_A = size_t(lda) * K1 * stride_scale;
    hipblasStride stride_B = size_t(ldb) * K1 * stride_scale;
    hipblasStride stride_C = size_t(ldc) * N * stride_scale;
    size_t        A_size   = stride_A * batch_count;
    size_t        B_size   = stride_B * batch_count;
    size_t        C_size   = stride_C * batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && (lda < N || ldb < N))
       || (transA != HIPBLAS_OP_N && (lda < K || ldb < K)) || batch_count < 0)
    {
        return;
    }
    else if(batch_count == 0)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hC_host(C_size);
    host_vector<T> hC_device(C_size);
    host_vector<T> hC_gold(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, N, K1, lda, stride_A, batch_count, hipblas_client_never_set_nan, true);
    hipblas_init_matrix(
        hB, arg, N, K1, ldb, stride_B, batch_count, hipblas_client_never_set_nan, false, true);
    hipblas_init_matrix(
        hC_host, arg, N, N, ldc, stride_C, batch_count, hipblas_client_never_set_nan);

    hC_device = hC_host;
    hC_gold   = hC_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC_host, sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasSyrk2StridedBatchedFn(handle,
                                                         uplo,
                                                         transA,
                                                         N,
                                                         K,
                                                         &h_alpha,
                                                         dA,
                                                         lda,
                                                         stride_A,
                                                         dB,
                                                         ldb,
                                                         stride_B,
                                                         &h_beta,
                                                         dC,
                                                         ldc,
                                                         stride_C,
                                                         batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(T) * C_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasSyrk2StridedBatchedFn(handle,
                                                         uplo,
                                                         transA,
                                                         N,
                                                         K,
                                                         d_alpha,
                                                         dA,
                                                         lda,
                                                         stride_A,
                                                         dB,
                                                         ldb,
                                                         stride_B,
                                                         d_beta,
                                                         dC,
                                                         ldc,
                                                         stride_C,
                                                         batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            ref_syr2k<T>(uplo,
                         transA,
                         N,
                         K,
                         h_alpha,
                         hA.data() + b * stride_A,
                         lda,
                         hB.data() + b * stride_B,
                         ldb,
                         h_beta,
                         hC_gold.data() + b * stride_C,
                         ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, ldc, stride_C, hC_gold, hC_host);
            unit_check_general<T>(N, N, batch_count, ldc, stride_C, hC_gold, hC_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', N, N, ldc, stride_C, hC_gold, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', N, N, ldc, stride_C, hC_gold, hC_device, batch_count);
        }
    }

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

            CHECK_HIPBLAS_ERROR(hipblasSyrk2StridedBatchedFn(handle,
                                                             uplo,
                                                             transA,
                                                             N,
                                                             K,
                                                             d_alpha,
                                                             dA,
                                                             lda,
                                                             stride_A,
                                                             dB,
                                                             ldb,
                                                             stride_B,
                                                             d_beta,
                                                             dC,
                                                             ldc,
                                                             stride_C,
                                                             batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasSyr2kStridedBatchedModel{}.log_args<T>(std::cout,
                                                      arg,
                                                      gpu_time_used,
                                                      syr2k_gflop_count<T>(N, K),
                                                      syr2k_gbyte_count<T>(N, K),
                                                      hipblas_error_host,
                                                      hipblas_error_device);
    }
}

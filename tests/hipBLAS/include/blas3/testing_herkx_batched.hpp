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

using hipblasHerkxBatchedModel = ArgumentModel<e_a_type,
                                               e_uplo,
                                               e_transA,
                                               e_N,
                                               e_K,
                                               e_alpha,
                                               e_lda,
                                               e_ldb,
                                               e_beta,
                                               e_ldc,
                                               e_batch_count>;

inline void testname_herkx_batched(const Arguments& arg, std::string& name)
{
    hipblasHerkxBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_herkx_batched_bad_arg(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHerkxBatchedFn
        = FORTRAN ? hipblasHerkxBatched<T, U, true> : hipblasHerkxBatched<T, U, false>;

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

    device_batch_vector<T> dA(cols * lda, 1, batch_count);
    device_batch_vector<T> dB(cols * ldb, 1, batch_count);
    device_batch_vector<T> dC(N * ldc, 1, batch_count);

    device_vector<T> d_alpha(1), d_zero(1);
    device_vector<U> d_beta(1), d_one(1);
    const T          h_alpha(1), h_zero(0);
    const U          h_beta(2), h_one(1);

    const T* alpha = &h_alpha;
    const U* beta  = &h_beta;
    const U* one   = &h_one;
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

        EXPECT_HIPBLAS_STATUS(hipblasHerkxBatchedFn(nullptr,
                                                    uplo,
                                                    transA,
                                                    N,
                                                    K,
                                                    alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dB.ptr_on_device(),
                                                    ldb,
                                                    beta,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count),
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(hipblasHerkxBatchedFn(handle,
                                                    HIPBLAS_FILL_MODE_FULL,
                                                    transA,
                                                    N,
                                                    K,
                                                    alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dB.ptr_on_device(),
                                                    ldb,
                                                    beta,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasHerkxBatchedFn(handle,
                                                    (hipblasFillMode_t)HIPBLAS_OP_N,
                                                    transA,
                                                    N,
                                                    K,
                                                    alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dB.ptr_on_device(),
                                                    ldb,
                                                    beta,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasHerkxBatchedFn(handle,
                                                    uplo,
                                                    HIPBLAS_OP_T,
                                                    N,
                                                    K,
                                                    alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dB.ptr_on_device(),
                                                    ldb,
                                                    beta,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasHerkxBatchedFn(handle,
                                                    uplo,
                                                    (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                                    N,
                                                    K,
                                                    alpha,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    dB.ptr_on_device(),
                                                    ldb,
                                                    beta,
                                                    dC.ptr_on_device(),
                                                    ldc,
                                                    batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(hipblasHerkxBatchedFn(handle,
                                                        uplo,
                                                        transA,
                                                        N,
                                                        K,
                                                        nullptr,
                                                        dA.ptr_on_device(),
                                                        lda,
                                                        dB.ptr_on_device(),
                                                        ldb,
                                                        beta,
                                                        dC.ptr_on_device(),
                                                        ldc,
                                                        batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasHerkxBatchedFn(handle,
                                                        uplo,
                                                        transA,
                                                        N,
                                                        K,
                                                        alpha,
                                                        dA.ptr_on_device(),
                                                        lda,
                                                        dB,
                                                        ldb,
                                                        nullptr,
                                                        dC.ptr_on_device(),
                                                        ldc,
                                                        batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                EXPECT_HIPBLAS_STATUS(hipblasHerkxBatchedFn(handle,
                                                            uplo,
                                                            transA,
                                                            N,
                                                            K,
                                                            alpha,
                                                            nullptr,
                                                            lda,
                                                            dB.ptr_on_device(),
                                                            ldb,
                                                            beta,
                                                            dC.ptr_on_device(),
                                                            ldc,
                                                            batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasHerkxBatchedFn(handle,
                                                            uplo,
                                                            transA,
                                                            N,
                                                            K,
                                                            alpha,
                                                            dA.ptr_on_device(),
                                                            lda,
                                                            nullptr,
                                                            ldb,
                                                            beta,
                                                            dC.ptr_on_device(),
                                                            ldc,
                                                            batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasHerkxBatchedFn(handle,
                                                            uplo,
                                                            transA,
                                                            N,
                                                            K,
                                                            alpha,
                                                            dA.ptr_on_device(),
                                                            lda,
                                                            dB.ptr_on_device(),
                                                            ldb,
                                                            beta,
                                                            nullptr,
                                                            ldc,
                                                            batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
            }

            // If k == 0 && beta == 1, A, B, C may be nullptr
            CHECK_HIPBLAS_ERROR(hipblasHerkxBatchedFn(handle,
                                                      uplo,
                                                      transA,
                                                      N,
                                                      0,
                                                      alpha,
                                                      nullptr,
                                                      lda,
                                                      nullptr,
                                                      ldb,
                                                      one,
                                                      nullptr,
                                                      ldc,
                                                      batch_count));

            // If alpha == 0 && beta == 1, A, B, C may be nullptr
            CHECK_HIPBLAS_ERROR(hipblasHerkxBatchedFn(handle,
                                                      uplo,
                                                      transA,
                                                      N,
                                                      K,
                                                      zero,
                                                      nullptr,
                                                      lda,
                                                      nullptr,
                                                      ldb,
                                                      one,
                                                      nullptr,
                                                      ldc,
                                                      batch_count));
        }

        // If N == 0 || batch_count == 0, can have nullptrs
        CHECK_HIPBLAS_ERROR(hipblasHerkxBatchedFn(handle,
                                                  uplo,
                                                  transA,
                                                  0,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  lda,
                                                  nullptr,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  ldc,
                                                  batch_count));
        CHECK_HIPBLAS_ERROR(hipblasHerkxBatchedFn(handle,
                                                  uplo,
                                                  transA,
                                                  N,
                                                  K,
                                                  nullptr,
                                                  nullptr,
                                                  lda,
                                                  nullptr,
                                                  ldb,
                                                  nullptr,
                                                  nullptr,
                                                  ldc,
                                                  0));
    }
}

template <typename T>
void testing_herkx_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHerkxBatchedFn
        = FORTRAN ? hipblasHerkxBatched<T, U, true> : hipblasHerkxBatched<T, U, false>;

    int N           = arg.N;
    int K           = arg.K;
    int lda         = arg.lda;
    int ldb         = arg.ldb;
    int ldc         = arg.ldc;
    int batch_count = arg.batch_count;

    hipblasFillMode_t  uplo   = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA = char2hipblas_operation(arg.transA);

    T h_alpha = arg.get_alpha<T>();
    U h_beta  = arg.get_beta<U>();

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

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    int    K1     = (transA == HIPBLAS_OP_N ? K : N);
    size_t A_size = size_t(lda) * K1;
    size_t B_size = size_t(ldb) * K1;
    size_t C_size = size_t(ldc) * N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hB(B_size, 1, batch_count);
    host_batch_vector<T> hC_host(C_size, 1, batch_count);
    host_batch_vector<T> hC_device(C_size, 1, batch_count);
    host_batch_vector<T> hC_gold(C_size, 1, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dB(B_size, 1, batch_count);
    device_batch_vector<T> dC(C_size, 1, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<U>       d_beta(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    hipblas_init(hA, true);
    hipblas_init(hB);
    hipblas_init(hC_host);

    hC_device.copy_from(hC_host);
    hC_gold.copy_from(hC_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(U), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasHerkxBatchedFn(handle,
                                                  uplo,
                                                  transA,
                                                  N,
                                                  K,
                                                  &h_alpha,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dB.ptr_on_device(),
                                                  ldb,
                                                  &h_beta,
                                                  dC.ptr_on_device(),
                                                  ldc,
                                                  batch_count));

        CHECK_HIP_ERROR(hC_host.transfer_from(dC));
        CHECK_HIP_ERROR(dC.transfer_from(hC_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasHerkxBatchedFn(handle,
                                                  uplo,
                                                  transA,
                                                  N,
                                                  K,
                                                  d_alpha,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dB.ptr_on_device(),
                                                  ldb,
                                                  d_beta,
                                                  dC.ptr_on_device(),
                                                  ldc,
                                                  batch_count));

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            ref_herkx<T>(
                uplo, transA, N, K, h_alpha, hA[b], lda, hB[b], ldb, h_beta, hC_gold[b], ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, ldc, hC_gold, hC_host);
            unit_check_general<T>(N, N, batch_count, ldc, hC_gold, hC_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', N, N, ldc, hC_gold, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', N, N, ldc, hC_gold, hC_device, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasHerkxBatchedFn(handle,
                                                      uplo,
                                                      transA,
                                                      N,
                                                      K,
                                                      d_alpha,
                                                      dA.ptr_on_device(),
                                                      lda,
                                                      dB.ptr_on_device(),
                                                      ldb,
                                                      d_beta,
                                                      dC.ptr_on_device(),
                                                      ldc,
                                                      batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasHerkxBatchedModel{}.log_args<T>(std::cout,
                                               arg,
                                               gpu_time_used,
                                               herkx_gflop_count<T>(N, K),
                                               herkx_gbyte_count<T>(N, K),
                                               hipblas_error_host,
                                               hipblas_error_device);
    }
}

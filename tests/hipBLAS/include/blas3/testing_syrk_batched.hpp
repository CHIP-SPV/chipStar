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

using hipblasSyrkBatchedModel = ArgumentModel<e_a_type,
                                              e_uplo,
                                              e_transA,
                                              e_N,
                                              e_K,
                                              e_alpha,
                                              e_lda,
                                              e_beta,
                                              e_ldc,
                                              e_batch_count>;

inline void testname_syrk_batched(const Arguments& arg, std::string& name)
{
    hipblasSyrkBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_syrk_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSyrkBatchedFn
        = FORTRAN ? hipblasSyrkBatched<T, true> : hipblasSyrkBatched<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t            N           = 101;
    int64_t            K           = 100;
    int64_t            lda         = 102;
    int64_t            ldc         = 104;
    int64_t            batch_count = 2;
    hipblasOperation_t transA      = HIPBLAS_OP_N;
    hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_LOWER;

    int64_t cols = transA == HIPBLAS_OP_N ? K : N;

    device_batch_vector<T> dA(cols * lda, 1, batch_count);
    device_batch_vector<T> dC(N * ldc, 1, batch_count);

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

        EXPECT_HIPBLAS_STATUS(hipblasSyrkBatchedFn(nullptr,
                                                   uplo,
                                                   transA,
                                                   N,
                                                   K,
                                                   alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   beta,
                                                   dC.ptr_on_device(),
                                                   ldc,
                                                   batch_count),
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(hipblasSyrkBatchedFn(handle,
                                                   HIPBLAS_FILL_MODE_FULL,
                                                   transA,
                                                   N,
                                                   K,
                                                   alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   beta,
                                                   dC.ptr_on_device(),
                                                   ldc,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasSyrkBatchedFn(handle,
                                                   (hipblasFillMode_t)HIPBLAS_OP_N,
                                                   transA,
                                                   N,
                                                   K,
                                                   alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   beta,
                                                   dC.ptr_on_device(),
                                                   ldc,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasSyrkBatchedFn(handle,
                                                   uplo,
                                                   (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                                   N,
                                                   K,
                                                   alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   beta,
                                                   dC.ptr_on_device(),
                                                   ldc,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(hipblasSyrkBatchedFn(handle,
                                                       uplo,
                                                       transA,
                                                       N,
                                                       K,
                                                       nullptr,
                                                       dA.ptr_on_device(),
                                                       lda,
                                                       beta,
                                                       dC.ptr_on_device(),
                                                       ldc,
                                                       batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasSyrkBatchedFn(handle,
                                                       uplo,
                                                       transA,
                                                       N,
                                                       K,
                                                       alpha,
                                                       dA.ptr_on_device(),
                                                       lda,
                                                       nullptr,
                                                       dC.ptr_on_device(),
                                                       ldc,
                                                       batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                EXPECT_HIPBLAS_STATUS(hipblasSyrkBatchedFn(handle,
                                                           uplo,
                                                           transA,
                                                           N,
                                                           K,
                                                           alpha,
                                                           nullptr,
                                                           lda,
                                                           beta,
                                                           dC.ptr_on_device(),
                                                           ldc,
                                                           batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasSyrkBatchedFn(handle,
                                                           uplo,
                                                           transA,
                                                           N,
                                                           K,
                                                           alpha,
                                                           dA.ptr_on_device(),
                                                           lda,
                                                           beta,
                                                           nullptr,
                                                           ldc,
                                                           batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
            }

            // If k == 0 && beta == 1, A, C may be nullptr
            CHECK_HIPBLAS_ERROR(hipblasSyrkBatchedFn(
                handle, uplo, transA, N, 0, alpha, nullptr, lda, one, nullptr, ldc, batch_count));

            // If alpha == 0 && beta == 1, A, C may be nullptr
            CHECK_HIPBLAS_ERROR(hipblasSyrkBatchedFn(
                handle, uplo, transA, N, K, zero, nullptr, lda, one, nullptr, ldc, batch_count));
        }

        // If N == 0 batch_count == 0, can have nullptrs
        CHECK_HIPBLAS_ERROR(hipblasSyrkBatchedFn(
            handle, uplo, transA, 0, K, nullptr, nullptr, lda, nullptr, nullptr, ldc, batch_count));
        CHECK_HIPBLAS_ERROR(hipblasSyrkBatchedFn(
            handle, uplo, transA, N, K, nullptr, nullptr, lda, nullptr, nullptr, ldc, 0));
    }
}

template <typename T>
void testing_syrk_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSyrkBatchedFn
        = FORTRAN ? hipblasSyrkBatched<T, true> : hipblasSyrkBatched<T, false>;

    hipblasFillMode_t  uplo        = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA      = char2hipblas_operation(arg.transA);
    int                N           = arg.N;
    int                K           = arg.K;
    int                lda         = arg.lda;
    int                ldc         = arg.ldc;
    int                batch_count = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && lda < N)
       || (transA != HIPBLAS_OP_N && lda < K) || batch_count < 0)
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
    size_t C_size = size_t(ldc) * N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hC_host(C_size, 1, batch_count);
    host_batch_vector<T> hC_device(C_size, 1, batch_count);
    host_batch_vector<T> hC_gold(C_size, 1, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dC(C_size, 1, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    hipblas_init_vector(hA, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hC_host, arg, hipblas_client_beta_sets_nan);
    hC_device.copy_from(hC_host);
    hC_gold.copy_from(hC_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasSyrkBatchedFn(handle,
                                                 uplo,
                                                 transA,
                                                 N,
                                                 K,
                                                 &h_alpha,
                                                 dA.ptr_on_device(),
                                                 lda,
                                                 &h_beta,
                                                 dC.ptr_on_device(),
                                                 ldc,
                                                 batch_count));

        CHECK_HIP_ERROR(hC_host.transfer_from(dC));
        CHECK_HIP_ERROR(dC.transfer_from(hC_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasSyrkBatchedFn(handle,
                                                 uplo,
                                                 transA,
                                                 N,
                                                 K,
                                                 d_alpha,
                                                 dA.ptr_on_device(),
                                                 lda,
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
            ref_syrk<T>(uplo, transA, N, K, h_alpha, hA[b], lda, h_beta, hC_gold[b], ldc);
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

            CHECK_HIPBLAS_ERROR(hipblasSyrkBatchedFn(handle,
                                                     uplo,
                                                     transA,
                                                     N,
                                                     K,
                                                     d_alpha,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     d_beta,
                                                     dC.ptr_on_device(),
                                                     ldc,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasSyrkBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              syrk_gflop_count<T>(N, K),
                                              syrk_gbyte_count<T>(N, K),
                                              hipblas_error_host,
                                              hipblas_error_device);
    }
}

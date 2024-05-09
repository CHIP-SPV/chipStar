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

#include "arg_check.h"
#include "testing_common.hpp"
#include <typeinfo>

/* ============================================================================================ */

using hipblasGemmBatchedModel = ArgumentModel<e_a_type,
                                              e_transA,
                                              e_transB,
                                              e_M,
                                              e_N,
                                              e_K,
                                              e_alpha,
                                              e_lda,
                                              e_ldb,
                                              e_beta,
                                              e_ldc,
                                              e_batch_count>;

inline void testname_gemm_batched(const Arguments& arg, std::string& name)
{
    hipblasGemmBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_gemm_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGemmBatchedFn
        = FORTRAN ? hipblasGemmBatched<T, true> : hipblasGemmBatched<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t M           = 101;
    int64_t N           = 100;
    int64_t K           = 102;
    int64_t lda         = 103;
    int64_t ldb         = 104;
    int64_t ldc         = 105;
    int64_t batch_count = 2;

    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasOperation_t transB = HIPBLAS_OP_N;

    int64_t colsA = transA == HIPBLAS_OP_N ? N : M;
    int64_t colsB = transB == HIPBLAS_OP_N ? N : M;

    device_batch_vector<T> dA(colsA * lda, 1, batch_count);
    device_batch_vector<T> dB(colsB * ldb, 1, batch_count);
    device_batch_vector<T> dC(N * ldc, 1, batch_count);

    device_vector<T> d_alpha(1), d_beta(1), d_one(1), d_zero(1);
    T                h_alpha(1), h_beta(2), h_one(1), h_zero(0);

    if constexpr(std::is_same_v<T, hipblasHalf>)
        h_one = float_to_half(1.0f);

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

        EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedFn(nullptr,
                                                   transA,
                                                   transB,
                                                   M,
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

        EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedFn(handle,
                                                   (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                                   transB,
                                                   M,
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
        EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedFn(handle,
                                                   transA,
                                                   (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                                   M,
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
            EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedFn(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       alpha,
                                                       dA.ptr_on_device(),
                                                       lda,
                                                       dB.ptr_on_device(),
                                                       ldb,
                                                       nullptr,
                                                       dC.ptr_on_device(),
                                                       ldc,
                                                       batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                // alpha check only for host mode. rocBLAS can handle this in device mode too but shouldn't assume in case this changes.
                EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedFn(handle,
                                                           transA,
                                                           transB,
                                                           M,
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

                // again, rocBLAS can handle this in device mode but shouldn't assume
                EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedFn(handle,
                                                           transA,
                                                           transB,
                                                           M,
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
                EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedFn(handle,
                                                           transA,
                                                           transB,
                                                           M,
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
                EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedFn(handle,
                                                           transA,
                                                           transB,
                                                           M,
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

            // If alpha == 0 && beta == 1, can have A, B, C be nullptr
            CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                     transA,
                                                     transB,
                                                     M,
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

            // If alpha == 0, A and B can be nullptr
            CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     zero,
                                                     nullptr,
                                                     lda,
                                                     nullptr,
                                                     ldb,
                                                     beta,
                                                     dC.ptr_on_device(),
                                                     ldc,
                                                     batch_count));

            // If K == 0, alpha, A, and B can be nullptr
            CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     0,
                                                     nullptr,
                                                     nullptr,
                                                     lda,
                                                     nullptr,
                                                     ldb,
                                                     beta,
                                                     dC.ptr_on_device(),
                                                     ldc,
                                                     batch_count));
        }

        // If M == 0 || N == 0 || batch_count == 0, can have nullptrs
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                 transA,
                                                 transB,
                                                 0,
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
                                                 batch_count));
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                 transA,
                                                 transB,
                                                 M,
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
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                 transA,
                                                 transB,
                                                 M,
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
void testing_gemm_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGemmBatchedFn
        = FORTRAN ? hipblasGemmBatched<T, true> : hipblasGemmBatched<T, false>;

    hipblasOperation_t transA      = char2hipblas_operation(arg.transA);
    hipblasOperation_t transB      = char2hipblas_operation(arg.transB);
    int                M           = arg.M;
    int                N           = arg.N;
    int                K           = arg.K;
    int                lda         = arg.lda;
    int                ldb         = arg.ldb;
    int                ldc         = arg.ldc;
    int                batch_count = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // bad arg checks
    if(batch_count < 0 || M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0)
    {
        hipblasStatus_t    status = HIPBLAS_STATUS_SUCCESS;
        hipblasLocalHandle handle(arg);

        const T *dA_array[1], *dB_array[1];
        T*       dC1_array[1];

        status = hipblasGemmBatchedFn(handle,
                                      transA,
                                      transB,
                                      M,
                                      N,
                                      K,
                                      &h_alpha,
                                      dA_array,
                                      lda,
                                      dB_array,
                                      ldb,
                                      &h_beta,
                                      dC1_array,
                                      ldc,
                                      batch_count);

        verify_hipblas_status_invalid_value(
            status,
            "ERROR: batch_count < 0 || M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0 ");

        return;
    }

    int A_row, A_col, B_row, B_col;

    if(transA == HIPBLAS_OP_N)
    {
        A_row = M;
        A_col = K;
    }
    else
    {
        A_row = K;
        A_col = M;
    }

    if(transB == HIPBLAS_OP_N)
    {
        B_row = K;
        B_col = N;
    }
    else
    {
        B_row = N;
        B_col = K;
    }

    if(lda < A_row || ldb < B_row || ldc < M)
    {
        return;
    }

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    size_t A_size = size_t(lda) * A_col;
    size_t B_size = size_t(ldb) * B_col;
    size_t C_size = size_t(ldc) * N;

    // host arrays
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hB(B_size, 1, batch_count);
    host_batch_vector<T> hC_host(C_size, 1, batch_count);
    host_batch_vector<T> hC_device(C_size, 1, batch_count);
    host_batch_vector<T> hC_copy(C_size, 1, batch_count);

    // device arrays
    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dB(B_size, 1, batch_count);
    device_batch_vector<T> dC(C_size, 1, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    hipblas_init_vector(hA, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hB, arg, hipblas_client_alpha_sets_nan);
    hipblas_init_vector(hC_host, arg, hipblas_client_beta_sets_nan);

    hC_device.copy_from(hC_host);
    hC_copy.copy_from(hC_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // calculate "golden" result on CPU
        for(int i = 0; i < batch_count; i++)
        {
            ref_gemm<T>(transA,
                        transB,
                        M,
                        N,
                        K,
                        h_alpha,
                        (T*)hA[i],
                        lda,
                        (T*)hB[i],
                        ldb,
                        h_beta,
                        (T*)hC_copy[i],
                        ldc);
        }

        // test hipBLAS batched gemm with alpha and beta pointers on device
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 d_alpha,
                                                 (const T* const*)dA.ptr_on_device(),
                                                 lda,
                                                 (const T* const*)dB.ptr_on_device(),
                                                 ldb,
                                                 d_beta,
                                                 dC.ptr_on_device(),
                                                 ldc,
                                                 batch_count));

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        // test hipBLAS batched gemm with alpha and beta pointers on host
        CHECK_HIP_ERROR(dC.transfer_from(hC_host));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 &h_alpha,
                                                 (const T* const*)dA.ptr_on_device(),
                                                 lda,
                                                 (const T* const*)dB.ptr_on_device(),
                                                 ldb,
                                                 &h_beta,
                                                 dC.ptr_on_device(),
                                                 ldc,
                                                 batch_count));

        CHECK_HIP_ERROR(hC_host.transfer_from(dC));

        if(arg.unit_check)
        {
            if(std::is_same_v<T, hipblasHalf> && (getArchMajor() == 11))
            {
                const double tol = K * sum_error_tolerance_for_gfx11<T, T, T>;
                near_check_general<T>(M, N, batch_count, ldc, hC_copy, hC_host, tol);
                near_check_general<T>(M, N, batch_count, ldc, hC_copy, hC_device, tol);
            }
            else
            {
                unit_check_general<T>(M, N, batch_count, ldc, hC_copy, hC_host);
                unit_check_general<T>(M, N, batch_count, ldc, hC_copy, hC_device);
            }
        }

        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldc, hC_copy, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldc, hC_copy, hC_device, batch_count);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        // gemm has better performance in host mode. In rocBLAS in device mode
        // we need to copy alpha and beta to the host.
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     &h_alpha,
                                                     (const T* const*)dA.ptr_on_device(),
                                                     lda,
                                                     (const T* const*)dB.ptr_on_device(),
                                                     ldb,
                                                     &h_beta,
                                                     dC.ptr_on_device(),
                                                     ldc,
                                                     batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemmBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              gemm_gflop_count<T>(M, N, K),
                                              gemm_gbyte_count<T>(M, N, K),
                                              hipblas_error_host,
                                              hipblas_error_device);
    }
}

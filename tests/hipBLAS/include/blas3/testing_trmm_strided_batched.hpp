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

using hipblasTrmmStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_side,
                                                     e_uplo,
                                                     e_transA,
                                                     e_diag,
                                                     e_M,
                                                     e_N,
                                                     e_alpha,
                                                     e_lda,
                                                     e_ldb,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_trmm_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasTrmmStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_trmm_strided_batched_bad_arg(const Arguments& arg)
{
    auto hipblasTrmmStridedBatchedFn = arg.api == hipblas_client_api::FORTRAN
                                           ? hipblasTrmmStridedBatched<T, true>
                                           : hipblasTrmmStridedBatched<T, false>;
    bool inplace                     = arg.inplace;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_DEVICE, HIPBLAS_POINTER_MODE_HOST})
    {
        hipblasSideMode_t  side        = HIPBLAS_SIDE_LEFT;
        hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_LOWER;
        hipblasOperation_t transA      = HIPBLAS_OP_N;
        hipblasDiagType_t  diag        = HIPBLAS_DIAG_NON_UNIT;
        int64_t            M           = 100;
        int64_t            N           = 101;
        int64_t            lda         = 102;
        int64_t            ldb         = 103;
        int64_t            ldc         = 104;
        int64_t            batch_count = 2;
        int64_t            ldOut       = inplace ? ldb : ldc;
        int64_t            K           = M;

        device_vector<T> alpha_d(1), zero_d(1);

        const T alpha_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* zero  = &zero_h;

        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        hipblasStride strideA = lda * K;
        hipblasStride strideB = ldb * N;
        hipblasStride strideC = ldc * N;
        size_t        A_size  = strideA * batch_count;
        size_t        B_size  = strideB * batch_count;

        size_t C_size = inplace ? 1 : strideC * batch_count;

        device_vector<T> dA(A_size);
        device_vector<T> dB(B_size);
        device_vector<T> dC(C_size);

        device_vector<T>* dOut      = inplace ? &dB : &dC;
        hipblasStride     strideOut = inplace ? strideB : strideC;

        // invalid enums
        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          HIPBLAS_SIDE_BOTH,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          M,
                                                          N,
                                                          alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dB,
                                                          ldb,
                                                          strideB,
                                                          *dOut,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          side,
                                                          HIPBLAS_FILL_MODE_FULL,
                                                          transA,
                                                          diag,
                                                          M,
                                                          N,
                                                          alpha,
                                                          dA,
                                                          strideA,
                                                          lda,
                                                          dB,
                                                          ldb,
                                                          strideB,
                                                          *dOut,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          side,
                                                          uplo,
                                                          (hipblasOperation_t)HIPBLAS_SIDE_BOTH,
                                                          diag,
                                                          M,
                                                          N,
                                                          alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dB,
                                                          ldb,
                                                          strideB,
                                                          *dOut,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);

        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          side,
                                                          uplo,
                                                          transA,
                                                          (hipblasDiagType_t)HIPBLAS_SIDE_BOTH,
                                                          M,
                                                          N,
                                                          alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dB,
                                                          ldb,
                                                          strideB,
                                                          *dOut,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);

        // nullptr checks
        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(nullptr,
                                                          side,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          M,
                                                          N,
                                                          alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dB,
                                                          ldb,
                                                          strideB,
                                                          *dOut,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          side,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          M,
                                                          N,
                                                          nullptr,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dB,
                                                          ldb,
                                                          strideB,
                                                          *dOut,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          side,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          M,
                                                          N,
                                                          alpha,
                                                          nullptr,
                                                          lda,
                                                          strideA,
                                                          dB,
                                                          ldb,
                                                          strideB,
                                                          *dOut,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          side,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          M,
                                                          N,
                                                          alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          nullptr,
                                                          ldb,
                                                          strideB,
                                                          *dOut,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          side,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          M,
                                                          N,
                                                          alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dB,
                                                          ldb,
                                                          strideB,
                                                          nullptr,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);

        // quick return: if alpha == 0, both A & B can be nullptr
        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          side,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          M,
                                                          N,
                                                          zero,
                                                          nullptr,
                                                          lda,
                                                          strideA,
                                                          nullptr,
                                                          ldb,
                                                          strideB,
                                                          *dOut,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_SUCCESS);

        // quick return: if M == 0, then all ptrs can be nullptr
        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          side,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          0,
                                                          N,
                                                          nullptr,
                                                          nullptr,
                                                          lda,
                                                          strideA,
                                                          nullptr,
                                                          ldb,
                                                          strideB,
                                                          nullptr,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_SUCCESS);

        // quick return: if N == 0, then all ptrs can be nullptr
        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          side,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          M,
                                                          0,
                                                          nullptr,
                                                          nullptr,
                                                          lda,
                                                          strideA,
                                                          nullptr,
                                                          ldb,
                                                          strideB,
                                                          nullptr,
                                                          ldOut,
                                                          strideOut,
                                                          batch_count),
                              HIPBLAS_STATUS_SUCCESS);

        // quick return: if batch_count == 0, then all ptrs can be nullptr
        EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                          side,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          M,
                                                          N,
                                                          nullptr,
                                                          nullptr,
                                                          lda,
                                                          strideA,
                                                          nullptr,
                                                          ldb,
                                                          strideB,
                                                          nullptr,
                                                          ldOut,
                                                          strideOut,
                                                          0),
                              HIPBLAS_STATUS_SUCCESS);

        // in-place only checks
        if(inplace)
        {
            // if inplace, must have ldb == ldc
            EXPECT_HIPBLAS_STATUS(hipblasTrmmStridedBatchedFn(handle,
                                                              side,
                                                              uplo,
                                                              transA,
                                                              diag,
                                                              M,
                                                              N,
                                                              alpha,
                                                              dA,
                                                              lda,
                                                              strideA,
                                                              dB,
                                                              ldb,
                                                              strideB,
                                                              *dOut,
                                                              ldb + 1,
                                                              strideOut,
                                                              batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);
        }
    }
}

template <typename T>
void testing_trmm_strided_batched(const Arguments& arg)
{
    auto hipblasTrmmStridedBatchedFn = arg.api == hipblas_client_api::FORTRAN
                                           ? hipblasTrmmStridedBatched<T, true>
                                           : hipblasTrmmStridedBatched<T, false>;
    bool inplace                     = arg.inplace;

    hipblasSideMode_t  side         = char2hipblas_side(arg.side);
    hipblasFillMode_t  uplo         = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(arg.diag);
    int                M            = arg.M;
    int                N            = arg.N;
    int                lda          = arg.lda;
    int                ldb          = arg.ldb;
    int                ldc          = arg.ldc;
    int                ldOut        = inplace ? ldb : ldc;
    double             stride_scale = arg.stride_scale;
    int                batch_count  = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int           K          = (side == HIPBLAS_SIDE_LEFT ? M : N);
    hipblasStride stride_A   = size_t(lda) * K * stride_scale;
    hipblasStride stride_B   = size_t(ldb) * N * stride_scale;
    hipblasStride stride_C   = inplace ? 1 : size_t(ldc) * N * stride_scale;
    hipblasStride stride_out = inplace ? stride_B : stride_C;

    size_t A_size   = stride_A * batch_count;
    size_t B_size   = stride_B * batch_count;
    size_t C_size   = stride_C * batch_count;
    size_t out_size = stride_out * batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < K || ldb < M || ldOut < M || batch_count < 0)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hC(C_size);

    host_vector<T> hOut_host(out_size);
    host_vector<T> hOut_device(out_size);
    host_vector<T> hOut_gold(out_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);
    device_vector<T> d_alpha(1);

    device_vector<T>* dOut = inplace ? &dB : &dC;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasLocalHandle handle(arg);

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, K, K, lda, stride_A, batch_count, hipblas_client_alpha_sets_nan, true);
    hipblas_init_matrix(
        hB, arg, M, N, ldb, stride_B, batch_count, hipblas_client_alpha_sets_nan, false, true);
    if(!inplace)
        hipblas_init_matrix(
            hC, arg, M, N, ldc, stride_C, batch_count, hipblas_client_alpha_sets_nan, false, true);

    hOut_host   = inplace ? hB : hC;
    hOut_device = hOut_host;
    hOut_gold   = hOut_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasTrmmStridedBatchedFn(handle,
                                                        side,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        M,
                                                        N,
                                                        &h_alpha,
                                                        dA,
                                                        lda,
                                                        stride_A,
                                                        dB,
                                                        ldb,
                                                        stride_B,
                                                        *dOut,
                                                        ldOut,
                                                        stride_out,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hOut_host, *dOut, sizeof(T) * out_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * B_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC, sizeof(T) * C_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasTrmmStridedBatchedFn(handle,
                                                        side,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        M,
                                                        N,
                                                        d_alpha,
                                                        dA,
                                                        lda,
                                                        stride_A,
                                                        dB,
                                                        ldb,
                                                        stride_B,
                                                        *dOut,
                                                        ldOut,
                                                        stride_out,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hOut_device, *dOut, sizeof(T) * out_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            ref_trmm<T>(side,
                        uplo,
                        transA,
                        diag,
                        M,
                        N,
                        h_alpha,
                        hA.data() + b * stride_A,
                        lda,
                        hB.data() + b * stride_B,
                        ldb);
        }

        copy_matrix_with_different_leading_dimensions(
            hB, hOut_gold, M, N, ldb, ldOut, stride_B, stride_out, batch_count);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldOut, stride_out, hOut_gold, hOut_host);
            unit_check_general<T>(M, N, batch_count, ldOut, stride_out, hOut_gold, hOut_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>(
                'F', M, N, ldOut, stride_out, hOut_gold, hOut_host, batch_count);
            hipblas_error_device = norm_check_general<T>(
                'F', M, N, ldOut, stride_out, hOut_gold, hOut_device, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasTrmmStridedBatchedFn(handle,
                                                            side,
                                                            uplo,
                                                            transA,
                                                            diag,
                                                            M,
                                                            N,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_A,
                                                            dB,
                                                            ldb,
                                                            stride_B,
                                                            *dOut,
                                                            ldOut,
                                                            stride_out,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrmmStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     trmm_gflop_count<T>(M, N, K),
                                                     trmm_gbyte_count<T>(M, N, K),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}

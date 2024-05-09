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
#include <limits>
#include <stdlib.h>
#include <typeinfo>
#include <vector>

#include "hipblas_unique_ptr.hpp"
#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasGeamStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_transA,
                                                     e_transB,
                                                     e_M,
                                                     e_N,
                                                     e_alpha,
                                                     e_lda,
                                                     e_beta,
                                                     e_ldb,
                                                     e_ldc,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_geam_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasGeamStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_geam_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGeamStridedBatchedFn
        = FORTRAN ? hipblasGeamStridedBatched<T, true> : hipblasGeamStridedBatched<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t M           = 101;
    int64_t N           = 100;
    int64_t lda         = 102;
    int64_t ldb         = 103;
    int64_t ldc         = 104;
    int64_t batch_count = 2;

    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasOperation_t transB = HIPBLAS_OP_N;

    int64_t colsA = transA == HIPBLAS_OP_N ? N : M;
    int64_t colsB = transB == HIPBLAS_OP_N ? N : M;

    hipblasStride strideA = colsA * lda;
    hipblasStride strideB = colsB * ldb;
    hipblasStride strideC = N * ldc;

    device_vector<T> dA(strideA * batch_count);
    device_vector<T> dB(strideB * batch_count);
    device_vector<T> dC(strideC * batch_count);

    device_vector<T> d_alpha(1), d_beta(1), d_zero(1);
    const T          h_alpha(1), h_beta(2), h_zero(0);

    const T* alpha = &h_alpha;
    const T* beta  = &h_beta;
    const T* zero  = &h_zero;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_beta, beta, sizeof(*beta), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            beta  = d_beta;
            zero  = d_zero;
        }

        EXPECT_HIPBLAS_STATUS(hipblasGeamStridedBatchedFn(nullptr,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          alpha,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          beta,
                                                          dB,
                                                          ldb,
                                                          strideB,
                                                          dC,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(
            hipblasGeamStridedBatchedFn(handle,
                                        (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                        transB,
                                        M,
                                        N,
                                        alpha,
                                        dA,
                                        lda,
                                        strideA,
                                        beta,
                                        dB,
                                        ldb,
                                        strideB,
                                        dC,
                                        ldc,
                                        strideC,
                                        batch_count),
            HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(
            hipblasGeamStridedBatchedFn(handle,
                                        transA,
                                        (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                        M,
                                        N,
                                        alpha,
                                        dA,
                                        lda,
                                        strideA,
                                        beta,
                                        dB,
                                        ldb,
                                        strideB,
                                        dC,
                                        ldc,
                                        strideC,
                                        batch_count),
            HIPBLAS_STATUS_INVALID_ENUM);

        if(arg.bad_arg_all)
        {
            // (dA == dC) => (lda == ldc) else invalid_value, note no checks on strides
            EXPECT_HIPBLAS_STATUS(hipblasGeamStridedBatchedFn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              alpha,
                                                              dA,
                                                              lda,
                                                              strideA,
                                                              beta,
                                                              dB,
                                                              ldb,
                                                              strideB,
                                                              dA,
                                                              lda + 1,
                                                              strideA,
                                                              batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);

            // (dB == dC) => (ldb == ldc) else invalid value
            EXPECT_HIPBLAS_STATUS(hipblasGeamStridedBatchedFn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              alpha,
                                                              dA,
                                                              lda,
                                                              strideA,
                                                              beta,
                                                              dB,
                                                              ldb,
                                                              strideB,
                                                              dB,
                                                              ldb + 1,
                                                              strideA,
                                                              batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);

            EXPECT_HIPBLAS_STATUS(hipblasGeamStridedBatchedFn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              nullptr,
                                                              dA,
                                                              lda,
                                                              strideA,
                                                              beta,
                                                              dB,
                                                              ldb,
                                                              strideB,
                                                              dC,
                                                              ldc,
                                                              strideC,
                                                              batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasGeamStridedBatchedFn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              alpha,
                                                              dA,
                                                              lda,
                                                              strideA,
                                                              nullptr,
                                                              dB,
                                                              ldb,
                                                              strideB,
                                                              dC,
                                                              ldc,
                                                              strideC,
                                                              batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasGeamStridedBatchedFn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              alpha,
                                                              dA,
                                                              lda,
                                                              strideA,
                                                              beta,
                                                              dB,
                                                              ldb,
                                                              strideB,
                                                              nullptr,
                                                              ldc,
                                                              strideC,
                                                              batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                EXPECT_HIPBLAS_STATUS(hipblasGeamStridedBatchedFn(handle,
                                                                  transA,
                                                                  transB,
                                                                  M,
                                                                  N,
                                                                  alpha,
                                                                  nullptr,
                                                                  lda,
                                                                  strideA,
                                                                  beta,
                                                                  dB,
                                                                  ldb,
                                                                  strideB,
                                                                  dC,
                                                                  ldc,
                                                                  strideC,
                                                                  batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasGeamStridedBatchedFn(handle,
                                                                  transA,
                                                                  transB,
                                                                  M,
                                                                  N,
                                                                  alpha,
                                                                  dA,
                                                                  lda,
                                                                  strideA,
                                                                  beta,
                                                                  nullptr,
                                                                  ldb,
                                                                  strideB,
                                                                  dC,
                                                                  ldc,
                                                                  strideC,
                                                                  batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
            }

            // alpha == 0, can have A be nullptr. beta == 0 can have B be nullptr
            CHECK_HIPBLAS_ERROR(hipblasGeamStridedBatchedFn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            zero,
                                                            nullptr,
                                                            lda,
                                                            strideA,
                                                            beta,
                                                            dB,
                                                            ldb,
                                                            strideB,
                                                            dC,
                                                            ldc,
                                                            strideC,
                                                            batch_count));
            CHECK_HIPBLAS_ERROR(hipblasGeamStridedBatchedFn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            alpha,
                                                            dA,
                                                            lda,
                                                            strideA,
                                                            zero,
                                                            nullptr,
                                                            ldb,
                                                            strideB,
                                                            dC,
                                                            ldc,
                                                            strideC,
                                                            batch_count));
        }

        // If M == 0 || N == 0 || batch_count == 0, can have nullptrs
        CHECK_HIPBLAS_ERROR(hipblasGeamStridedBatchedFn(handle,
                                                        transA,
                                                        transB,
                                                        0,
                                                        N,
                                                        nullptr,
                                                        nullptr,
                                                        lda,
                                                        strideA,
                                                        nullptr,
                                                        nullptr,
                                                        lda,
                                                        strideB,
                                                        nullptr,
                                                        lda,
                                                        strideC,
                                                        batch_count));
        CHECK_HIPBLAS_ERROR(hipblasGeamStridedBatchedFn(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        0,
                                                        nullptr,
                                                        nullptr,
                                                        lda,
                                                        strideA,
                                                        nullptr,
                                                        nullptr,
                                                        lda,
                                                        strideB,
                                                        nullptr,
                                                        lda,
                                                        strideC,
                                                        batch_count));
        CHECK_HIPBLAS_ERROR(hipblasGeamStridedBatchedFn(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        N,
                                                        nullptr,
                                                        nullptr,
                                                        lda,
                                                        strideA,
                                                        nullptr,
                                                        nullptr,
                                                        lda,
                                                        strideB,
                                                        nullptr,
                                                        lda,
                                                        strideC,
                                                        0));
    }
}

template <typename T>
void testing_geam_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGeamStridedBatchedFn
        = FORTRAN ? hipblasGeamStridedBatched<T, true> : hipblasGeamStridedBatched<T, false>;

    hipblasOperation_t transA       = char2hipblas_operation(arg.transA);
    hipblasOperation_t transB       = char2hipblas_operation(arg.transB);
    int                M            = arg.M;
    int                N            = arg.N;
    int                lda          = arg.lda;
    int                ldb          = arg.ldb;
    int                ldc          = arg.ldc;
    double             stride_scale = arg.stride_scale;
    int                batch_count  = arg.batch_count;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    int           A_row, A_col, B_row, B_col;
    hipblasStride stride_A, stride_B, stride_C;

    if(transA == HIPBLAS_OP_N)
    {
        A_row = M;
        A_col = N;
    }
    else
    {
        A_row = N;
        A_col = M;
    }
    if(transB == HIPBLAS_OP_N)
    {
        B_row = M;
        B_col = N;
    }
    else
    {
        B_row = N;
        B_col = M;
    }

    stride_A = size_t(lda) * A_col * stride_scale;
    stride_B = size_t(ldb) * B_col * stride_scale;
    stride_C = size_t(ldc) * N * stride_scale;

    size_t A_size = stride_A * batch_count;
    size_t B_size = stride_B * batch_count;
    size_t C_size = stride_C * batch_count;

    // check here to prevent undefined memory allocation error
    if(M <= 0 || N <= 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0)
    {
        return;
    }

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    // allocate memory on device
    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    if(!dA || !dB || !dC || !d_alpha || !d_beta)
    {
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hC1(C_size);
    host_vector<T> hC2(C_size);
    host_vector<T> hC_copy(C_size);

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, A_row, A_col, lda, stride_A, batch_count, hipblas_client_alpha_sets_nan, true);
    hipblas_init_matrix(
        hB, arg, B_row, B_col, ldb, stride_B, batch_count, hipblas_client_beta_sets_nan);
    hipblas_init_matrix(hC1, arg, M, N, ldc, stride_C, batch_count, hipblas_client_beta_sets_nan);

    hC2     = hC1;
    hC_copy = hC1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC1.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.norm_check || arg.unit_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        {
            // &h_alpha and &h_beta are host pointers
            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
            CHECK_HIPBLAS_ERROR(hipblasGeamStridedBatchedFn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            &h_alpha,
                                                            dA,
                                                            lda,
                                                            stride_A,
                                                            &h_beta,
                                                            dB,
                                                            ldb,
                                                            stride_B,
                                                            dC,
                                                            ldc,
                                                            stride_C,
                                                            batch_count));

            CHECK_HIP_ERROR(hipMemcpy(hC1.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));
        }
        {
            CHECK_HIP_ERROR(hipMemcpy(dC, hC2.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));

            // d_alpha and d_beta are device pointers
            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
            CHECK_HIPBLAS_ERROR(hipblasGeamStridedBatchedFn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_A,
                                                            d_beta,
                                                            dB,
                                                            ldb,
                                                            stride_B,
                                                            dC,
                                                            ldc,
                                                            stride_C,
                                                            batch_count));

            CHECK_HIP_ERROR(hipMemcpy(hC2.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));
        }

        /* =====================================================================
                CPU BLAS
        =================================================================== */
        // reference calculation
        for(int b = 0; b < batch_count; b++)
        {
            ref_geam(transA,
                     transB,
                     M,
                     N,
                     &h_alpha,
                     (T*)hA + b * stride_A,
                     lda,
                     &h_beta,
                     (T*)hB + b * stride_B,
                     ldb,
                     (T*)hC_copy + b * stride_C,
                     ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_copy, hC1);
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_copy, hC2);
        }

        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_copy, hC1, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_copy, hC2, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasGeamStridedBatchedFn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_A,
                                                            d_beta,
                                                            dB,
                                                            ldb,
                                                            stride_B,
                                                            dC,
                                                            ldc,
                                                            stride_C,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasGeamStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     geam_gflop_count<T>(M, N),
                                                     geam_gbyte_count<T>(M, N),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}

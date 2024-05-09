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

using hipblasHerkModel
    = ArgumentModel<e_a_type, e_uplo, e_transA, e_N, e_K, e_alpha, e_lda, e_beta, e_ldc>;

inline void testname_herk(const Arguments& arg, std::string& name)
{
    hipblasHerkModel{}.test_name(arg, name);
}

template <typename T>
void testing_herk_bad_arg(const Arguments& arg)
{
    using U            = real_t<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHerkFn = FORTRAN ? hipblasHerk<T, U, true> : hipblasHerk<T, U, false>;

    hipblasLocalHandle handle(arg);

    int64_t            N      = 101;
    int64_t            K      = 100;
    int64_t            lda    = 102;
    int64_t            ldc    = 104;
    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasFillMode_t  uplo   = HIPBLAS_FILL_MODE_LOWER;

    int64_t cols = transA == HIPBLAS_OP_N ? K : N;

    device_vector<T> dA(cols * lda);
    device_vector<T> dC(N * ldc);

    device_vector<U> d_alpha(1), d_zero(1), d_beta(1), d_one(1);
    const U          h_alpha(1), h_zero(0), h_beta(2), h_one(1);

    const U* alpha = &h_alpha;
    const U* beta  = &h_beta;
    const U* one   = &h_one;
    const U* zero  = &h_zero;

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

        EXPECT_HIPBLAS_STATUS(
            hipblasHerkFn(nullptr, uplo, transA, N, K, alpha, dA, lda, beta, dC, ldc),
            HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(
            hipblasHerkFn(
                handle, HIPBLAS_FILL_MODE_FULL, transA, N, K, alpha, dA, lda, beta, dC, ldc),
            HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(hipblasHerkFn(handle,
                                            (hipblasFillMode_t)HIPBLAS_OP_N,
                                            transA,
                                            N,
                                            K,
                                            alpha,
                                            dA,
                                            lda,
                                            beta,
                                            dC,
                                            ldc),
                              HIPBLAS_STATUS_INVALID_ENUM);

        // TODO: Supported in cuBLAS but not in rocBLAS? Need to investigate.
        // EXPECT_HIPBLAS_STATUS(
        //     hipblasHerkFn(handle, uplo, HIPBLAS_OP_T, N, K, alpha, dA, lda, beta, dC, ldc),
        //     HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(hipblasHerkFn(handle,
                                            uplo,
                                            (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                            N,
                                            K,
                                            alpha,
                                            dA,
                                            lda,
                                            beta,
                                            dC,
                                            ldc),
                              HIPBLAS_STATUS_INVALID_ENUM);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasHerkFn(handle, uplo, transA, N, K, nullptr, dA, lda, beta, dC, ldc),
                HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(
                hipblasHerkFn(handle, uplo, transA, N, K, alpha, dA, lda, nullptr, dC, ldc),
                HIPBLAS_STATUS_INVALID_VALUE);

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                EXPECT_HIPBLAS_STATUS(
                    hipblasHerkFn(handle, uplo, transA, N, K, alpha, nullptr, lda, beta, dC, ldc),
                    HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(
                    hipblasHerkFn(handle, uplo, transA, N, K, alpha, dA, lda, beta, nullptr, ldc),
                    HIPBLAS_STATUS_INVALID_VALUE);
            }

            // If k == 0 && beta == 1, A, C may be nullptr
            CHECK_HIPBLAS_ERROR(
                hipblasHerkFn(handle, uplo, transA, N, 0, alpha, nullptr, lda, one, nullptr, ldc));

            // If alpha == 0 && beta == 1, A, C may be nullptr
            CHECK_HIPBLAS_ERROR(
                hipblasHerkFn(handle, uplo, transA, N, K, zero, nullptr, lda, one, nullptr, ldc));
        }

        // If N == 0, can have nullptrs
        CHECK_HIPBLAS_ERROR(hipblasHerkFn(
            handle, uplo, transA, 0, K, nullptr, nullptr, lda, nullptr, nullptr, ldc));
    }
}

template <typename T>
void testing_herk(const Arguments& arg)
{
    using U            = real_t<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHerkFn = FORTRAN ? hipblasHerk<T, U, true> : hipblasHerk<T, U, false>;

    int N   = arg.N;
    int K   = arg.K;
    int lda = arg.lda;
    int ldc = arg.ldc;

    hipblasFillMode_t  uplo   = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA = char2hipblas_operation(arg.transA);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && lda < N)
       || (transA != HIPBLAS_OP_N && lda < K))
    {
        return;
    }

    int    K1     = (transA == HIPBLAS_OP_N ? K : N);
    size_t A_size = size_t(lda) * K1;
    size_t C_size = size_t(ldc) * N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hC_host(C_size);
    host_vector<T> hC_device(C_size);
    host_vector<T> hC_gold(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dC(C_size);
    device_vector<U> d_alpha(1);
    device_vector<U> d_beta(1);

    U h_alpha = arg.get_alpha<U>();
    U h_beta  = arg.get_beta<U>();

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, N, K1, lda, 0, 1, hipblas_client_alpha_sets_nan, true);
    hipblas_init_matrix(hC_host, arg, N, N, ldc, 0, 1, hipblas_client_beta_sets_nan, false, true);
    hC_device = hC_host;
    hC_gold   = hC_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC_host, sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(U), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(U), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(
            hipblasHerkFn(handle, uplo, transA, N, K, &h_alpha, dA, lda, &h_beta, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(T) * C_size, hipMemcpyHostToDevice));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(
            hipblasHerkFn(handle, uplo, transA, N, K, d_alpha, dA, lda, d_beta, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        ref_herk<T>(uplo, transA, N, K, h_alpha, hA, lda, h_beta, hC_gold, ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, ldc, hC_gold, hC_host);
            unit_check_general<T>(N, N, ldc, hC_gold, hC_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host   = norm_check_general<T>('F', N, N, ldc, hC_gold, hC_host);
            hipblas_error_device = norm_check_general<T>('F', N, N, ldc, hC_gold, hC_device);
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

            CHECK_HIPBLAS_ERROR(
                hipblasHerkFn(handle, uplo, transA, N, K, d_alpha, dA, lda, d_beta, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasHerkModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       herk_gflop_count<T>(N, K),
                                       herk_gbyte_count<T>(N, K),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}

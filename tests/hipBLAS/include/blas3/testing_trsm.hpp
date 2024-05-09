/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

using hipblasTrsmModel
    = ArgumentModel<e_a_type, e_side, e_uplo, e_transA, e_diag, e_M, e_N, e_alpha, e_lda, e_ldb>;

inline void testname_trsm(const Arguments& arg, std::string& name)
{
    hipblasTrsmModel{}.test_name(arg, name);
}

template <typename T>
void testing_trsm_bad_arg(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrsmFn = FORTRAN ? hipblasTrsm<T, true> : hipblasTrsm<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t            M      = 101;
    int64_t            N      = 100;
    int64_t            lda    = 102;
    int64_t            ldb    = 103;
    hipblasSideMode_t  side   = HIPBLAS_SIDE_LEFT;
    hipblasFillMode_t  uplo   = HIPBLAS_FILL_MODE_LOWER;
    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasDiagType_t  diag   = HIPBLAS_DIAG_NON_UNIT;

    int64_t K = side == HIPBLAS_SIDE_LEFT ? M : N;

    device_vector<T> dA(K * lda);
    device_vector<T> dB(N * ldb);

    device_vector<T> d_alpha(1), d_zero(1);
    const T          h_alpha(1), h_zero(0);

    const T* alpha = &h_alpha;
    const T* zero  = &h_zero;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        EXPECT_HIPBLAS_STATUS(
            hipblasTrsmFn(nullptr, side, uplo, transA, diag, M, N, alpha, dA, lda, dB, ldb),
            HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(
            hipblasTrsmFn(
                handle, HIPBLAS_SIDE_BOTH, uplo, transA, diag, M, N, alpha, dA, lda, dB, ldb),
#ifdef __HIP_PLATFORM_NVCC__
            HIPBLAS_STATUS_INVALID_ENUM);
#else
            HIPBLAS_STATUS_INVALID_VALUE);
#endif
        EXPECT_HIPBLAS_STATUS(
            hipblasTrsmFn(
                handle, side, HIPBLAS_FILL_MODE_FULL, transA, diag, M, N, alpha, dA, lda, dB, ldb),
            HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasTrsmFn(handle,
                                            side,
                                            uplo,
                                            (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                            diag,
                                            M,
                                            N,
                                            alpha,
                                            dA,
                                            lda,
                                            dB,
                                            ldb),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasTrsmFn(handle,
                                            side,
                                            uplo,
                                            transA,
                                            (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                                            M,
                                            N,
                                            alpha,
                                            dA,
                                            lda,
                                            dB,
                                            ldb),
                              HIPBLAS_STATUS_INVALID_ENUM);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasTrsmFn(handle, side, uplo, transA, diag, M, N, nullptr, dA, lda, dB, ldb),
                HIPBLAS_STATUS_INVALID_VALUE);

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                EXPECT_HIPBLAS_STATUS(
                    hipblasTrsmFn(
                        handle, side, uplo, transA, diag, M, N, alpha, nullptr, lda, dB, ldb),
                    HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(
                    hipblasTrsmFn(
                        handle, side, uplo, transA, diag, M, N, alpha, dA, lda, nullptr, ldb),
                    HIPBLAS_STATUS_INVALID_VALUE);
            }

            // If alpha == 0, then A can be nullptr
            CHECK_HIPBLAS_ERROR(
                hipblasTrsmFn(handle, side, uplo, transA, diag, M, N, zero, nullptr, lda, dB, ldb));
        }

        // If M == 0 || N == 0, can have nullptrs
        CHECK_HIPBLAS_ERROR(hipblasTrsmFn(
            handle, side, uplo, transA, diag, 0, N, nullptr, nullptr, lda, nullptr, ldb));
        CHECK_HIPBLAS_ERROR(hipblasTrsmFn(
            handle, side, uplo, transA, diag, M, 0, nullptr, nullptr, lda, nullptr, ldb));
    }
}

template <typename T>
void testing_trsm(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrsmFn = FORTRAN ? hipblasTrsm<T, true> : hipblasTrsm<T, false>;

    hipblasSideMode_t  side   = char2hipblas_side(arg.side);
    hipblasFillMode_t  uplo   = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(arg.diag);
    int                M      = arg.M;
    int                N      = arg.N;
    int                lda    = arg.lda;
    int                ldb    = arg.ldb;

    T h_alpha = arg.get_alpha<T>();

    int    K      = (side == HIPBLAS_SIDE_LEFT ? M : N);
    size_t A_size = size_t(lda) * K;
    size_t B_size = size_t(ldb) * N;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < K || ldb < M)
    {
        return;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB_host(B_size);
    host_vector<T> hB_device(B_size);
    host_vector<T> hB_gold(B_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> d_alpha(1);

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    // Initial hA on CPU
    hipblas_init_matrix_type(hipblas_diagonally_dominant_triangular_matrix,
                             (T*)hA,
                             arg,
                             K,
                             K,
                             lda,
                             0,
                             1,
                             hipblas_client_never_set_nan,
                             true);
    hipblas_init_matrix(hB_host, arg, M, N, ldb, 0, 1, hipblas_client_never_set_nan);

    if(diag == HIPBLAS_DIAG_UNIT)
    {
        make_unit_diagonal(uplo, (T*)hA, lda, K);
    }

    hB_gold = hB_host; // original solution hX

    // Calculate hB = hA*hX;
    ref_trmm<T>(side, uplo, transA, diag, M, N, T(1.0) / h_alpha, (const T*)hA, lda, hB_host, ldb);

    hB_device = hB_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB_host, sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(
            hipblasTrsmFn(handle, side, uplo, transA, diag, M, N, &h_alpha, dA, lda, dB, ldb));

        CHECK_HIP_ERROR(hipMemcpy(hB_host, dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dB, hB_device, sizeof(T) * B_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(
            hipblasTrsmFn(handle, side, uplo, transA, diag, M, N, d_alpha, dA, lda, dB, ldb));

        CHECK_HIP_ERROR(hipMemcpy(hB_device, dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // ref_trsm<T>(
        //     side, uplo, transA, diag, M, N, h_alpha, (const T*)hA, lda, hB_gold, ldb);

        // if enable norm check, norm check is invasive
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        hipblas_error_host   = norm_check_general<T>('F', M, N, ldb, hB_gold, hB_host);
        hipblas_error_device = norm_check_general<T>('F', M, N, ldb, hB_gold, hB_device);
        if(arg.unit_check)
        {
            unit_check_error(hipblas_error_host, tolerance);
            unit_check_error(hipblas_error_device, tolerance);
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
                hipblasTrsmFn(handle, side, uplo, transA, diag, M, N, d_alpha, dA, lda, dB, ldb));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrsmModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       trsm_gflop_count<T>(M, N, K),
                                       trsm_gbyte_count<T>(M, N, K),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}

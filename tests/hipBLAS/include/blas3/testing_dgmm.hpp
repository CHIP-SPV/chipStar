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

using hipblasDgmmModel = ArgumentModel<e_a_type, e_side, e_M, e_N, e_lda, e_incx, e_ldc>;

inline void testname_dgmm(const Arguments& arg, std::string& name)
{
    hipblasDgmmModel{}.test_name(arg, name);
}

template <typename T>
void testing_dgmm_bad_arg(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasDgmmFn = FORTRAN ? hipblasDgmm<T, true> : hipblasDgmm<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t M    = 101;
    int64_t N    = 100;
    int64_t lda  = 102;
    int64_t incx = 1;
    int64_t ldc  = 103;

    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;

    int64_t K = side == HIPBLAS_SIDE_LEFT ? M : N;

    device_vector<T> dA(N * lda);
    device_vector<T> dx(incx * K);
    device_vector<T> dC(N * ldc);

    EXPECT_HIPBLAS_STATUS(hipblasDgmmFn(nullptr, side, M, N, dA, lda, dx, incx, dC, ldc),
                          HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(
        hipblasDgmmFn(
            handle, (hipblasSideMode_t)HIPBLAS_FILL_MODE_FULL, M, N, dA, lda, dx, incx, dC, ldc),
        HIPBLAS_STATUS_INVALID_ENUM);

    if(arg.bad_arg_all)
    {
        EXPECT_HIPBLAS_STATUS(hipblasDgmmFn(handle, side, M, N, nullptr, lda, dx, incx, dC, ldc),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasDgmmFn(handle, side, M, N, dA, lda, nullptr, incx, dC, ldc),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasDgmmFn(handle, side, M, N, dA, lda, dx, incx, nullptr, ldc),
                              HIPBLAS_STATUS_INVALID_VALUE);
    }

    // If M == 0 || N == 0, can have nullptrs
    CHECK_HIPBLAS_ERROR(
        hipblasDgmmFn(handle, side, 0, N, nullptr, lda, nullptr, incx, nullptr, ldc));
    CHECK_HIPBLAS_ERROR(
        hipblasDgmmFn(handle, side, M, 0, nullptr, lda, nullptr, incx, nullptr, ldc));
}

template <typename T>
void testing_dgmm(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasDgmmFn = FORTRAN ? hipblasDgmm<T, true> : hipblasDgmm<T, false>;

    hipblasSideMode_t side = char2hipblas_side(arg.side);

    int M    = arg.M;
    int N    = arg.N;
    int lda  = arg.lda;
    int incx = arg.incx;
    int ldc  = arg.ldc;

    int    abs_incx = incx >= 0 ? incx : -incx;
    size_t A_size   = size_t(lda) * N;
    size_t C_size   = size_t(ldc) * N;
    int    k        = (side == HIPBLAS_SIDE_RIGHT ? N : M);
    size_t X_size   = size_t(abs_incx) * k;
    if(!X_size)
        X_size = 1;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || ldc < M || lda < M;
    if(invalid_size || !N || !M)
    {
        hipblasStatus_t actual
            = hipblasDgmmFn(handle, side, M, N, nullptr, lda, nullptr, incx, nullptr, ldc);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA_copy(A_size);
    host_vector<T> hx(X_size);
    host_vector<T> hx_copy(X_size);
    host_vector<T> hC(C_size);
    host_vector<T> hC_1(C_size);
    host_vector<T> hC_gold(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size);
    device_vector<T> dC(C_size);

    double gpu_time_used, hipblas_error;

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, M, N, lda, 0, 1, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hx, arg, k, abs_incx, 0, 1, hipblas_client_never_set_nan, false, true);
    hipblas_init_matrix(hC, arg, M, N, ldc, 0, 1, hipblas_client_never_set_nan);
    hA_copy = hA;
    hx_copy = hx;
    hC_1    = hC;
    hC_gold = hC;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasDgmmFn(handle, side, M, N, dA, lda, dx, incx, dC, ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation
        ptrdiff_t shift_x = incx < 0 ? -ptrdiff_t(incx) * (N - 1) : 0;
        for(size_t i1 = 0; i1 < M; i1++)
        {
            for(size_t i2 = 0; i2 < N; i2++)
            {
                if(HIPBLAS_SIDE_RIGHT == side)
                {
                    hC_gold[i1 + i2 * ldc] = hA_copy[i1 + i2 * lda] * hx_copy[shift_x + i2 * incx];
                }
                else
                {
                    hC_gold[i1 + i2 * ldc] = hA_copy[i1 + i2 * lda] * hx_copy[shift_x + i1 * incx];
                }
            }
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
        }

        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1);
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

            CHECK_HIPBLAS_ERROR(hipblasDgmmFn(handle, side, M, N, dA, lda, dx, incx, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasDgmmModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       dgmm_gflop_count<T>(M, N),
                                       dgmm_gbyte_count<T>(M, N, k),
                                       hipblas_error);
    }
}

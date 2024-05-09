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

using hipblasTrmvModel = ArgumentModel<e_a_type, e_uplo, e_transA, e_diag, e_N, e_lda, e_incx>;

inline void testname_trmv(const Arguments& arg, std::string& name)
{
    hipblasTrmvModel{}.test_name(arg, name);
}

template <typename T>
void testing_trmv_bad_arg(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrmvFn = FORTRAN ? hipblasTrmv<T, true> : hipblasTrmv<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasOperation_t transA = HIPBLAS_OP_N;
        hipblasFillMode_t  uplo   = HIPBLAS_FILL_MODE_UPPER;
        hipblasDiagType_t  diag   = HIPBLAS_DIAG_NON_UNIT;
        int64_t            N      = 100;
        int64_t            lda    = 100;
        int64_t            incx   = 1;

        device_vector<T> dA(N * lda);
        device_vector<T> dx(N * incx);

        EXPECT_HIPBLAS_STATUS(hipblasTrmvFn(nullptr, uplo, transA, diag, N, dA, lda, dx, incx),
                              HIPBLAS_STATUS_NOT_INITIALIZED);
        EXPECT_HIPBLAS_STATUS(
            hipblasTrmvFn(handle, HIPBLAS_FILL_MODE_FULL, transA, diag, N, dA, lda, dx, incx),
            HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(
            hipblasTrmvFn(
                handle, (hipblasFillMode_t)HIPBLAS_OP_N, transA, diag, N, dA, lda, dx, incx),
            HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasTrmvFn(handle,
                                            uplo,
                                            (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                            diag,
                                            N,
                                            dA,
                                            lda,
                                            dx,
                                            incx),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasTrmvFn(handle,
                                            uplo,
                                            transA,
                                            (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                                            N,
                                            dA,
                                            lda,
                                            dx,
                                            incx),
                              HIPBLAS_STATUS_INVALID_ENUM);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasTrmvFn(handle, uplo, transA, diag, N, nullptr, lda, dx, incx),
                HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(
                hipblasTrmvFn(handle, uplo, transA, diag, N, dA, lda, nullptr, incx),
                HIPBLAS_STATUS_INVALID_VALUE);
        }

        // With N == 0, can have all nullptrs
        CHECK_HIPBLAS_ERROR(
            hipblasTrmvFn(handle, uplo, transA, diag, 0, nullptr, lda, nullptr, incx));
    }
}

template <typename T>
void testing_trmv(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrmvFn = FORTRAN ? hipblasTrmv<T, true> : hipblasTrmv<T, false>;

    hipblasFillMode_t  uplo   = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(arg.diag);
    int                N      = arg.N;
    int                lda    = arg.lda;
    int                incx   = arg.incx;

    int    abs_incx = incx >= 0 ? incx : -incx;
    size_t x_size   = size_t(N) * abs_incx;
    size_t A_size   = size_t(lda) * N;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx;
    if(invalid_size || !N)
    {
        hipblasStatus_t actual
            = hipblasTrmvFn(handle, uplo, transA, diag, N, nullptr, lda, nullptr, incx);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(x_size);
    host_vector<T> hres(x_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(x_size);

    double gpu_time_used, hipblas_error;

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, N, N, lda, 0, 1, hipblas_client_never_set_nan, true, false);
    hipblas_init_vector(hx, arg, N, abs_incx, 0, 1, hipblas_client_never_set_nan, false, true);

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hres = hx;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * x_size, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasTrmvFn(handle, uplo, transA, diag, N, dA, lda, dx, incx));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hres.data(), dx, sizeof(T) * x_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        ref_trmv<T>(uplo, transA, diag, N, hA.data(), lda, hx.data(), incx);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, abs_incx, hx, hres);
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, N, abs_incx, hx.data(), hres.data());
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

            CHECK_HIPBLAS_ERROR(hipblasTrmvFn(handle, uplo, transA, diag, N, dA, lda, dx, incx));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrmvModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       trmv_gflop_count<T>(N),
                                       trmv_gbyte_count<T>(N),
                                       hipblas_error);
    }
}

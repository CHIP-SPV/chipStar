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

using hipblasGbmvModel = ArgumentModel<e_a_type,
                                       e_transA,
                                       e_M,
                                       e_N,
                                       e_KL,
                                       e_KU,
                                       e_alpha,
                                       e_lda,
                                       e_incx,
                                       e_beta,
                                       e_incy>;

inline void testname_gbmv(const Arguments& arg, std::string& name)
{
    hipblasGbmvModel{}.test_name(arg, name);
}

template <typename T>
void testing_gbmv_bad_arg(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGbmvFn = FORTRAN ? hipblasGbmv<T, true> : hipblasGbmv<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasOperation_t transA = HIPBLAS_OP_N;
        int64_t            N      = 100;
        int64_t            M      = 100;
        int64_t            KL     = 5;
        int64_t            KU     = 5;
        int64_t            lda    = 100;
        int64_t            incx   = 1;
        int64_t            incy   = 1;

        device_vector<T> d_alpha(1), d_beta(1), d_one(1), d_zero(1);

        const T  h_alpha(1), h_beta(2), h_one(1), h_zero(0);
        const T* alpha = &h_alpha;
        const T* beta  = &h_beta;
        const T* one   = &h_one;
        const T* zero  = &h_zero;

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

        device_vector<T> dA(N * lda);
        device_vector<T> dx(N * incx);
        device_vector<T> dy(M * incy);

        EXPECT_HIPBLAS_STATUS(
            hipblasGbmvFn(nullptr, transA, M, N, KL, KU, alpha, dA, lda, dx, incx, beta, dy, incy),
            HIPBLAS_STATUS_NOT_INITIALIZED);
        EXPECT_HIPBLAS_STATUS(hipblasGbmvFn(handle,
                                            (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                            M,
                                            N,
                                            KL,
                                            KU,
                                            alpha,
                                            dA,
                                            lda,
                                            dx,
                                            incx,
                                            beta,
                                            dy,
                                            incy),
                              HIPBLAS_STATUS_INVALID_ENUM);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasGbmvFn(
                    handle, transA, M, N, KL, KU, nullptr, dA, lda, dx, incx, beta, dy, incy),
                HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(
                hipblasGbmvFn(
                    handle, transA, M, N, KL, KU, alpha, dA, lda, dx, incx, nullptr, dy, incy),
                HIPBLAS_STATUS_INVALID_VALUE);

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                // For device mode in rocBLAS we don't have checks for dA, dx, dy as we may be able to quick return
                EXPECT_HIPBLAS_STATUS(hipblasGbmvFn(handle,
                                                    transA,
                                                    M,
                                                    N,
                                                    KL,
                                                    KU,
                                                    alpha,
                                                    nullptr,
                                                    lda,
                                                    dx,
                                                    incx,
                                                    beta,
                                                    dy,
                                                    incy),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasGbmvFn(handle,
                                                    transA,
                                                    M,
                                                    N,
                                                    KL,
                                                    KU,
                                                    alpha,
                                                    dA,
                                                    lda,
                                                    nullptr,
                                                    incx,
                                                    beta,
                                                    dy,
                                                    incy),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasGbmvFn(handle,
                                                    transA,
                                                    M,
                                                    N,
                                                    KL,
                                                    KU,
                                                    alpha,
                                                    dA,
                                                    lda,
                                                    dx,
                                                    incx,
                                                    beta,
                                                    nullptr,
                                                    incy),
                                      HIPBLAS_STATUS_INVALID_VALUE);
            }

            // With alpha == 0 can have x nullptr
            CHECK_HIPBLAS_ERROR(hipblasGbmvFn(
                handle, transA, M, N, KL, KU, zero, nullptr, lda, nullptr, incx, beta, dy, incy));

            // With alpha == 0 && beta == 1, all other ptrs can be nullptr
            CHECK_HIPBLAS_ERROR(hipblasGbmvFn(handle,
                                              transA,
                                              M,
                                              N,
                                              KL,
                                              KU,
                                              zero,
                                              nullptr,
                                              lda,
                                              nullptr,
                                              incx,
                                              one,
                                              nullptr,
                                              incy));
        }

        // With M == 0 || N == 0, can have all nullptrs
        CHECK_HIPBLAS_ERROR(hipblasGbmvFn(handle,
                                          transA,
                                          0,
                                          N,
                                          KL,
                                          KU,
                                          nullptr,
                                          nullptr,
                                          lda,
                                          nullptr,
                                          incx,
                                          nullptr,
                                          nullptr,
                                          incy));
        CHECK_HIPBLAS_ERROR(hipblasGbmvFn(handle,
                                          transA,
                                          M,
                                          0,
                                          KL,
                                          KU,
                                          nullptr,
                                          nullptr,
                                          lda,
                                          nullptr,
                                          incx,
                                          nullptr,
                                          nullptr,
                                          incy));
    }
}

template <typename T>
void testing_gbmv(const Arguments& arg)
{
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGbmvFn = FORTRAN ? hipblasGbmv<T, true> : hipblasGbmv<T, false>;

    int M    = arg.M;
    int N    = arg.N;
    int KL   = arg.KL;
    int KU   = arg.KU;
    int lda  = arg.lda;
    int incx = arg.incx;
    int incy = arg.incy;

    size_t A_size = size_t(lda) * N;
    int    dim_x;
    int    dim_y;

    hipblasOperation_t transA = char2hipblas_operation(arg.transA);

    if(transA == HIPBLAS_OP_N)
    {
        dim_x = N;
        dim_y = M;
    }
    else
    {
        dim_x = M;
        dim_y = N;
    }

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || lda < KL + KU + 1 || !incx || !incy || KL < 0 || KU < 0;
    if(invalid_size || !M || !N)
    {
        hipblasStatus_t actual = hipblasGbmvFn(handle,
                                               transA,
                                               M,
                                               N,
                                               KL,
                                               KU,
                                               nullptr,
                                               nullptr,
                                               lda,
                                               nullptr,
                                               incx,
                                               nullptr,
                                               nullptr,
                                               incy);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return;
    }

    int abs_incx = incx >= 0 ? incx : -incx;
    int abs_incy = incy >= 0 ? incy : -incy;

    size_t X_size = size_t(dim_x) * abs_incx;
    size_t Y_size = size_t(dim_y) * abs_incy;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(X_size);
    host_vector<T> hy(Y_size);
    host_vector<T> hy_host(Y_size);
    host_vector<T> hy_device(Y_size);
    host_vector<T> hy_cpu(Y_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size);
    device_vector<T> dy(Y_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, lda, N, lda, 0, 1, hipblas_client_alpha_sets_nan, true, false);
    hipblas_init_vector(hx, arg, dim_x, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, arg, dim_y, abs_incy, 0, 1, hipblas_client_beta_sets_nan);

    // copy vector is easy in STL; hy_cpu = hy: save a copy in hy_cpu which will be output of CPU BLAS
    hy_cpu = hy;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasGbmvFn(
            handle, transA, M, N, KL, KU, (T*)&h_alpha, dA, lda, dx, incx, (T*)&h_beta, dy, incy));

        CHECK_HIP_ERROR(hipMemcpy(hy_host.data(), dy, sizeof(T) * Y_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasGbmvFn(
            handle, transA, M, N, KL, KU, d_alpha, dA, lda, dx, incx, d_beta, dy, incy));

        CHECK_HIP_ERROR(hipMemcpy(hy_device.data(), dy, sizeof(T) * Y_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        ref_gbmv<T>(transA,
                    M,
                    N,
                    KL,
                    KU,
                    h_alpha,
                    hA.data(),
                    lda,
                    hx.data(),
                    incx,
                    h_beta,
                    hy_cpu.data(),
                    incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, dim_y, abs_incy, hy_cpu, hy_host);
            unit_check_general<T>(1, dim_y, abs_incy, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_cpu.data(), hy_host.data());
            hipblas_error_device
                = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_cpu.data(), hy_device.data());
        }
    }

    if(arg.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGbmvFn(
                handle, transA, M, N, KL, KU, d_alpha, dA, lda, dx, incx, d_beta, dy, incy));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGbmvModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       gbmv_gflop_count<T>(transA, M, N, KL, KU),
                                       gbmv_gbyte_count<T>(transA, M, N, KL, KU),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}

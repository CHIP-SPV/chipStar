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

#include "gtest/gtest.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using hipblasGeqrfModel = ArgumentModel<e_a_type, e_M, e_N, e_lda>;

inline void testname_geqrf(const Arguments& arg, std::string& name)
{
    hipblasGeqrfModel{}.test_name(arg, name);
}

template <typename T>
void setup_geqrf_testing(
    host_vector<T>& hA, device_vector<T>& dA, device_vector<T>& dIpiv, int M, int N, int lda)
{
    size_t A_size = size_t(lda) * N;
    int    K      = std::min(M, N);

    // Initial hA on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda);

    // scale A to avoid singularities
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            if(i == j)
                hA[i + j * lda] += 400;
            else
                hA[i + j * lda] -= 4;
        }
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, K * sizeof(T)));
}

template <typename T>
void testing_geqrf_bad_arg(const Arguments& arg)
{
    auto hipblasGeqrfFn
        = arg.api == hipblas_client_api::FORTRAN ? hipblasGeqrf<T, true> : hipblasGeqrf<T, false>;

    hipblasLocalHandle handle(arg);
    const int          M      = 100;
    const int          N      = 101;
    const int          lda    = 102;
    const size_t       A_size = size_t(N) * lda;
    const int          K      = std::min(M, N);

    host_vector<T> hA(A_size);

    device_vector<T> dA(A_size);
    device_vector<T> dIpiv(K);
    int              info         = 0;
    int              expectedInfo = 0;

    setup_geqrf_testing(hA, dA, dIpiv, M, N, lda);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfFn(handle, M, N, dA, lda, dIpiv, nullptr),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfFn(handle, -1, N, dA, lda, dIpiv, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -1;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfFn(handle, M, -1, dA, lda, dIpiv, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -2;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfFn(handle, M, N, nullptr, lda, dIpiv, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -3;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfFn(handle, M, N, dA, M - 1, dIpiv, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -4;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfFn(handle, M, N, dA, lda, nullptr, &info),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -5;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // If M == 0 || N == 0, A and ipiv can be nullptr
    EXPECT_HIPBLAS_STATUS(hipblasGeqrfFn(handle, 0, N, nullptr, lda, nullptr, &info),
                          HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGeqrfFn(handle, M, 0, nullptr, lda, nullptr, &info),
                          HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);
}

template <typename T>
void testing_geqrf(const Arguments& arg)
{
    using U             = real_t<T>;
    bool FORTRAN        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGeqrfFn = FORTRAN ? hipblasGeqrf<T, true> : hipblasGeqrf<T, false>;

    int M   = arg.M;
    int N   = arg.N;
    int K   = std::min(M, N);
    int lda = arg.lda;

    size_t A_size    = size_t(lda) * N;
    int    Ipiv_size = K;
    int    info;

    hipblasLocalHandle handle(arg);

    // Check to prevent memory allocation error
    bool invalid_size = M < 0 || N < 0 || lda < std::max(1, M);
    if(invalid_size || !M || !N)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA1(A_size);
    host_vector<T> hIpiv(Ipiv_size);
    host_vector<T> hIpiv1(Ipiv_size);

    device_vector<T> dA(A_size);
    device_vector<T> dIpiv(Ipiv_size);

    double gpu_time_used, hipblas_error;

    setup_geqrf_testing(hA, dA, dIpiv, M, N, lda);

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasGeqrfFn(handle, M, N, dA, lda, dIpiv, &info));

    // Copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA1, dA, A_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hIpiv1, dIpiv, Ipiv_size * sizeof(T), hipMemcpyDeviceToHost));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        // Workspace query
        host_vector<T> work(1);
        ref_geqrf(M, N, hA.data(), lda, hIpiv.data(), work.data(), -1);
        int lwork = type2int(work[0]);

        // Perform factorization
        work = host_vector<T>(lwork);
        ref_geqrf(M, N, hA.data(), lda, hIpiv.data(), work.data(), lwork);

        double e1     = norm_check_general<T>('F', M, N, lda, hA, hA1);
        double e2     = norm_check_general<T>('F', K, 1, K, hIpiv, hIpiv1);
        hipblas_error = e1 + e2;

        if(arg.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = eps * 2000;

            unit_check_error(e1, tolerance);
            unit_check_error(e2, tolerance);
            int zero = 0;
            unit_check_general(1, 1, 1, &zero, &info);
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

            CHECK_HIPBLAS_ERROR(hipblasGeqrfFn(handle, M, N, dA, lda, dIpiv, &info));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGeqrfModel{}.log_args<T>(std::cout,
                                        arg,
                                        gpu_time_used,
                                        geqrf_gflop_count<T>(N, M),
                                        ArgumentLogging::NA_value,
                                        hipblas_error);
    }
}

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

using hipblasGetrfModel = ArgumentModel<e_a_type, e_N, e_lda>;

inline void testname_getrf(const Arguments& arg, std::string& name)
{
    hipblasGetrfModel{}.test_name(arg, name);
}

template <typename T>
void testing_getrf_bad_arg(const Arguments& arg)
{
    bool FORTRAN        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGetrfFn = FORTRAN ? hipblasGetrf<T, true> : hipblasGetrf<T, false>;

    hipblasLocalHandle handle(arg);
    int64_t            N         = 101;
    int64_t            M         = N;
    int64_t            lda       = 102;
    int64_t            A_size    = N * lda;
    int64_t            Ipiv_size = std::min(M, N);

    device_vector<T>   dA(A_size);
    device_vector<int> dIpiv(Ipiv_size);
    device_vector<int> dInfo(1);

    EXPECT_HIPBLAS_STATUS(hipblasGetrfFn(nullptr, N, dA, lda, dIpiv, dInfo),
                          HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(hipblasGetrfFn(handle, -1, dA, lda, dIpiv, dInfo),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasGetrfFn(handle, N, dA, N - 1, dIpiv, dInfo),
                          HIPBLAS_STATUS_INVALID_VALUE);

    // If N == 0, A and ipiv can be nullptr
    CHECK_HIPBLAS_ERROR(hipblasGetrfFn(handle, 0, nullptr, lda, nullptr, dInfo));

    if(arg.bad_arg_all)
    {
        EXPECT_HIPBLAS_STATUS(hipblasGetrfFn(handle, N, nullptr, lda, dIpiv, dInfo),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasGetrfFn(handle, N, dA, lda, dIpiv, nullptr),
                              HIPBLAS_STATUS_INVALID_VALUE);
    }
}

template <typename T>
void testing_getrf(const Arguments& arg)
{
    using U             = real_t<T>;
    bool FORTRAN        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGetrfFn = FORTRAN ? hipblasGetrf<T, true> : hipblasGetrf<T, false>;

    int M   = arg.N;
    int N   = arg.N;
    int lda = arg.lda;

    size_t A_size    = size_t(lda) * N;
    int    Ipiv_size = std::min(M, N);

    // Check to prevent memory allocation error
    if(M < 0 || N < 0 || lda < M)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T>   hA(A_size);
    host_vector<T>   hA1(A_size);
    host_vector<int> hIpiv(Ipiv_size);
    host_vector<int> hIpiv1(Ipiv_size);
    host_vector<int> hInfo(1);
    host_vector<int> hInfo1(1);

    device_vector<T>   dA(A_size);
    device_vector<int> dIpiv(Ipiv_size);
    device_vector<int> dInfo(1);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(arg);

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
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, Ipiv_size * sizeof(int)));
    CHECK_HIP_ERROR(hipMemset(dInfo, 0, sizeof(int)));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasGetrfFn(handle, N, dA, lda, dIpiv, dInfo));

        // Copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hA1, dA, A_size * sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hIpiv1, dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hInfo1, dInfo, sizeof(int), hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */
        hInfo[0] = ref_getrf(M, N, hA.data(), lda, hIpiv.data());

        hipblas_error = norm_check_general<T>('F', M, N, lda, hA, hA1);
        if(arg.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = eps * 2000;

            unit_check_error(hipblas_error, tolerance);
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

            CHECK_HIPBLAS_ERROR(hipblasGetrfFn(handle, N, dA, lda, dIpiv, dInfo));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGetrfModel{}.log_args<T>(std::cout,
                                        arg,
                                        gpu_time_used,
                                        getrf_gflop_count<T>(N, M),
                                        ArgumentLogging::NA_value,
                                        hipblas_error);
    }
}

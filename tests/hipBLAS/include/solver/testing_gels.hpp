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

using hipblasGelsModel = ArgumentModel<e_a_type, e_transA, e_M, e_N, e_lda, e_ldb>;

inline void testname_gels(const Arguments& arg, std::string& name)
{
    hipblasGelsModel{}.test_name(arg, name);
}

template <typename T>
void testing_gels_bad_arg(const Arguments& arg)
{
    auto hipblasGelsFn
        = arg.api == hipblas_client_api::FORTRAN ? hipblasGels<T, true> : hipblasGels<T, false>;

    hipblasLocalHandle       handle(arg);
    const int                M     = 100;
    const int                N     = 101;
    const int                nrhs  = 10;
    const int                lda   = 102;
    const int                ldb   = 103;
    const hipblasOperation_t opN   = HIPBLAS_OP_N;
    const hipblasOperation_t opBad = is_complex<T> ? HIPBLAS_OP_T : HIPBLAS_OP_C;

    const size_t A_size = size_t(lda) * N;
    const size_t B_size = size_t(ldb) * nrhs;

    device_vector<T>   dA(A_size);
    device_vector<T>   dB(B_size);
    device_vector<int> dInfo(1);
    int                info = 0;
    int                expectedInfo;

    EXPECT_HIPBLAS_STATUS(hipblasGelsFn(handle, opN, M, N, nrhs, dA, lda, dB, ldb, nullptr, dInfo),
                          HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(hipblasGelsFn(handle, opBad, M, N, nrhs, dA, lda, dB, ldb, &info, dInfo),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -1;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGelsFn(handle, opN, -1, N, nrhs, dA, lda, dB, ldb, &info, dInfo),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -2;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGelsFn(handle, opN, M, -1, nrhs, dA, lda, dB, ldb, &info, dInfo),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -3;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGelsFn(handle, opN, M, N, -1, dA, lda, dB, ldb, &info, dInfo),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -4;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsFn(handle, opN, M, N, nrhs, nullptr, lda, dB, ldb, &info, dInfo),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -5;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGelsFn(handle, opN, M, N, nrhs, dA, M - 1, dB, ldb, &info, dInfo),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -6;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsFn(handle, opN, M, N, nrhs, dA, lda, nullptr, ldb, &info, dInfo),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -7;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // Explicit values to check for ldb < M and ldb < N
    EXPECT_HIPBLAS_STATUS(
        hipblasGelsFn(handle, opN, 100, 200, nrhs, dA, lda, dB, 199, &info, dInfo),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -8;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsFn(handle, opN, 200, 100, nrhs, dA, 201, dB, 199, &info, dInfo),
        HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -8;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(hipblasGelsFn(handle, opN, M, N, nrhs, dA, lda, dB, ldb, &info, nullptr),
                          HIPBLAS_STATUS_INVALID_VALUE);
    expectedInfo = -10;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // If M == 0 || N == 0, A can be nullptr
    EXPECT_HIPBLAS_STATUS(
        hipblasGelsFn(handle, opN, 0, N, nrhs, nullptr, lda, dB, ldb, &info, dInfo),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    EXPECT_HIPBLAS_STATUS(
        hipblasGelsFn(handle, opN, M, 0, nrhs, nullptr, lda, dB, ldb, &info, dInfo),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // If nrhs == 0, B can be nullptr
    EXPECT_HIPBLAS_STATUS(hipblasGelsFn(handle, opN, M, N, 0, dA, lda, nullptr, ldb, &info, dInfo),
                          HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);

    // If M == 0 && N == 0, B can be nullptr
    EXPECT_HIPBLAS_STATUS(
        hipblasGelsFn(handle, opN, 0, 0, nrhs, nullptr, lda, nullptr, ldb, &info, dInfo),
        HIPBLAS_STATUS_SUCCESS);
    expectedInfo = 0;
    unit_check_general(1, 1, 1, &expectedInfo, &info);
}

template <typename T>
void testing_gels(const Arguments& arg)
{
    using U            = real_t<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGelsFn = FORTRAN ? hipblasGels<T, true> : hipblasGels<T, false>;

    char transc = arg.transA;
    int  N      = arg.N;
    int  M      = arg.M;
    int  nrhs   = arg.K;
    int  lda    = arg.lda;
    int  ldb    = arg.ldb;

    if(is_complex<T> && transc == 'T')
        transc = 'C';
    else if(!is_complex<T> && transc == 'C')
        transc = 'T';

    hipblasOperation_t trans = char2hipblas_operation(transc);

    size_t A_size = size_t(lda) * N;
    size_t B_size = size_t(ldb) * nrhs;

    // Check to prevent memory allocation error
    if(M < 0 || N < 0 || nrhs < 0 || lda < M || ldb < M || ldb < N)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hB_res(B_size);
    int            info, info_res;
    int            info_input(-1);

    device_vector<T>   dA(A_size);
    device_vector<T>   dB(B_size);
    device_vector<int> dInfo(1);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(arg);

    // Initial hA, hB, hX on CPU
    srand(1);
    hipblas_init<T>(hA, true);
    hipblas_init<T>(hB);
    hB_res = hB;

    // scale A to avoid singularities
    for(int i = 0; i < N; i++)
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
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, B_size * sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(
            hipblasGelsFn(handle, trans, M, N, nrhs, dA, lda, dB, ldb, &info_input, dInfo));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hB_res, dB, B_size * sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(&info_res, dInfo, sizeof(int), hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */
        int            sizeW = std::max(1, std::min(M, N) + std::max(std::min(M, N), nrhs));
        host_vector<T> hW(sizeW);

        info = ref_gels(transc, M, N, nrhs, hA.data(), lda, hB.data(), ldb, hW.data(), sizeW);

        hipblas_error
            = norm_check_general<T>('F', std::max(M, N), nrhs, ldb, hB.data(), hB_res.data());

        if(info != info_res)
            hipblas_error += 1.0;
        if(info_input != 0)
            hipblas_error += 1.0;

        if(arg.unit_check)
        {
            double eps       = std::numeric_limits<U>::epsilon();
            double tolerance = N * eps * 100;
            int    zero      = 0;

            unit_check_error(hipblas_error, tolerance);
            unit_check_general(1, 1, 1, &zero, &info_input);
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

            CHECK_HIPBLAS_ERROR(
                hipblasGelsFn(handle, trans, M, N, nrhs, dA, lda, dB, ldb, &info_input, dInfo));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGelsModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       ArgumentLogging::NA_value,
                                       ArgumentLogging::NA_value,
                                       hipblas_error);
    }
}

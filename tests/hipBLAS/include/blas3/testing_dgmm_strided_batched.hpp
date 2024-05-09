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

using hipblasDgmmStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_side,
                                                     e_M,
                                                     e_N,
                                                     e_lda,
                                                     e_incx,
                                                     e_ldc,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_dgmm_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasDgmmStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_dgmm_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasDgmmStridedBatchedFn
        = FORTRAN ? hipblasDgmmStridedBatched<T, true> : hipblasDgmmStridedBatched<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t M           = 101;
    int64_t N           = 100;
    int64_t lda         = 102;
    int64_t incx        = 1;
    int64_t ldc         = 103;
    int64_t batch_count = 2;

    hipblasSideMode_t side = HIPBLAS_SIDE_LEFT;

    int64_t K = side == HIPBLAS_SIDE_LEFT ? M : N;

    hipblasStride strideA = N * lda;
    hipblasStride stridex = K * incx;
    hipblasStride strideC = N * ldc;

    device_vector<T> dA(strideA * batch_count);
    device_vector<T> dx(stridex * batch_count);
    device_vector<T> dC(strideC * batch_count);

    EXPECT_HIPBLAS_STATUS(hipblasDgmmStridedBatchedFn(nullptr,
                                                      side,
                                                      M,
                                                      N,
                                                      dA,
                                                      lda,
                                                      strideA,
                                                      dx,
                                                      incx,
                                                      stridex,
                                                      dC,
                                                      ldc,
                                                      strideC,
                                                      batch_count),
                          HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(hipblasDgmmStridedBatchedFn(handle,
                                                      (hipblasSideMode_t)HIPBLAS_FILL_MODE_FULL,
                                                      M,
                                                      N,
                                                      dA,
                                                      lda,
                                                      strideA,
                                                      dx,
                                                      incx,
                                                      stridex,
                                                      dC,
                                                      ldc,
                                                      strideC,
                                                      batch_count),
                          HIPBLAS_STATUS_INVALID_ENUM);

    if(arg.bad_arg_all)
    {
        EXPECT_HIPBLAS_STATUS(hipblasDgmmStridedBatchedFn(handle,
                                                          side,
                                                          M,
                                                          N,
                                                          nullptr,
                                                          lda,
                                                          strideA,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          dC,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasDgmmStridedBatchedFn(handle,
                                                          side,
                                                          M,
                                                          N,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          nullptr,
                                                          incx,
                                                          stridex,
                                                          dC,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasDgmmStridedBatchedFn(handle,
                                                          side,
                                                          M,
                                                          N,
                                                          dA,
                                                          lda,
                                                          strideA,
                                                          dx,
                                                          incx,
                                                          stridex,
                                                          nullptr,
                                                          ldc,
                                                          strideC,
                                                          batch_count),
                              HIPBLAS_STATUS_INVALID_VALUE);
    }

    // If M == 0 || N == 0 || batch_count == 0, can have nullptrs
    CHECK_HIPBLAS_ERROR(hipblasDgmmStridedBatchedFn(handle,
                                                    side,
                                                    0,
                                                    N,
                                                    nullptr,
                                                    lda,
                                                    strideA,
                                                    nullptr,
                                                    incx,
                                                    stridex,
                                                    nullptr,
                                                    ldc,
                                                    strideC,
                                                    batch_count));
    CHECK_HIPBLAS_ERROR(hipblasDgmmStridedBatchedFn(handle,
                                                    side,
                                                    M,
                                                    0,
                                                    nullptr,
                                                    lda,
                                                    strideA,
                                                    nullptr,
                                                    incx,
                                                    stridex,
                                                    nullptr,
                                                    ldc,
                                                    strideC,
                                                    batch_count));
    CHECK_HIPBLAS_ERROR(hipblasDgmmStridedBatchedFn(handle,
                                                    side,
                                                    M,
                                                    N,
                                                    nullptr,
                                                    lda,
                                                    strideA,
                                                    nullptr,
                                                    incx,
                                                    stridex,
                                                    nullptr,
                                                    ldc,
                                                    strideC,
                                                    0));
}

template <typename T>
void testing_dgmm_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasDgmmStridedBatchedFn
        = FORTRAN ? hipblasDgmmStridedBatched<T, true> : hipblasDgmmStridedBatched<T, false>;

    hipblasSideMode_t side = char2hipblas_side(arg.side);

    int    M            = arg.M;
    int    N            = arg.N;
    int    lda          = arg.lda;
    int    incx         = arg.incx;
    int    ldc          = arg.ldc;
    int    batch_count  = arg.batch_count;
    double stride_scale = arg.stride_scale;
    int    k            = (side == HIPBLAS_SIDE_RIGHT ? N : M);

    int           abs_incx = incx >= 0 ? incx : -incx;
    hipblasStride stride_A = size_t(lda) * N * stride_scale;
    hipblasStride stride_x = size_t(abs_incx) * k * stride_scale;
    hipblasStride stride_C = size_t(ldc) * N * stride_scale;
    if(!stride_x)
        stride_x = 1;

    size_t A_size = size_t(stride_A) * batch_count;
    size_t C_size = size_t(stride_C) * batch_count;
    size_t X_size = size_t(stride_x) * batch_count;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || ldc < M || lda < M || batch_count < 0;
    if(invalid_size || !N || !M || !batch_count)
    {
        hipblasStatus_t actual = hipblasDgmmStridedBatchedFn(handle,
                                                             side,
                                                             M,
                                                             N,
                                                             nullptr,
                                                             lda,
                                                             stride_A,
                                                             nullptr,
                                                             incx,
                                                             stride_x,
                                                             nullptr,
                                                             ldc,
                                                             stride_C,
                                                             batch_count);
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
    hipblas_init_matrix(
        hA, arg, M, N, lda, stride_A, batch_count, hipblas_client_never_set_nan, true);
    hipblas_init_vector(
        hx, arg, k, abs_incx, stride_x, batch_count, hipblas_client_never_set_nan, false, true);
    hipblas_init_matrix(hC, arg, M, N, ldc, stride_C, batch_count, hipblas_client_never_set_nan);
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
        CHECK_HIPBLAS_ERROR(hipblasDgmmStridedBatchedFn(handle,
                                                        side,
                                                        M,
                                                        N,
                                                        dA,
                                                        lda,
                                                        stride_A,
                                                        dx,
                                                        incx,
                                                        stride_x,
                                                        dC,
                                                        ldc,
                                                        stride_C,
                                                        batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation
        ptrdiff_t shift_x = incx < 0 ? -ptrdiff_t(incx) * (N - 1) : 0;
        for(int b = 0; b < batch_count; b++)
        {
            auto hC_goldb = hC_gold + b * stride_C;
            auto hA_copyb = hA_copy + b * stride_A;
            auto hx_copyb = hx_copy + b * stride_x;
            for(size_t i1 = 0; i1 < M; i1++)
            {
                for(size_t i2 = 0; i2 < N; i2++)
                {
                    if(HIPBLAS_SIDE_RIGHT == side)
                    {
                        hC_goldb[i1 + i2 * ldc]
                            = hA_copyb[i1 + i2 * lda] * hx_copyb[shift_x + i2 * incx];
                    }
                    else
                    {
                        hC_goldb[i1 + i2 * ldc]
                            = hA_copyb[i1 + i2 * lda] * hx_copyb[shift_x + i1 * incx];
                    }
                }
            }
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_gold, hC_1);
        }

        if(arg.norm_check)
        {
            hipblas_error
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_gold, hC_1, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasDgmmStridedBatchedFn(handle,
                                                            side,
                                                            M,
                                                            N,
                                                            dA,
                                                            lda,
                                                            stride_A,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            dC,
                                                            ldc,
                                                            stride_C,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        hipblasDgmmStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     dgmm_gflop_count<T>(M, N),
                                                     dgmm_gbyte_count<T>(M, N, k),
                                                     hipblas_error);
    }
}

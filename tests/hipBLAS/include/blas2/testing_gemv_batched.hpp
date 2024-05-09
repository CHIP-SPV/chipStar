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

using hipblasGemvBatchedModel = ArgumentModel<e_a_type,
                                              e_transA,
                                              e_M,
                                              e_N,
                                              e_alpha,
                                              e_lda,
                                              e_incx,
                                              e_beta,
                                              e_incy,
                                              e_batch_count>;

inline void testname_gemv_batched(const Arguments& arg, std::string& name)
{
    hipblasGemvBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_gemv_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGemvBatchedFn
        = FORTRAN ? hipblasGemvBatched<T, true> : hipblasGemvBatched<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasOperation_t transA      = HIPBLAS_OP_N;
        int64_t            N           = 100;
        int64_t            M           = 100;
        int64_t            lda         = 100;
        int64_t            incx        = 1;
        int64_t            incy        = 1;
        int64_t            batch_count = 2;

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

        device_batch_vector<T> dA(N * lda, 1, batch_count);
        device_batch_vector<T> dx(N, incx, batch_count);
        device_batch_vector<T> dy(M, incy, batch_count);

        EXPECT_HIPBLAS_STATUS(hipblasGemvBatchedFn(handle,
                                                   (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                                   M,
                                                   N,
                                                   alpha,
                                                   dA.ptr_on_device(),
                                                   lda,
                                                   dx.ptr_on_device(),
                                                   incx,
                                                   beta,
                                                   dy.ptr_on_device(),
                                                   incy,
                                                   batch_count),
                              HIPBLAS_STATUS_INVALID_ENUM);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(hipblasGemvBatchedFn(nullptr,
                                                       transA,
                                                       M,
                                                       N,
                                                       alpha,
                                                       dA.ptr_on_device(),
                                                       lda,
                                                       dx.ptr_on_device(),
                                                       incx,
                                                       beta,
                                                       dy.ptr_on_device(),
                                                       incy,
                                                       batch_count),
                                  HIPBLAS_STATUS_NOT_INITIALIZED);

            EXPECT_HIPBLAS_STATUS(hipblasGemvBatchedFn(handle,
                                                       transA,
                                                       M,
                                                       N,
                                                       nullptr,
                                                       dA.ptr_on_device(),
                                                       lda,
                                                       dx.ptr_on_device(),
                                                       incx,
                                                       beta,
                                                       dy.ptr_on_device(),
                                                       incy,
                                                       batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasGemvBatchedFn(handle,
                                                       transA,
                                                       M,
                                                       N,
                                                       alpha,
                                                       dA.ptr_on_device(),
                                                       lda,
                                                       dx.ptr_on_device(),
                                                       incx,
                                                       nullptr,
                                                       dy.ptr_on_device(),
                                                       incy,
                                                       batch_count),
                                  HIPBLAS_STATUS_INVALID_VALUE);

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                // For device mode in rocBLAS we don't have checks for dA, dx, dy as we may be able to quick return
                EXPECT_HIPBLAS_STATUS(hipblasGemvBatchedFn(handle,
                                                           transA,
                                                           M,
                                                           N,
                                                           alpha,
                                                           nullptr,
                                                           lda,
                                                           dx.ptr_on_device(),
                                                           incx,
                                                           beta,
                                                           dy.ptr_on_device(),
                                                           incy,
                                                           batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasGemvBatchedFn(handle,
                                                           transA,
                                                           M,
                                                           N,
                                                           alpha,
                                                           dA.ptr_on_device(),
                                                           lda,
                                                           nullptr,
                                                           incx,
                                                           beta,
                                                           dy.ptr_on_device(),
                                                           incy,
                                                           batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasGemvBatchedFn(handle,
                                                           transA,
                                                           M,
                                                           N,
                                                           alpha,
                                                           dA.ptr_on_device(),
                                                           lda,
                                                           dx.ptr_on_device(),
                                                           incx,
                                                           beta,
                                                           nullptr,
                                                           incy,
                                                           batch_count),
                                      HIPBLAS_STATUS_INVALID_VALUE);
            }

            // With alpha == 0 can have x nullptr
            CHECK_HIPBLAS_ERROR(hipblasGemvBatchedFn(handle,
                                                     transA,
                                                     M,
                                                     N,
                                                     zero,
                                                     nullptr,
                                                     lda,
                                                     nullptr,
                                                     incx,
                                                     beta,
                                                     dy.ptr_on_device(),
                                                     incy,
                                                     batch_count));

            // With alpha == 0 && beta == 1, all other ptrs can be nullptr
            CHECK_HIPBLAS_ERROR(hipblasGemvBatchedFn(handle,
                                                     transA,
                                                     M,
                                                     N,
                                                     zero,
                                                     nullptr,
                                                     lda,
                                                     nullptr,
                                                     incx,
                                                     one,
                                                     nullptr,
                                                     incy,
                                                     batch_count));
        }

        // With M == 0 || N == 0, can have all nullptrs
        CHECK_HIPBLAS_ERROR(hipblasGemvBatchedFn(handle,
                                                 transA,
                                                 0,
                                                 N,
                                                 nullptr,
                                                 nullptr,
                                                 lda,
                                                 nullptr,
                                                 incx,
                                                 nullptr,
                                                 nullptr,
                                                 incy,
                                                 batch_count));
        CHECK_HIPBLAS_ERROR(hipblasGemvBatchedFn(handle,
                                                 transA,
                                                 M,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 lda,
                                                 nullptr,
                                                 incx,
                                                 nullptr,
                                                 nullptr,
                                                 incy,
                                                 batch_count));
        CHECK_HIPBLAS_ERROR(hipblasGemvBatchedFn(
            handle, transA, M, N, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy, 0));
    }
}

template <typename T>
void testing_gemv_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGemvBatchedFn
        = FORTRAN ? hipblasGemvBatched<T, true> : hipblasGemvBatched<T, false>;

    int M    = arg.M;
    int N    = arg.N;
    int lda  = arg.lda;
    int incx = arg.incx;
    int incy = arg.incy;

    size_t A_size = size_t(lda) * N;
    size_t dim_x;
    size_t dim_y;

    int batch_count = arg.batch_count;

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
    bool invalid_size = M < 0 || N < 0 || lda < M || lda < 1 || !incx || !incy || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        if(!invalid_size || arg.bad_arg_all)
        {
            // cublas backend doesn't support nullptrs with bad input sizes
            hipblasStatus_t actual = hipblasGemvBatchedFn(handle,
                                                          transA,
                                                          M,
                                                          N,
                                                          nullptr,
                                                          nullptr,
                                                          lda,
                                                          nullptr,
                                                          incx,
                                                          nullptr,
                                                          nullptr,
                                                          incy,
                                                          batch_count);
            EXPECT_HIPBLAS_STATUS(
                actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        }

        return;
    }

    int abs_incy = incy >= 0 ? incy : -incy;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // arrays of pointers-to-host on host
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hx(dim_x, incx, batch_count);
    host_batch_vector<T> hy(dim_y, incy, batch_count);
    host_batch_vector<T> hy_cpu(dim_y, incy, batch_count);
    host_batch_vector<T> hy_host(dim_y, incy, batch_count);
    host_batch_vector<T> hy_device(dim_y, incy, batch_count);

    // device pointers
    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dx(dim_x, incx, batch_count);
    device_batch_vector<T> dy(dim_y, incy, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    // Initial Data on CPU
    hipblas_init_vector(hA, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, arg, hipblas_client_beta_sets_nan);

    hy_cpu.copy_from(hy);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasGemvBatchedFn(handle,
                                                 transA,
                                                 M,
                                                 N,
                                                 (T*)&h_alpha,
                                                 dA.ptr_on_device(),
                                                 lda,
                                                 dx.ptr_on_device(),
                                                 incx,
                                                 (T*)&h_beta,
                                                 dy.ptr_on_device(),
                                                 incy,
                                                 batch_count));

        CHECK_HIP_ERROR(hy_host.transfer_from(dy));
        CHECK_HIP_ERROR(dy.transfer_from(hy));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasGemvBatchedFn(handle,
                                                 transA,
                                                 M,
                                                 N,
                                                 d_alpha,
                                                 dA.ptr_on_device(),
                                                 lda,
                                                 dx.ptr_on_device(),
                                                 incx,
                                                 d_beta,
                                                 dy.ptr_on_device(),
                                                 incy,
                                                 batch_count));

        CHECK_HIP_ERROR(hy_device.transfer_from(dy));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            ref_gemv<T>(transA, M, N, h_alpha, hA[b], lda, hx[b], incx, h_beta, hy_cpu[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, dim_y, batch_count, abs_incy, hy_cpu, hy_host);
            unit_check_general<T>(1, dim_y, batch_count, abs_incy, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_cpu, hy_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', 1, dim_y, abs_incy, hy_cpu, hy_device, batch_count);
        }
    }

    if(arg.timing)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
            {
                gpu_time_used = get_time_us_sync(stream);
            }
            CHECK_HIPBLAS_ERROR(hipblasGemvBatchedFn(handle,
                                                     transA,
                                                     M,
                                                     N,
                                                     d_alpha,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     dx.ptr_on_device(),
                                                     incx,
                                                     d_beta,
                                                     dy.ptr_on_device(),
                                                     incy,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemvBatchedModel{}.log_args<T>(std::cout,
                                              arg,
                                              gpu_time_used,
                                              gemv_gflop_count<T>(transA, M, N),
                                              gemv_gbyte_count<T>(transA, M, N),
                                              hipblas_error_host,
                                              hipblas_error_device);
    }
}

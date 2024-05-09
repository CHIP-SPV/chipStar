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

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "hipblas_iamax_iamin_ref.hpp"
#include "testing_common.hpp"

using hipblasIamaxIaminBatchedModel = ArgumentModel<e_a_type, e_N, e_incx, e_batch_count>;

template <typename T, typename R, typename FUNC>
void testing_iamax_iamin_batched_bad_arg(const Arguments& arg, FUNC func)
{
    hipblasLocalHandle handle(arg);
    int64_t            N           = 100;
    int64_t            incx        = 1;
    int64_t            batch_count = 2;

    device_batch_vector<T> dx(N, incx, batch_count);
    device_vector<R>       d_res(batch_count);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // None of these test cases will write to result so using device pointer is fine for both modes
        EXPECT_HIPBLAS_STATUS(func(nullptr, N, dx, incx, batch_count, d_res),
                              HIPBLAS_STATUS_NOT_INITIALIZED);
        EXPECT_HIPBLAS_STATUS(func(handle, N, nullptr, incx, batch_count, d_res),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(func(handle, N, dx, incx, batch_count, nullptr),
                              HIPBLAS_STATUS_INVALID_VALUE);
    }
}

template <typename T>
void testing_iamax_batched_bad_arg(const Arguments& arg)
{
    auto hipblasIamaxBatchedFn
        = arg.api == FORTRAN ? hipblasIamaxBatched<T, true> : hipblasIamaxBatched<T, false>;
    auto hipblasIamaxBatchedFn_64 = arg.api == FORTRAN_64 ? hipblasIamaxBatched_64<T, true>
                                                          : hipblasIamaxBatched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_batched_bad_arg<T, int64_t>(arg, hipblasIamaxBatchedFn_64);
    else
        testing_iamax_iamin_batched_bad_arg<T, int>(arg, hipblasIamaxBatchedFn);
}

template <typename T>
void testing_iamin_batched_bad_arg(const Arguments& arg)
{
    auto hipblasIaminBatchedFn
        = arg.api == FORTRAN ? hipblasIaminBatched<T, true> : hipblasIaminBatched<T, false>;
    auto hipblasIaminBatchedFn_64 = arg.api == FORTRAN_64 ? hipblasIaminBatched_64<T, true>
                                                          : hipblasIaminBatched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_batched_bad_arg<T, int64_t>(arg, hipblasIaminBatchedFn_64);
    else
        testing_iamax_iamin_batched_bad_arg<T, int>(arg, hipblasIaminBatchedFn);
}

template <typename T,
          void REFBLAS_FUNC(int64_t, const T*, int64_t, int64_t*),
          typename R,
          typename FUNC>
void testing_iamax_iamin_batched(const Arguments& arg, FUNC func)
{
    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t batch_count = arg.batch_count;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(batch_count <= 0 || N <= 0 || incx <= 0)
    {
        // quick return success
        int64_t          batches = std::max(int64_t(1), batch_count);
        device_vector<R> d_hipblas_result_0(batches);
        host_vector<R>   h_hipblas_result_0(batches);
        hipblas_init_nan(h_hipblas_result_0.data(), batches);
        CHECK_HIP_ERROR(hipMemcpy(
            d_hipblas_result_0, h_hipblas_result_0, sizeof(R) * batches, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(func(handle, N, nullptr, incx, batch_count, d_hipblas_result_0));

        if(batch_count > 0)
        {
            host_vector<R> cpu_0(batch_count);
            host_vector<R> gpu_0(batch_count);
            CHECK_HIP_ERROR(hipMemcpy(
                gpu_0, d_hipblas_result_0, sizeof(R) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<R>(1, batch_count, 1, cpu_0, gpu_0);
        }

        return;
    }

    host_batch_vector<T> hx(N, incx, batch_count);
    host_vector<R>       cpu_result(batch_count);
    host_vector<int64_t> cpu_result_64(batch_count);
    host_vector<R>       hipblas_result_host(batch_count);
    host_vector<R>       hipblas_result_device(batch_count);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_vector<R>       d_hipblas_result_device(batch_count);
    CHECK_HIP_ERROR(dx.memcheck());

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used;
    R      hipblas_error_host = 0, hipblas_error_device = 0;

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        // device_pointer
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(
            func(handle, N, dx.ptr_on_device(), incx, batch_count, d_hipblas_result_device));
        CHECK_HIP_ERROR(hipMemcpy(hipblas_result_device,
                                  d_hipblas_result_device,
                                  sizeof(R) * batch_count,
                                  hipMemcpyDeviceToHost));

        // host_pointer
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(
            func(handle, N, dx.ptr_on_device(), incx, batch_count, hipblas_result_host));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            REFBLAS_FUNC(N, hx[b], incx, cpu_result_64 + b);
            cpu_result[b] = cpu_result_64[b];
        }

        if(arg.unit_check)
        {
            unit_check_general<R>(1, 1, batch_count, cpu_result, hipblas_result_host);
            unit_check_general<R>(1, 1, batch_count, cpu_result, hipblas_result_device);
        }
        if(arg.norm_check)
        {
            for(int64_t b = 0; b < batch_count; b++)
            {
                hipblas_error_host
                    = std::max(int64_t(hipblas_error_host),
                               int64_t(hipblas_abs(hipblas_result_host[b] - cpu_result[b])));
                hipblas_error_device
                    = std::max(int64_t(hipblas_error_device),
                               int64_t(hipblas_abs(hipblas_result_device[b] - cpu_result[b])));
            }
        }
    } // end of if unit/norm check

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
                func(handle, N, dx.ptr_on_device(), incx, batch_count, d_hipblas_result_device));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasIamaxIaminBatchedModel{}.log_args<T>(std::cout,
                                                    arg,
                                                    gpu_time_used,
                                                    iamax_gflop_count<T>(N),
                                                    iamax_gbyte_count<T>(N),
                                                    hipblas_error_host,
                                                    hipblas_error_device);
    }
}

inline void testname_iamax_batched(const Arguments& arg, std::string& name)
{
    hipblasIamaxIaminBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_iamax_batched(const Arguments& arg)
{
    auto hipblasIamaxBatchedFn
        = arg.api == FORTRAN ? hipblasIamaxBatched<T, true> : hipblasIamaxBatched<T, false>;
    auto hipblasIamaxBatchedFn_64 = arg.api == FORTRAN_64 ? hipblasIamaxBatched_64<T, true>
                                                          : hipblasIamaxBatched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_batched<T, hipblas_iamax_iamin_ref::iamax<T>, int64_t>(
            arg, hipblasIamaxBatchedFn_64);
    else
        testing_iamax_iamin_batched<T, hipblas_iamax_iamin_ref::iamax<T>, int>(
            arg, hipblasIamaxBatchedFn);
}

inline void testname_iamin_batched(const Arguments& arg, std::string& name)
{
    hipblasIamaxIaminBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_iamin_batched(const Arguments& arg)
{
    auto hipblasIaminBatchedFn
        = arg.api == FORTRAN ? hipblasIaminBatched<T, true> : hipblasIaminBatched<T, false>;
    auto hipblasIaminBatchedFn_64 = arg.api == FORTRAN_64 ? hipblasIaminBatched_64<T, true>
                                                          : hipblasIaminBatched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_batched<T, hipblas_iamax_iamin_ref::iamin<T>, int64_t>(
            arg, hipblasIaminBatchedFn_64);
    else
        testing_iamax_iamin_batched<T, hipblas_iamax_iamin_ref::iamin<T>, int>(
            arg, hipblasIaminBatchedFn);
}

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

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasRotStridedBatchedModel = ArgumentModel<e_a_type,
                                                    e_c_type,
                                                    e_compute_type,
                                                    e_N,
                                                    e_incx,
                                                    e_incy,
                                                    e_stride_scale,
                                                    e_batch_count>;

inline void testname_rot_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasRotStridedBatchedModel{}.test_name(arg, name);
}

template <typename T, typename U = T, typename V = T>
void testing_rot_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN                       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotStridedBatchedFn    = FORTRAN ? hipblasRotStridedBatched<T, U, V, true>
                                                 : hipblasRotStridedBatched<T, U, V, false>;
    auto hipblasRotStridedBatchedFn_64 = arg.api == FORTRAN_64
                                             ? hipblasRotStridedBatched_64<T, U, V, true>
                                             : hipblasRotStridedBatched_64<T, U, V, false>;

    int64_t       N           = 100;
    int64_t       incx        = 1;
    int64_t       incy        = 1;
    int64_t       batch_count = 2;
    hipblasStride stride_x    = N * incx;
    hipblasStride stride_y    = N * incy;

    hipblasLocalHandle handle(arg);

    device_vector<T> dx(stride_x * batch_count);
    device_vector<T> dy(stride_y * batch_count);
    device_vector<U> dc(1);
    device_vector<V> ds(1);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasRotStridedBatchedFn,
                (nullptr, N, dx, incx, stride_x, dy, incy, stride_y, dc, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotStridedBatchedFn,
                (handle, N, nullptr, incx, stride_x, dy, incy, stride_y, dc, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotStridedBatchedFn,
                (handle, N, dx, incx, stride_x, nullptr, incy, stride_y, dc, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotStridedBatchedFn,
                (handle, N, dx, incx, stride_x, dy, incy, stride_y, nullptr, ds, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasRotStridedBatchedFn,
                (handle, N, dx, incx, stride_x, dy, incy, stride_y, dc, nullptr, batch_count));
}

template <typename T, typename U = T, typename V = T>
void testing_rot_strided_batched(const Arguments& arg)
{
    bool FORTRAN                       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotStridedBatchedFn    = FORTRAN ? hipblasRotStridedBatched<T, U, V, true>
                                                 : hipblasRotStridedBatched<T, U, V, false>;
    auto hipblasRotStridedBatchedFn_64 = arg.api == FORTRAN_64
                                             ? hipblasRotStridedBatched_64<T, U, V, true>
                                             : hipblasRotStridedBatched_64<T, U, V, false>;

    int64_t N            = arg.N;
    int64_t incx         = arg.incx;
    int64_t incy         = arg.incy;
    double  stride_scale = arg.stride_scale;
    int64_t batch_count  = arg.batch_count;

    int64_t       abs_incx = incx >= 0 ? incx : -incx;
    int64_t       abs_incy = incy >= 0 ? incy : -incy;
    hipblasStride stride_x = size_t(N) * abs_incx * stride_scale;
    hipblasStride stride_y = size_t(N) * abs_incy * stride_scale;

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasRotStridedBatchedFn,
                   (handle,
                    N,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    incy,
                    stride_y,
                    nullptr,
                    nullptr,
                    batch_count));

        return;
    }

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    size_t size_x = N * size_t(abs_incx) + size_t(stride_x) * size_t(batch_count - 1);
    size_t size_y = N * size_t(abs_incy) + size_t(stride_y) * size_t(batch_count - 1);
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;

    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<U> dc(1);
    device_vector<V> ds(1);

    // Initial Data on CPU
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);
    host_vector<U> hc(1);
    host_vector<V> hs(1);

    // Random alpha (0 - 10)
    host_vector<int> alpha(1);

    hipblas_init_vector(
        hx, arg, N, abs_incx, stride_x, batch_count, hipblas_client_never_set_nan, true);
    hipblas_init_vector(
        hy, arg, N, abs_incy, stride_y, batch_count, hipblas_client_never_set_nan, false);
    hipblas_init_vector(alpha, arg, 1, 1, 0, 1, hipblas_client_never_set_nan, false);

    // cos and sin of alpha (in rads)
    hc[0] = cos(alpha[0]);
    hs[0] = sin(alpha[0]);

    // CPU BLAS reference data
    host_vector<T> cx = hx;
    host_vector<T> cy = hy;
    // ref_rotg<T, U>(cx, cy, hc, hs);
    // cx[0] = hx[0];
    // cy[0] = hy[0];
    for(int64_t b = 0; b < batch_count; b++)
    {
        ref_rot<T, U, V>(
            N, cx.data() + b * stride_x, incx, cy.data() + b * stride_y, incy, *hc, *hs);
    }

    if(arg.unit_check || arg.norm_check)
    {
        // Test host
        {
            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
            DAPI_CHECK(hipblasRotStridedBatchedFn,
                       (handle, N, dx, incx, stride_x, dy, incy, stride_y, hc, hs, batch_count));

            host_vector<T> rx(size_x);
            host_vector<T> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                near_check_general<T>(1, N, batch_count, abs_incx, stride_x, cx, rx, rel_error);
                near_check_general<T>(1, N, batch_count, abs_incy, stride_y, cy, ry, rel_error);
            }
            if(arg.norm_check)
            {
                hipblas_error_host
                    = norm_check_general<T>('F', 1, N, abs_incx, stride_x, cx, rx, batch_count);
                hipblas_error_host
                    += norm_check_general<T>('F', 1, N, abs_incy, stride_y, cy, ry, batch_count);
            }
        }

        // Test device
        {
            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));
            DAPI_CHECK(hipblasRotStridedBatchedFn,
                       (handle, N, dx, incx, stride_x, dy, incy, stride_y, dc, ds, batch_count));

            host_vector<T> rx(size_x);
            host_vector<T> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                near_check_general<T>(1, N, batch_count, abs_incx, stride_x, cx, rx, rel_error);
                near_check_general<T>(1, N, batch_count, abs_incy, stride_y, cy, ry, rel_error);
            }
            if(arg.norm_check)
            {
                hipblas_error_device
                    = norm_check_general<T>('F', 1, N, abs_incx, stride_x, cx, rx, batch_count);
                hipblas_error_device
                    += norm_check_general<T>('F', 1, N, abs_incy, stride_y, cy, ry, batch_count);
            }
        }
    }

    if(arg.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_CHECK(hipblasRotStridedBatchedFn,
                       (handle, N, dx, incx, stride_x, dy, incy, stride_y, dc, ds, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotStridedBatchedModel{}.log_args<T>(std::cout,
                                                    arg,
                                                    gpu_time_used,
                                                    rot_gflop_count<T, T, U, V>(N),
                                                    rot_gbyte_count<T>(N),
                                                    hipblas_error_host,
                                                    hipblas_error_device);
    }
}

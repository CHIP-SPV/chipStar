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

using hipblasNrm2StridedBatchedModel
    = ArgumentModel<e_a_type, e_N, e_incx, e_stride_scale, e_batch_count>;

inline void testname_nrm2_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasNrm2StridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_nrm2_strided_batched_bad_arg(const Arguments& arg)
{
    using Tr                            = real_t<T>;
    bool FORTRAN                        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasNrm2StridedBatchedFn    = FORTRAN ? hipblasNrm2StridedBatched<T, Tr, true>
                                                  : hipblasNrm2StridedBatched<T, Tr, false>;
    auto hipblasNrm2StridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasNrm2StridedBatched_64<T, Tr, true>
                                              : hipblasNrm2StridedBatched_64<T, Tr, false>;

    int64_t       N           = 100;
    int64_t       incx        = 1;
    int64_t       batch_count = 2;
    hipblasStride stride_x    = N * incx;

    hipblasLocalHandle handle(arg);

    device_vector<T>  dx(stride_x * batch_count);
    device_vector<Tr> d_res(batch_count);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // None of these test cases will write to result so using device pointer is fine for both modes
        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasNrm2StridedBatchedFn,
                    (nullptr, N, dx, incx, stride_x, batch_count, d_res));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasNrm2StridedBatchedFn,
                    (handle, N, nullptr, incx, stride_x, batch_count, d_res));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasNrm2StridedBatchedFn,
                    (handle, N, dx, incx, stride_x, batch_count, nullptr));
    }
}

template <typename T>
void testing_nrm2_strided_batched(const Arguments& arg)
{
    using Tr                            = real_t<T>;
    bool FORTRAN                        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasNrm2StridedBatchedFn    = FORTRAN ? hipblasNrm2StridedBatched<T, Tr, true>
                                                  : hipblasNrm2StridedBatched<T, Tr, false>;
    auto hipblasNrm2StridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasNrm2StridedBatched_64<T, Tr, true>
                                              : hipblasNrm2StridedBatched_64<T, Tr, false>;

    int64_t N            = arg.N;
    int64_t incx         = arg.incx;
    double  stride_scale = arg.stride_scale;
    int64_t batch_count  = arg.batch_count;

    hipblasStride stridex = size_t(N) * incx * stride_scale;
    size_t        sizeX   = stridex * batch_count;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        int64_t           batches = std::max(int64_t(1), batch_count);
        device_vector<Tr> d_hipblas_result_0(batches);
        host_vector<Tr>   h_hipblas_result_0(batches);
        hipblas_init_nan(h_hipblas_result_0.data(), batches);
        CHECK_HIP_ERROR(hipMemcpy(
            d_hipblas_result_0, h_hipblas_result_0, sizeof(Tr) * batches, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasNrm2StridedBatchedFn(
            handle, N, nullptr, incx, stridex, batch_count, d_hipblas_result_0));

        if(batch_count > 0)
        {
            host_vector<Tr> cpu_0(batch_count);
            host_vector<Tr> gpu_0(batch_count);
            CHECK_HIP_ERROR(hipMemcpy(
                gpu_0, d_hipblas_result_0, sizeof(Tr) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<Tr>(1, batch_count, 1, cpu_0, gpu_0);
        }

        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T>  hx(sizeX);
    host_vector<Tr> h_hipblas_result_host(batch_count);
    host_vector<Tr> h_hipblas_result_device(batch_count);
    host_vector<Tr> h_cpu_result(batch_count);

    device_vector<T>  dx(sizeX);
    device_vector<Tr> d_hipblas_result(batch_count);

    double gpu_time_used;
    double hipblas_error_host = 0, hipblas_error_device = 0;

    // Initial Data on CPU
    hipblas_init_vector(
        hx, arg, N, incx, stridex, batch_count, hipblas_client_alpha_sets_nan, true);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // hipblasNrm2 accept both dev/host pointer for the scalar
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasNrm2StridedBatchedFn,
                   (handle, N, dx, incx, stridex, batch_count, d_hipblas_result));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasNrm2StridedBatchedFn,
                   (handle, N, dx, incx, stridex, batch_count, h_hipblas_result_host));

        CHECK_HIP_ERROR(hipMemcpy(h_hipblas_result_device,
                                  d_hipblas_result,
                                  sizeof(Tr) * batch_count,
                                  hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_nrm2<T, Tr>(N, hx.data() + b * stridex, incx, &(h_cpu_result[b]));
        }

        if(arg.unit_check)
        {
            unit_check_nrm2<Tr>(batch_count, h_cpu_result, h_hipblas_result_host, N);
            unit_check_nrm2<Tr>(batch_count, h_cpu_result, h_hipblas_result_device, N);
        }
        if(arg.norm_check)
        {
            for(int64_t b = 0; b < batch_count; b++)
            {
                hipblas_error_host
                    = std::max(vector_norm_1(1, 1, &(h_cpu_result[b]), &(h_hipblas_result_host[b])),
                               hipblas_error_host);
                hipblas_error_device = std::max(
                    vector_norm_1(1, 1, &(h_cpu_result[b]), &(h_hipblas_result_device[b])),
                    hipblas_error_device);
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

            DAPI_CHECK(hipblasNrm2StridedBatchedFn,
                       (handle, N, dx, incx, stridex, batch_count, d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasNrm2StridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     nrm2_gflop_count<T>(N),
                                                     nrm2_gbyte_count<T>(N),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}

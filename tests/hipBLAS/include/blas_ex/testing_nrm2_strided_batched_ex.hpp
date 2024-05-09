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

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasNrm2StridedBatchedExModel
    = ArgumentModel<e_a_type, e_b_type, e_compute_type, e_N, e_incx, e_stride_scale, e_batch_count>;

inline void testname_nrm2_strided_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasNrm2StridedBatchedExModel{}.test_name(arg, name);
}

template <typename Tx, typename Tr = Tx, typename Tex = Tr>
void testing_nrm2_strided_batched_ex_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasNrm2StridedBatchedExFn
        = FORTRAN ? hipblasNrm2StridedBatchedExFortran : hipblasNrm2StridedBatchedEx;

    int64_t N           = 100;
    int64_t incx        = 1;
    int64_t batch_count = 2;

    hipblasStride stridex = N * incx;

    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t resultType    = arg.b_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasLocalHandle handle(arg);

    device_vector<Tx> dx(stridex * batch_count);
    device_vector<Tr> d_res(batch_count);

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        // None of these test cases will write to result so using device pointer is fine for both modes
        EXPECT_HIPBLAS_STATUS(hipblasNrm2StridedBatchedExFn(nullptr,
                                                            N,
                                                            dx,
                                                            xType,
                                                            incx,
                                                            stridex,
                                                            batch_count,
                                                            d_res,
                                                            resultType,
                                                            executionType),
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(hipblasNrm2StridedBatchedExFn(handle,
                                                                N,
                                                                nullptr,
                                                                xType,
                                                                incx,
                                                                stridex,
                                                                batch_count,
                                                                d_res,
                                                                resultType,
                                                                executionType),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasNrm2StridedBatchedExFn(handle,
                                                                N,
                                                                dx,
                                                                xType,
                                                                incx,
                                                                stridex,
                                                                batch_count,
                                                                nullptr,
                                                                resultType,
                                                                executionType),
                                  HIPBLAS_STATUS_INVALID_VALUE);
        }
    }
}

template <typename Tx, typename Tr = Tx, typename Tex = Tr>
void testing_nrm2_strided_batched_ex(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasNrm2StridedBatchedExFn
        = FORTRAN ? hipblasNrm2StridedBatchedExFortran : hipblasNrm2StridedBatchedEx;

    int    N            = arg.N;
    int    incx         = arg.incx;
    double stride_scale = arg.stride_scale;
    int    batch_count  = arg.batch_count;

    hipblasStride stridex = size_t(N) * incx * stride_scale;
    size_t        sizeX   = stridex * batch_count;

    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t resultType    = arg.b_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasLocalHandle handle(arg);

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        device_vector<Tr> d_hipblas_result_0(std::max(batch_count, 1));
        host_vector<Tr>   h_hipblas_result_0(std::max(1, batch_count));
        hipblas_init_nan(h_hipblas_result_0.data(), std::max(1, batch_count));
        CHECK_HIP_ERROR(hipMemcpy(d_hipblas_result_0,
                                  h_hipblas_result_0,
                                  sizeof(Tr) * std::max(1, batch_count),
                                  hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasNrm2StridedBatchedExFn(handle,
                                                          N,
                                                          nullptr,
                                                          xType,
                                                          incx,
                                                          stridex,
                                                          batch_count,
                                                          d_hipblas_result_0,
                                                          resultType,
                                                          executionType));

        if(batch_count > 0)
        {
            // TODO: error in rocBLAS - only setting the first element to 0, not for all batches
            // host_vector<Tr> cpu_0(batch_count);
            // host_vector<Tr> gpu_0(batch_count);
            // CHECK_HIP_ERROR(hipMemcpy(
            //     gpu_0, d_hipblas_result_0, sizeof(Tr) * batch_count, hipMemcpyDeviceToHost));
            // unit_check_general<Tr>(1, batch_count, 1, cpu_0, gpu_0);
        }
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(sizeX);
    host_vector<Tr> h_hipblas_result_host(batch_count);
    host_vector<Tr> h_hipblas_result_device(batch_count);
    host_vector<Tr> h_cpu_result(batch_count);

    device_vector<Tx> dx(sizeX);
    device_vector<Tr> d_hipblas_result(batch_count);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(
        hx, arg, N, incx, stridex, batch_count, hipblas_client_alpha_sets_nan, true);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(Tx) * sizeX, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        // hipblasNrm2 accept both dev/host pointer for the scalar
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasNrm2StridedBatchedExFn(handle,
                                                          N,
                                                          dx,
                                                          xType,
                                                          incx,
                                                          stridex,
                                                          batch_count,
                                                          d_hipblas_result,
                                                          resultType,
                                                          executionType));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasNrm2StridedBatchedExFn(handle,
                                                          N,
                                                          dx,
                                                          xType,
                                                          incx,
                                                          stridex,
                                                          batch_count,
                                                          h_hipblas_result_host,
                                                          resultType,
                                                          executionType));

        CHECK_HIP_ERROR(hipMemcpy(h_hipblas_result_device,
                                  d_hipblas_result,
                                  sizeof(Tr) * batch_count,
                                  hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            ref_nrm2<Tx, Tr>(N, hx.data() + b * stridex, incx, &(h_cpu_result[b]));
        }

        double abs_result = h_cpu_result[0] > 0 ? h_cpu_result[0] : -h_cpu_result[0];
        double abs_error;

        abs_error = abs_result > 0 ? hipblas_type_epsilon<Tr> * N * abs_result
                                   : hipblas_type_epsilon<Tr> * N;

        double tolerance = 2.0; //  accounts for rounding in reduction sum. depends on n.
            //  If test fails, try decreasing n or increasing tolerance.
        abs_error *= tolerance;

        if(arg.unit_check)
        {
            near_check_general<Tr>(
                batch_count, 1, 1, h_cpu_result.data(), h_hipblas_result_host.data(), abs_error);
            near_check_general<Tr>(
                batch_count, 1, 1, h_cpu_result.data(), h_hipblas_result_device.data(), abs_error);
        }
        if(arg.norm_check)
        {
            for(int b = 0; b < batch_count; b++)
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

            CHECK_HIPBLAS_ERROR(hipblasNrm2StridedBatchedExFn(handle,
                                                              N,
                                                              dx,
                                                              xType,
                                                              incx,
                                                              stridex,
                                                              batch_count,
                                                              d_hipblas_result,
                                                              resultType,
                                                              executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasNrm2StridedBatchedExModel{}.log_args<Tx>(std::cout,
                                                        arg,
                                                        gpu_time_used,
                                                        nrm2_gflop_count<Tx>(N),
                                                        nrm2_gbyte_count<Tx>(N),
                                                        hipblas_error_host,
                                                        hipblas_error_device);
    }
}

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

using hipblasDotExModel
    = ArgumentModel<e_a_type, e_b_type, e_c_type, e_compute_type, e_N, e_incx, e_incy>;

inline void testname_dot_ex(const Arguments& arg, std::string& name)
{
    hipblasDotExModel{}.test_name(arg, name);
}

inline void testname_dotc_ex(const Arguments& arg, std::string& name)
{
    hipblasDotExModel{}.test_name(arg, name);
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_ex_bad_arg(const Arguments& arg)
{
    bool FORTRAN        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasDotExFn = FORTRAN ? (CONJ ? hipblasDotcExFortran : hipblasDotExFortran)
                                  : (CONJ ? hipblasDotcEx : hipblasDotEx);

    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t resultType    = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        int64_t N    = 100;
        int64_t incx = 1;
        int64_t incy = 1;

        device_vector<Tx> dx(N * incx);
        device_vector<Ty> dy(N * incy);
        device_vector<Tr> d_res(1);

        // None of these test cases will write to result so using device pointer is fine for both modes
        EXPECT_HIPBLAS_STATUS(
            hipblasDotExFn(
                nullptr, N, dx, xType, incx, dy, yType, incy, d_res, resultType, executionType),
            HIPBLAS_STATUS_NOT_INITIALIZED);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(hipblasDotExFn(handle,
                                                 N,
                                                 nullptr,
                                                 xType,
                                                 incx,
                                                 dy,
                                                 yType,
                                                 incy,
                                                 d_res,
                                                 resultType,
                                                 executionType),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasDotExFn(handle,
                                                 N,
                                                 dx,
                                                 xType,
                                                 incx,
                                                 nullptr,
                                                 yType,
                                                 incy,
                                                 d_res,
                                                 resultType,
                                                 executionType),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasDotExFn(handle,
                                                 N,
                                                 dx,
                                                 xType,
                                                 incx,
                                                 dy,
                                                 yType,
                                                 incy,
                                                 nullptr,
                                                 resultType,
                                                 executionType),
                                  HIPBLAS_STATUS_INVALID_VALUE);
        }
    }
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr>
void testing_dotc_ex_bad_arg(const Arguments& arg)
{
    testing_dot_ex_bad_arg<Tx, Ty, Tr, Tex, true>(arg);
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_ex(const Arguments& arg)
{
    bool FORTRAN        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasDotExFn = FORTRAN ? (CONJ ? hipblasDotcExFortran : hipblasDotExFortran)
                                  : (CONJ ? hipblasDotcEx : hipblasDotEx);

    int N    = arg.N;
    int incx = arg.incx;
    int incy = arg.incy;

    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t resultType    = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        device_vector<Tr> d_hipblas_result_0(1);
        host_vector<Tr>   h_hipblas_result_0(1);
        hipblas_init_nan(h_hipblas_result_0.data(), 1);
        CHECK_HIP_ERROR(
            hipMemcpy(d_hipblas_result_0, h_hipblas_result_0, sizeof(Tr), hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasDotExFn(handle,
                                           N,
                                           nullptr,
                                           xType,
                                           incx,
                                           nullptr,
                                           yType,
                                           incy,
                                           d_hipblas_result_0,
                                           resultType,
                                           executionType));

        host_vector<Tr> cpu_0(1);
        host_vector<Tr> gpu_0(1);
        CHECK_HIP_ERROR(hipMemcpy(gpu_0, d_hipblas_result_0, sizeof(Tr), hipMemcpyDeviceToHost));
        unit_check_general<Tr>(1, 1, 1, cpu_0, gpu_0);
        return;
    }

    int    abs_incx = incx >= 0 ? incx : -incx;
    int    abs_incy = incy >= 0 ? incy : -incy;
    size_t sizeX    = size_t(N) * abs_incx;
    size_t sizeY    = size_t(N) * abs_incy;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(sizeX);
    host_vector<Ty> hy(sizeY);

    device_vector<Tx> dx(sizeX);
    device_vector<Ty> dy(sizeY);
    device_vector<Tr> d_hipblas_result(1);

    Tr cpu_result, hipblas_result_host, hipblas_result_device;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, N, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, true, true);
    hipblas_init_vector(hy, arg, N, abs_incy, 0, 1, hipblas_client_alpha_sets_nan, false);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(Ty) * sizeY, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasDotExFn(handle,
                                           N,
                                           dx,
                                           xType,
                                           incx,
                                           dy,
                                           yType,
                                           incy,
                                           &hipblas_result_host,
                                           resultType,
                                           executionType));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasDotExFn(handle,
                                           N,
                                           dx,
                                           xType,
                                           incx,
                                           dy,
                                           yType,
                                           incy,
                                           d_hipblas_result,
                                           resultType,
                                           executionType));

        CHECK_HIP_ERROR(
            hipMemcpy(&hipblas_result_device, d_hipblas_result, sizeof(Tr), hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        (CONJ ? ref_dotc<Tx> : ref_dot<Tx>)(N, hx.data(), incx, hy.data(), incy, &cpu_result);

        if(arg.unit_check)
        {
            if(std::is_same<Tr, hipblasHalf>{})
            {
                double tol = error_tolerance<Tr> * N;
                near_check_general(1, 1, 1, &cpu_result, &hipblas_result_host, tol);
                near_check_general(1, 1, 1, &cpu_result, &hipblas_result_device, tol);
            }
            else
            {
                unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_host);
                unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_device);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tr>('F', 1, 1, 1, &cpu_result, &hipblas_result_host);
            hipblas_error_device
                = norm_check_general<Tr>('F', 1, 1, 1, &cpu_result, &hipblas_result_device);
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

            CHECK_HIPBLAS_ERROR(hipblasDotExFn(handle,
                                               N,
                                               dx,
                                               xType,
                                               incx,
                                               dy,
                                               yType,
                                               incy,
                                               d_hipblas_result,
                                               resultType,
                                               executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasDotExModel{}.log_args<Tx>(std::cout,
                                         arg,
                                         gpu_time_used,
                                         dot_gflop_count<CONJ, Tx>(N),
                                         dot_gbyte_count<Tx>(N),
                                         hipblas_error_host,
                                         hipblas_error_device);
    }
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr>
void testing_dotc_ex(const Arguments& arg)
{
    testing_dot_ex<Tx, Ty, Tr, Tex, true>(arg);
}

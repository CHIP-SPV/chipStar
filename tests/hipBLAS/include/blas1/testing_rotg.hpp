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

using hipblasRotgModel = ArgumentModel<e_a_type>;

inline void testname_rotg(const Arguments& arg, std::string& name)
{
    hipblasRotgModel{}.test_name(arg, name);
}

template <typename T>
void testing_rotg_bad_arg(const Arguments& arg)
{
    using U            = real_t<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotgFn = FORTRAN ? hipblasRotg<T, U, true> : hipblasRotg<T, U, false>;
    auto hipblasRotgFn_64
        = arg.api == FORTRAN_64 ? hipblasRotg_64<T, U, true> : hipblasRotg_64<T, U, false>;

    hipblasLocalHandle handle(arg);

    device_vector<T> da(1);
    device_vector<T> db(1);
    device_vector<U> dc(1);
    device_vector<T> ds(1);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED, hipblasRotgFn, (nullptr, da, db, dc, ds));

    if(arg.bad_arg_all)
    {
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, hipblasRotgFn, (handle, nullptr, db, dc, ds));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, hipblasRotgFn, (handle, da, nullptr, dc, ds));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, hipblasRotgFn, (handle, da, db, nullptr, ds));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE, hipblasRotgFn, (handle, da, db, dc, nullptr));
    }
}

template <typename T>
void testing_rotg(const Arguments& arg)
{
    using U            = real_t<T>;
    bool FORTRAN       = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotgFn = FORTRAN ? hipblasRotg<T, U, true> : hipblasRotg<T, U, false>;
    auto hipblasRotgFn_64
        = arg.api == FORTRAN_64 ? hipblasRotg_64<T, U, true> : hipblasRotg_64<T, U, false>;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasLocalHandle handle(arg);

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    host_vector<T> ha(1);
    host_vector<T> hb(1);
    host_vector<U> hc(1);
    host_vector<T> hs(1);

    // Initial data on CPU
    hipblas_init_vector(ha, arg, 1, 1, 0, 1, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hb, arg, 1, 1, 0, 1, hipblas_client_alpha_sets_nan, false);
    hipblas_init_vector(hc, arg, 1, 1, 0, 1, hipblas_client_alpha_sets_nan, false);
    hipblas_init_vector(hs, arg, 1, 1, 0, 1, hipblas_client_alpha_sets_nan, false);

    // CPU BLAS
    host_vector<T> ca = ha;
    host_vector<T> cb = hb;
    host_vector<U> cc = hc;
    host_vector<T> cs = hs;

    // result hipBLAS device
    host_vector<T> ra = ha;
    host_vector<T> rb = hb;
    host_vector<U> rc = hc;
    host_vector<T> rs = hs;

    device_vector<T> da(1);
    device_vector<T> db(1);
    device_vector<U> dc(1);
    device_vector<T> ds(1);

    CHECK_HIP_ERROR(hipMemcpy(da, ha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasRotgFn, (handle, ha, hb, hc, hs));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasRotgFn, (handle, da, db, dc, ds));

        CHECK_HIP_ERROR(hipMemcpy(ra, da, sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rb, db, sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rc, dc, sizeof(U), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rs, ds, sizeof(T), hipMemcpyDeviceToHost));

        ref_rotg<T, U>(ca, cb, cc, cs);

        if(arg.unit_check)
        {
            near_check_general(1, 1, 1, ca.data(), ha.data(), rel_error);
            near_check_general(1, 1, 1, cb.data(), hb.data(), rel_error);
            near_check_general(1, 1, 1, cc.data(), hc.data(), rel_error);
            near_check_general(1, 1, 1, cs.data(), hs.data(), rel_error);

            near_check_general(1, 1, 1, ca.data(), ra.data(), rel_error);
            near_check_general(1, 1, 1, cb.data(), rb.data(), rel_error);
            near_check_general(1, 1, 1, cc.data(), rc.data(), rel_error);
            near_check_general(1, 1, 1, cs.data(), rs.data(), rel_error);
        }

        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>('F', 1, 1, 1, ca, ha);
            hipblas_error_host += norm_check_general<T>('F', 1, 1, 1, cb, hb);
            hipblas_error_host += norm_check_general<U>('F', 1, 1, 1, cc, hc);
            hipblas_error_host += norm_check_general<T>('F', 1, 1, 1, cs, hs);

            hipblas_error_device = norm_check_general<T>('F', 1, 1, 1, ca, ra);
            hipblas_error_device += norm_check_general<T>('F', 1, 1, 1, cb, rb);
            hipblas_error_device += norm_check_general<U>('F', 1, 1, 1, cc, rc);
            hipblas_error_device += norm_check_general<T>('F', 1, 1, 1, cs, rs);
        }
    }

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

            DAPI_CHECK(hipblasRotgFn, (handle, da, db, dc, ds));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotgModel{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       ArgumentLogging::NA_value,
                                       ArgumentLogging::NA_value,
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}

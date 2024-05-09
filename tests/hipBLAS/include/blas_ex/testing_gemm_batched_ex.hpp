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
#include <limits>
#include <stdlib.h>
#include <typeinfo>
#include <vector>

#include "hipblas_unique_ptr.hpp"
#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasGemmBatchedExModel = ArgumentModel<e_a_type,
                                                e_c_type,
                                                e_compute_type,
                                                e_transA,
                                                e_transB,
                                                e_M,
                                                e_N,
                                                e_K,
                                                e_alpha,
                                                e_lda,
                                                e_ldb,
                                                e_beta,
                                                e_ldc,
                                                e_batch_count,
                                                e_with_flags,
                                                e_flags>;

inline void testname_gemm_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasGemmBatchedExModel{}.test_name(arg, name);
}

template <typename Ti, typename To = Ti, typename Tex = To>
void testing_gemm_batched_ex_bad_arg(const Arguments& arg)
{
    // Note: hipblasGemmEx and hipblasGemmExWithFlags are essentially the exact same.
    //       Only testing WithFlags version as it has slightly more functionality.
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGemmBatchedExFn
        = FORTRAN ? hipblasGemmBatchedExWithFlagsFortran : hipblasGemmBatchedExWithFlags;

    hipblasLocalHandle handle(arg);

    hipblasDatatype_t    aType           = arg.a_type;
    hipblasDatatype_t    bType           = arg.b_type;
    hipblasDatatype_t    cType           = arg.c_type;
    hipblasDatatype_t    computeType     = arg.compute_type;
    hipblasComputeType_t computeTypeGemm = arg.compute_type_gemm;
    hipblasGemmFlags_t   flags           = HIPBLAS_GEMM_FLAGS_NONE;
    hipblasGemmAlgo_t    algo            = HIPBLAS_GEMM_DEFAULT;

    int64_t M           = 101;
    int64_t N           = 100;
    int64_t K           = 102;
    int64_t lda         = 103;
    int64_t ldb         = 104;
    int64_t ldc         = 105;
    int64_t batch_count = 2;

    hipblasOperation_t transA = HIPBLAS_OP_N;
    hipblasOperation_t transB = HIPBLAS_OP_N;

    int64_t colsA = transA == HIPBLAS_OP_N ? N : M;
    int64_t colsB = transB == HIPBLAS_OP_N ? N : M;

    device_batch_vector<Ti> dA(colsA * lda, 1, batch_count);
    device_batch_vector<Ti> dB(colsB * ldb, 1, batch_count);
    device_batch_vector<To> dC(N * ldc, 1, batch_count);

    device_vector<Tex> d_alpha(1), d_beta(1), d_one(1), d_zero(1);
    Tex                h_alpha(1), h_beta(2), h_one(1), h_zero(0);

    if constexpr(std::is_same_v<Tex, hipblasHalf>)
        h_one = float_to_half(1.0f);

    const Tex* alpha = &h_alpha;
    const Tex* beta  = &h_beta;
    const Tex* one   = &h_one;
    const Tex* zero  = &h_zero;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

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

        // clang-format off

        EXPECT_HIPBLAS_STATUS(
            hipblasGemmBatchedExFn(nullptr, transA, transB, M, N, K, alpha,
                           (const void**)dA.ptr_on_device(), aType, lda,
                           (const void**)dB.ptr_on_device(), bType, ldb, beta,
                           (void**)dC.ptr_on_device(), cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                           computeTypeGemm,
#else
                           computeType,
#endif
                           algo, flags),
            HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedExFn(handle,
                                            (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                            transB, M, N, K, alpha,
                                            (const void**)dA.ptr_on_device(), aType, lda,
                                            (const void**)dB.ptr_on_device(), bType, ldb, beta,
                                            (void**)dC.ptr_on_device(), cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                                            computeTypeGemm,
#else
                                            computeType,
#endif
                                            algo, flags),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedExFn(handle, transA,
                                            (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                            M, N, K, alpha,
                                            (const void**)dA.ptr_on_device(), aType, lda,
                                            (const void**)dB.ptr_on_device(), bType, ldb, beta,
                                            (void**)dC.ptr_on_device(), cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                                            computeTypeGemm,
#else
                                            computeType,
#endif
                                            algo, flags),
                              HIPBLAS_STATUS_INVALID_ENUM);

                EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedExFn(handle, transA, transB, M, N, K, alpha,
                                            (const void**)dA.ptr_on_device(), aType, lda,
                                            (const void**)dB.ptr_on_device(), bType, ldb, beta,
                                            (void**)dC.ptr_on_device(), cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                                            computeTypeGemm,
#else
                                            computeType,
#endif
                                            (hipblasGemmAlgo_t)HIPBLAS_OP_N,
                                            flags),
                              HIPBLAS_STATUS_INVALID_ENUM);

        if(arg.bad_arg_all)
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasGemmBatchedExFn(
                    handle, transA, transB, M, N, K, alpha,
                    (const void**)dA.ptr_on_device(), aType, lda,
                    (const void**)dB.ptr_on_device(), bType, ldb, nullptr,
                    (void**)dC.ptr_on_device(), cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                    computeTypeGemm,
#else
                    computeType,
#endif
                    algo, flags),
                HIPBLAS_STATUS_INVALID_VALUE);

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                // alpha check only for host mode. rocBLAS can handle this in device mode too but shouldn't assume in case this changes.
                EXPECT_HIPBLAS_STATUS(
                    hipblasGemmBatchedExFn(
                        handle, transA, transB, M, N, K, nullptr,
                        (const void**)dA.ptr_on_device(), aType, lda,
                        (const void**)dB.ptr_on_device(), bType, ldb, beta,
                        (void**)dC.ptr_on_device(), cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                        computeTypeGemm,
#else
                        computeType,
#endif
                        algo, flags),
                    HIPBLAS_STATUS_INVALID_VALUE);

                // again, rocBLAS can handle this in device mode but shouldn't assume
                EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedExFn(handle, transA, transB, M, N, K, alpha,
                                                    nullptr, aType, lda,
                                                    (const void**)dB.ptr_on_device(), bType, ldb, beta,
                                                    (void**)dC.ptr_on_device(), cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                                                    computeTypeGemm,
#else
                                                    computeType,
#endif
                                                    algo, flags),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedExFn(handle, transA, transB, M, N, K, alpha,
                                                    (const void**)dA.ptr_on_device(), aType, lda,
                                                    nullptr, bType, ldb, beta,
                                                    (void**)dC.ptr_on_device(), cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                                                    computeTypeGemm,
#else
                                                    computeType,
#endif
                                                    algo, flags),
                                      HIPBLAS_STATUS_INVALID_VALUE);
                EXPECT_HIPBLAS_STATUS(hipblasGemmBatchedExFn(handle, transA, transB, M, N, K, alpha,
                                                    (const void**)dA.ptr_on_device(), aType, lda,
                                                    (const void**)dB.ptr_on_device(), bType, ldb, beta,
                                                    nullptr, cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                                                    computeTypeGemm,
#else
                                                    computeType,
#endif
                                                    algo, flags),
                                      HIPBLAS_STATUS_INVALID_VALUE);
            }

            // If alpha == 0, A and B can be nullptr
            CHECK_HIPBLAS_ERROR(hipblasGemmBatchedExFn(
                handle, transA, transB, M, N, K, zero,
                nullptr, aType, lda,
                nullptr, bType, ldb, beta,
                (void**)dC.ptr_on_device(), cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                computeTypeGemm,
#else
                computeType,
#endif
                algo, flags));

            // If K == 0, alpha, A, and B can be nullptr
            CHECK_HIPBLAS_ERROR(hipblasGemmBatchedExFn(handle, transA, transB, M, N, 0, nullptr,
                                              nullptr, aType, lda,
                                              nullptr, bType, ldb, beta,
                                              (void**)dC.ptr_on_device(), cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                                              computeTypeGemm,
#else
                                              computeType,
#endif
                                              algo, flags));
        }

        // If M == 0 || N == 0 || batch_count == 0, can have nullptrs
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedExFn(handle, transA, transB, 0, N, K, nullptr,
                                          nullptr, aType, lda,
                                          nullptr, bType, ldb, nullptr,
                                          nullptr, cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                                          computeTypeGemm,
#else
                                          computeType,
#endif
                                          algo, flags));
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedExFn(handle, transA, transB, M, 0, K, nullptr,
                                          nullptr, aType, lda,
                                          nullptr, bType, ldb, nullptr,
                                          nullptr, cType, ldc, batch_count,
#ifdef HIPBLAS_V2
                                          computeTypeGemm,
#else
                                          computeType,
#endif
                                          algo, flags));
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedExFn(handle, transA, transB, M, N, K, nullptr,
                                          nullptr, aType, lda,
                                          nullptr, bType, ldb, nullptr,
                                          nullptr, cType, ldc, 0,
#ifdef HIPBLAS_V2
                                          computeTypeGemm,
#else
                                          computeType,
#endif
                                          algo, flags));

        // clang-format on
    }
}

template <typename Ti, typename To = Ti, typename Tex = To>
void testing_gemm_batched_ex(const Arguments& arg)
{
    bool FORTRAN                = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasGemmBatchedExFn = FORTRAN ? hipblasGemmBatchedExFortran : hipblasGemmBatchedEx;
    auto hipblasGemmBatchedExWithFlagsFn
        = FORTRAN ? hipblasGemmBatchedExWithFlagsFortran : hipblasGemmBatchedExWithFlags;

    hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;

    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    hipblasOperation_t transB = char2hipblas_operation(arg.transB);

    int M = arg.M;
    int N = arg.N;
    int K = arg.K;

    int lda = arg.lda;
    int ldb = arg.ldb;
    int ldc = arg.ldc;

    int batch_count = arg.batch_count;

    hipblasDatatype_t    a_type            = arg.a_type;
    hipblasDatatype_t    b_type            = arg.b_type;
    hipblasDatatype_t    c_type            = arg.c_type;
    hipblasDatatype_t    compute_type      = arg.compute_type;
    hipblasComputeType_t compute_type_gemm = arg.compute_type_gemm;
    hipblasGemmFlags_t   flags             = hipblasGemmFlags_t(arg.flags);

    Tex h_alpha_Tex = arg.get_alpha<Tex>();
    Tex h_beta_Tex  = arg.get_beta<Tex>();

    int norm_check = arg.norm_check;
    int unit_check = arg.unit_check;
    int timing     = arg.timing;

    int A_row = transA == HIPBLAS_OP_N ? M : K;
    int A_col = transA == HIPBLAS_OP_N ? K : M;
    int B_row = transB == HIPBLAS_OP_N ? K : N;
    int B_col = transB == HIPBLAS_OP_N ? N : K;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0)
    {
        return;
    }

    const size_t size_A = static_cast<size_t>(lda) * static_cast<size_t>(A_col);
    const size_t size_B = static_cast<size_t>(ldb) * static_cast<size_t>(B_col);
    const size_t size_C = static_cast<size_t>(ldc) * static_cast<size_t>(N);

    device_batch_vector<Ti> dA(size_A, 1, batch_count);
    device_batch_vector<Ti> dB(size_B, 1, batch_count);
    device_batch_vector<To> dC(size_C, 1, batch_count);
    device_vector<Tex>      d_alpha(1);
    device_vector<Tex>      d_beta(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    host_batch_vector<Ti> hA(size_A, 1, batch_count);
    host_batch_vector<Ti> hB(size_B, 1, batch_count);
    host_batch_vector<To> hC_host(size_C, 1, batch_count);
    host_batch_vector<To> hC_device(size_C, 1, batch_count);
    host_batch_vector<To> hC_gold(size_C, 1, batch_count);

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    hipblas_init_vector(hA, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hB, arg, hipblas_client_alpha_sets_nan);
    hipblas_init_vector(hC_host, arg, hipblas_client_beta_sets_nan);

    hC_device.copy_from(hC_host);
    hC_gold.copy_from(hC_host);

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }

    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha_Tex, sizeof(Tex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta_Tex, sizeof(Tex), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        // hipBLAS
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        if(!arg.with_flags)
        {
            CHECK_HIPBLAS_ERROR(hipblasGemmBatchedExFn(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       &h_alpha_Tex,
                                                       (const void**)(Ti**)dA.ptr_on_device(),
                                                       a_type,
                                                       lda,
                                                       (const void**)(Ti**)dB.ptr_on_device(),
                                                       b_type,
                                                       ldb,
                                                       &h_beta_Tex,
                                                       (void**)(To**)dC.ptr_on_device(),
                                                       c_type,
                                                       ldc,
                                                       batch_count,
#ifdef HIPBLAS_V2
                                                       compute_type_gemm,
#else
                                                       compute_type,
#endif
                                                       algo));
        }
        else
        {
            CHECK_HIPBLAS_ERROR(
                hipblasGemmBatchedExWithFlagsFn(handle,
                                                transA,
                                                transB,
                                                M,
                                                N,
                                                K,
                                                &h_alpha_Tex,
                                                (const void**)(Ti**)dA.ptr_on_device(),
                                                a_type,
                                                lda,
                                                (const void**)(Ti**)dB.ptr_on_device(),
                                                b_type,
                                                ldb,
                                                &h_beta_Tex,
                                                (void**)(To**)dC.ptr_on_device(),
                                                c_type,
                                                ldc,
                                                batch_count,
#ifdef HIPBLAS_V2
                                                compute_type_gemm,
#else
                                                compute_type,
#endif
                                                algo,
                                                flags));
        }

        CHECK_HIP_ERROR(hC_host.transfer_from(dC));
        CHECK_HIP_ERROR(dC.transfer_from(hC_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        if(!arg.with_flags)
        {
            CHECK_HIPBLAS_ERROR(hipblasGemmBatchedExFn(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       d_alpha,
                                                       (const void**)(Ti**)dA.ptr_on_device(),
                                                       a_type,
                                                       lda,
                                                       (const void**)(Ti**)dB.ptr_on_device(),
                                                       b_type,
                                                       ldb,
                                                       d_beta,
                                                       (void**)(To**)dC.ptr_on_device(),
                                                       c_type,
                                                       ldc,
                                                       batch_count,
#ifdef HIPBLAS_V2
                                                       compute_type_gemm,
#else
                                                       compute_type,
#endif
                                                       algo));
        }
        else
        {
            CHECK_HIPBLAS_ERROR(
                hipblasGemmBatchedExWithFlagsFn(handle,
                                                transA,
                                                transB,
                                                M,
                                                N,
                                                K,
                                                d_alpha,
                                                (const void**)(Ti**)dA.ptr_on_device(),
                                                a_type,
                                                lda,
                                                (const void**)(Ti**)dB.ptr_on_device(),
                                                b_type,
                                                ldb,
                                                d_beta,
                                                (void**)(To**)dC.ptr_on_device(),
                                                c_type,
                                                ldc,
                                                batch_count,
#ifdef HIPBLAS_V2
                                                compute_type_gemm,
#else
                                                compute_type,
#endif
                                                algo,
                                                flags));
        }

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        // CPU BLAS
        for(int b = 0; b < batch_count; b++)
        {
            ref_gemm<Ti, To, Tex>(transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  h_alpha_Tex,
                                  hA[b],
                                  lda,
                                  hB[b],
                                  ldb,
                                  h_beta_Tex,
                                  hC_gold[b],
                                  ldc);
        }

        if(unit_check)
        {
            // check for float16/bfloat16 input
            if((getArchMajor() == 11)
               && ((std::is_same<Tex, float>{} && std::is_same<Ti, hipblasBfloat16>{})
                   || (std::is_same<Tex, float>{} && std::is_same<Ti, hipblasHalf>{})
                   || (std::is_same<Tex, hipblasHalf>{} && std::is_same<Ti, hipblasHalf>{})))
            {
                const double tol = K * sum_error_tolerance_for_gfx11<Tex, Ti, To>;
                near_check_general<To>(M, N, batch_count, ldc, hC_gold, hC_host, tol);
                near_check_general<To>(M, N, batch_count, ldc, hC_gold, hC_device, tol);
            }
            else
            {
                unit_check_general<To>(M, N, batch_count, ldc, hC_gold, hC_host);
                unit_check_general<To>(M, N, batch_count, ldc, hC_gold, hC_device);
            }
        }

        if(norm_check)
        {
            hipblas_error_host
                = norm_check_general<To>('F', M, N, ldc, hC_gold, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<To>('F', M, N, ldc, hC_gold, hC_device, batch_count);
        }
    }

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            if(!arg.with_flags)
            {
                CHECK_HIPBLAS_ERROR(hipblasGemmBatchedExFn(handle,
                                                           transA,
                                                           transB,
                                                           M,
                                                           N,
                                                           K,
                                                           &h_alpha_Tex,
                                                           (const void**)(Ti**)dA.ptr_on_device(),
                                                           a_type,
                                                           lda,
                                                           (const void**)(Ti**)dB.ptr_on_device(),
                                                           b_type,
                                                           ldb,
                                                           &h_beta_Tex,
                                                           (void**)(To**)dC.ptr_on_device(),
                                                           c_type,
                                                           ldc,
                                                           batch_count,
#ifdef HIPBLAS_V2
                                                           compute_type_gemm,
#else
                                                           compute_type,
#endif
                                                           algo));
            }
            else
            {
                CHECK_HIPBLAS_ERROR(
                    hipblasGemmBatchedExWithFlagsFn(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    K,
                                                    &h_alpha_Tex,
                                                    (const void**)(Ti**)dA.ptr_on_device(),
                                                    a_type,
                                                    lda,
                                                    (const void**)(Ti**)dB.ptr_on_device(),
                                                    b_type,
                                                    ldb,
                                                    &h_beta_Tex,
                                                    (void**)(To**)dC.ptr_on_device(),
                                                    c_type,
                                                    ldc,
                                                    batch_count,
#ifdef HIPBLAS_V2
                                                    compute_type_gemm,
#else
                                                    compute_type,
#endif
                                                    algo,
                                                    flags));
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemmBatchedExModel{}.log_args<To>(std::cout,
                                                 arg,
                                                 gpu_time_used,
                                                 gemm_gflop_count<Tex>(M, N, K),
                                                 gemm_gbyte_count<Tex>(M, N, K),
                                                 hipblas_error_host,
                                                 hipblas_error_device);
    }
}

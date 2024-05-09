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

#include <hipblas/hipblas.h>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(error)                              \
    if(error != HIPBLAS_STATUS_SUCCESS)                         \
    {                                                           \
        fprintf(stderr, "hipBLAS error: ");                     \
        if(error == HIPBLAS_STATUS_NOT_INITIALIZED)             \
            fprintf(stderr, "HIPBLAS_STATUS_NOT_INITIALIZED");  \
        if(error == HIPBLAS_STATUS_ALLOC_FAILED)                \
            fprintf(stderr, "HIPBLAS_STATUS_ALLOC_FAILED");     \
        if(error == HIPBLAS_STATUS_INVALID_VALUE)               \
            fprintf(stderr, "HIPBLAS_STATUS_INVALID_VALUE");    \
        if(error == HIPBLAS_STATUS_MAPPING_ERROR)               \
            fprintf(stderr, "HIPBLAS_STATUS_MAPPING_ERROR");    \
        if(error == HIPBLAS_STATUS_EXECUTION_FAILED)            \
            fprintf(stderr, "HIPBLAS_STATUS_EXECUTION_FAILED"); \
        if(error == HIPBLAS_STATUS_INTERNAL_ERROR)              \
            fprintf(stderr, "HIPBLAS_STATUS_INTERNAL_ERROR");   \
        if(error == HIPBLAS_STATUS_NOT_SUPPORTED)               \
            fprintf(stderr, "HIPBLAS_STATUS_NOT_SUPPORTED");    \
        if(error == HIPBLAS_STATUS_INVALID_ENUM)                \
            fprintf(stderr, "HIPBLAS_STATUS_INVALID_ENUM");     \
        if(error == HIPBLAS_STATUS_UNKNOWN)                     \
            fprintf(stderr, "HIPBLAS_STATUS_UNKNOWN");          \
        fprintf(stderr, "\n");                                  \
        exit(EXIT_FAILURE);                                     \
    }
#endif

#define DIM1 1023
#define DIM2 1024
#define DIM3 1025

void mat_mat_mult_hpa_half(float   alpha,
                           float   beta,
                           int     M,
                           int     N,
                           int     K,
                           __half* A,
                           size_t  As1,
                           size_t  As2,
                           __half* B,
                           size_t  Bs1,
                           size_t  Bs2,
                           __half* C,
                           size_t  Cs1,
                           size_t  Cs2)
{
    float beta_float  = beta;
    float alpha_float = alpha;
    for(int i1 = 0; i1 < M; i1++)
    {
        for(int i2 = 0; i2 < N; i2++)
        {
            float t     = 0.0;
            float C_val = __half2float(C[i1 * Cs1 + i2 * Cs2]);
            for(int i3 = 0; i3 < K; i3++)
            {
                float A_val = __half2float(A[i1 * As1 + i3 * As2]);
                float B_val = __half2float(B[i3 * Bs1 + i2 * Bs2]);
                t += A_val * B_val;
            }
            C[i1 * Cs1 + i2 * Cs2] = __float2half(beta_float * C_val + alpha_float * t);
        }
    }
}

int main()
{
    hipblasOperation_t transa = HIPBLAS_OP_N, transb = HIPBLAS_OP_T;

    int    m = DIM1, n = DIM2, k = DIM3;
    int    lda, ldb, ldc;
    size_t size_a, size_b, size_c;
    int    a_stride_1, a_stride_2, b_stride_1, b_stride_2;

    hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;

    // HPA
    using T   = __half;
    using Tex = float;
#ifdef HIPBLAS_V2
    // Compiling with HIPBLAS_V2 in cmake
    hipDataType          a_type       = HIP_R_16F;
    hipDataType          b_type       = HIP_R_16F;
    hipDataType          c_type       = HIP_R_16F;
    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
#else
    hipblasDatatype_t a_type       = HIPBLAS_R_16F;
    hipblasDatatype_t b_type       = HIPBLAS_R_16F;
    hipblasDatatype_t c_type       = HIPBLAS_R_16F;
    hipblasDatatype_t compute_type = HIPBLAS_R_32F;
#endif

    std::cout << "hipblasGemmEx example" << std::endl;
    if(transa == HIPBLAS_OP_N)
    {
        lda        = m;
        size_a     = k * size_t(lda);
        a_stride_1 = 1;
        a_stride_2 = lda;
        std::cout << "N";
    }
    else
    {
        lda        = k;
        size_a     = m * size_t(lda);
        a_stride_1 = lda;
        a_stride_2 = 1;
        std::cout << "T";
    }
    if(transb == HIPBLAS_OP_N)
    {
        ldb        = k;
        size_b     = n * size_t(ldb);
        b_stride_1 = 1;
        b_stride_2 = ldb;
        std::cout << "N: ";
    }
    else
    {
        ldb        = n;
        size_b     = k * size_t(ldb);
        b_stride_1 = ldb;
        b_stride_2 = 1;
        std::cout << "T: ";
    }
    ldc    = m;
    size_c = n * size_t(ldc);

    Tex alpha = 1.1, beta = 0.9;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<T> ha(size_a);
    std::vector<T> hb(size_b);
    std::vector<T> hc(size_c);
    std::vector<T> hc_gold(size_c);

    // initial data on host
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        // random number in [-2, 2]
        ha[i] = __float2half(rand() % 5 - 2.0f);
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = __float2half(rand() % 5 - 2.0f);
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = __float2half(rand() % 5 - 2.0f);
    }
    hc_gold = hc;

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(T)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(T) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(T) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(T) * size_c, hipMemcpyHostToDevice));

    hipblasHandle_t handle;
    CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

    CHECK_HIPBLAS_ERROR(hipblasGemmEx(handle,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      da,
                                      a_type,
                                      lda,
                                      db,
                                      b_type,
                                      ldb,
                                      &beta,
                                      dc,
                                      c_type,
                                      ldc,
                                      compute_type,
                                      algo));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(T) * size_c, hipMemcpyDeviceToHost));

    std::cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << lda << ", "
              << ldb << ", " << ldc << std::endl;

    float max_relative_error = std::numeric_limits<float>::min();

    // calculate golden or correct result
    mat_mat_mult_hpa_half(alpha,
                          beta,
                          m,
                          n,
                          k,
                          ha.data(),
                          a_stride_1,
                          a_stride_2,
                          hb.data(),
                          b_stride_1,
                          b_stride_2,
                          hc_gold.data(),
                          1,
                          ldc);

    for(int i = 0; i < size_c; i++)
    {
        float hc_gold_val    = __half2float(hc_gold[i]);
        float hc_val         = __half2float(hc[i]);
        float relative_error = hc_gold_val == 0 ? std::abs(hc_gold_val - hc_val)
                                                : (hc_gold_val - hc_val) / hc_gold_val;
        relative_error       = relative_error > 0 ? relative_error : -relative_error;
        max_relative_error
            = relative_error < max_relative_error ? max_relative_error : relative_error;
    }

    // 2 ^ -14, smallest positive normal number for IEEE16
    float eps = 0.000061035;

    // tolerance is arbitrary right now
    float tolerance = 20;
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        std::cout << "FAIL: max_relative_error = " << max_relative_error << std::endl;
    }
    else
    {
        std::cout << "PASS: max_relative_error = " << max_relative_error << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));
    return EXIT_SUCCESS;
}

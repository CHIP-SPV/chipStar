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

#include "unit.h"
#include "hipblas.h"
#include "hipblas_vector.hpp"
#include "utility.h"

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// This returns from the current function if an error occurs

#ifndef GOOGLE_TEST

#define UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, UNIT_ASSERT_EQ)
#define UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, UNIT_ASSERT_EQ)

#else // GOOGLE_TEST

#define UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, UNIT_ASSERT_EQ)      \
    do                                                                               \
    {                                                                                \
        for(size_t k = 0; k < batch_count; k++)                                      \
            for(size_t j = 0; j < N; j++)                                            \
                for(size_t i = 0; i < M; i++)                                        \
                    if(hipblas_isnan(hCPU[i + j * lda + k * strideA]))               \
                    {                                                                \
                        ASSERT_TRUE(hipblas_isnan(hGPU[i + j * lda + k * strideA])); \
                    }                                                                \
                    else                                                             \
                    {                                                                \
                        UNIT_ASSERT_EQ(hCPU[i + j * lda + k * strideA],              \
                                       hGPU[i + j * lda + k * strideA]);             \
                    }                                                                \
    } while(0)

#define UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, UNIT_ASSERT_EQ)            \
    do                                                                              \
    {                                                                               \
        for(size_t k = 0; k < batch_count; k++)                                     \
            for(size_t j = 0; j < N; j++)                                           \
                for(size_t i = 0; i < M; i++)                                       \
                    if(hipblas_isnan(hCPU[k][i + j * lda]))                         \
                    {                                                               \
                        ASSERT_TRUE(hipblas_isnan(hGPU[k][i + j * lda]));           \
                    }                                                               \
                    else                                                            \
                    {                                                               \
                        UNIT_ASSERT_EQ(hCPU[k][i + j * lda], hGPU[k][i + j * lda]); \
                    }                                                               \
    } while(0)

#endif // GOOGLE_TEST

#define ASSERT_HALF_EQ(a, b) ASSERT_FLOAT_EQ(half_to_float(a), half_to_float(b))
#define ASSERT_BFLOAT16_EQ(a, b) ASSERT_FLOAT_EQ(bfloat16_to_float(a), bfloat16_to_float(b))

#define ASSERT_FLOAT_COMPLEX_EQ(a, b)        \
    do                                       \
    {                                        \
        ASSERT_FLOAT_EQ(a.real(), b.real()); \
        ASSERT_FLOAT_EQ(a.imag(), b.imag()); \
    } while(0)

#define ASSERT_DOUBLE_COMPLEX_EQ(a, b)        \
    do                                        \
    {                                         \
        ASSERT_DOUBLE_EQ(a.real(), b.real()); \
        ASSERT_DOUBLE_EQ(a.imag(), b.imag()); \
    } while(0)

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, hipblasHalf* hCPU, hipblasHalf* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, int64_t lda, hipblasBfloat16* hCPU, hipblasBfloat16* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, float* hCPU, float* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, double* hCPU, double* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, int64_t lda, hipblasComplex* hCPU, hipblasComplex* hGPU)
{
#ifdef GOOGLE_TEST
    for(int64_t j = 0; j < N; j++)
        for(int64_t i = 0; i < M; i++)
        {
            ASSERT_FLOAT_EQ(hCPU[i + j * lda].real(), hGPU[i + j * lda].real());
            ASSERT_FLOAT_EQ(hCPU[i + j * lda].imag(), hGPU[i + j * lda].imag());
        }
#endif
}

template <>
void unit_check_general(
    int64_t M, int64_t N, int64_t lda, hipblasDoubleComplex* hCPU, hipblasDoubleComplex* hGPU)
{
#ifdef GOOGLE_TEST
    for(int64_t j = 0; j < N; j++)
        for(int64_t i = 0; i < M; i++)
        {
            ASSERT_DOUBLE_EQ(hCPU[i + j * lda].real(), hGPU[i + j * lda].real());
            ASSERT_DOUBLE_EQ(hCPU[i + j * lda].imag(), hGPU[i + j * lda].imag());
        }
#endif
}

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, int* hCPU, int* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_EQ);
}

template <>
void unit_check_general(int64_t M, int64_t N, int64_t lda, int64_t* hCPU, int64_t* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_EQ);
}

// batched checks
template <>
void unit_check_general(
    int64_t M, int64_t N, int64_t batch_count, int64_t lda, hipblasHalf** hCPU, hipblasHalf** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
void unit_check_general(int64_t           M,
                        int64_t           N,
                        int64_t           batch_count,
                        int64_t           lda,
                        hipblasBfloat16** hCPU,
                        hipblasBfloat16** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, int64_t batch_count, int64_t lda, float** hCPU, float** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, int64_t batch_count, int64_t lda, double** hCPU, double** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, int64_t batch_count, int64_t lda, int** hCPU, int** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, int64_t batch_count, int64_t lda, int64_t** hCPU, int64_t** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_EQ);
}

template <>
void unit_check_general(int64_t          M,
                        int64_t          N,
                        int64_t          batch_count,
                        int64_t          lda,
                        hipblasComplex** hCPU,
                        hipblasComplex** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
void unit_check_general(int64_t                M,
                        int64_t                N,
                        int64_t                batch_count,
                        int64_t                lda,
                        hipblasDoubleComplex** hCPU,
                        hipblasDoubleComplex** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

// batched checks for host_vector[]s
template <>
void unit_check_general(int64_t                  M,
                        int64_t                  N,
                        int64_t                  batch_count,
                        int64_t                  lda,
                        host_vector<hipblasHalf> hCPU[],
                        host_vector<hipblasHalf> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
void unit_check_general(int64_t                      M,
                        int64_t                      N,
                        int64_t                      batch_count,
                        int64_t                      lda,
                        host_vector<hipblasBfloat16> hCPU[],
                        host_vector<hipblasBfloat16> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
void unit_check_general(int64_t          M,
                        int64_t          N,
                        int64_t          batch_count,
                        int64_t          lda,
                        host_vector<int> hCPU[],
                        host_vector<int> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_EQ);
}

template <>
void unit_check_general(int64_t            M,
                        int64_t            N,
                        int64_t            batch_count,
                        int64_t            lda,
                        host_vector<float> hCPU[],
                        host_vector<float> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(int64_t             M,
                        int64_t             N,
                        int64_t             batch_count,
                        int64_t             lda,
                        host_vector<double> hCPU[],
                        host_vector<double> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
void unit_check_general(int64_t                     M,
                        int64_t                     N,
                        int64_t                     batch_count,
                        int64_t                     lda,
                        host_vector<hipblasComplex> hCPU[],
                        host_vector<hipblasComplex> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
void unit_check_general(int64_t                           M,
                        int64_t                           N,
                        int64_t                           batch_count,
                        int64_t                           lda,
                        host_vector<hipblasDoubleComplex> hCPU[],
                        host_vector<hipblasDoubleComplex> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

// strided_batched checks
template <>
void unit_check_general(int64_t       M,
                        int64_t       N,
                        int64_t       batch_count,
                        int64_t       lda,
                        hipblasStride strideA,
                        hipblasHalf*  hCPU,
                        hipblasHalf*  hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
void unit_check_general(int64_t          M,
                        int64_t          N,
                        int64_t          batch_count,
                        int64_t          lda,
                        hipblasStride    strideA,
                        hipblasBfloat16* hCPU,
                        hipblasBfloat16* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
void unit_check_general(int64_t       M,
                        int64_t       N,
                        int64_t       batch_count,
                        int64_t       lda,
                        hipblasStride strideA,
                        float*        hCPU,
                        float*        hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(int64_t       M,
                        int64_t       N,
                        int64_t       batch_count,
                        int64_t       lda,
                        hipblasStride strideA,
                        double*       hCPU,
                        double*       hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
void unit_check_general(int64_t         M,
                        int64_t         N,
                        int64_t         batch_count,
                        int64_t         lda,
                        hipblasStride   strideA,
                        hipblasComplex* hCPU,
                        hipblasComplex* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
void unit_check_general(int64_t               M,
                        int64_t               N,
                        int64_t               batch_count,
                        int64_t               lda,
                        hipblasStride         strideA,
                        hipblasDoubleComplex* hCPU,
                        hipblasDoubleComplex* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
void unit_check_general(int64_t       M,
                        int64_t       N,
                        int64_t       batch_count,
                        int64_t       lda,
                        hipblasStride strideA,
                        int*          hCPU,
                        int*          hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_EQ);
}

template <>
void unit_check_general(int64_t       M,
                        int64_t       N,
                        int64_t       batch_count,
                        int64_t       lda,
                        hipblasStride strideA,
                        int64_t*      hCPU,
                        int64_t*      hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_EQ);
}

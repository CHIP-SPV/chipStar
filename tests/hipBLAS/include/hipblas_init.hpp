/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif
#include <assert.h>

#include "hipblas.h"
#include "hipblas_vector.hpp"

//!
//! @brief enum to check for NaN initialization of the Input vector/matrix
//!
typedef enum hipblas_client_nan_init_
{
    // Alpha sets NaN
    hipblas_client_alpha_sets_nan,

    // Beta sets NaN
    hipblas_client_beta_sets_nan,

    //  Never set NaN
    hipblas_client_never_set_nan

} hipblas_client_nan_init;

/*************************************************************************************************************************
//! @brief enum for the type of matrix
 ************************************************************************************************************************/
typedef enum hipblas_matrix_type_
{
    // General matrix
    hipblas_general_matrix,

    // Hermitian matrix
    hipblas_hermitian_matrix,

    // Symmetric matrix
    hipblas_symmetric_matrix,

    // Triangular matrix
    hipblas_triangular_matrix,

    // Diagonally dominant triangular matrix
    hipblas_diagonally_dominant_triangular_matrix,

} hipblas_matrix_type;

//!
//! @brief Template for initializing a host (non_batched|batched|strided_batched)vector.
//! @param that That vector.
//! @param rand_gen The random number generator
//! @param seedReset Reset the seed if true, do not reset the seed otherwise.
//!
template <typename U, typename T>
void hipblas_init_template(U& that, T rand_gen(), bool seedReset, bool alternating_sign = false)
{
    if(seedReset)
        hipblas_seedrand();

    for(int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
    {
        auto*   batched_data = that[batch_index];
        int64_t inc          = that.inc();
        auto    n            = that.n();
        if(inc < 0)
            batched_data -= (n - 1) * inc;

        if(alternating_sign)
        {
            for(size_t i = 0; i < n; i++)
            {
                auto value            = rand_gen();
                batched_data[i * inc] = (i ^ 0) & 1 ? value : hipblas_negate(value);
            }
        }
        else
        {
            for(size_t i = 0; i < n; ++i)
                batched_data[i * inc] = rand_gen();
        }
    }
}

//!
//! @brief Initialize a host_batch_vector with NaNs.
//! @param that The host_batch_vector to be initialized.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void hipblas_init_nan(host_batch_vector<T>& that, bool seedReset = false)
{
    hipblas_init_template(that, random_nan_generator<T>, seedReset);
}

// //!
// //! @brief Initialize a host_vector with NaNs.
// //! @param that The host_vector to be initialized.
// //! @param seedReset reset he seed if true, do not reset the seed otherwise.
// //!
// template <typename T>
// inline void hipblas_init_nan(host_vector<T>& that, bool seedReset = false)
// {
//     hipblas_init_template(that, random_nan_generator<T>, seedReset);
// }

template <typename T>
inline void hipblas_init_nan(host_vector<T>& A,
                             size_t          M,
                             size_t          N,
                             size_t          lda,
                             hipblasStride   stride      = 0,
                             int64_t         batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(hipblas_nan_rng());
}

//!
//! @brief Initialize a host_batch_vector.
//! @param that The host_batch_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void hipblas_init_hpl(host_batch_vector<T>& that,
                             bool                  seedReset        = false,
                             bool                  alternating_sign = false)
{
    hipblas_init_template(that, random_hpl_generator<T>, seedReset, alternating_sign);
}

//!
//! @brief Initialize a host_batch_vector.
//! @param that The host_batch_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void
    hipblas_init(host_batch_vector<T>& that, bool seedReset = false, bool alternating_sign = false)
{
    hipblas_init_template(that, random_generator<T>, seedReset, alternating_sign);
}

//!
//! @brief Initialize a host_vector.
//! @param that The host_vector.
//! @param seedReset reset the seed if true, do not reset the seed otherwise.
//!
template <typename T>
inline void hipblas_init(host_vector<T>& that, bool seedReset = false)
{
    if(seedReset)
        hipblas_seedrand();
    hipblas_init(that, that.size(), 1, 1);
}

//!
//! @brief trig Initialize of a host_batch_vector.
//! @param that The host_batch_vector.
//! @param init_cos cos initialize if true, else sin initialize.
//!
template <typename T>
inline void hipblas_init_trig(host_batch_vector<T>& that, bool init_cos = false)
{
    if(init_cos)
    {
        for(int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
        {
            auto*     batched_data = that[batch_index];
            ptrdiff_t inc          = that.inc();
            auto      n            = that.n();

            if(inc < 0)
                batched_data -= (n - 1) * inc;

            hipblas_init_cos(batched_data, 1, n, inc, 0, 1);
        }
    }
    else
    {
        for(int batch_index = 0; batch_index < that.batch_count(); ++batch_index)
        {
            auto*     batched_data = that[batch_index];
            ptrdiff_t inc          = that.inc();
            auto      n            = that.n();

            if(inc < 0)
                batched_data -= (n - 1) * inc;

            hipblas_init_sin(batched_data, 1, n, inc, 0, 1);
        }
    }
}

//!
//! @brief Initialize a host_vector.
//! @param hx The host_vector.
//! @param arg Specifies the argument class.
//! @param N Length of the host vector.
//! @param incx Increment for the host vector.
//! @param stride_x Incement between the host vector.
//! @param batch_count number of instances in the batch.
//! @param nan_init Initialize vector with Nan's depending upon the hipblas_client_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize vector so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblas_init_vector(host_vector<T>&         hx,
                                const Arguments&        arg,
                                size_t                  N,
                                int64_t                 incx,
                                hipblasStride           stride_x,
                                int64_t                 batch_count,
                                hipblas_client_nan_init nan_init,
                                bool                    seedReset        = false,
                                bool                    alternating_sign = false)
{
    if(seedReset)
        hipblas_seedrand();

    if(nan_init == hipblas_client_alpha_sets_nan && hipblas_isnan(arg.alpha))
    {
        hipblas_init_nan(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(nan_init == hipblas_client_beta_sets_nan && hipblas_isnan(arg.beta))
    {
        hipblas_init_nan(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == hipblas_initialization::hpl)
    {
        if(alternating_sign)
            hipblas_init_hpl_alternating_sign(hx, 1, N, incx, stride_x, batch_count);
        else
            hipblas_init_hpl(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == hipblas_initialization::rand_int)
    {
        if(alternating_sign)
            hipblas_init_alternating_sign(hx, 1, N, incx, stride_x, batch_count);
        else
            hipblas_init(hx, 1, N, incx, stride_x, batch_count);
    }
    else if(arg.initialization == hipblas_initialization::trig_float)
    {
        if(seedReset)
            hipblas_init_cos(hx, 1, N, incx, stride_x, batch_count);
        else
            hipblas_init_sin(hx, 1, N, incx, stride_x, batch_count);
    }
}

//!
//! @brief Initialize a host_batch_vector.
//! @param hx The host_batch_vector.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize vector with Nan's depending upon the hipblas_client_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize vector so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblas_init_vector(host_batch_vector<T>&   hx,
                                const Arguments&        arg,
                                hipblas_client_nan_init nan_init,
                                bool                    seedReset        = false,
                                bool                    alternating_sign = false)
{
    if(nan_init == hipblas_client_alpha_sets_nan && hipblas_isnan(arg.alpha))
    {
        hipblas_init_nan(hx, seedReset);
    }
    else if(nan_init == hipblas_client_beta_sets_nan && hipblas_isnan(arg.beta))
    {
        hipblas_init_nan(hx, seedReset);
    }
    else if(arg.initialization == hipblas_initialization::hpl)
    {
        hipblas_init_hpl(hx, seedReset, alternating_sign);
    }
    else if(arg.initialization == hipblas_initialization::rand_int)
    {
        hipblas_init(hx, seedReset, alternating_sign);
    }
    else if(arg.initialization == hipblas_initialization::trig_float)
    {
        hipblas_init_trig(hx, seedReset);
    }
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alternating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.

template <typename T>
void hipblas_init_matrix_alternating_sign(hipblas_matrix_type matrix_type,
                                          const char          uplo,
                                          T                   rand_gen(),
                                          T*                  A,
                                          size_t              M,
                                          size_t              N,
                                          size_t              lda,
                                          hipblasStride       stride      = 0,
                                          int64_t             batch_count = 1)
{
    if(matrix_type == hipblas_general_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value                  = rand_gen();
                    A[i + j * lda + b * stride] = (i ^ j) & 1 ? T(value) : T(hipblas_negate(value));
                }
    }
    else if(matrix_type == hipblas_triangular_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : 0) : (j <= i ? rand_gen() : 0);
                    A[i + j * lda + b * stride] = (i ^ j) & 1 ? T(value) : T(hipblas_negate(value));
                }
    }
}

/* ============================================================================================ */
/*! \brief  Trigonometric matrix initialization: */
// Initialize matrix with rand_int/hpl/NaN values

template <typename T>
void hipblas_init_matrix_trig(hipblas_matrix_type matrix_type,
                              const char          uplo,
                              T*                  A,
                              size_t              M,
                              size_t              N,
                              size_t              lda,
                              hipblasStride       stride      = 0,
                              int64_t             batch_count = 1,
                              bool                use_cosine  = false)
{
    if(matrix_type == hipblas_general_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t j = 0; j < N; ++j)
                for(size_t i = 0; i < M; ++i)
                    A[i + j * lda + b * stride] = T(use_cosine ? cos(i + j * lda + b * stride)
                                                               : sin(i + j * lda + b * stride));
    }
    else if(matrix_type == hipblas_hermitian_matrix)
    {
        for(size_t b = 0; b < batch_count; ++b)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = T(use_cosine ? cos(i + j * lda + b * stride)
                                              : sin(i + j * lda + b * stride));

                    if(i == j)
                        A[b * stride + j + i * lda] = hipblas_real(value);
                    else if(uplo == 'U')
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[b * stride + j + i * lda] = T(0);
                        A[b * stride + i + j * lda] = value;
                    }
                    else
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = hipblas_conjugate(value);
                    }
                }
    }
    else if(matrix_type == hipblas_symmetric_matrix)
    {
        for(size_t b = 0; b < batch_count; ++b)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = T(use_cosine ? cos(i + j * lda + b * stride)
                                              : sin(i + j * lda + b * stride));
                    if(i == j)
                        A[b * stride + j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[b * stride + j + i * lda] = T(0);
                        A[b * stride + i + j * lda] = value;
                    }
                    else
                    {
                        A[b * stride + j + i * lda] = value;
                        A[b * stride + i + j * lda] = value;
                    }
                }
    }
    else if(matrix_type == hipblas_triangular_matrix)
    {
        for(size_t b = 0; b < batch_count; b++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t j = 0; j < N; ++j)
            {
                size_t offset = j * lda + b * stride;
                for(size_t i = 0; i < M; ++i)
                {
                    auto value
                        = uplo == 'U'
                              ? (j >= i ? T(use_cosine ? cos(i + offset) : sin(i + offset)) : T(0))
                              : (j <= i ? T(use_cosine ? cos(i + offset) : sin(i + offset)) : T(0));
                    A[i + offset] = value;
                }
            }
    }
    else
    {
        assert(false);
    }
}

/* ============================================================================================ */
/*! \brief  matrix initialization: */
// Initialize matrix according to the matrix_types

template <typename T>
void hipblas_fill_matrix_type(hipblas_matrix_type matrix_type,
                              const char          uplo,
                              T                   rand_gen(),
                              T*                  A_data,
                              int64_t             M,
                              int64_t             N,
                              size_t              lda,
                              hipblasStride       stride      = 0,
                              int64_t             batch_count = 1)
{
    for(size_t b = 0; b < batch_count; b++)
    {
        T* A = A_data + b * stride;

        if(matrix_type == hipblas_general_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t j = 0; j < N; ++j)
                for(size_t i = 0; i < M; ++i)
                    A[i + j * lda] = rand_gen();
        }
        else if(matrix_type == hipblas_hermitian_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[j + i * lda] = hipblas_real(value);
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = T(0);
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = hipblas_conjugate(value);
                    }
                }
        }
        else if(matrix_type == hipblas_symmetric_matrix)
        {

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = T(0);
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = value;
                    }
                }
        }
        else if(matrix_type == hipblas_triangular_matrix)
        {

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : T(0)) : (j <= i ? rand_gen() : T(0));
                    A[i + j * lda + b * stride] = value;
                }
        }
        else if(matrix_type == hipblas_diagonally_dominant_triangular_matrix)
        {
            /*An n x n triangle matrix with random entries has a condition number that grows exponentially with n ("Condition numbers of random triangular matrices" D. Viswanath and L.N.Trefethen).
    Here we use a triangle matrix with random values that is strictly row and column diagonal dominant.
    This matrix should have a lower condition number. An alternative is to calculate the Cholesky factor of an SPD matrix with random values and make it diagonal dominant.
    This approach is not used because it is slow.*/

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int64_t j = 0; j < N; ++j)
                for(int64_t i = 0; i < M; ++i)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : T(0)) : (j <= i ? rand_gen() : T(0));
                    A[i + j * lda] = value;
                }

            const T multiplier = T(
                1.01); // Multiplying factor to slightly increase the base value of (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) dominant diagonal element. If tests fail and it seems that there are numerical stability problems, try increasing multiplier, it should decrease the condition number of the matrix and thereby avoid numerical stability issues.

            if(uplo == 'U') // hipblas_fill_upper
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for(int64_t i = 0; i < N; i++)
                {
                    T abs_sum_off_diagonal_row = T(
                        0); //store absolute sum of entire row of the particular diagonal element
                    T abs_sum_off_diagonal_col = T(
                        0); //store absolute sum of entire column of the particular diagonal element

                    for(int64_t j = i + 1; j < N; j++)
                        abs_sum_off_diagonal_row += hipblas_abs(A[i + j * lda]);
                    for(size_t j = 0; j < i; j++)
                        abs_sum_off_diagonal_col += hipblas_abs(A[j + i * lda]);

                    A[i + i * lda] = (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) == T(0)
                                         ? T(1)
                                         : T((abs_sum_off_diagonal_row + abs_sum_off_diagonal_col)
                                             * multiplier);
                }
            }
            else // hipblas_fill_lower
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for(int64_t j = 0; j < N; j++)
                {
                    T abs_sum_off_diagonal_row = T(
                        0); //store absolute sum of entire row of the particular diagonal element
                    T abs_sum_off_diagonal_col = T(
                        0); //store absolute sum of entire column of the particular diagonal element

                    for(int64_t i = j + 1; i < N; i++)
                        abs_sum_off_diagonal_col += hipblas_abs(A[i + j * lda]);

                    for(int64_t i = 0; i < j; i++)
                        abs_sum_off_diagonal_row += hipblas_abs(A[j + i * lda]);

                    A[j + j * lda] = (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) == T(0)
                                         ? T(1)
                                         : T((abs_sum_off_diagonal_row + abs_sum_off_diagonal_col)
                                             * multiplier);
                }
            }
        }
        else
        {
            assert(false);
        }
    } // for batch
}

template <typename T>
void hipblas_init_matrix(const char      uplo,
                         T               rand_gen(),
                         host_vector<T>& A,
                         size_t          M,
                         size_t          N,
                         size_t          lda,
                         hipblasStride   stride      = 0,
                         int64_t         batch_count = 1)
{
    hipblas_fill_matrix_type(
        hipblas_general_matrix, uplo, rand_gen(), A, M, N, lda, stride, batch_count);
}

template <typename T>
void hipblas_init_matrix(
    const char uplo, T rand_gen(), host_batch_vector<T>& A, size_t M, size_t N, size_t lda)
{
    for(int64_t b = 0; b < A.batch_count(); b++)
        hipblas_fill_matrix_type(
            hipblas_general_matrix, uplo, rand_gen(), (T*)A[b], M, N, lda, 0, 1);
}

/*
template <typename U, typename T>
void hipblas_init_matrix(hipblas_matrix_type matrix_type,
                         const char                uplo,
                         T                         rand_gen(),
                         U&                        hA,
                         int)
{
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto*   A   = hA[batch_index];
        int64_t M   = hA.m();
        int64_t N   = hA.n();
        int64_t lda = hA.lda();
        if(matrix_type == hipblas_general_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t j = 0; j < N; ++j)
                for(size_t i = 0; i < M; ++i)
                    A[i + j * lda] = rand_gen();
        }
        else if(matrix_type == hipblas_hermitian_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[j + i * lda] = hipblas_real(value);
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = T(0);
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = hipblas_conjugate(value);
                    }
                }
        }
        else if(matrix_type == hipblas_symmetric_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = rand_gen();
                    if(i == j)
                        A[j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = T(0);
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = value;
                    }
                }
        }
        else if(matrix_type == hipblas_triangular_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t j = 0; j < N; ++j)
                for(size_t i = 0; i < M; ++i)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : T(0)) : (j <= i ? rand_gen() : T(0));
                    A[i + j * lda] = value;
                }
        }
        else if(matrix_type == hipblas_diagonally_dominant_triangular_matrix)
        {
            //An n x n triangle matrix with random entries has a condition number that grows exponentially with n ("Condition numbers of random triangular matrices" D. Viswanath and L.N.Trefethen).
            //Here we use a triangle matrix with random values that is strictly row and column diagonal dominant.
            //This matrix should have a lower condition number. An alternative is to calculate the Cholesky factor of an SPD matrix with random values and make it diagonal dominant.
            //This approach is not used because it is slow.

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t j = 0; j < N; ++j)
                for(size_t i = 0; i < M; ++i)
                {
                    auto value
                        = uplo == 'U' ? (j >= i ? rand_gen() : T(0)) : (j <= i ? rand_gen() : T(0));
                    A[i + j * lda] = value;
                }

            const T multiplier = T(
                1.01); // Multiplying factor to slightly increase the base value of (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) dominant diagonal element. If tests fail and it seems that there are numerical stability problems, try increasing multiplier, it should decrease the condition number of the matrix and thereby avoid numerical stability issues.

            if(uplo == 'U') // hipblas_fill_upper
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for(size_t i = 0; i < N; i++)
                {
                    T abs_sum_off_diagonal_row = T(
                        0); //store absolute sum of entire row of the particular diagonal element
                    T abs_sum_off_diagonal_col = T(
                        0); //store absolute sum of entire column of the particular diagonal element

                    for(size_t j = i + 1; j < N; j++)
                        abs_sum_off_diagonal_row += hipblas_abs(A[i + j * lda]);
                    for(size_t j = 0; j < i; j++)
                        abs_sum_off_diagonal_col += hipblas_abs(A[j + i * lda]);

                    A[i + i * lda] = (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) == T(0)
                                         ? T(1)
                                         : T((abs_sum_off_diagonal_row + abs_sum_off_diagonal_col)
                                             * multiplier);
                }
            }
            else // hipblas_fill_lower
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for(size_t j = 0; j < N; j++)
                {
                    T abs_sum_off_diagonal_row = T(
                        0); //store absolute sum of entire row of the particular diagonal element
                    T abs_sum_off_diagonal_col = T(
                        0); //store absolute sum of entire column of the particular diagonal element

                    for(size_t i = j + 1; i < N; i++)
                        abs_sum_off_diagonal_col += hipblas_abs(A[i + j * lda]);

                    for(size_t i = 0; i < j; i++)
                        abs_sum_off_diagonal_row += hipblas_abs(A[j + i * lda]);

                    A[j + j * lda] = (abs_sum_off_diagonal_row + abs_sum_off_diagonal_col) == T(0)
                                         ? T(1)
                                         : T((abs_sum_off_diagonal_row + abs_sum_off_diagonal_col)
                                             * multiplier);
                }
            }
        }
    }
}


template <typename T, typename U>
void hipblas_init_matrix_trig(hipblas_matrix_type matrix_type,
                              const char                uplo,
                              U&                        hA,
                              bool                      seedReset = false)
{
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

        if(matrix_type == hipblas_general_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                    A[i + j * lda] = T(seedReset ? cos(i + j * lda) : sin(i + j * lda));
        }
        else if(matrix_type == hipblas_hermitian_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = T(seedReset ? cos(i + j * lda) : sin(i + j * lda));

                    if(i == j)
                        A[j + i * lda] = hipblas_real(value);
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = T(0);
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = hipblas_conjugate(value);
                    }
                }
        }
        else if(matrix_type == hipblas_symmetric_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < N; ++i)
                for(size_t j = 0; j <= i; ++j)
                {
                    auto value = T(seedReset ? cos(i + j * lda) : sin(i + j * lda));
                    if(i == j)
                        A[j + i * lda] = value;
                    else if(uplo == 'U')
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = T(0);
                    }
                    else if(uplo == 'L')
                    {
                        A[j + i * lda] = T(0);
                        A[i + j * lda] = value;
                    }
                    else
                    {
                        A[j + i * lda] = value;
                        A[i + j * lda] = value;
                    }
                }
        }
        else if(matrix_type == hipblas_triangular_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value
                        = uplo == 'U'
                              ? (j >= i ? T(seedReset ? cos(i + j * lda) : sin(i + j * lda)) : T(0))
                              : (j <= i ? T(seedReset ? cos(i + j * lda) : sin(i + j * lda))
                                        : T(0));
                    A[i + j * lda] = value;
                }
        }
    }
}
*/

//!
//! @brief Initialize a host matrix.
//! @param hA The host matrix.
//! @param arg Specifies the argument class.
//! @param M Length of the host matrix.
//! @param N Length of the host matrix.
//! @param lda Leading dimension of the host matrix.
//! @param stride_A Incement between the host matrix.
//! @param batch_count number of instances in the batch.
//! @param nan_init Initialize matrix with Nan's depending upon the hipblas_client_nan_init enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblas_init_matrix_type(hipblas_matrix_type     matrix_type,
                                     T*                      hA,
                                     const Arguments&        arg,
                                     size_t                  M,
                                     size_t                  N,
                                     size_t                  lda,
                                     hipblasStride           stride_A,
                                     int                     batch_count,
                                     hipblas_client_nan_init nan_init,
                                     bool                    seedReset        = false,
                                     bool                    alternating_sign = false)
{
    if(seedReset)
        hipblas_seedrand();

    if(nan_init == hipblas_client_alpha_sets_nan && hipblas_isnan(arg.alpha))
    {
        hipblas_fill_matrix_type(
            matrix_type, arg.uplo, random_nan_generator<T>, hA, M, N, lda, stride_A, batch_count);
    }
    else if(nan_init == hipblas_client_beta_sets_nan && hipblas_isnan(arg.beta))
    {
        hipblas_fill_matrix_type(
            matrix_type, arg.uplo, random_nan_generator<T>, hA, M, N, lda, stride_A, batch_count);
    }
    else if(arg.initialization == hipblas_initialization::hpl)
    {
        if(alternating_sign)
            hipblas_init_matrix_alternating_sign(matrix_type,
                                                 arg.uplo,
                                                 random_hpl_generator<T>,
                                                 hA,
                                                 M,
                                                 N,
                                                 lda,
                                                 stride_A,
                                                 batch_count);
        else
            hipblas_fill_matrix_type(matrix_type,
                                     arg.uplo,
                                     random_hpl_generator<T>,
                                     hA,
                                     M,
                                     N,
                                     lda,
                                     stride_A,
                                     batch_count);
    }
    else if(arg.initialization == hipblas_initialization::rand_int)
    {
        if(alternating_sign)
            hipblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_generator<T>, hA, M, N, lda, stride_A, batch_count);
        else
            hipblas_fill_matrix_type(
                matrix_type, arg.uplo, random_generator<T>, hA, M, N, lda, stride_A, batch_count);
    }
    else if(arg.initialization == hipblas_initialization::trig_float)
    {
        hipblas_init_matrix_trig<T>(
            matrix_type, arg.uplo, hA, seedReset, M, N, lda, stride_A, batch_count);
    }
}

template <typename T>
inline void hipblas_init_matrix(host_batch_vector<T>&   hA,
                                const Arguments&        arg,
                                size_t                  M,
                                size_t                  N,
                                size_t                  lda,
                                hipblas_client_nan_init nan_init,
                                hipblas_matrix_type     matrix_type,
                                bool                    seed_reset       = false,
                                bool                    alternating_sign = false)
{
    for(int64_t b = 0; b < hA.batch_count(); b++)
        hipblas_init_matrix_type(matrix_type,
                                 (T*)hA[b],
                                 arg,
                                 M,
                                 N,
                                 lda,
                                 0,
                                 1,
                                 nan_init,
                                 seed_reset && b == 0,
                                 alternating_sign);
}

template <typename T>
inline void hipblas_init_matrix(host_vector<T>&         hA,
                                const Arguments&        arg,
                                size_t                  M,
                                size_t                  N,
                                size_t                  lda,
                                hipblasStride           stride_A,
                                int                     batch_count,
                                hipblas_client_nan_init nan_init,
                                bool                    seed_reset       = false,
                                bool                    alternating_sign = false)
{
    hipblas_init_matrix_type(hipblas_general_matrix,
                             (T*)hA,
                             arg,
                             M,
                             N,
                             lda,
                             stride_A,
                             batch_count,
                             nan_init,
                             seed_reset,
                             alternating_sign);
}

//!
//! @brief Template for initializing a host (non_batched|batched|strided_batched)vector.
//! @param that That vector.
//! @param rand_gen The random number generator for odd elements
//! @param rand_gen_alt The random number generator for even elements
//! @param seedReset Reset the seed if true, do not reset the seed otherwise.
//!
template <typename U, typename T>
void hipblas_init_alternating_template(U& that, T rand_gen(), T rand_gen_alt(), bool seedReset)
{
    if(seedReset)
        hipblas_seedrand();

    for(int64_t b = 0; b < that.batch_count(); ++b)
    {
        auto*   batched_data = that[b];
        int64_t inc          = that.inc();
        auto    n            = that.n();
        if(inc < 0)
            batched_data -= (n - 1) * inc;

        for(int64_t i = 0; i < n; ++i)
        {
            if(i % 2)
                batched_data[i * inc] = rand_gen();
            else
                batched_data[i * inc] = rand_gen_alt();
        }
    }
}

template <typename T>
void hipblas_init_alternating_sign(host_batch_vector<T>& that, bool seedReset = false)
{
    hipblas_init_alternating_template(
        that, random_generator<T>, random_generator_negative<T>, seedReset);
}

/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#include "blas1_ex_gtest.hpp"

#include "blas_ex/testing_scal_batched_ex.hpp"
#include "blas_ex/testing_scal_ex.hpp"
#include "blas_ex/testing_scal_strided_batched_ex.hpp"

namespace
{
    // ----------------------------------------------------------------------------
    // BLAS1_ex testing template
    // ----------------------------------------------------------------------------
    template <template <typename...> class FILTER, blas1_ex BLAS1_EX>
    struct scal_ex_test_template
        : public HipBLAS_Test<scal_ex_test_template<FILTER, BLAS1_EX>, FILTER>
    {
        template <typename... T>
        struct type_filter_functor
        {
            bool operator()(const Arguments& args)
            {
                // additional global filters applied first
                if(!hipblas_client_global_filters(args))
                    return false;

                // type filters
                return static_cast<bool>(FILTER<T...>{});
            }
        };

        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return hipblas_blas1_ex_dispatch<scal_ex_test_template::template type_filter_functor>(
                arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg);

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(BLAS1_EX == blas1_ex::scal_ex)
                testname_scal_ex(arg, name);
            else if constexpr(BLAS1_EX == blas1_ex::scal_batched_ex)
                testname_scal_batched_ex(arg, name);
            else if constexpr(BLAS1_EX == blas1_ex::scal_strided_batched_ex)
                testname_scal_strided_batched_ex(arg, name);
            return std::move(name);
        }
    };

    // This tells whether the BLAS1_EX tests are enabled
    // Appears that we will need up to 4 template variables (see dot)
    template <blas1_ex BLAS1_EX, typename T1, typename T2, typename T3>
    using blas1_ex_enabled = std::integral_constant<
        bool,
        // scal_ex
        // T1 is alpha_type T2 is x_type T3 is execution_type
        ((BLAS1_EX == blas1_ex::scal_ex || BLAS1_EX == blas1_ex::scal_batched_ex
          || BLAS1_EX == blas1_ex::scal_strided_batched_ex)
         && ((std::is_same_v<
                  T1,
                  T2> && std::is_same_v<T2, T3> && (std::is_same_v<T1, float> || std::is_same_v<T1, double> || std::is_same_v<T1, hipblasHalf> || std::is_same_v<T1, hipblasComplex> || std::is_same_v<T1, hipblasDoubleComplex>))
             || (std::is_same_v<
                     T2,
                     T3> && std::is_same_v<T1, float> && std::is_same_v<T2, hipblasComplex>)
             || (std::is_same_v<
                     T2,
                     T3> && std::is_same_v<T1, double> && std::is_same_v<T2, hipblasDoubleComplex>)
             || (std::is_same_v<T1,
                                T2> && std::is_same_v<T1, hipblasHalf> && std::is_same_v<T3, float>)
             || (std::is_same_v<T1,
                                T3> && std::is_same_v<T1, float> && std::is_same_v<T2, hipblasHalf>)
             || (std::is_same_v<
                     T1,
                     hipblasBfloat16> && std::is_same_v<T1, T2> && std::is_same_v<T3, float>)
             || (std::is_same_v<
                     T1,
                     float> && std::is_same_v<T2, hipblasBfloat16> && std::is_same_v<T1, T3>)))>;

// Creates tests for one of the BLAS 1 functions
// ARG passes 1-3 template arguments to the testing_* function
#define BLAS1_EX_TESTING(NAME, ARG)                                                        \
    struct blas1_ex_##NAME                                                                 \
    {                                                                                      \
        template <typename Ta, typename Tb = Ta, typename Tex = Tb, typename = void>       \
        struct testing : hipblas_test_invalid                                              \
        {                                                                                  \
        };                                                                                 \
                                                                                           \
        template <typename Ta, typename Tb, typename Tex>                                  \
        struct testing<Ta,                                                                 \
                       Tb,                                                                 \
                       Tex,                                                                \
                       std::enable_if_t<blas1_ex_enabled<blas1_ex::NAME, Ta, Tb, Tex>{}>>  \
            : hipblas_test_valid                                                           \
        {                                                                                  \
            void operator()(const Arguments& arg)                                          \
            {                                                                              \
                if(!strcmp(arg.function, #NAME))                                           \
                    testing_##NAME<ARG(Ta, Tb, Tex)>(arg);                                 \
                else if(!strcmp(arg.function, #NAME "_bad_arg"))                           \
                    testing_##NAME##_bad_arg<ARG(Ta, Tb, Tex)>(arg);                       \
                else                                                                       \
                    FAIL() << "Internal error: Test called with unknown function: "        \
                           << arg.function;                                                \
            }                                                                              \
        };                                                                                 \
    };                                                                                     \
                                                                                           \
    using NAME = scal_ex_test_template<blas1_ex_##NAME::template testing, blas1_ex::NAME>; \
                                                                                           \
    template <>                                                                            \
    inline bool NAME::function_filter(const Arguments& arg)                                \
    {                                                                                      \
        return !strcmp(arg.function, #NAME) || !strcmp(arg.function, #NAME "_bad_arg");    \
    }                                                                                      \
                                                                                           \
    TEST_P(NAME, blas1_ex)                                                                 \
    {                                                                                      \
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(                                          \
            hipblas_blas1_ex_dispatch<blas1_ex_##NAME::template testing>(GetParam()));     \
    }                                                                                      \
                                                                                           \
    INSTANTIATE_TEST_CATEGORIES(NAME)

#define ARG3(Ta, Tb, Tex) Ta, Tb, Tex

    BLAS1_EX_TESTING(scal_ex, ARG3)
    BLAS1_EX_TESTING(scal_batched_ex, ARG3)
    BLAS1_EX_TESTING(scal_strided_batched_ex, ARG3)

} // namespace

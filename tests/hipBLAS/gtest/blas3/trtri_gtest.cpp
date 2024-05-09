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

#include "blas3/testing_trtri.hpp"
#include "blas3/testing_trtri_batched.hpp"
#include "blas3/testing_trtri_strided_batched.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible trtri test cases
    enum trtri_test_type
    {
        TRTRI,
        TRTRI_BATCHED,
        TRTRI_STRIDED_BATCHED,
    };

    // trtri test template
    template <template <typename...> class FILTER, trtri_test_type TRTRI_TYPE>
    struct trtri_template : HipBLAS_Test<trtri_template<FILTER, TRTRI_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<trtri_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(TRTRI_TYPE)
            {
            case TRTRI:
                return !strcmp(arg.function, "trtri") || !strcmp(arg.function, "trtri_bad_arg");
            case TRTRI_BATCHED:
                return !strcmp(arg.function, "trtri_batched")
                       || !strcmp(arg.function, "trtri_batched_bad_arg");
            case TRTRI_STRIDED_BATCHED:
                return !strcmp(arg.function, "trtri_strided_batched")
                       || !strcmp(arg.function, "trtri_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(TRTRI_TYPE == TRTRI)
                testname_trtri(arg, name);
            else if constexpr(TRTRI_TYPE == TRTRI_BATCHED)
                testname_trtri_batched(arg, name);
            else if constexpr(TRTRI_TYPE == TRTRI_STRIDED_BATCHED)
                testname_trtri_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct trtri_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct trtri_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, hipblasComplex> || std::is_same_v<T, hipblasDoubleComplex>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "trtri"))
                testing_trtri<T>(arg);
            else if(!strcmp(arg.function, "trtri_bad_arg"))
                testing_trtri_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trtri_batched"))
                testing_trtri_batched<T>(arg);
            else if(!strcmp(arg.function, "trtri_batched_bad_arg"))
                testing_trtri_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "trtri_strided_batched"))
                testing_trtri_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "trtri_strided_batched_bad_arg"))
                testing_trtri_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using trtri = trtri_template<trtri_testing, TRTRI>;
    TEST_P(trtri, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<trtri_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trtri);

    using trtri_batched = trtri_template<trtri_testing, TRTRI_BATCHED>;
    TEST_P(trtri_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<trtri_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trtri_batched);

    using trtri_strided_batched = trtri_template<trtri_testing, TRTRI_STRIDED_BATCHED>;
    TEST_P(trtri_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            hipblas_simple_dispatch<trtri_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(trtri_strided_batched);

} // namespace

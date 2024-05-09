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

#include "blas3/testing_dgmm.hpp"
#include "blas3/testing_dgmm_batched.hpp"
#include "blas3/testing_dgmm_strided_batched.hpp"
#include "hipblas_data.hpp"
#include "hipblas_test.hpp"
#include "type_dispatch.hpp"

namespace
{
    // possible dgmm test cases
    enum dgmm_test_type
    {
        DGMM,
        DGMM_BATCHED,
        DGMM_STRIDED_BATCHED,
    };

    // dgmm test template
    template <template <typename...> class FILTER, dgmm_test_type DGMM_TYPE>
    struct dgmm_template : HipBLAS_Test<dgmm_template<FILTER, DGMM_TYPE>, FILTER>
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
            return hipblas_simple_dispatch<dgmm_template::template type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            switch(DGMM_TYPE)
            {
            case DGMM:
                return !strcmp(arg.function, "dgmm") || !strcmp(arg.function, "dgmm_bad_arg");
            case DGMM_BATCHED:
                return !strcmp(arg.function, "dgmm_batched")
                       || !strcmp(arg.function, "dgmm_batched_bad_arg");
            case DGMM_STRIDED_BATCHED:
                return !strcmp(arg.function, "dgmm_strided_batched")
                       || !strcmp(arg.function, "dgmm_strided_batched_bad_arg");
            }
            return false;
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            std::string name;
            if constexpr(DGMM_TYPE == DGMM)
                testname_dgmm(arg, name);
            else if constexpr(DGMM_TYPE == DGMM_BATCHED)
                testname_dgmm_batched(arg, name);
            else if constexpr(DGMM_TYPE == DGMM_STRIDED_BATCHED)
                testname_dgmm_strided_batched(arg, name);
            return std::move(name);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct dgmm_testing : hipblas_test_invalid
    {
    };

    // When the condition in the second argument is satisfied, the type combination
    // is valid. When the condition is false, this specialization does not apply.
    template <typename T>
    struct dgmm_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                float> || std::is_same_v<T, double> || std::is_same_v<T, hipblasComplex> || std::is_same_v<T, hipblasDoubleComplex>>>
        : hipblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "dgmm"))
                testing_dgmm<T>(arg);
            else if(!strcmp(arg.function, "dgmm_bad_arg"))
                testing_dgmm_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "dgmm_batched"))
                testing_dgmm_batched<T>(arg);
            else if(!strcmp(arg.function, "dgmm_batched_bad_arg"))
                testing_dgmm_batched_bad_arg<T>(arg);
            else if(!strcmp(arg.function, "dgmm_strided_batched"))
                testing_dgmm_strided_batched<T>(arg);
            else if(!strcmp(arg.function, "dgmm_strided_batched_bad_arg"))
                testing_dgmm_strided_batched_bad_arg<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    using dgmm = dgmm_template<dgmm_testing, DGMM>;
    TEST_P(dgmm, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<dgmm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(dgmm);

    using dgmm_batched = dgmm_template<dgmm_testing, DGMM_BATCHED>;
    TEST_P(dgmm_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<dgmm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(dgmm_batched);

    using dgmm_strided_batched = dgmm_template<dgmm_testing, DGMM_STRIDED_BATCHED>;
    TEST_P(dgmm_strided_batched, blas3)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(hipblas_simple_dispatch<dgmm_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(dgmm_strided_batched);

} // namespace

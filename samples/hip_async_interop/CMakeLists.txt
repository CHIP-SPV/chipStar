# async interop
add_chip_binary(hip_async_binomial BinomialOption.cpp BinomialOptionOpenCL.cpp BinomialOptionLevel0.cpp kernel.cpp)

add_test(NAME "hip_async_binomial"
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hip_async_binomial"
         -q -e -t -x 2048 -i 32)

set_tests_properties("hip_async_binomial" PROPERTIES
                     PASS_REGULAR_EXPRESSION "Passed"
                     LABELS internal)

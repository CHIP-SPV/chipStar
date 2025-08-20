# async interop
set(SOURCES BinomialOption.cpp kernel.cpp)
set(OPTS "")

if(OpenCL_LIBRARY)
  list(APPEND SOURCES BinomialOptionOpenCL.cpp)
  list(APPEND OPTS "HAVE_OPENCL")
endif()

if(LevelZero_LIBRARY)
  list(APPEND SOURCES BinomialOptionLevel0.cpp)
  list(APPEND OPTS "HAVE_LEVEL0")
endif()

add_chip_binary(hip_async_binomial ${SOURCES})
target_compile_definitions(hip_async_binomial PRIVATE ${OPTS})

add_test(NAME "hip_async_binomial"
         COMMAND ${HIP_PROFILE_TESTS_COMMAND} ${SKIP_DOUBLE_TESTS} "${CMAKE_CURRENT_BINARY_DIR}/hip_async_binomial"
         -q -e -t -x 1024 -i 16)

set_tests_properties("hip_async_binomial" PROPERTIES
                     PASS_REGULAR_EXPRESSION "Passed"
                     SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST"
                     LABELS "internal;SAMPLE")

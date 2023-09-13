
function(add_legacy_compile_test)
  set(options)
  set(args NAME COMPILE_OPTIONS)
  set(list_args SOURCES)
  cmake_parse_arguments(
    PARSE_ARGV 0
    "TEST" # variable prefix
    "${options}"
    "${args}"
    "${list_args}")

  add_test(NAME ${TEST_NAME}
    COMMAND bash -c "${CMAKE_BINARY_DIR}/bin/cuspvc ${TEST_COMPILE_OPTIONS} ${TEST_SOURCES} 2>&1"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction()

function(add_compile_test)
  set(options)
  set(args NAME COMPILE_OPTIONS)
  set(list_args SOURCES)
  cmake_parse_arguments(
    PARSE_ARGV 0
    "TEST" # variable prefix
    "${options}"
    "${args}"
    "${list_args}")

  add_test(NAME ${TEST_NAME}
    COMMAND bash -c "${CMAKE_BINARY_DIR}/bin/cucc ${TEST_COMPILE_OPTIONS} ${TEST_SOURCES} 2>&1"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction()

add_legacy_compile_test(NAME activemask SOURCES activemask.cu
  # '-c' option is a kludge fix for C++ based hipcc. This test worked
  # without the option with the previous perl-based hipcc.
  COMPILE_OPTIONS "-fsyntax-only -c")
set_tests_properties(activemask PROPERTIES
  PASS_REGULAR_EXPRESSION "error: '__activemask' is unavailable: unsupported in chipStar")

add_compile_test(NAME cucc-hello COMPILE_OPTIONS "hello.cu -o /dev/null")

# nvcc allows and ignores '-std=c++##' for C inputs.
add_compile_test(NAME cucc-std-plus-c-input
  COMPILE_OPTIONS "-std=c++17 some.c -E -o /dev/null")

# Other nvcc specific options - check thay aren't passed to hipcc (as is).
add_compile_test(NAME cucc-nvcc-specific-options
  COMPILE_OPTIONS
  "-default-stream null -extended-lambda -use_fast_math \
  -fmad true -maxrregcount=32 --generate-code arch=compute_50,code=sm_50 \
  -expt-relaxed-constexpr --keep some.cu --keep-dir some-dir -c -o /dev/null")

add_compile_test(NAME cucc-cucc-debug-env
  COMPILE_OPTIONS "some.cu -E -o /dev/null")
set_tests_properties(cucc-cucc-debug-env PROPERTIES
  ENVIRONMENT CHIP_CUCC_VERBOSE=1
  PASS_REGULAR_EXPRESSION "Executing: .*hipcc .*")

# Check -Xcompiler passes options to the driven compiler.
add_compile_test(NAME cucc-xcompiler-opt
  COMPILE_OPTIONS
  "-Xcompiler -DFOO=123 -Xcompiler=-DBAR=321 check-macros.cu -E -o /dev/null")

# Check --version response can be user defined. A use case is for
# masquerading as another compiler (see docs/Usage.md: Configuring
# CMake-based CUDA Projects).
add_compile_test(NAME cucc-alt-version-string
  COMPILE_OPTIONS "--version")
set_tests_properties(cucc-alt-version-string PROPERTIES
  ENVIRONMENT CUCC_VERSION_STRING="My custom version string"
  PASS_REGULAR_EXPRESSION "My custom version string")

# Match nvcc's -x behavior. It applies globally unlike in other compilers.
# DISABLED: hipcc does not currently process '-x hip cuda-inside.cpp
#           cuda-inside.c' line correctly.
#add_compile_test(NAME cucc-x-behavior
#  COMPILE_OPTIONS "cuda-inside.cpp cuda-inside.c -x cu")

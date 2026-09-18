# The host code must use the configured compiler's target at compile and link time.
# Cross-built tests may run on a target without the build-host compiler.
if(NOT EXISTS "${CXX_COMPILER}")
  message(STATUS "HIP_SKIP_THIS_TEST: no configured compiler at ${CXX_COMPILER}")
  return()
endif()
set(COMPILER "${CXX_COMPILER}")
if(CXX_COMPILER_TARGET)
  list(APPEND COMPILER "--target=${CXX_COMPILER_TARGET}")
endif()
execute_process(COMMAND ${COMPILER} --print-target-triple
  OUTPUT_VARIABLE EXPECTED OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE RC ERROR_VARIABLE ERR)
if(NOT RC EQUAL 0 OR EXPECTED STREQUAL "")
  message(FATAL_ERROR "Cannot query compiler target: ${ERR}")
endif()

foreach(PHASE COMPILE LINK)
  file(STRINGS "${HIP_INFO}" OPTIONS
    REGEX "^HIP_OFFLOAD_${PHASE}_OPTIONS=")
  if(NOT OPTIONS)
    message(FATAL_ERROR "Missing ${PHASE} options in ${HIP_INFO}")
  endif()
  string(REPLACE "HIP_OFFLOAD_${PHASE}_OPTIONS=" "" OPTIONS "${OPTIONS}")
  separate_arguments(OPTIONS UNIX_COMMAND "${OPTIONS}")
  execute_process(COMMAND "${CXX_COMPILER}" ${OPTIONS} --print-target-triple
    OUTPUT_VARIABLE ACTUAL OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE RC ERROR_VARIABLE ERR)
  if(NOT RC EQUAL 0 OR NOT ACTUAL STREQUAL EXPECTED)
    message(FATAL_ERROR
      "${PHASE} target '${ACTUAL}' differs from compiler target '${EXPECTED}': ${ERR}")
  endif()
endforeach()
message(STATUS "TestFix1635HostTarget passed: ${EXPECTED}")

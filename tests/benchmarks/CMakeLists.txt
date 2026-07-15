set_source_files_properties(manySmallKernels.hip PROPERTIES LANGUAGE CXX)
add_executable(manySmallKernels manySmallKernels.hip)
set_target_properties(manySmallKernels PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(manySmallKernels CHIP deviceInternal)
target_include_directories(manySmallKernels
  PRIVATE $<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>)
add_test(NAME manySmallKernels COMMAND ${HIP_PROFILE_TESTS_COMMAND} ${SKIP_DOUBLE_TESTS} ${CMAKE_CURRENT_BINARY_DIR}/manySmallKernels)

add_executable(manySmallKernelsOnlyLaunch manySmallKernels.hip)
set_target_properties(manySmallKernelsOnlyLaunch PROPERTIES CXX_STANDARD_REQUIRED ON)
target_compile_definitions(manySmallKernelsOnlyLaunch PRIVATE CHECK_ONLY_LAUNCH)
target_link_libraries(manySmallKernelsOnlyLaunch CHIP deviceInternal)
target_include_directories(manySmallKernelsOnlyLaunch
  PRIVATE $<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>)
add_test(NAME manySmallKernelsOnlyLaunch COMMAND ${HIP_PROFILE_TESTS_COMMAND} ${SKIP_DOUBLE_TESTS} ${CMAKE_CURRENT_BINARY_DIR}/manySmallKernelsOnlyLaunch)

set_source_files_properties(manyMallocMemcopies.hip PROPERTIES LANGUAGE CXX)
add_executable(manyMallocMemcopies manyMallocMemcopies.hip)
set_target_properties(manyMallocMemcopies PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(manyMallocMemcopies CHIP deviceInternal)
target_include_directories(manyMallocMemcopies
  PRIVATE $<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>)
add_test(NAME manyMallocMemcopies COMMAND ${HIP_PROFILE_TESTS_COMMAND} ${SKIP_DOUBLE_TESTS} ${CMAKE_CURRENT_BINARY_DIR}/manyMallocMemcopies)

set_source_files_properties(hostRegisterOverhead.hip PROPERTIES LANGUAGE CXX)
add_executable(hostRegisterOverhead hostRegisterOverhead.hip)
set_target_properties(hostRegisterOverhead PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(hostRegisterOverhead CHIP deviceInternal)
target_include_directories(hostRegisterOverhead
  PRIVATE $<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>)
add_test(NAME hostRegisterOverhead COMMAND ${HIP_PROFILE_TESTS_COMMAND} ${SKIP_DOUBLE_TESTS} ${CMAKE_CURRENT_BINARY_DIR}/hostRegisterOverhead)

set_source_files_properties(hostRegisterMemcpyOverhead.hip PROPERTIES LANGUAGE CXX)
add_executable(hostRegisterMemcpyOverhead hostRegisterMemcpyOverhead.hip)
set_target_properties(hostRegisterMemcpyOverhead PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(hostRegisterMemcpyOverhead CHIP deviceInternal)
target_include_directories(hostRegisterMemcpyOverhead
  PRIVATE $<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>)
add_test(NAME hostRegisterMemcpyOverhead COMMAND ${HIP_PROFILE_TESTS_COMMAND} ${SKIP_DOUBLE_TESTS} ${CMAKE_CURRENT_BINARY_DIR}/hostRegisterMemcpyOverhead)

# These are timing benchmarks: they measure per-op latency and per-launch
# overhead in nanoseconds. check.py runs the suite with `ctest -j N`, so without
# this every benchmark would contend for the single GPU with whatever other
# tests happen to run alongside it -- corrupting its timings (e.g. a memcpy
# appearing slower WITH hipHostRegister than without, purely from contention).
# RUN_SERIAL makes ctest run each of these alone, giving clean measurements.
set_tests_properties(
  manySmallKernels
  manySmallKernelsOnlyLaunch
  manyMallocMemcopies
  hostRegisterOverhead
  hostRegisterMemcpyOverhead
  PROPERTIES RUN_SERIAL ON)

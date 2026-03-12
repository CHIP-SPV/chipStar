set_source_files_properties(manySmallKernels.hip PROPERTIES LANGUAGE CXX)
add_executable(manySmallKernels manySmallKernels.hip)
set_target_properties(manySmallKernels PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(manySmallKernels CHIP deviceInternal)
target_include_directories(manySmallKernels
  PRIVATE $<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>)

set_source_files_properties(timer_bench.hip PROPERTIES LANGUAGE CXX)
add_executable(timer_bench timer_bench.hip)
target_link_libraries(timer_bench CHIP deviceInternal)
target_include_directories(timer_bench PRIVATE $<TARGET_PROPERTY:CHIP,INCLUDE_DIRECTORIES>)

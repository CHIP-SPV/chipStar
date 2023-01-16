set(SOURCE shuffles.cc)
set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)
add_executable(shuffles ${SOURCE})
target_link_libraries(shuffles PUBLIC CHIP deviceInternal)
set_target_properties(shuffles PROPERTIES CXX_STANDARD_REQUIRED ON)
target_include_directories(shuffles PUBLIC
  ${CMAKE_SOURCE_DIR}/HIP/include ${CMAKE_SOURCE_DIR}/include)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run-shuffles.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/run-shuffles.cmake
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/shuffles.xstdout
  ${CMAKE_CURRENT_BINARY_DIR}/shuffles.xstdout
  COPYONLY)
add_test(
  NAME shuffles
  COMMAND "${CMAKE_COMMAND}" -P ${CMAKE_CURRENT_BINARY_DIR}/run-shuffles.cmake)

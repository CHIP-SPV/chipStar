set(SOURCE abort.cc)
set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)
add_executable(abort ${SOURCE})
target_compile_options(abort PRIVATE ${OFFLOAD_ARCH_STR})
set_target_properties(abort PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(abort CHIP)
target_include_directories(abort PUBLIC
  ${CMAKE_SOURCE_DIR}/HIP/include ${CMAKE_SOURCE_DIR}/include)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run-abort.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/run-abort.cmake
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/abort.xstdout
  ${CMAKE_CURRENT_BINARY_DIR}/abort.xstdout
  COPYONLY)
add_test(
  NAME abort
  COMMAND "${CMAKE_COMMAND}" -P ${CMAKE_CURRENT_BINARY_DIR}/run-abort.cmake)

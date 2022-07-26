set(SOURCE abort.cc)
set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)
add_executable(abort ${SOURCE})
target_link_libraries(abort PUBLIC CHIP device)
set_target_properties(abort PROPERTIES CXX_STANDARD_REQUIRED ON)
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


set(SOURCE abort2.cc)
set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)
add_executable(abort2 ${SOURCE})
target_link_libraries(abort2 PUBLIC CHIP device)
set_target_properties(abort2 PROPERTIES CXX_STANDARD_REQUIRED ON)
target_include_directories(abort2 PUBLIC
  ${CMAKE_SOURCE_DIR}/HIP/include ${CMAKE_SOURCE_DIR}/include)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run-abort2.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/run-abort2.cmake
  COPYONLY)
add_test(
  NAME abort2
  COMMAND "${CMAKE_COMMAND}" -P ${CMAKE_CURRENT_BINARY_DIR}/run-abort2.cmake)


#add_chip_test(abort2 abort2 PASSED abort2.cc)

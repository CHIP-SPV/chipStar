add_custom_command(
  OUTPUT saxpy.o DEPENDS ${CMAKE_SOURCE_DIR}/samples/ccompat/saxpy.hip
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc
  ${CMAKE_SOURCE_DIR}/samples/ccompat/saxpy.hip -c -g -pthread
  -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/HIP/include)
add_custom_command(
  OUTPUT ccompat.o DEPENDS ${CMAKE_SOURCE_DIR}/samples/ccompat/ccompat.c
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc
  ${CMAKE_SOURCE_DIR}/samples/ccompat/ccompat.c -c -g -pthread
  -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/HIP/include)

add_custom_target(
  ccompat ALL DEPENDS saxpy.o ccompat.o
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc ./ccompat.o ./saxpy.o -o ccompat
  -g -pthread)

add_dependencies(ccompat hipcc.bin CHIP)
add_test(NAME ccompat COMMAND ${CMAKE_BINARY_DIR}/samples/ccompat/ccompat)
